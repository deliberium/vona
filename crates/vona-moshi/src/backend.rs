//! [`MoshiBackend`] — a [`SpeechToSpeechBackend`] that connects to a running
//! Kyutai Moshi inference server over WebSocket.
//!
//! # Architecture
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────┐
//!  │  MoshiSession                                        │
//!  │                                                      │
//!  │  step() ──┐                                          │
//!  │           │ PCM f32 (any rate/ch)                   │
//!  │           ▼                                          │
//!  │    mix_to_mono + resample_to_24k                     │
//!  │           │                                          │
//!  │           ▼                                          │
//!  │    Opus encode (960-sample frames)                   │
//!  │           │                                          │
//!  │           ▼                                          │
//!  │    OGG mux  ──► WS MT=1 ──────────────► Moshi server│
//!  │                                                      │
//!  │  Moshi server ──► WS MT=1 ──► (background tasks)    │
//!  │                      │                              │
//!  │                      ▼                              │
//!  │              OGG demux + Opus decode                │
//!  │                      │                              │
//!  │                      ▼  mpsc                        │
//!  │             audio_rx channel ──► AudioOutputFrame   │
//!  │                                                      │
//!  │  Moshi server ──► WS MT=2 ──► text_rx ──► Transcript│
//!  └──────────────────────────────────────────────────────┘
//! ```

use crate::config::MoshiConfig;
use crate::proto::{ctrl, mt, OGG_SERIAL, MOSHI_SAMPLE_RATE, OPUS_FRAME_SAMPLES};
use async_trait::async_trait;
use byteorder::WriteBytesExt;
use futures_util::{SinkExt, StreamExt};
use ogg::PacketWriteEndInfo;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};
use tokio::io::AsyncWriteExt as _;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use vona::{
    AudioInputFrame, AudioOutputFrame, BackendCapabilities, BackendError, BackendStep,
    ControlEvent, ExternalContextEvent, SessionConfig, SpeechToSpeechBackend,
};

// ── WebSocket type aliases ──────────────────────────────────────────────────

type WsStream = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;
type WsSink = futures_util::stream::SplitSink<WsStream, Message>;
type SharedWsTx = Arc<tokio::sync::Mutex<WsSink>>;

// ── Public structs ──────────────────────────────────────────────────────────

/// Connects to a Kyutai Moshi inference server and wraps it as a
/// [`SpeechToSpeechBackend`].
///
/// # Example
/// ```rust,no_run
/// use vona_moshi::{MoshiBackend, MoshiConfig};
///
/// let backend = MoshiBackend::new(MoshiConfig {
///     url: "wss://127.0.0.1:8998/api/chat".into(),
///     accept_invalid_certs: true, // only for local dev!
/// });
/// ```
pub struct MoshiBackend {
    config: MoshiConfig,
}

impl MoshiBackend {
    pub fn new(config: MoshiConfig) -> Self {
        Self { config }
    }
}

/// Live state for a single Moshi conversation session.
pub struct MoshiSession {
    /// Shared handle to the WebSocket sender.
    ws_tx: SharedWsTx,
    /// Opus encoder wrapped in a Mutex to satisfy `Sync` (only accessed
    /// in `step()` which holds `&mut self`).
    encoder: Mutex<opus::Encoder>,
    /// Accumulates partial frames until we have `OPUS_FRAME_SAMPLES` samples.
    pcm_pending: VecDeque<f32>,
    /// Running count of samples sent (used for OGG granule positions).
    total_samples: u64,
    /// OGG muxer for outgoing audio; inner `Vec<u8>` is drained after each send.
    ogg_writer: ogg::PacketWriter<'static, Vec<u8>>,
    /// Reusable buffer for the Opus encoder output.
    encode_buf: Vec<u8>,

    /// Decoded PCM chunks from the receive pipeline.
    audio_rx: mpsc::Receiver<Vec<f32>>,
    /// Transcript / inner-monologue text from the server.
    text_rx: mpsc::Receiver<String>,
    /// Monotonically increasing counter for [`AudioOutputFrame::sequence`].
    output_seq: u64,
    /// WS receive + OGG/Opus decode tasks.  Aborted in [`end_session`].
    recv_task: tokio::task::JoinHandle<()>,
    decode_task: tokio::task::JoinHandle<()>,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Write a valid OGG OpusHead identification header.
/// Spec: <https://wiki.xiph.org/OggOpus#ID_Header>
fn write_opus_head<W: Write>(w: &mut W) -> std::io::Result<()> {
    w.write_all(b"OpusHead")?;
    w.write_u8(1)?; // version
    w.write_u8(1)?; // channel count (mono)
    w.write_u16::<byteorder::LittleEndian>(3840)?; // pre-skip
    w.write_u32::<byteorder::LittleEndian>(48_000)?; // original sample rate hint
    w.write_i16::<byteorder::LittleEndian>(0)?; // output gain Q7.8
    w.write_u8(0)?; // channel map (RTP mapping)
    Ok(())
}

/// Write a minimal OGG OpusTags comment header.
fn write_opus_tags<W: Write>(w: &mut W) -> std::io::Result<()> {
    let vendor = b"VonaRS/MoshiBackend";
    w.write_all(b"OpusTags")?;
    w.write_u32::<byteorder::LittleEndian>(vendor.len() as u32)?;
    w.write_all(vendor)?;
    w.write_u32::<byteorder::LittleEndian>(0)?; // no user comments
    Ok(())
}

/// Mix a multi-channel PCM buffer down to mono by averaging channels.
fn mix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels as usize)
        .map(|c| c.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Linear-interpolation resample a mono PCM buffer from `src_hz` to `dst_hz`.
fn resample_mono(samples: &[f32], src_hz: u32, dst_hz: u32) -> Vec<f32> {
    if src_hz == dst_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len =
        (((samples.len() as f64) / src_hz as f64 * dst_hz as f64).round() as usize).max(1);
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 * src_hz as f64 / dst_hz as f64;
            let idx = src_pos as usize;
            let frac = (src_pos - idx as f64) as f32;
            let a = samples.get(idx).copied().unwrap_or(0.0);
            let b = samples.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

/// Prepend the MT byte to OGG data and wrap in a WebSocket binary message.
#[inline]
fn ogg_ws_message(data: &[u8]) -> Message {
    let mut msg = Vec::with_capacity(1 + data.len());
    msg.push(mt::AUDIO);
    msg.extend_from_slice(data);
    Message::Binary(msg)
}

// ── SpeechToSpeechBackend impl ───────────────────────────────────────────────

#[async_trait]
impl SpeechToSpeechBackend for MoshiBackend {
    type Session = MoshiSession;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_full_duplex: true,
            supports_control_stream: false,
            supports_context_injection: false,
            supports_pause_resume: false,
            supports_style_conditioning: false,
            supports_word_timestamps: false,
        }
    }

    async fn start_session(&self, _config: SessionConfig) -> Result<MoshiSession, BackendError> {
        // ── Connect WebSocket ──────────────────────────────────────────────
        let ws_stream = if self.config.accept_invalid_certs {
            let tls = native_tls::TlsConnector::builder()
                .danger_accept_invalid_certs(true)
                .build()
                .map_err(|e| BackendError::Start(format!("TLS connector build failed: {e}")))?;
            let connector = tokio_tungstenite::Connector::NativeTls(tls);
            let (stream, _) = tokio_tungstenite::connect_async_tls_with_config(
                self.config.url.as_str(),
                None,
                false,
                Some(connector),
            )
            .await
            .map_err(|e| BackendError::Start(format!("WebSocket connect failed: {e}")))?;
            stream
        } else {
            let (stream, _) = tokio_tungstenite::connect_async(self.config.url.as_str())
                .await
                .map_err(|e| BackendError::Start(format!("WebSocket connect failed: {e}")))?;
            stream
        };

        let (ws_sink, mut ws_source) = ws_stream.split();
        let ws_tx: SharedWsTx = Arc::new(tokio::sync::Mutex::new(ws_sink));

        // ── Handshake: wait for server MT=0, then reply ────────────────────
        loop {
            match ws_source.next().await {
                None => {
                    return Err(BackendError::Start(
                        "server closed connection before handshake".into(),
                    ));
                }
                Some(Err(e)) => {
                    return Err(BackendError::Start(format!("WS error during handshake: {e}")));
                }
                Some(Ok(Message::Binary(bin)))
                    if !bin.is_empty() && bin[0] == mt::HANDSHAKE =>
                {
                    // Respond with our own handshake
                    let mut reply = vec![mt::HANDSHAKE];
                    reply.extend_from_slice(&0u32.to_le_bytes()); // protocol version
                    reply.extend_from_slice(&0u32.to_le_bytes()); // model version
                    ws_tx
                        .lock()
                        .await
                        .send(Message::Binary(reply))
                        .await
                        .map_err(|e| {
                            BackendError::Start(format!("handshake send failed: {e}"))
                        })?;
                    break;
                }
                // Ignore pings / text before handshake completes
                Some(Ok(_)) => continue,
            }
        }

        tracing::info!(url = %self.config.url, "Moshi handshake complete");

        // ── Background tasks ───────────────────────────────────────────────
        let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(128);
        let (text_tx, text_rx) = mpsc::channel::<String>(64);

        // Duplex pipe: recv_task writes OGG bytes, decode_task reads them.
        let (mut ogg_write_half, ogg_read_half) = tokio::io::duplex(512 * 1024);

        // Task 1: WebSocket → dispatch incoming messages
        let recv_task = tokio::spawn(async move {
            while let Some(msg) = ws_source.next().await {
                match msg {
                    Ok(Message::Binary(bin)) if !bin.is_empty() => {
                        match bin[0] {
                            mt::AUDIO => {
                                if ogg_write_half.write_all(&bin[1..]).await.is_err() {
                                    break; // decode_task exited; shutdown
                                }
                            }
                            mt::TEXT => {
                                let text = String::from_utf8_lossy(&bin[1..]).into_owned();
                                let _ = text_tx.send(text).await;
                            }
                            mt::ERROR => {
                                let msg = String::from_utf8_lossy(&bin[1..]);
                                tracing::error!(moshi_error = %msg, "Moshi server error");
                            }
                            _ => {}
                        }
                    }
                    Ok(Message::Close(_)) | Err(_) => break,
                    _ => {}
                }
            }
        });

        // Task 2: OGG/Opus decode → PCM
        let decode_task = tokio::spawn(async move {
            let mut pr = ogg::reading::async_api::PacketReader::new(ogg_read_half);
            let decoder = opus::Decoder::new(MOSHI_SAMPLE_RATE, opus::Channels::Mono);
            let mut decoder = match decoder {
                Ok(d) => d,
                Err(e) => {
                    tracing::error!("Moshi: Opus decoder init failed: {e}");
                    return;
                }
            };
            // Large buffer: 10 s of f32 at 24 kHz
            let mut pcm_buf = vec![0f32; MOSHI_SAMPLE_RATE as usize * 10];

            while let Some(packet_result) = pr.next().await {
                match packet_result {
                    Err(e) => {
                        tracing::warn!("Moshi: OGG read error: {e}");
                        break;
                    }
                    Ok(packet) => {
                        // Skip OGG header pages
                        if packet.data.starts_with(b"OpusHead")
                            || packet.data.starts_with(b"OpusTags")
                        {
                            continue;
                        }
                        match decoder.decode_float(&packet.data, &mut pcm_buf, false) {
                            Ok(size) if size > 0 => {
                                if audio_tx.send(pcm_buf[..size].to_vec()).await.is_err() {
                                    break; // receiver dropped
                                }
                            }
                            Ok(_) => {}
                            Err(e) => tracing::warn!("Moshi: Opus decode error: {e}"),
                        }
                    }
                }
            }
        });

        // ── Encoder + OGG muxer (outgoing path) ────────────────────────────
        let encoder =
            opus::Encoder::new(MOSHI_SAMPLE_RATE, opus::Channels::Mono, opus::Application::Voip)
                .map_err(|e| BackendError::Start(format!("Opus encoder init: {e}")))?;

        let mut ogg_writer = ogg::PacketWriter::new(Vec::new());

        // Write and immediately flush OGG identification headers to the server
        let mut head_buf = Vec::new();
        write_opus_head(&mut head_buf)
            .map_err(|e| BackendError::Start(format!("OpusHead write: {e}")))?;
        ogg_writer
            .write_packet(head_buf, OGG_SERIAL, PacketWriteEndInfo::EndPage, 0)
            .map_err(|e| BackendError::Start(format!("OGG head packet: {e}")))?;

        let mut tags_buf = Vec::new();
        write_opus_tags(&mut tags_buf)
            .map_err(|e| BackendError::Start(format!("OpusTags write: {e}")))?;
        ogg_writer
            .write_packet(tags_buf, OGG_SERIAL, PacketWriteEndInfo::EndPage, 0)
            .map_err(|e| BackendError::Start(format!("OGG tags packet: {e}")))?;

        {
            let data = ogg_writer.inner_mut();
            if !data.is_empty() {
                let msg = ogg_ws_message(data);
                ws_tx
                    .lock()
                    .await
                    .send(msg)
                    .await
                    .map_err(|e| BackendError::Start(format!("OGG header send: {e}")))?;
                data.clear();
            }
        }

        Ok(MoshiSession {
            ws_tx,
            encoder: Mutex::new(encoder),
            pcm_pending: VecDeque::new(),
            total_samples: 0,
            ogg_writer,
            encode_buf: vec![0u8; 8192],
            audio_rx,
            text_rx,
            output_seq: 0,
            recv_task,
            decode_task,
        })
    }

    async fn step(
        &self,
        session: &mut MoshiSession,
        input: AudioInputFrame,
    ) -> Result<BackendStep, BackendError> {
        // ── Normalise input to 24 kHz mono ────────────────────────────────
        let mono = mix_to_mono(&input.samples, input.channels);
        let resampled = resample_mono(&mono, input.sample_rate_hz, MOSHI_SAMPLE_RATE);
        session.pcm_pending.extend(resampled.iter().copied());

        // ── Encode and send complete 960-sample Opus frames ───────────────
        while session.pcm_pending.len() >= OPUS_FRAME_SAMPLES {
            let chunk: Vec<f32> = session.pcm_pending.drain(..OPUS_FRAME_SAMPLES).collect();
            session.total_samples += OPUS_FRAME_SAMPLES as u64;

            let size = {
                let enc = &session.encoder;
                let buf = &mut session.encode_buf;
                enc.lock().unwrap()
                    .encode_float(&chunk, buf)
                    .map_err(|e| BackendError::Step(format!("Opus encode: {e}")))?            
            };

            if size > 0 {
                let packet = session.encode_buf[..size].to_vec();
                session
                    .ogg_writer
                    .write_packet(
                        packet,
                        OGG_SERIAL,
                        PacketWriteEndInfo::EndPage,
                        session.total_samples,
                    )
                    .map_err(|e| BackendError::Step(format!("OGG write: {e}")))?;

                let data = session.ogg_writer.inner_mut();
                if !data.is_empty() {
                    let msg = ogg_ws_message(data);
                    session
                        .ws_tx
                        .lock()
                        .await
                        .send(msg)
                        .await
                        .map_err(|e| BackendError::Step(format!("WS send: {e}")))?;
                    data.clear();
                }
            }
        }

        // ── Collect decoded output audio (non-blocking) ───────────────────
        let mut output_audio = Vec::new();
        while let Ok(pcm) = session.audio_rx.try_recv() {
            let seq = session.output_seq;
            session.output_seq += 1;
            output_audio.push(AudioOutputFrame {
                sequence: seq,
                sample_rate_hz: MOSHI_SAMPLE_RATE,
                channels: 1,
                samples: pcm,
                is_filler: false,
            });
        }

        // ── Collect transcript fragments (inner monologue) ─────────────────
        let mut control_events = Vec::new();
        while let Ok(text) = session.text_rx.try_recv() {
            control_events.push(ControlEvent::TranscriptFragment {
                text,
                final_fragment: false,
            });
        }

        Ok(BackendStep {
            output_audio,
            control_events,
            transcript: None,
            finished: false,
            debug_payload: None,
        })
    }

    async fn inject_event(
        &self,
        _session: &mut MoshiSession,
        _event: ExternalContextEvent,
    ) -> Result<(), BackendError> {
        // Moshi does not support external context injection.
        Ok(())
    }

    async fn end_session(&self, session: MoshiSession) -> Result<(), BackendError> {
        // Tell the server the turn is over, then close.
        let end_turn = vec![mt::CONTROL, ctrl::END_TURN];
        let mut tx = session.ws_tx.lock().await;
        let _ = tx.send(Message::Binary(end_turn)).await;
        let _ = tx.send(Message::Close(None)).await;
        drop(tx);

        // Stop background tasks immediately.
        session.recv_task.abort();
        session.decode_task.abort();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MoshiConfig;

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn backend_new_stores_config() {
        let cfg = MoshiConfig {
            url: "wss://test.local/api/chat".into(),
            accept_invalid_certs: true,
        };
        let backend = MoshiBackend::new(cfg.clone());
        assert_eq!(backend.config.url, cfg.url);
        assert_eq!(backend.config.accept_invalid_certs, cfg.accept_invalid_certs);
    }

    // ── capabilities() — no network required ────────────────────────────────

    #[test]
    fn backend_capabilities_full_duplex() {
        let backend = MoshiBackend::new(MoshiConfig::default());
        let caps = backend.capabilities();
        assert!(caps.supports_full_duplex, "Moshi is a full-duplex backend");
    }

    #[test]
    fn backend_capabilities_no_context_injection() {
        let backend = MoshiBackend::new(MoshiConfig::default());
        let caps = backend.capabilities();
        assert!(
            !caps.supports_context_injection,
            "Moshi does not support external context injection"
        );
    }

    #[test]
    fn backend_capabilities_no_style_conditioning() {
        let backend = MoshiBackend::new(MoshiConfig::default());
        let caps = backend.capabilities();
        assert!(!caps.supports_style_conditioning);
    }

    // ── start_session fails gracefully when server is unreachable ────────────

    #[tokio::test]
    async fn start_session_fails_when_server_unreachable() {
        let backend = MoshiBackend::new(MoshiConfig {
            // Use a port that is guaranteed to refuse connections.
            url: "ws://127.0.0.1:19998/api/chat".into(),
            accept_invalid_certs: false,
        });
        let result = backend.start_session(SessionConfig::default()).await;
        assert!(
            result.is_err(),
            "expected BackendError when Moshi server is not reachable"
        );
        if let Err(BackendError::Start(msg)) = result {
            assert!(
                !msg.is_empty(),
                "error message should describe the failure"
            );
        }
    }
}
