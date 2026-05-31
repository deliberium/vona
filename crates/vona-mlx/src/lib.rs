use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{path::PathBuf, sync::Arc};
use thiserror::Error;
use vona_core::{
    AudioInputFrame, AudioOutputFrame, AudioProcessingError, AudioStreamingTranscriber,
    AudioSynthesisConfig, AudioSynthesizer, AudioTranscriber, BackendCapabilities, BackendError,
    BackendStep, ControlEvent, ExternalContextEvent, SessionConfig, SpeechToSpeechBackend,
    StreamingTranscriptKind, StreamingTranscriptUpdate, StreamingTranscriptionConfig,
    StreamingTranscriptionSession,
};

#[cfg(feature = "native-mlx")]
pub type MlxArray = mlx_rs::Array;

#[cfg(not(feature = "native-mlx"))]
#[derive(Debug, Clone)]
pub struct MlxArray {
    samples: Vec<f32>,
    shape: Vec<i32>,
}

#[cfg(not(feature = "native-mlx"))]
impl MlxArray {
    fn from_samples(samples: &[f32]) -> Result<Self, MlxAudioError> {
        let len = i32::try_from(samples.len()).map_err(|_| {
            MlxAudioError::InvalidInput("audio frame is too large for mlx shape".to_string())
        })?;
        Ok(Self {
            samples: samples.to_vec(),
            shape: vec![len],
        })
    }

    pub fn eval(&self) -> Result<(), MlxAudioError> {
        Ok(())
    }

    pub fn as_slice<T>(&self) -> &[f32] {
        let _ = std::marker::PhantomData::<T>;
        &self.samples
    }

    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
}

pub const DEFAULT_STT_MODEL_ID: &str = "distil-whisper/distil-large-v3";
pub const DEFAULT_TTS_MODEL_ID: &str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16";
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 24_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxAudioConfig {
    pub stt_model_id: String,
    pub tts_model_id: String,
    pub output_sample_rate_hz: u32,
}

impl Default for MlxAudioConfig {
    fn default() -> Self {
        Self {
            stt_model_id: DEFAULT_STT_MODEL_ID.to_string(),
            tts_model_id: DEFAULT_TTS_MODEL_ID.to_string(),
            output_sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
        }
    }
}

impl MlxAudioConfig {
    pub fn from_env() -> Self {
        Self {
            stt_model_id: std::env::var("VONA_MLX_STT_MODEL")
                .unwrap_or_else(|_| DEFAULT_STT_MODEL_ID.to_string()),
            tts_model_id: std::env::var("VONA_MLX_TTS_MODEL")
                .unwrap_or_else(|_| DEFAULT_TTS_MODEL_ID.to_string()),
            output_sample_rate_hz: std::env::var("VONA_MLX_OUTPUT_SAMPLE_RATE")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_SAMPLE_RATE_HZ),
        }
    }
}

pub trait MlxSpeechModel: Send + Sync {
    fn transcribe(&self, audio: &MlxArray, sample_rate_hz: u32) -> Result<String, MlxAudioError>;
    fn synthesize(&self, text: &str, sample_rate_hz: u32) -> Result<MlxArray, MlxAudioError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxModelKind {
    Speech,
    WhisperSpeech,
    Qwen3TtsSpeech,
    TransformerText,
    Qwen3NextText,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxModelLoadRequest {
    pub model_id: String,
    pub local_path: Option<PathBuf>,
    pub kind: MlxModelKind,
}

impl MlxModelLoadRequest {
    pub fn local(
        kind: MlxModelKind,
        model_id: impl Into<String>,
        local_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            local_path: Some(local_path.into()),
            kind,
        }
    }
}

pub enum LoadedMlxModel {
    Speech(Arc<dyn MlxSpeechModel>),
    #[cfg(feature = "mlx-models-loader")]
    TransformerText {
        model: mlx_models::Model,
        tokenizer: tokenizers::Tokenizer,
    },
    #[cfg(feature = "mlx-models-loader")]
    Qwen3NextText {
        model: mlx_models::Qwen3NextCausalLM,
        tokenizer: tokenizers::Tokenizer,
    },
}

pub trait MlxModelLoader: Send + Sync {
    fn load_model(&self, request: MlxModelLoadRequest) -> Result<LoadedMlxModel, MlxAudioError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MlxModelsLoader;

impl MlxModelLoader for MlxModelsLoader {
    fn load_model(&self, request: MlxModelLoadRequest) -> Result<LoadedMlxModel, MlxAudioError> {
        load_with_mlx_models(request)
    }
}

#[derive(Clone)]
pub struct MlxAudioEngine {
    config: MlxAudioConfig,
    device_label: String,
    model: Arc<dyn MlxSpeechModel>,
}

impl MlxAudioEngine {
    pub fn init() -> Result<Self, MlxAudioError> {
        Self::with_model(MlxAudioConfig::default(), Arc::new(UnloadedMlxSpeechModel))
    }

    pub fn from_env() -> Result<Self, MlxAudioError> {
        Self::with_model(MlxAudioConfig::from_env(), Arc::new(UnloadedMlxSpeechModel))
    }

    pub fn with_model(
        config: MlxAudioConfig,
        model: Arc<dyn MlxSpeechModel>,
    ) -> Result<Self, MlxAudioError> {
        let device_label = assert_mlx_gpu_available()?;
        Ok(Self {
            config,
            device_label,
            model,
        })
    }

    pub fn with_loader(
        config: MlxAudioConfig,
        loader: &dyn MlxModelLoader,
        request: MlxModelLoadRequest,
    ) -> Result<Self, MlxAudioError> {
        match loader.load_model(request)? {
            LoadedMlxModel::Speech(model) => Self::with_model(config, model),
            #[cfg(feature = "mlx-models-loader")]
            LoadedMlxModel::TransformerText { .. } | LoadedMlxModel::Qwen3NextText { .. } => {
                Err(MlxAudioError::ModelUnavailable(
                    "loaded mlx-models text model cannot satisfy Vona speech traits".to_string(),
                ))
            }
        }
    }

    pub fn config(&self) -> &MlxAudioConfig {
        &self.config
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }

    pub fn audio_array_from_frame(frame: &AudioInputFrame) -> Result<MlxArray, MlxAudioError> {
        #[cfg(feature = "native-mlx")]
        {
            let len = i32::try_from(frame.samples.len()).map_err(|_| {
                MlxAudioError::InvalidInput("audio frame is too large for mlx shape".to_string())
            })?;
            return Ok(MlxArray::from_slice(&frame.samples, &[len]));
        }

        #[cfg(not(feature = "native-mlx"))]
        {
            MlxArray::from_samples(&frame.samples)
        }
    }
}

#[cfg(feature = "mlx-models-loader")]
fn load_with_mlx_models(request: MlxModelLoadRequest) -> Result<LoadedMlxModel, MlxAudioError> {
    let local_path = request.local_path.ok_or_else(|| {
        MlxAudioError::InvalidInput(
            "mlx-models loader currently requires a local model directory".to_string(),
        )
    })?;

    match request.kind {
        MlxModelKind::Speech | MlxModelKind::WhisperSpeech | MlxModelKind::Qwen3TtsSpeech => {
            Err(MlxAudioError::ModelUnavailable(
                "mlx-models 0.1.x does not expose speech model loaders; use a Vona speech-model crate"
                    .to_string(),
            ))
        }
        MlxModelKind::TransformerText => {
            let model = mlx_models::transformer::load_model(&local_path)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
            let tokenizer = mlx_models::load_tokenizer(&local_path)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
            Ok(LoadedMlxModel::TransformerText { model, tokenizer })
        }
        MlxModelKind::Qwen3NextText => {
            let model = mlx_models::qwen3_next::load_qwen3_next_model(&local_path)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
            let tokenizer = mlx_models::load_tokenizer(&local_path)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
            Ok(LoadedMlxModel::Qwen3NextText { model, tokenizer })
        }
    }
}

#[cfg(not(feature = "mlx-models-loader"))]
fn load_with_mlx_models(_request: MlxModelLoadRequest) -> Result<LoadedMlxModel, MlxAudioError> {
    Err(MlxAudioError::Runtime(
        "enable the mlx-models-loader feature to use mlx-models loading".to_string(),
    ))
}

#[cfg(feature = "native-mlx")]
fn assert_mlx_gpu_available() -> Result<String, MlxAudioError> {
    use mlx_rs::{Array, Device};
    use std::panic::AssertUnwindSafe;

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let device = Device::gpu();
        Device::set_default(&device);
        let probe = Array::from_slice(&[0.0_f32], &[1]);
        probe.eval()?;
        mlx_rs::error::Result::Ok(format!("{device}"))
    }))
    .map_err(|_| MlxAudioError::Runtime("MLX GPU initialization panicked".to_string()))?;

    result.map_err(|error| MlxAudioError::Runtime(error.to_string()))
}

#[cfg(not(feature = "native-mlx"))]
fn assert_mlx_gpu_available() -> Result<String, MlxAudioError> {
    Ok("native-mlx feature disabled".to_string())
}

#[cfg(feature = "native-mlx")]
fn eval_array(array: &MlxArray) -> Result<(), MlxAudioError> {
    array
        .eval()
        .map_err(|error| MlxAudioError::Inference(error.to_string()))
}

#[cfg(not(feature = "native-mlx"))]
fn eval_array(array: &MlxArray) -> Result<(), MlxAudioError> {
    array.eval()
}

fn samples_from_array(array: &MlxArray) -> Vec<f32> {
    array.as_slice::<f32>().to_vec()
}

#[cfg(test)]
fn test_array_from_samples(samples: &[f32]) -> MlxArray {
    #[cfg(feature = "native-mlx")]
    {
        let len = i32::try_from(samples.len()).unwrap();
        return MlxArray::from_slice(samples, &[len]);
    }

    #[cfg(not(feature = "native-mlx"))]
    {
        MlxArray::from_samples(samples).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct MlxAudioSession {
    pub config: SessionConfig,
    pub pending_events: Vec<ExternalContextEvent>,
}

pub struct MlxStreamingTranscriptionSession {
    engine: MlxAudioEngine,
    config: StreamingTranscriptionConfig,
    pcm_buffer: Vec<f32>,
    last_inference_samples: usize,
    recent_hypotheses: Vec<String>,
    committed_prefix: String,
    latest_update: Option<StreamingTranscriptUpdate>,
    pending_decode: Option<tokio::task::JoinHandle<Result<String, AudioProcessingError>>>,
    pending_decode_samples: usize,
}

#[derive(Debug, Error)]
pub enum MlxAudioError {
    #[error("MLX runtime is unavailable: {0}")]
    Runtime(String),
    #[error("MLX model is not loaded: {0}")]
    ModelUnavailable(String),
    #[error("MLX input is invalid: {0}")]
    InvalidInput(String),
    #[error("MLX inference failed: {0}")]
    Inference(String),
}

impl From<MlxAudioError> for BackendError {
    fn from(value: MlxAudioError) -> Self {
        match value {
            MlxAudioError::Runtime(message) | MlxAudioError::ModelUnavailable(message) => {
                BackendError::Start(message)
            }
            MlxAudioError::InvalidInput(message) | MlxAudioError::Inference(message) => {
                BackendError::Step(message)
            }
        }
    }
}

impl From<MlxAudioError> for AudioProcessingError {
    fn from(value: MlxAudioError) -> Self {
        match value {
            MlxAudioError::Runtime(message) => AudioProcessingError::Runtime(message),
            MlxAudioError::ModelUnavailable(message) => {
                AudioProcessingError::ModelUnavailable(message)
            }
            MlxAudioError::InvalidInput(message) => AudioProcessingError::InvalidInput(message),
            MlxAudioError::Inference(message) => AudioProcessingError::Inference(message),
        }
    }
}

struct UnloadedMlxSpeechModel;

impl MlxSpeechModel for UnloadedMlxSpeechModel {
    fn transcribe(&self, _audio: &MlxArray, _sample_rate_hz: u32) -> Result<String, MlxAudioError> {
        Err(MlxAudioError::ModelUnavailable(
            "Distil-Whisper MLX graph loader is not implemented in mlx-models 0.1.x".to_string(),
        ))
    }

    fn synthesize(&self, _text: &str, _sample_rate_hz: u32) -> Result<MlxArray, MlxAudioError> {
        Err(MlxAudioError::ModelUnavailable(
            "Qwen3-TTS MLX graph loader is not implemented in mlx-models 0.1.x".to_string(),
        ))
    }
}

fn event_text(events: &[ExternalContextEvent]) -> Option<String> {
    events.iter().find_map(|event| match event.source.as_str() {
        "vona.plan_result" | "vona.precomputed_reply" => event
            .spoken_summary
            .as_ref()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        "vona.tts_text" => event
            .payload
            .as_str()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        _ => None,
    })
}

#[async_trait]
impl AudioTranscriber for MlxAudioEngine {
    async fn transcribe_audio(
        &self,
        input: AudioInputFrame,
    ) -> Result<String, AudioProcessingError> {
        let audio = Self::audio_array_from_frame(&input).map_err(AudioProcessingError::from)?;
        self.model
            .transcribe(&audio, input.sample_rate_hz)
            .map_err(AudioProcessingError::from)
    }
}

#[async_trait]
impl AudioStreamingTranscriber for MlxAudioEngine {
    async fn start_streaming_transcription(
        &self,
        config: StreamingTranscriptionConfig,
    ) -> Result<Box<dyn StreamingTranscriptionSession>, AudioProcessingError> {
        if config.sample_rate_hz == 0 {
            return Err(AudioProcessingError::InvalidInput(
                "streaming transcription sample rate must be non-zero".to_string(),
            ));
        }
        if config.channels == 0 {
            return Err(AudioProcessingError::InvalidInput(
                "streaming transcription channel count must be non-zero".to_string(),
            ));
        }

        Ok(Box::new(MlxStreamingTranscriptionSession {
            engine: self.clone(),
            config,
            pcm_buffer: Vec::new(),
            last_inference_samples: 0,
            recent_hypotheses: Vec::new(),
            committed_prefix: String::new(),
            latest_update: None,
            pending_decode: None,
            pending_decode_samples: 0,
        }))
    }
}

#[async_trait]
impl StreamingTranscriptionSession for MlxStreamingTranscriptionSession {
    async fn push_audio(
        &mut self,
        input: AudioInputFrame,
    ) -> Result<Option<StreamingTranscriptUpdate>, AudioProcessingError> {
        if input.sample_rate_hz != self.config.sample_rate_hz {
            return Err(AudioProcessingError::InvalidInput(format!(
                "streaming transcription sample rate changed from {} to {}",
                self.config.sample_rate_hz, input.sample_rate_hz
            )));
        }
        if input.channels != self.config.channels {
            return Err(AudioProcessingError::InvalidInput(format!(
                "streaming transcription channel count changed from {} to {}",
                self.config.channels, input.channels
            )));
        }
        if input.samples.is_empty() {
            return Ok(None);
        }

        self.pcm_buffer.extend(input.samples);
        self.enforce_buffer_limit();

        if let Some(update) = self.collect_finished_decode(false).await? {
            return Ok(Some(update));
        }

        let min_samples = samples_for_ms(self.config.sample_rate_hz, self.config.min_buffer_ms);
        if self.pcm_buffer.len() < min_samples {
            return Ok(None);
        }

        let step_samples = samples_for_ms(self.config.sample_rate_hz, self.config.step_ms);
        if self
            .pcm_buffer
            .len()
            .saturating_sub(self.last_inference_samples)
            < step_samples
        {
            return Ok(None);
        }

        if self.pending_decode.is_none() {
            self.last_inference_samples = self.pcm_buffer.len();
            self.pending_decode_samples = self.pcm_buffer.len();
            self.pending_decode = Some(spawn_mlx_streaming_decode(
                self.engine.clone(),
                self.config.sample_rate_hz,
                self.config.channels,
                self.pcm_buffer.clone(),
            ));
        }
        Ok(None)
    }

    async fn finish(&mut self) -> Result<Option<StreamingTranscriptUpdate>, AudioProcessingError> {
        if self.pcm_buffer.is_empty() {
            return Ok(None);
        }
        if let Some(update) = self.collect_finished_decode(false).await?
            && !update.text.trim().is_empty()
        {
            self.latest_update = Some(update);
        }

        if self.pending_decode.is_some() {
            if self.pending_decode_samples == self.pcm_buffer.len() {
                if let Some(update) = self.collect_finished_decode(true).await? {
                    return Ok(Some(StreamingTranscriptUpdate {
                        kind: StreamingTranscriptKind::Final,
                        text: update.text,
                        stability_passes: update.stability_passes,
                        total_audio_ms: self.total_audio_ms(),
                    }));
                }
            } else {
                self.pending_decode = None;
                self.pending_decode_samples = 0;
            }
        }
        self.transcribe_current(true).await
    }
}

impl MlxStreamingTranscriptionSession {
    async fn collect_finished_decode(
        &mut self,
        wait: bool,
    ) -> Result<Option<StreamingTranscriptUpdate>, AudioProcessingError> {
        let should_collect = self
            .pending_decode
            .as_ref()
            .is_some_and(|handle| wait || handle.is_finished());
        if !should_collect {
            return Ok(None);
        }

        let handle = self.pending_decode.take().expect("checked pending decode");
        self.pending_decode_samples = 0;
        let transcript = handle.await.map_err(|err| {
            AudioProcessingError::Runtime(format!("MLX streaming STT task join failed: {err}"))
        })??;
        self.accept_transcript(transcript, false)
    }

    fn enforce_buffer_limit(&mut self) {
        let max_samples = samples_for_ms(self.config.sample_rate_hz, self.config.max_buffer_ms);
        if self.pcm_buffer.len() <= max_samples {
            return;
        }

        let overflow = self.pcm_buffer.len().saturating_sub(max_samples);
        self.pcm_buffer.drain(0..overflow);
        self.last_inference_samples = self.last_inference_samples.saturating_sub(overflow);
        self.pending_decode_samples = self.pending_decode_samples.saturating_sub(overflow);
    }

    async fn transcribe_current(
        &mut self,
        is_final: bool,
    ) -> Result<Option<StreamingTranscriptUpdate>, AudioProcessingError> {
        let transcript = self.decode_current_buffer().await?;
        self.last_inference_samples = self.pcm_buffer.len();
        self.accept_transcript(transcript, is_final)
    }

    async fn decode_current_buffer(&self) -> Result<String, AudioProcessingError> {
        self.engine
            .transcribe_audio(AudioInputFrame {
                sequence: 0,
                sample_rate_hz: self.config.sample_rate_hz,
                channels: self.config.channels,
                samples: self.pcm_buffer.clone(),
            })
            .await
            .map(|value| value.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    fn accept_transcript(
        &mut self,
        transcript: String,
        is_final: bool,
    ) -> Result<Option<StreamingTranscriptUpdate>, AudioProcessingError> {
        self.recent_hypotheses.push(transcript.clone());
        let max_hypotheses = self.config.stability_passes.max(1) as usize;
        if self.recent_hypotheses.len() > max_hypotheses {
            self.recent_hypotheses.remove(0);
        }

        if is_final {
            self.committed_prefix = transcript.clone();
            let update = StreamingTranscriptUpdate {
                kind: StreamingTranscriptKind::Final,
                text: transcript,
                stability_passes: self.recent_hypotheses.len() as u32,
                total_audio_ms: self.total_audio_ms(),
            };
            self.latest_update = Some(update.clone());
            return Ok(Some(update));
        }

        if self.recent_hypotheses.len() < max_hypotheses {
            return Ok(None);
        }

        let committed_candidate = longest_common_word_prefix(&self.recent_hypotheses);
        if committed_candidate.is_empty() || committed_candidate == self.committed_prefix {
            return Ok(None);
        }
        if !self.committed_prefix.is_empty()
            && !committed_candidate.starts_with(&self.committed_prefix)
        {
            return Ok(None);
        }

        self.committed_prefix = committed_candidate.clone();
        let update = StreamingTranscriptUpdate {
            kind: StreamingTranscriptKind::Partial,
            text: committed_candidate,
            stability_passes: self.recent_hypotheses.len() as u32,
            total_audio_ms: self.total_audio_ms(),
        };
        self.latest_update = Some(update.clone());
        Ok(Some(update))
    }

    fn total_audio_ms(&self) -> u64 {
        ((self.pcm_buffer.len() as u64) * 1000) / self.config.sample_rate_hz as u64
    }
}

fn spawn_mlx_streaming_decode(
    engine: MlxAudioEngine,
    sample_rate_hz: u32,
    channels: u16,
    samples: Vec<f32>,
) -> tokio::task::JoinHandle<Result<String, AudioProcessingError>> {
    tokio::spawn(async move {
        engine
            .transcribe_audio(AudioInputFrame {
                sequence: 0,
                sample_rate_hz,
                channels,
                samples,
            })
            .await
            .map(|value| value.split_whitespace().collect::<Vec<_>>().join(" "))
    })
}

fn samples_for_ms(sample_rate_hz: u32, ms: u32) -> usize {
    ((sample_rate_hz as u64 * ms as u64) / 1000) as usize
}

fn longest_common_word_prefix(hypotheses: &[String]) -> String {
    let Some(first) = hypotheses.first() else {
        return String::new();
    };
    let mut prefix: Vec<&str> = first.split_whitespace().collect();
    for hypothesis in hypotheses.iter().skip(1) {
        let words: Vec<&str> = hypothesis.split_whitespace().collect();
        let common_len = prefix
            .iter()
            .zip(words.iter())
            .take_while(|(left, right)| left.eq_ignore_ascii_case(right))
            .count();
        prefix.truncate(common_len);
        if prefix.is_empty() {
            break;
        }
    }
    prefix.join(" ")
}

#[async_trait]
impl AudioSynthesizer for MlxAudioEngine {
    async fn synthesize_audio(
        &self,
        text: String,
        config: AudioSynthesisConfig,
    ) -> Result<AudioOutputFrame, AudioProcessingError> {
        let speech = self
            .model
            .synthesize(&text, config.sample_rate_hz)
            .map_err(AudioProcessingError::from)?;
        eval_array(&speech).map_err(AudioProcessingError::from)?;

        Ok(AudioOutputFrame {
            sequence: config.sequence,
            sample_rate_hz: config.sample_rate_hz,
            channels: config.channels,
            samples: samples_from_array(&speech),
            is_filler: false,
        })
    }
}

#[async_trait]
impl SpeechToSpeechBackend for MlxAudioEngine {
    type Session = MlxAudioSession;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_context_injection: true,
            supports_style_conditioning: false,
            supports_word_timestamps: false,
            ..BackendCapabilities::default()
        }
    }

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, BackendError> {
        Ok(MlxAudioSession {
            config,
            pending_events: Vec::new(),
        })
    }

    async fn step(
        &self,
        session: &mut Self::Session,
        input: AudioInputFrame,
    ) -> Result<BackendStep, BackendError> {
        let sequence = input.sequence;
        let transcript = self
            .transcribe_audio(input)
            .await
            .map_err(|error| BackendError::Step(error.to_string()))?;

        let pending_events = std::mem::take(&mut session.pending_events);
        let reply_text = event_text(&pending_events).unwrap_or_else(|| transcript.clone());
        let output_audio = self
            .synthesize_audio(
                reply_text,
                AudioSynthesisConfig {
                    sequence,
                    sample_rate_hz: self.config.output_sample_rate_hz,
                    channels: session.config.channels,
                },
            )
            .await
            .map_err(|error| BackendError::Step(error.to_string()))?;

        Ok(BackendStep {
            output_audio: vec![output_audio],
            control_events: vec![ControlEvent::Diagnostic {
                message: format!("vona-mlx device {}", self.device_label),
            }],
            transcript: Some(transcript),
            finished: false,
            debug_payload: Some(json!({
                "stt_model_id": self.config.stt_model_id,
                "tts_model_id": self.config.tts_model_id,
            })),
        })
    }

    async fn inject_event(
        &self,
        session: &mut Self::Session,
        event: ExternalContextEvent,
    ) -> Result<(), BackendError> {
        session.pending_events.push(event);
        Ok(())
    }

    async fn end_session(&self, _session: Self::Session) -> Result<(), BackendError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoModel;

    impl MlxSpeechModel for EchoModel {
        fn transcribe(
            &self,
            _audio: &MlxArray,
            _sample_rate_hz: u32,
        ) -> Result<String, MlxAudioError> {
            Ok("hello".to_string())
        }

        fn synthesize(&self, _text: &str, _sample_rate_hz: u32) -> Result<MlxArray, MlxAudioError> {
            Ok(test_array_from_samples(&[0.0_f32, 0.25, -0.25]))
        }
    }

    struct LengthAwareModel;

    impl MlxSpeechModel for LengthAwareModel {
        fn transcribe(
            &self,
            audio: &MlxArray,
            _sample_rate_hz: u32,
        ) -> Result<String, MlxAudioError> {
            if audio.shape()[0] < 3_000 {
                Ok("stale partial".to_string())
            } else {
                Ok("final transcript".to_string())
            }
        }

        fn synthesize(&self, _text: &str, _sample_rate_hz: u32) -> Result<MlxArray, MlxAudioError> {
            Ok(test_array_from_samples(&[0.0]))
        }
    }

    #[test]
    fn builds_mlx_audio_array_from_vona_frame() {
        let frame = AudioInputFrame {
            sequence: 0,
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.1, 0.2],
        };

        let array = MlxAudioEngine::audio_array_from_frame(&frame).unwrap();
        assert_eq!(array.shape(), &[2]);
    }

    #[test]
    fn extracts_tts_text_from_events() {
        let events = vec![ExternalContextEvent {
            source: "vona.tts_text".to_string(),
            spoken_summary: None,
            payload: json!("speak this"),
        }];

        assert_eq!(event_text(&events), Some("speak this".to_string()));
    }

    #[test]
    fn mlx_models_loader_is_feature_gated() {
        #[cfg(not(feature = "mlx-models-loader"))]
        {
            let request = MlxModelLoadRequest::local(
                MlxModelKind::TransformerText,
                "local-test-model",
                "/tmp/model",
            );
            let result = MlxModelsLoader.load_model(request);
            assert!(matches!(result, Err(MlxAudioError::Runtime(_))));
        }
    }

    #[test]
    fn streaming_common_prefix_is_word_stable() {
        let hypotheses = vec![
            "focus on the first task today".to_string(),
            "focus on the first useful task".to_string(),
        ];

        assert_eq!(
            longest_common_word_prefix(&hypotheses),
            "focus on the first"
        );
    }

    #[tokio::test]
    async fn injected_model_runs_backend_step() {
        let engine = MlxAudioEngine {
            config: MlxAudioConfig::default(),
            device_label: "test".to_string(),
            model: Arc::new(EchoModel),
        };
        let mut session = engine
            .start_session(SessionConfig::default())
            .await
            .unwrap();
        let step = engine
            .step(
                &mut session,
                AudioInputFrame {
                    sequence: 7,
                    sample_rate_hz: 16_000,
                    channels: 1,
                    samples: vec![0.0, 1.0],
                },
            )
            .await
            .unwrap();

        assert_eq!(step.transcript, Some("hello".to_string()));
        assert_eq!(step.output_audio[0].samples, vec![0.0, 0.25, -0.25]);
    }

    #[tokio::test]
    async fn streaming_finish_decodes_current_buffer_instead_of_stale_partial() {
        let engine = MlxAudioEngine {
            config: MlxAudioConfig::default(),
            device_label: "test".to_string(),
            model: Arc::new(LengthAwareModel),
        };
        let mut session = engine
            .start_streaming_transcription(StreamingTranscriptionConfig {
                sample_rate_hz: 16_000,
                channels: 1,
                step_ms: 50,
                min_buffer_ms: 50,
                max_buffer_ms: 30_000,
                stability_passes: 1,
            })
            .await
            .unwrap();

        let _ = session
            .push_audio(AudioInputFrame {
                sequence: 1,
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![0.0; 1_000],
            })
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let _ = session
            .push_audio(AudioInputFrame {
                sequence: 2,
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![0.0; 3_000],
            })
            .await
            .unwrap();

        let final_update = session.finish().await.unwrap().unwrap();
        assert_eq!(final_update.kind, StreamingTranscriptKind::Final);
        assert_eq!(final_update.text, "final transcript");
    }
}
