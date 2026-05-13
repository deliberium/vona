use anyhow::Context;
use axum::{Json, Router, routing::post};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::watch;
use vona_core::ExternalContextEvent;
use vona_seamless::{
    SeamlessM4tRemoteStepRequest, SeamlessM4tRemoteStepResponse, SeamlessM4tRemoteTransport,
};
use vona_transport_local::{
    LocalIpcSeamlessM4tTransport, LocalIpcStepEnvelope, LocalIpcTransportConfig,
};

#[derive(Debug, Clone)]
struct BenchConfig {
    base_url: String,
    socket_path: String,
    iterations: usize,
    sample_count: usize,
    live: bool,
    mock_live: bool,
}

impl BenchConfig {
    fn from_args() -> Self {
        let mut config = Self {
            base_url: "http://127.0.0.1:9090".to_string(),
            socket_path: "/tmp/vona-sts.sock".to_string(),
            iterations: 8,
            sample_count: 320,
            live: false,
            mock_live: false,
        };

        for arg in std::env::args().skip(1) {
            if let Some(value) = arg.strip_prefix("--base-url=") {
                config.base_url = value.to_string();
            } else if let Some(value) = arg.strip_prefix("--socket-path=") {
                config.socket_path = value.to_string();
            } else if let Some(value) = arg.strip_prefix("--iterations=") {
                if let Ok(iterations) = value.parse::<usize>() {
                    config.iterations = iterations.max(1);
                }
            } else if let Some(value) = arg.strip_prefix("--sample-count=") {
                if let Ok(sample_count) = value.parse::<usize>() {
                    config.sample_count = sample_count;
                }
            } else if arg == "--live" {
                config.live = true;
            } else if arg == "--mock-live" {
                config.mock_live = true;
            }
        }

        config
    }
}

fn mock_response(request: &SeamlessM4tRemoteStepRequest) -> SeamlessM4tRemoteStepResponse {
    SeamlessM4tRemoteStepResponse {
        output_samples: vec![0.0; request.input_samples.len().clamp(32, 160)],
        output_sample_rate_hz: 16_000,
        transcript: Some("benchmark transcript".to_string()),
        control_events: vec![],
        finished: true,
        debug_payload: Some(serde_json::json!({
            "reply_text": "benchmark reply",
            "transport": "mock",
        })),
    }
}

fn bench_request(sample_count: usize, session_id: &str) -> SeamlessM4tRemoteStepRequest {
    SeamlessM4tRemoteStepRequest {
        session_id: session_id.to_string(),
        sample_rate_hz: 16_000,
        channels: 1,
        input_samples: (0..sample_count)
            .map(|idx| ((idx % 32) as f32 / 32.0) - 0.5)
            .collect(),
        model: Some("facebook/hf-seamless-m4t-medium".to_string()),
        session_metadata: serde_json::json!({
            "user_id": "benchmark-user",
            "thread_id": "benchmark-thread",
            "provenance": "transport_benchmark",
            "session_key": session_id,
        }),
        style_profile: None,
        pending_events: vec![
            ExternalContextEvent {
                source: "vona.transcript_override".to_string(),
                spoken_summary: None,
                payload: serde_json::json!("benchmark request"),
            },
            ExternalContextEvent {
                source: "vona.precomputed_reply".to_string(),
                spoken_summary: Some("Benchmark reply context".to_string()),
                payload: serde_json::json!("Benchmark reply context"),
            },
        ],
    }
}

fn average_duration_ms(total: Duration, iterations: usize) -> f64 {
    (total.as_secs_f64() * 1000.0) / iterations as f64
}

async fn measure_http_round_trip(
    base_url: &str,
    request: &SeamlessM4tRemoteStepRequest,
    iterations: usize,
) -> anyhow::Result<f64> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(90))
        .build()
        .context("failed building reqwest client")?;
    let endpoint = format!("{}/v1/seamless-m4t/step", base_url.trim_end_matches('/'));
    let mut total = Duration::ZERO;

    for idx in 0..iterations {
        let mut candidate = request.clone();
        candidate.session_id = format!("bench-http-{idx}");
        let started = Instant::now();
        let response = client
            .post(&endpoint)
            .json(&candidate)
            .send()
            .await
            .with_context(|| format!("HTTP request failed for {endpoint}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("HTTP benchmark request failed: {status} {body}");
        }
        let _payload: serde_json::Value = response
            .json()
            .await
            .context("failed decoding HTTP benchmark response")?;
        total += started.elapsed();
    }

    Ok(average_duration_ms(total, iterations))
}

async fn measure_ipc_round_trip(
    socket_path: &str,
    request: &SeamlessM4tRemoteStepRequest,
    iterations: usize,
) -> anyhow::Result<f64> {
    let transport =
        LocalIpcSeamlessM4tTransport::new(LocalIpcTransportConfig::unix_socket(socket_path))
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let mut total = Duration::ZERO;

    for idx in 0..iterations {
        let mut candidate = request.clone();
        candidate.session_id = format!("bench-ipc-{idx}");
        let started = Instant::now();
        let _response = transport
            .step(candidate)
            .await
            .with_context(|| format!("IPC request failed for {socket_path}"))?;
        total += started.elapsed();
    }

    Ok(average_duration_ms(total, iterations))
}

async fn start_mock_http_server(
    base_url: &str,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<tokio::task::JoinHandle<anyhow::Result<()>>> {
    async fn handle_step(
        Json(request): Json<SeamlessM4tRemoteStepRequest>,
    ) -> Json<SeamlessM4tRemoteStepResponse> {
        Json(mock_response(&request))
    }

    let bind = base_url
        .trim()
        .strip_prefix("http://")
        .ok_or_else(|| anyhow::anyhow!("mock benchmark only supports http:// base URLs"))?;
    let listener = TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed binding mock HTTP benchmark server to {bind}"))?;
    let app = Router::new().route("/v1/seamless-m4t/step", post(handle_step));

    Ok(tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                loop {
                    match shutdown_rx.changed().await {
                        Ok(()) if *shutdown_rx.borrow() => break,
                        Ok(()) => continue,
                        Err(_) => break,
                    }
                }
            })
            .await
            .context("mock HTTP benchmark server failed")
    }))
}

#[cfg(unix)]
async fn measure_mock_ipc_round_trip(
    request: &SeamlessM4tRemoteStepRequest,
    iterations: usize,
) -> anyhow::Result<f64> {
    let mut total = Duration::ZERO;
    for idx in 0..iterations {
        let mut candidate = request.clone();
        candidate.session_id = format!("bench-ipc-{idx}");
        let started = Instant::now();
        let request_payload =
            serde_cbor::to_vec(&candidate).context("failed encoding mock IPC request")?;
        let decoded_request: SeamlessM4tRemoteStepRequest =
            serde_cbor::from_slice(&request_payload).context("failed decoding mock IPC request")?;
        let envelope = LocalIpcStepEnvelope {
            response: Some(mock_response(&decoded_request)),
            error: None,
        };
        let response_payload =
            serde_cbor::to_vec(&envelope).context("failed encoding mock IPC response")?;
        let decoded_envelope: LocalIpcStepEnvelope = serde_cbor::from_slice(&response_payload)
            .context("failed decoding mock IPC response")?;
        if let Some(error) = decoded_envelope.error {
            anyhow::bail!("mock IPC server returned an error: {error}");
        }
        total += started.elapsed();
    }
    Ok(average_duration_ms(total, iterations))
}

#[cfg(not(unix))]
async fn measure_mock_ipc_round_trip(
    _request: &SeamlessM4tRemoteStepRequest,
    _iterations: usize,
) -> anyhow::Result<f64> {
    anyhow::bail!("mock IPC benchmark is unsupported on this platform")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = BenchConfig::from_args();
    let request = bench_request(config.sample_count, "bench-template");
    let pcm_payload_bytes = request.input_samples.len() * std::mem::size_of::<f32>();
    let http_json =
        serde_json::to_vec(&request).context("failed encoding HTTP benchmark payload")?;
    let ipc_frame =
        serde_cbor::to_vec(&request).context("failed encoding IPC benchmark payload")?;
    let http_encode_started = Instant::now();
    for _ in 0..config.iterations {
        let _ =
            serde_json::to_vec(&request).context("failed encoding HTTP payload during timing")?;
    }
    let http_encode_total = http_encode_started.elapsed();
    let ipc_encode_started = Instant::now();
    for _ in 0..config.iterations {
        let _ =
            serde_cbor::to_vec(&request).context("failed encoding IPC payload during timing")?;
    }
    let ipc_encode_total = ipc_encode_started.elapsed();

    println!("=== seamless_m4t_transport_bench ===");
    println!("sample_count={}", config.sample_count);
    println!("pcm_payload_bytes={pcm_payload_bytes}");
    println!("http_json_bytes={}", http_json.len());
    println!("ipc_cbor_bytes={}", ipc_frame.len());
    println!("ipc_framed_bytes={}", ipc_frame.len() + 4);
    println!(
        "http_json_over_pcm_ratio={:.2}",
        http_json.len() as f64 / pcm_payload_bytes.max(1) as f64
    );
    println!(
        "http_json_over_ipc_ratio={:.2}",
        http_json.len() as f64 / (ipc_frame.len() + 4) as f64
    );
    println!(
        "http_encode_avg_ms={:.3}",
        average_duration_ms(http_encode_total, config.iterations)
    );
    println!(
        "ipc_encode_avg_ms={:.3}",
        average_duration_ms(ipc_encode_total, config.iterations)
    );

    if config.live {
        let mut mock_http_handle = None;
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        if config.mock_live {
            mock_http_handle =
                Some(start_mock_http_server(&config.base_url, shutdown_rx.clone()).await?);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let http_ms =
            measure_http_round_trip(&config.base_url, &request, config.iterations).await?;
        let ipc_ms = if config.mock_live {
            measure_mock_ipc_round_trip(&request, config.iterations).await?
        } else {
            measure_ipc_round_trip(&config.socket_path, &request, config.iterations).await?
        };
        let _ = shutdown_tx.send(true);
        if let Some(handle) = mock_http_handle {
            handle
                .await
                .context("mock HTTP benchmark task join failed")??;
        }
        println!("http_round_trip_avg_ms={http_ms:.3}");
        println!("ipc_round_trip_avg_ms={ipc_ms:.3}");
        println!("live_iterations={}", config.iterations);
        println!(
            "live_latency_ratio_http_over_ipc={:.2}",
            http_ms / ipc_ms.max(0.001)
        );
        println!(
            "live_mode={}",
            if config.mock_live {
                "mock_http_plus_ipc_codec"
            } else {
                "real_sidecar"
            }
        );
    } else {
        println!("live_round_trip=skipped");
        println!(
            "hint=run with --live --mock-live for transport-only RTT, or --live against a real sidecar"
        );
    }

    Ok(())
}
