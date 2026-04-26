use anyhow::Context;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, Request, StatusCode, header::AUTHORIZATION},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use axum::extract::DefaultBodyLimit;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, watch};
use tracing::{info, warn};
use vona::{AudioInputFrame, SessionConfig, SpeechToSpeechBackend};
use vona_seamless::{
    SeamlessM4tLocalBackend, SeamlessM4tLocalConfig, SeamlessM4tLocalSession,
    SeamlessM4tRemoteStepRequest, SeamlessM4tRemoteStepResponse,
};
use vona_transport_local::{
    LocalIpcStepEnvelope, read_length_prefixed_frame, write_length_prefixed_frame,
};

#[derive(Clone)]
struct AppState {
    backend: Arc<SeamlessM4tLocalBackend>,
    sessions: Arc<Mutex<HashMap<String, SeamlessM4tLocalSession>>>,
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "model": std::env::var("VONA_STS_MODEL").ok().unwrap_or_else(|| SeamlessM4tLocalConfig::default().model_id),
        "sessions": state.sessions.lock().await.len(),
    }))
}

async fn process_step_request(
    state: &AppState,
    request: SeamlessM4tRemoteStepRequest,
) -> Result<SeamlessM4tRemoteStepResponse, String> {
    let input_sample_count = request.input_samples.len();
    let mut session = {
        let mut sessions = state.sessions.lock().await;
        sessions.remove(&request.session_id).unwrap_or_else(|| {
            SeamlessM4tLocalSession {
                config: SessionConfig {
                    session_id: request.session_id.clone(),
                    sample_rate_hz: request.sample_rate_hz,
                    channels: request.channels,
                    style_profile: request.style_profile.clone(),
                    metadata: request.session_metadata.clone(),
                },
                pending_events: Vec::new(),
            }
        })
    };

    for event in request.pending_events {
        state
            .backend
            .inject_event(&mut session, event)
            .await
            .map_err(|err| {
                let message = err.to_string();
                warn!(
                    session_id = %request.session_id,
                    sample_rate_hz = request.sample_rate_hz,
                    channels = request.channels,
                    input_sample_count,
                    error = %message,
                    "Vona Seamless sidecar rejected pending session context"
                );
                message
            })?;
    }

    let step = state
        .backend
        .step(
            &mut session,
            AudioInputFrame {
                sequence: 1,
                sample_rate_hz: request.sample_rate_hz,
                channels: request.channels,
                samples: request.input_samples,
            },
        )
        .await
        .map_err(|err| {
            let message = err.to_string();
            warn!(
                session_id = %request.session_id,
                sample_rate_hz = request.sample_rate_hz,
                channels = request.channels,
                input_sample_count,
                error = %message,
                "Vona Seamless sidecar backend step failed"
            );
            message
        })?;

    if !step.finished {
        state
            .sessions
            .lock()
            .await
            .insert(request.session_id, session);
    }

    let frame = step.output_audio.first().cloned();

    Ok(SeamlessM4tRemoteStepResponse {
        output_samples: frame
            .as_ref()
            .map(|value| value.samples.clone())
            .unwrap_or_default(),
        output_sample_rate_hz: frame
            .as_ref()
            .map(|value| value.sample_rate_hz)
            .unwrap_or(16_000),
        transcript: step.transcript,
        control_events: step.control_events,
        finished: step.finished,
        debug_payload: step.debug_payload,
    })
}

async fn step_http(
    State(state): State<AppState>,
    Json(request): Json<SeamlessM4tRemoteStepRequest>,
) -> Result<Json<SeamlessM4tRemoteStepResponse>, (axum::http::StatusCode, String)> {
    process_step_request(&state, request)
        .await
        .map(Json)
        .map_err(|message| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, message))
}

#[cfg(unix)]
struct UnixSocketCleanupGuard {
    path: PathBuf,
}

#[cfg(unix)]
impl UnixSocketCleanupGuard {
    fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[cfg(unix)]
impl Drop for UnixSocketCleanupGuard {
    fn drop(&mut self) {
        if let Err(err) = std::fs::remove_file(&self.path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            warn!(path = %self.path.display(), error = %err, "failed cleaning up Vona IPC socket");
        }
    }
}

#[cfg(unix)]
async fn serve_unix_socket(
    state: AppState,
    socket_path: String,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    use tokio::net::UnixListener;

    if let Some(parent) = Path::new(&socket_path).parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("failed to create parent directory for Vona IPC socket at {socket_path}")
        })?;
    }

    if Path::new(&socket_path).exists() {
        std::fs::remove_file(&socket_path)
            .with_context(|| format!("failed to remove stale Vona IPC socket at {socket_path}"))?;
    }

    let _cleanup_guard = UnixSocketCleanupGuard::new(&socket_path);
    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("failed to bind Vona IPC socket at {socket_path}"))?;
    info!(path = %socket_path, "Vona Seamless sidecar IPC listener active");

    loop {
        let accept_result = tokio::select! {
            changed = shutdown_rx.changed() => {
                match changed {
                    Ok(()) if *shutdown_rx.borrow() => break,
                    Ok(()) => continue,
                    Err(_) => break,
                }
            }
            accept = listener.accept() => accept,
        };
        let (mut stream, _addr) = accept_result
            .with_context(|| format!("failed to accept IPC connection on {socket_path}"))?;
        let state = state.clone();
        tokio::spawn(async move {
            loop {
                let request = match read_length_prefixed_frame::<_, SeamlessM4tRemoteStepRequest>(
                    &mut stream,
                )
                .await
                {
                    Ok(Some(request)) => request,
                    Ok(None) => break,
                    Err(err) => {
                        warn!(error = %err, "Vona IPC sidecar failed reading request frame");
                        break;
                    }
                };

                let envelope = match process_step_request(&state, request).await {
                    Ok(response) => LocalIpcStepEnvelope {
                        response: Some(response),
                        error: None,
                    },
                    Err(error) => LocalIpcStepEnvelope {
                        response: None,
                        error: Some(error),
                    },
                };

                if let Err(err) = write_length_prefixed_frame(&mut stream, &envelope).await {
                    warn!(error = %err, "Vona IPC sidecar failed writing response frame");
                    break;
                }
            }
        });
    }

    info!(path = %socket_path, "Vona Seamless sidecar IPC listener shutting down");
    Ok(())
}

/// Optional Bearer-token middleware. When `VONA_SIDECAR_AUTH_TOKEN` is set,
/// every request must include a matching `Authorization: Bearer <token>` header.
/// If the env var is not set, all requests are allowed through.
async fn bearer_auth(
    State(expected): State<Option<String>>,
    headers: HeaderMap,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if let Some(ref token) = expected {
        let authorized = headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(|v| v == format!("Bearer {token}"))
            .unwrap_or(false);

        if !authorized {
            return StatusCode::UNAUTHORIZED.into_response();
        }
    }
    next.run(request).await
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            warn!(%err, "Failed to install Ctrl+C handler for Vona sidecar");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                let _ = signal.recv().await;
            }
            Err(err) => {
                warn!(%err, "Failed to install SIGTERM handler for Vona sidecar");
            }
        }
    };

    #[cfg(unix)]
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    #[cfg(not(unix))]
    ctrl_c.await;

    info!("Shutdown signal received; draining Vona sidecar listeners");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let bind = std::env::var("VONA_STS_SIDECAR_BIND")
        .unwrap_or_else(|_| "127.0.0.1:9090".to_string());
    let ipc_socket_path = std::env::var("VONA_STS_SIDECAR_UNIX_SOCKET")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let auth_token = std::env::var("VONA_SIDECAR_AUTH_TOKEN").ok();
    let backend = Arc::new(SeamlessM4tLocalBackend::new(
        SeamlessM4tLocalConfig::from_env(),
    )?);
    let state = AppState {
        backend,
        sessions: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/healthz", get(health))
        .route("/v1/seamless-m4t/step", post(step_http))
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(auth_token, bearer_auth))
        .with_state(state.clone());

    let listener = TcpListener::bind(&bind)
        .await
        .with_context(|| format!("failed to bind Vona STS sidecar to {bind}"))?;
    info!(bind = %bind, "Vona Seamless sidecar HTTP listener active");
    if std::env::var("VONA_BACKEND").ok().as_deref() == Some("local-bridge") {
        warn!(
            "VONA_BACKEND still points at local-bridge; the sidecar will not be used until that backend is switched"
        );
    }

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    #[cfg(unix)]
    let mut ipc_handle = None;

    #[cfg(unix)]
    {
        if let Some(socket_path) = ipc_socket_path {
            ipc_handle = Some(tokio::spawn(serve_unix_socket(
                state.clone(),
                socket_path,
                shutdown_rx.clone(),
            )));
        }
    }

    #[cfg(not(unix))]
    if ipc_socket_path.is_some() {
        warn!(
            "VONA_STS_SIDECAR_UNIX_SOCKET was provided but Unix sockets are unsupported on this platform"
        );
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    let _ = shutdown_tx.send(true);

    #[cfg(unix)]
    if let Some(handle) = ipc_handle {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(err),
            Err(err) => return Err(anyhow::anyhow!("Vona IPC listener task failed: {err}")),
        }
    }

    Ok(())
}
