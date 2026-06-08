use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use vona_mlx::MlxSpeechModel;
use vona_mlx_whisper::{
    DEFAULT_WHISPER_SAMPLE_RATE_HZ, WhisperSpeechConfig, WhisperSpeechModel, WhisperTask,
    parse_transcript_hotwords,
};

#[derive(Debug, Deserialize)]
struct WorkerRequest {
    id: u64,
    sample_rate_hz: u32,
    channels: u16,
    samples: usize,
}

#[derive(Debug, Serialize)]
struct WorkerReady {
    ready: bool,
    model: String,
    weights: usize,
}

#[derive(Debug, Serialize)]
struct WorkerResponse {
    id: u64,
    transcript: Option<String>,
    error: Option<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = worker_config_from_args()?;
    let model_path = config.model_path.display().to_string();
    let model = WhisperSpeechModel::load(config)?;
    write_json_line(&WorkerReady {
        ready: true,
        model: model_path,
        weights: model.weight_count(),
    })
    .await?;

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).await? == 0 {
            break;
        }
        if header.trim().is_empty() {
            continue;
        }
        let request: WorkerRequest = match serde_json::from_str(&header) {
            Ok(request) => request,
            Err(error) => {
                write_json_line(&WorkerResponse {
                    id: 0,
                    transcript: None,
                    error: Some(format!("invalid request header: {error}")),
                })
                .await?;
                continue;
            }
        };
        let response = handle_request(&model, &mut reader, request).await;
        write_json_line(&response).await?;
    }

    Ok(())
}

async fn handle_request(
    model: &WhisperSpeechModel,
    reader: &mut BufReader<tokio::io::Stdin>,
    request: WorkerRequest,
) -> WorkerResponse {
    let result = async {
        if request.sample_rate_hz != DEFAULT_WHISPER_SAMPLE_RATE_HZ {
            return Err(format!(
                "Whisper worker expects {DEFAULT_WHISPER_SAMPLE_RATE_HZ} Hz audio, got {} Hz",
                request.sample_rate_hz
            ));
        }
        if request.channels != 1 {
            return Err(format!(
                "Whisper worker expects mono audio, got {} channels",
                request.channels
            ));
        }
        let mut bytes = vec![0_u8; request.samples.saturating_mul(4)];
        reader
            .read_exact(&mut bytes)
            .await
            .map_err(|error| format!("failed to read PCM payload: {error}"))?;
        let samples = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        let audio = mlx_rs::Array::from_slice(&samples, &[samples.len() as i32]);
        model
            .transcribe(&audio, DEFAULT_WHISPER_SAMPLE_RATE_HZ)
            .map_err(|error| error.to_string())
    }
    .await;

    match result {
        Ok(transcript) => WorkerResponse {
            id: request.id,
            transcript: Some(transcript),
            error: None,
        },
        Err(error) => WorkerResponse {
            id: request.id,
            transcript: None,
            error: Some(error),
        },
    }
}

async fn write_json_line<T: Serialize>(value: &T) -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = tokio::io::stdout();
    stdout
        .write_all(serde_json::to_string(value)?.as_bytes())
        .await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}

fn worker_config_from_args() -> Result<WhisperSpeechConfig, Box<dyn std::error::Error>> {
    let mut model_path = None;
    let mut language = None;
    let mut task = WhisperTask::Transcribe;
    let mut max_decode_tokens = None;
    let mut hotwords = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = args.next().map(PathBuf::from),
            "--language" => language = args.next(),
            "--task" => {
                task = match args.next().as_deref() {
                    Some("translate") => WhisperTask::Translate,
                    Some("transcribe") => WhisperTask::Transcribe,
                    Some(other) => return Err(format!("unsupported Whisper task {other:?}").into()),
                    None => return Err("--task requires a value".into()),
                };
            }
            "--max-decode-tokens" => {
                max_decode_tokens = Some(
                    args.next()
                        .ok_or("--max-decode-tokens requires a value")?
                        .parse::<usize>()?,
                );
            }
            "--hotwords" => {
                hotwords = Some(parse_transcript_hotwords(
                    &args.next().ok_or("--hotwords requires a value")?,
                )?);
            }
            "--help" | "-h" => {
                return Err("usage: vona_mlx_whisper_worker --model <dir> [--language en] [--task transcribe|translate] [--max-decode-tokens n] [--hotwords replacement=variant|variant]".into());
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    let mut config = WhisperSpeechConfig::new(model_path.ok_or("--model is required")?);
    config.language = language;
    config.task = task;
    if let Some(max_decode_tokens) = max_decode_tokens {
        config.max_decode_tokens = max_decode_tokens;
    }
    if let Some(hotwords) = hotwords {
        config.hotwords = hotwords;
    }
    Ok(config)
}
