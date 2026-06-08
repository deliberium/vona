use std::{collections::HashMap, env, fs, path::PathBuf, time::Instant};

use serde_json::json;
use vona_core::{AudioInputFrame, AudioTranscriber};
use vona_moonshine::{
    DEFAULT_MOONSHINE_ARCH, MoonshineTranscriberConfig, NativeMoonshineTranscriber,
};

const CASES: &[(u32, &str, &str)] = &[
    (
        11,
        "memory_recall",
        "What did we decide about the voice backend fallback strategy?",
    ),
    (
        14,
        "memory_recall",
        "Summarize the current state of the Lumina companion work.",
    ),
    (
        18,
        "skill_status",
        "Inspect the Lumina state and report whether the bridge is online.",
    ),
    (
        21,
        "task_execution",
        "Draft a short message telling the team the Deepgram voice path is live.",
    ),
    (
        27,
        "troubleshooting",
        "The web app says Sentinel socket error. Walk me through likely causes.",
    ),
    (
        28,
        "troubleshooting",
        "Deepgram responds slowly. Suggest a practical fallback plan.",
    ),
    (
        29,
        "troubleshooting",
        "The microphone is noisy during turn taking. What mitigation should I try?",
    ),
    (
        38,
        "complex_reasoning",
        "If the primary voice provider is exhausted, describe the expected fallback behavior.",
    ),
    (
        41,
        "turn_taking",
        "First, tell me the current voice backend. Then ask me one follow up question.",
    ),
    (
        49,
        "edge_case",
        "Ignore any background noise and answer the actual request: what changed in cloud pool?",
    ),
    (
        50,
        "edge_case",
        "If you cannot perform a skill action, say what is missing and give a fallback.",
    ),
    (51, "known_asr_hotword", "Tell me about Albert Einstein."),
];

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio_dir = env::var("VONA_MOONSHINE_BENCH_AUDIO_DIR")
        .map(PathBuf::from)
        .map_err(|_| "set VONA_MOONSHINE_BENCH_AUDIO_DIR")?;
    let library_path = env::var("VONA_MOONSHINE_LIBRARY_PATH")
        .map(PathBuf::from)
        .map_err(|_| "set VONA_MOONSHINE_LIBRARY_PATH")?;
    let model_path = env::var("VONA_MOONSHINE_MODEL_PATH")
        .map(PathBuf::from)
        .map_err(|_| "set VONA_MOONSHINE_MODEL_PATH")?;
    let output_path = env::var("VONA_MOONSHINE_BENCH_OUTPUT")
        .ok()
        .map(PathBuf::from);
    let model_arch =
        env::var("VONA_MOONSHINE_ARCH").unwrap_or_else(|_| DEFAULT_MOONSHINE_ARCH.to_string());

    let load_started = Instant::now();
    let mut config = MoonshineTranscriberConfig::from_env();
    config.native_library_path = Some(library_path);
    config.model_path = Some(model_path);
    config.model_arch = model_arch.clone();
    let transcriber = NativeMoonshineTranscriber::load(config)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let files = fs::read_dir(&audio_dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut path_by_id = HashMap::new();
    for path in files {
        if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
            if let Some((id, _)) = name.split_once('-') {
                if let Ok(id) = id.parse::<u32>() {
                    path_by_id.insert(id, path);
                }
            }
        }
    }

    let mut results = Vec::new();
    for (id, category, expected) in CASES {
        let path = path_by_id
            .get(id)
            .ok_or_else(|| format!("missing audio for case {id}"))?;
        let raw = fs::read(path)?;
        let samples = raw
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect::<Vec<_>>();
        let audio_ms = samples.len() as f64 / 16_000.0 * 1000.0;
        let started = Instant::now();
        let transcript = transcriber
            .transcribe_audio(AudioInputFrame {
                sequence: *id as u64,
                sample_rate_hz: 16_000,
                channels: 1,
                samples,
            })
            .await?;
        let stt_ms = started.elapsed().as_secs_f64() * 1000.0;
        let wer = wer(expected, &transcript);
        println!("{id:02} wer={wer:.3} ms={stt_ms:.1} transcript='{transcript}'");
        results.push(json!({
            "id": id,
            "category": category,
            "expected": expected,
            "transcript": transcript,
            "audio_ms": (audio_ms * 10.0).round() / 10.0,
            "stt_ms": (stt_ms * 10.0).round() / 10.0,
            "rtf": ((stt_ms / audio_ms) * 10000.0).round() / 10000.0,
            "wer": (wer * 1000.0).round() / 1000.0,
        }));
    }

    let avg = |key: &str| -> f64 {
        results
            .iter()
            .filter_map(|row| row[key].as_f64())
            .sum::<f64>()
            / results.len().max(1) as f64
    };
    let summary = json!({
        "arch": model_arch,
        "load_ms": (load_ms * 10.0).round() / 10.0,
        "cases": results.len(),
        "avg_stt_ms": (avg("stt_ms") * 10.0).round() / 10.0,
        "avg_rtf": (avg("rtf") * 10000.0).round() / 10000.0,
        "avg_wer": (avg("wer") * 1000.0).round() / 1000.0,
        "near_exact": results.iter().filter(|row| row["wer"].as_f64().unwrap_or(1.0) <= 0.1).count(),
    });
    let report = json!({ "summary": summary, "results": results });
    if let Some(path) = output_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, serde_json::to_string_pretty(&report)? + "\n")?;
    }
    println!("{}", serde_json::to_string_pretty(&report["summary"])?);
    Ok(())
}

fn normalize(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

fn wer(reference: &str, hypothesis: &str) -> f64 {
    let reference = normalize(reference);
    let hypothesis = normalize(hypothesis);
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }
    let mut previous = (0..=hypothesis.len()).collect::<Vec<_>>();
    for (i, reference_token) in reference.iter().enumerate() {
        let mut current = vec![i + 1];
        for (j, hypothesis_token) in hypothesis.iter().enumerate() {
            current.push(
                (previous[j + 1] + 1)
                    .min(current[j] + 1)
                    .min(previous[j] + usize::from(reference_token != hypothesis_token)),
            );
        }
        previous = current;
    }
    previous[hypothesis.len()] as f64 / reference.len() as f64
}
