use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use serde::Deserialize;
use serde_json::json;
use vona_core::{AudioInputFrame, AudioTranscriber};
use vona_moonshine::{
    DEFAULT_MOONSHINE_ARCH, MoonshineTranscriberConfig, NativeMoonshineTranscriber,
};

#[derive(Debug, Deserialize)]
struct ManifestCase {
    id: String,
    category: String,
    expected: String,
    audio_path: PathBuf,
    sample_rate_hz: u32,
    channels: u16,
    source_type: String,
    #[serde(default)]
    text_sha256_16: Option<String>,
}

#[derive(Debug, Default)]
struct CategoryStats {
    cases: usize,
    total_wer: f64,
    total_stt_ms: f64,
    total_rtf: f64,
    near_exact: usize,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = env::var("VONA_MOONSHINE_BENCH_MANIFEST")
        .map(PathBuf::from)
        .map_err(|_| "set VONA_MOONSHINE_BENCH_MANIFEST")?;
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
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let cases = load_manifest(&manifest_path)?;

    let load_started = Instant::now();
    let mut config = MoonshineTranscriberConfig::from_env();
    config.native_library_path = Some(library_path);
    config.model_path = Some(model_path);
    config.model_arch = model_arch.clone();
    let transcriber = NativeMoonshineTranscriber::load(config)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let started_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let mut results = Vec::with_capacity(cases.len());
    let mut categories = BTreeMap::<String, CategoryStats>::new();

    for (index, case) in cases.iter().enumerate() {
        let path = manifest_dir.join(&case.audio_path);
        let raw = fs::read(&path)?;
        let samples = raw
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect::<Vec<_>>();
        let audio_ms = samples.len() as f64 / case.sample_rate_hz as f64 * 1000.0;
        let started = Instant::now();
        let transcript = transcriber
            .transcribe_audio(AudioInputFrame {
                sequence: index as u64 + 1,
                sample_rate_hz: case.sample_rate_hz,
                channels: case.channels,
                samples,
            })
            .await?;
        let stt_ms = started.elapsed().as_secs_f64() * 1000.0;
        let rtf = stt_ms / audio_ms.max(1.0);
        let wer = wer(&case.expected, &transcript);
        let near_exact = wer <= 0.1;
        let stats = categories.entry(case.category.clone()).or_default();
        stats.cases += 1;
        stats.total_wer += wer;
        stats.total_stt_ms += stt_ms;
        stats.total_rtf += rtf;
        stats.near_exact += usize::from(near_exact);

        results.push(json!({
            "id": case.id,
            "category": case.category,
            "source_type": case.source_type,
            "expected": case.expected,
            "text_sha256_16": case.text_sha256_16,
            "transcript": transcript,
            "audio_ms": round1(audio_ms),
            "stt_ms": round1(stt_ms),
            "rtf": round4(rtf),
            "wer": round3(wer),
            "near_exact": near_exact,
        }));

        if index == 0 || (index + 1) % 100 == 0 || index + 1 == cases.len() {
            println!(
                "scored {}/{} avg_wer={:.3} avg_ms={:.1}",
                index + 1,
                cases.len(),
                average(&results, "wer"),
                average(&results, "stt_ms")
            );
        }
    }

    let mut stt_ms_values = results
        .iter()
        .filter_map(|row| row["stt_ms"].as_f64())
        .collect::<Vec<_>>();
    let mut wer_values = results
        .iter()
        .filter_map(|row| row["wer"].as_f64())
        .collect::<Vec<_>>();
    stt_ms_values.sort_by(f64::total_cmp);
    wer_values.sort_by(f64::total_cmp);

    let category_summary = categories
        .iter()
        .map(|(category, stats)| {
            (
                category.clone(),
                json!({
                    "cases": stats.cases,
                    "avg_stt_ms": round1(stats.total_stt_ms / stats.cases as f64),
                    "avg_rtf": round4(stats.total_rtf / stats.cases as f64),
                    "avg_wer": round3(stats.total_wer / stats.cases as f64),
                    "near_exact": stats.near_exact,
                    "near_exact_rate": round4(stats.near_exact as f64 / stats.cases as f64),
                }),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let summary = json!({
        "arch": model_arch,
        "source_types": source_types(&results),
        "started_unix_ms": started_unix_ms,
        "load_ms": round1(load_ms),
        "cases": results.len(),
        "avg_stt_ms": round1(average(&results, "stt_ms")),
        "p50_stt_ms": round1(percentile(&stt_ms_values, 0.50)),
        "p95_stt_ms": round1(percentile(&stt_ms_values, 0.95)),
        "p99_stt_ms": round1(percentile(&stt_ms_values, 0.99)),
        "avg_rtf": round4(average(&results, "rtf")),
        "avg_wer": round3(average(&results, "wer")),
        "p95_wer": round3(percentile(&wer_values, 0.95)),
        "near_exact": results.iter().filter(|row| row["near_exact"].as_bool().unwrap_or(false)).count(),
        "near_exact_rate": round4(results.iter().filter(|row| row["near_exact"].as_bool().unwrap_or(false)).count() as f64 / results.len().max(1) as f64),
        "category_summary": category_summary,
        "evidence_class": "generated_voice_regression",
        "human_recorded": false,
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

fn load_manifest(path: &Path) -> Result<Vec<ManifestCase>, Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for (line_index, line) in fs::read_to_string(path)?.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let case = serde_json::from_str::<ManifestCase>(line)
            .map_err(|error| format!("invalid manifest line {}: {error}", line_index + 1))?;
        cases.push(case);
    }
    if cases.is_empty() {
        return Err("manifest did not contain any cases".into());
    }
    Ok(cases)
}

fn average(results: &[serde_json::Value], key: &str) -> f64 {
    results
        .iter()
        .filter_map(|row| row[key].as_f64())
        .sum::<f64>()
        / results.len().max(1) as f64
}

fn source_types(results: &[serde_json::Value]) -> Vec<String> {
    let mut values = results
        .iter()
        .filter_map(|row| row["source_type"].as_str().map(str::to_string))
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn round4(value: f64) -> f64 {
    (value * 10000.0).round() / 10000.0
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
