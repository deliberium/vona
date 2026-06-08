use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use serde::Deserialize;
use vona_core::{AudioSynthesisConfig, AudioSynthesizer};
use vona_kokoro_onnx::{DEFAULT_KOKORO_SAMPLE_RATE_HZ, KokoroOnnxConfig, KokoroOnnxSynthesizer};

#[derive(Debug, Deserialize)]
struct ManifestCase {
    id: String,
    expected: String,
    audio_path: PathBuf,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = env::var("VONA_KOKORO_CORPUS_MANIFEST")
        .map(PathBuf::from)
        .map_err(|_| "set VONA_KOKORO_CORPUS_MANIFEST")?;
    let model_path = env::var("VONA_KOKORO_ONNX_MODEL")
        .map(PathBuf::from)
        .map_err(|_| "set VONA_KOKORO_ONNX_MODEL")?;
    let voices_path = env::var("VONA_KOKORO_VOICES")
        .map(PathBuf::from)
        .map_err(|_| "set VONA_KOKORO_VOICES")?;
    let target_sample_rate_hz = env::var("VONA_KOKORO_CORPUS_SAMPLE_RATE_HZ")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(16_000);
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let cases = load_manifest(&manifest_path)?;

    let mut config = KokoroOnnxConfig::new(model_path, voices_path);
    if let Ok(voice) = env::var("VONA_KOKORO_VOICE") {
        config.voice = voice;
    }

    let load_started = Instant::now();
    let synth = KokoroOnnxSynthesizer::load(config).await?;
    eprintln!(
        "loaded Kokoro in {:.1} ms",
        load_started.elapsed().as_secs_f64() * 1000.0
    );

    let started = Instant::now();
    for (index, case) in cases.iter().enumerate() {
        let frame = synth
            .synthesize_audio(
                case.expected.clone(),
                AudioSynthesisConfig {
                    sequence: index as u64 + 1,
                    sample_rate_hz: DEFAULT_KOKORO_SAMPLE_RATE_HZ,
                    channels: 1,
                },
            )
            .await?;
        let samples = if frame.sample_rate_hz == target_sample_rate_hz {
            frame.samples
        } else {
            resample_linear(&frame.samples, frame.sample_rate_hz, target_sample_rate_hz)
        };
        let output_path = manifest_dir.join(&case.audio_path);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        write_pcm16le(&output_path, &samples)?;
        if index == 0 || (index + 1) % 100 == 0 || index + 1 == cases.len() {
            eprintln!(
                "synthesized {}/{} elapsed_s={:.1}",
                index + 1,
                cases.len(),
                started.elapsed().as_secs_f64()
            );
            eprintln!("last_case={}", case.id);
        }
    }
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

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if samples.is_empty() || from_hz == 0 || to_hz == 0 {
        return Vec::new();
    }
    let output_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    let scale = from_hz as f64 / to_hz as f64;
    (0..output_len)
        .map(|index| {
            let position = index as f64 * scale;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let mix = (position - left as f64) as f32;
            samples[left] * (1.0 - mix) + samples[right] * mix
        })
        .collect()
}

fn write_pcm16le(path: &Path, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&pcm.to_le_bytes())?;
    }
    Ok(())
}
