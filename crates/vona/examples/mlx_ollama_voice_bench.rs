use std::{env, path::PathBuf, time::Instant};

use futures_util::StreamExt;
use vona::{
    DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ, DEFAULT_WHISPER_SAMPLE_RATE_HZ, MlxSpeechModel, OllamaConfig,
    OllamaTextEngine, Qwen3TtsSpeechConfig, Qwen3TtsSpeechModel, TextGenerationInput,
    TextGenerator, WhisperSpeechConfig, WhisperSpeechModel,
};

const DEFAULT_CASES: usize = 100;
const DEFAULT_MIN_AUDIO_TOKENS: usize = 24;
const DEFAULT_MAX_AUDIO_TOKENS: usize = 96;
const DEFAULT_MAX_DECODE_TOKENS: usize = 96;

#[derive(Debug)]
struct CaseResult {
    index: usize,
    prompt: String,
    tts_ms: u128,
    samples_24k: usize,
    peak: f32,
    stt_ms: u128,
    transcript: String,
    ollama_ms: u128,
    ollama_frames: usize,
    response_chars: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qwen_path = required_env_path("VONA_E2E_QWEN3_TTS_MODEL")?;
    let whisper_path = required_env_path("VONA_E2E_WHISPER_MODEL")?;
    let cases = env_usize("VONA_E2E_CASES", DEFAULT_CASES);
    let min_audio_tokens = env_usize("VONA_E2E_MIN_AUDIO_TOKENS", DEFAULT_MIN_AUDIO_TOKENS);
    let max_audio_tokens = env_usize("VONA_E2E_MAX_AUDIO_TOKENS", DEFAULT_MAX_AUDIO_TOKENS);
    let max_decode_tokens = env_usize("VONA_E2E_MAX_DECODE_TOKENS", DEFAULT_MAX_DECODE_TOKENS);
    let report_path = env::var_os("VONA_E2E_REPORT").map(PathBuf::from);

    eprintln!("loading Qwen3 TTS from {}", qwen_path.display());
    let qwen_load_start = Instant::now();
    let mut qwen_config = Qwen3TtsSpeechConfig::new(qwen_path);
    qwen_config.min_audio_tokens = min_audio_tokens.min(max_audio_tokens);
    qwen_config.max_audio_tokens = max_audio_tokens;
    let qwen = Qwen3TtsSpeechModel::load(qwen_config)?;
    let qwen_load_ms = qwen_load_start.elapsed().as_millis();

    eprintln!("loading Whisper from {}", whisper_path.display());
    let whisper_load_start = Instant::now();
    let mut whisper_config = WhisperSpeechConfig::new(whisper_path);
    whisper_config.max_decode_tokens = max_decode_tokens;
    let whisper = WhisperSpeechModel::load(whisper_config)?;
    let whisper_load_ms = whisper_load_start.elapsed().as_millis();

    let ollama_model = env::var("VONA_E2E_OLLAMA_MODEL")
        .or_else(|_| env::var("OLLAMA_MODEL"))
        .unwrap_or_else(|_| "phi4-mini".to_string());
    let ollama = OllamaTextEngine::new(OllamaConfig {
        model: ollama_model,
        ..OllamaConfig::from_env()
    });

    let prompts = benchmark_prompts(cases);
    let mut results = Vec::with_capacity(prompts.len());
    let run_start = Instant::now();

    for (index, prompt) in prompts.into_iter().enumerate() {
        let case_index = index + 1;
        eprintln!("case {case_index}/{cases}: {prompt}");

        let tts_start = Instant::now();
        let audio = qwen.synthesize(&prompt, DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ)?;
        let audio = audio.as_type::<f32>()?;
        audio.eval()?;
        let samples_24k = audio.as_slice::<f32>().to_vec();
        let tts_ms = tts_start.elapsed().as_millis();
        let peak = samples_24k
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);

        let samples_16k = resample_linear(
            &samples_24k,
            DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ,
            DEFAULT_WHISPER_SAMPLE_RATE_HZ,
        );
        let whisper_audio =
            vona::mlx::MlxArray::from_slice(&samples_16k, &[samples_16k.len() as i32]);

        let stt_start = Instant::now();
        let transcript = whisper.transcribe(&whisper_audio, DEFAULT_WHISPER_SAMPLE_RATE_HZ)?;
        let stt_ms = stt_start.elapsed().as_millis();

        let ollama_prompt = format!(
            "Answer in one concise sentence. Transcribed user speech: {}",
            transcript.trim()
        );
        let ollama_start = Instant::now();
        let (response, ollama_frames) = collect_ollama(&ollama, ollama_prompt).await?;
        let ollama_ms = ollama_start.elapsed().as_millis();

        results.push(CaseResult {
            index: case_index,
            prompt,
            tts_ms,
            samples_24k: samples_24k.len(),
            peak,
            stt_ms,
            transcript,
            ollama_ms,
            ollama_frames,
            response_chars: response.chars().count(),
        });
    }

    let markdown = render_markdown(
        qwen.weight_count(),
        qwen.vocoder_weight_count(),
        qwen_load_ms,
        whisper.weight_count(),
        whisper_load_ms,
        min_audio_tokens,
        max_audio_tokens,
        max_decode_tokens,
        run_start.elapsed().as_millis(),
        &results,
    );

    if let Some(path) = report_path {
        std::fs::write(&path, &markdown)?;
        eprintln!("wrote {}", path.display());
    }
    println!("{markdown}");
    Ok(())
}

async fn collect_ollama(
    ollama: &OllamaTextEngine,
    prompt: String,
) -> Result<(String, usize), Box<dyn std::error::Error>> {
    let mut stream = ollama.generate_text(TextGenerationInput::streaming(prompt));
    let mut response = String::new();
    let mut frames = 0_usize;
    while let Some(frame) = stream.next().await {
        let frame = frame?;
        frames += 1;
        response.push_str(&frame.text);
        if frame.final_fragment {
            break;
        }
    }
    Ok((response, frames))
}

fn required_env_path(name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let value = env::var_os(name).ok_or_else(|| format!("missing required env var {name}"))?;
    Ok(PathBuf::from(value))
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn resample_linear(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input.is_empty() || input_rate == output_rate {
        return input.to_vec();
    }
    let output_len = (input.len() as u64 * output_rate as u64 / input_rate as u64).max(1) as usize;
    let ratio = input_rate as f64 / output_rate as f64;
    (0..output_len)
        .map(|out_index| {
            let position = out_index as f64 * ratio;
            let left = position.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let frac = (position - left as f64) as f32;
            input[left] * (1.0 - frac) + input[right] * frac
        })
        .collect()
}

fn benchmark_prompts(cases: usize) -> Vec<String> {
    const PHRASES: &[&str] = &[
        "Hello from native Vona MLX",
        "Local voice test for Ollama",
        "Rust audio pipeline ready",
        "Apple Silicon speech check",
        "Streaming adapter test",
        "Whisper transcription pass",
        "Qwen speech synthesis pass",
        "Benchmark sample complete",
        "Realtime voice path active",
        "Vona local inference check",
    ];

    (0..cases)
        .map(|index| format!("{} case {}", PHRASES[index % PHRASES.len()], index + 1))
        .collect()
}

fn render_markdown(
    qwen_weights: usize,
    qwen_vocoder_weights: usize,
    qwen_load_ms: u128,
    whisper_weights: usize,
    whisper_load_ms: u128,
    min_audio_tokens: usize,
    max_audio_tokens: usize,
    max_decode_tokens: usize,
    total_ms: u128,
    results: &[CaseResult],
) -> String {
    let mut out = String::new();
    out.push_str("# MLX + Ollama Voice E2E Benchmark\n\n");
    out.push_str("| Metric | Value |\n|---|---:|\n");
    out.push_str(&format!("| Cases | {} |\n", results.len()));
    out.push_str(&format!(
        "| Non-empty transcripts | {} |\n",
        results
            .iter()
            .filter(|result| !result.transcript.trim().is_empty())
            .count()
    ));
    out.push_str(&format!("| Qwen3 TTS load | {qwen_load_ms} ms |\n"));
    out.push_str(&format!("| Qwen3 TTS weights | {qwen_weights} |\n"));
    out.push_str(&format!(
        "| Qwen3 vocoder weights | {qwen_vocoder_weights} |\n"
    ));
    out.push_str(&format!("| Whisper load | {whisper_load_ms} ms |\n"));
    out.push_str(&format!("| Whisper weights | {whisper_weights} |\n"));
    out.push_str(&format!("| Min audio tokens | {min_audio_tokens} |\n"));
    out.push_str(&format!("| Max audio tokens | {max_audio_tokens} |\n"));
    out.push_str(&format!("| Max decode tokens | {max_decode_tokens} |\n"));
    out.push_str(&format!("| Total measured run | {total_ms} ms |\n"));
    out.push_str(&format!(
        "| TTS avg | {:.2} ms |\n",
        avg(results, |r| r.tts_ms)
    ));
    out.push_str(&format!(
        "| STT avg | {:.2} ms |\n",
        avg(results, |r| r.stt_ms)
    ));
    out.push_str(&format!(
        "| Ollama avg | {:.2} ms |\n\n",
        avg(results, |r| r.ollama_ms)
    ));

    out.push_str("| # | TTS ms | STT ms | Ollama ms | Samples | Peak | Frames | Response chars | Prompt | Transcript |\n");
    out.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n");
    for result in results {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.4} | {} | {} | {} | {} |\n",
            result.index,
            result.tts_ms,
            result.stt_ms,
            result.ollama_ms,
            result.samples_24k,
            result.peak,
            result.ollama_frames,
            result.response_chars,
            markdown_cell(&result.prompt, 96),
            markdown_cell(&result.transcript, 96),
        ));
    }
    out
}

fn avg(results: &[CaseResult], f: impl Fn(&CaseResult) -> u128) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let total = results.iter().map(f).sum::<u128>() as f64;
    total / results.len() as f64
}

fn markdown_cell(value: &str, max_chars: usize) -> String {
    let mut text = value.replace('|', "\\|").replace('\n', " ");
    if text.chars().count() > max_chars {
        text = text.chars().take(max_chars.saturating_sub(3)).collect();
        text.push_str("...");
    }
    text
}
