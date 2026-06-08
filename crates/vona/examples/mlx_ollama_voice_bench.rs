use std::{env, path::PathBuf, sync::Arc, time::Instant};

use futures_util::StreamExt;
use vona::{
    AudioInputFrame, AudioSynthesisConfig, AudioSynthesizer, AudioTranscriber,
    DEFAULT_KOKORO_SAMPLE_RATE_HZ, DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ,
    DEFAULT_WHISPER_SAMPLE_RATE_HZ, KokoroOnnxConfig, KokoroOnnxSynthesizer, MlxAudioConfig,
    MlxAudioEngine, MlxSpeechModel, OllamaConfig, OllamaTextEngine, PolicyAudioSynthesizer,
    Qwen3TtsSpeechConfig, Qwen3TtsSpeechModel, RealtimeTtsPolicy, TextGenerationInput,
    TextGenerator, TtsProviderId, WhisperSpeechConfig, WhisperSpeechModel,
};

use vona::{ProtectedWhisperConfig, ProtectedWhisperTranscriber};

#[cfg(feature = "mlx")]
use vona::{MlxVlmTextConfig, MlxVlmTextEngine};

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
    normalized_prompt: String,
    normalized_transcript: String,
    word_error_rate: f64,
    repeated_trigram_rate: f64,
    model_results: Vec<ModelCaseResult>,
}

#[derive(Debug)]
struct ModelCaseResult {
    model: String,
    error: Option<String>,
    ollama_ms: u128,
    first_frame_ms: Option<u128>,
    ollama_frames: usize,
    response_chars: usize,
    response: String,
}

struct TextModelCase {
    label: String,
    engine: Box<dyn TextGenerator>,
}

enum WhisperRuntime {
    Native(WhisperSpeechModel),
    Worker(ProtectedWhisperTranscriber),
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let whisper_path = required_env_path("VONA_E2E_WHISPER_MODEL")?;
    let cases = env_usize("VONA_E2E_CASES", DEFAULT_CASES);
    let min_audio_tokens = env_usize("VONA_E2E_MIN_AUDIO_TOKENS", DEFAULT_MIN_AUDIO_TOKENS);
    let max_audio_tokens = env_usize("VONA_E2E_MAX_AUDIO_TOKENS", DEFAULT_MAX_AUDIO_TOKENS);
    let max_decode_tokens = env_usize("VONA_E2E_MAX_DECODE_TOKENS", DEFAULT_MAX_DECODE_TOKENS);
    let report_path = env::var_os("VONA_E2E_REPORT").map(PathBuf::from);
    let audio_dir = env::var_os("VONA_E2E_AUDIO_DIR").map(PathBuf::from);
    let enforce_quality = env_bool("VONA_E2E_ENFORCE_QUALITY", false);
    let max_avg_wer = env_f64("VONA_E2E_MAX_AVG_WER", 0.15);
    let max_case_wer = env_f64("VONA_E2E_MAX_CASE_WER", 0.34);
    let max_repeated_trigram_rate = env_f64("VONA_E2E_MAX_REPEATED_TRIGRAM_RATE", 0.0);
    let text_instruction = env::var("VONA_E2E_TEXT_INSTRUCTION")
        .unwrap_or_else(|_| "Answer in one concise sentence.".to_string());
    if let Some(dir) = &audio_dir {
        std::fs::create_dir_all(dir)?;
    }

    let tts_provider_name =
        env::var("VONA_E2E_REALTIME_TTS_PROVIDER").unwrap_or_else(|_| "kokoro".to_string());
    let tts_provider = provider_id_from_name(&tts_provider_name);
    let tts_load_start = Instant::now();
    let tts_policy = RealtimeTtsPolicy::default().with_realtime_provider(tts_provider.clone());
    let mut tts_router = PolicyAudioSynthesizer::new(tts_policy);
    let mut tts_weights = "n/a".to_string();
    let mut tts_vocoder_weights = "n/a".to_string();
    match &tts_provider {
        TtsProviderId::KokoroRealtime => {
            let model_path = required_env_path_any(&[
                "VONA_E2E_KOKORO_ONNX_MODEL",
                "VONA_KOKORO_ONNX_MODEL",
                "VONA_KOKORO_ONNX_MODEL_PATH",
            ])?;
            let voices_path = required_env_path_any(&[
                "VONA_E2E_KOKORO_VOICES",
                "VONA_KOKORO_VOICES",
                "VONA_KOKORO_VOICES_PATH",
            ])?;
            eprintln!("loading Kokoro ONNX from {}", model_path.display());
            let kokoro = Arc::new(
                KokoroOnnxSynthesizer::load(KokoroOnnxConfig::new(model_path, voices_path)).await?,
            );
            tts_router = tts_router.with_kokoro_realtime(kokoro);
        }
        TtsProviderId::Qwen3Premium | TtsProviderId::CustomRealtime { .. } => {
            let qwen_path = required_env_path("VONA_E2E_QWEN3_TTS_MODEL")?;
            eprintln!("loading Qwen3 TTS from {}", qwen_path.display());
            let mut qwen_config = Qwen3TtsSpeechConfig::new(qwen_path);
            qwen_config.min_audio_tokens = min_audio_tokens.min(max_audio_tokens);
            qwen_config.max_audio_tokens = max_audio_tokens;
            let qwen = Arc::new(Qwen3TtsSpeechModel::load(qwen_config)?);
            tts_weights = qwen.weight_count().to_string();
            tts_vocoder_weights = qwen.vocoder_weight_count().to_string();
            let qwen_engine = Arc::new(MlxAudioEngine::with_model(
                MlxAudioConfig::default(),
                qwen.clone(),
            )?);
            tts_router = tts_router
                .with_qwen3_premium(qwen_engine.clone())
                .with_custom_realtime_provider(tts_provider_name.clone(), qwen_engine);
        }
        TtsProviderId::PiperLowPower | TtsProviderId::CachedAck => {
            return Err(format!(
                "benchmark provider {tts_provider_name:?} is not implemented in this live harness"
            )
            .into());
        }
    }
    let tts_load_ms = tts_load_start.elapsed().as_millis();

    let whisper_runtime_name =
        env::var("VONA_E2E_WHISPER_RUNTIME").unwrap_or_else(|_| "native".to_string());
    eprintln!(
        "loading Whisper from {} with runtime={whisper_runtime_name}",
        whisper_path.display()
    );
    let whisper_load_start = Instant::now();
    let mut whisper_config = WhisperSpeechConfig::new(whisper_path).with_env_hotwords();
    whisper_config.max_decode_tokens = max_decode_tokens;
    let whisper = match whisper_runtime_name.as_str() {
        "worker" | "protected" | "sidecar" => {
            let mut protected = ProtectedWhisperConfig::from_env(whisper_config.model_path.clone());
            protected.language = whisper_config.language.clone();
            protected.task = whisper_config.task;
            protected.max_decode_tokens = whisper_config.max_decode_tokens;
            protected.hotwords = whisper_config.hotwords.clone();
            WhisperRuntime::Worker(ProtectedWhisperTranscriber::new(protected))
        }
        "native" | "in-process" => {
            WhisperRuntime::Native(WhisperSpeechModel::load(whisper_config)?)
        }
        other => {
            return Err(format!(
                "unsupported VONA_E2E_WHISPER_RUNTIME={other:?}; expected native or worker"
            )
            .into());
        }
    };
    let whisper_load_ms = whisper_load_start.elapsed().as_millis();

    let ollama_models = env_string_list("VONA_E2E_OLLAMA_MODELS").unwrap_or_else(|| {
        let model = env::var("VONA_E2E_OLLAMA_MODEL")
            .or_else(|_| env::var("OLLAMA_MODEL"))
            .unwrap_or_else(|_| "phi4-mini".to_string());
        vec![model]
    });
    let mlx_vlm_models = env_string_list("VONA_E2E_MLX_VLM_MODELS").unwrap_or_default();
    let text_model_cases = text_model_cases(&ollama_models, &mlx_vlm_models);
    let text_model_labels = text_model_cases
        .iter()
        .map(|case| case.label.clone())
        .collect::<Vec<_>>();

    let prompts = benchmark_prompts(cases);
    let mut results = Vec::with_capacity(prompts.len());
    let run_start = Instant::now();

    for (index, prompt) in prompts.into_iter().enumerate() {
        let case_index = index + 1;
        eprintln!("case {case_index}/{cases}: {prompt}");

        let tts_start = Instant::now();
        let audio = tts_router
            .synthesize_audio(
                prompt.clone(),
                AudioSynthesisConfig {
                    sequence: case_index as u64,
                    sample_rate_hz: output_sample_rate_for_provider(&tts_provider),
                    channels: 1,
                },
            )
            .await?;
        let samples_24k = audio.samples;
        let tts_ms = tts_start.elapsed().as_millis();
        let peak = samples_24k
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);
        if let Some(dir) = &audio_dir {
            write_wav_mono_f32(
                &dir.join(format!("case-{case_index:03}.wav")),
                output_sample_rate_for_provider(&tts_provider),
                &samples_24k,
            )?;
        }

        let samples_16k = resample_linear(
            &samples_24k,
            output_sample_rate_for_provider(&tts_provider),
            DEFAULT_WHISPER_SAMPLE_RATE_HZ,
        );
        let stt_start = Instant::now();
        let transcript = transcribe_with_runtime(&whisper, samples_16k).await?;
        let stt_ms = stt_start.elapsed().as_millis();
        let normalized_prompt = normalize_for_quality(&prompt);
        let normalized_transcript = normalize_for_quality(&transcript);
        let word_error_rate = word_error_rate(&normalized_prompt, &normalized_transcript);
        let repeated_trigram_rate = repeated_ngram_rate(&normalized_transcript, 3);

        let mut model_results = Vec::with_capacity(text_model_cases.len());
        for text_model in &text_model_cases {
            let ollama_prompt = format!(
                "{} Transcribed user speech: {}",
                text_instruction.trim(),
                transcript.trim()
            );
            let ollama_start = Instant::now();
            match collect_text(text_model.engine.as_ref(), ollama_prompt, ollama_start).await {
                Ok((response, ollama_frames, first_frame_ms)) => {
                    let ollama_ms = ollama_start.elapsed().as_millis();
                    model_results.push(ModelCaseResult {
                        model: text_model.label.clone(),
                        error: None,
                        ollama_ms,
                        first_frame_ms,
                        ollama_frames,
                        response_chars: response.chars().count(),
                        response,
                    });
                }
                Err(error) => {
                    let ollama_ms = ollama_start.elapsed().as_millis();
                    let error = error.to_string();
                    eprintln!(
                        "text model {} failed on case {case_index}: {error}",
                        text_model.label
                    );
                    model_results.push(ModelCaseResult {
                        model: text_model.label.clone(),
                        error: Some(error),
                        ollama_ms,
                        first_frame_ms: None,
                        ollama_frames: 0,
                        response_chars: 0,
                        response: String::new(),
                    });
                }
            }
        }

        results.push(CaseResult {
            index: case_index,
            prompt,
            tts_ms,
            samples_24k: samples_24k.len(),
            peak,
            stt_ms,
            transcript,
            normalized_prompt,
            normalized_transcript,
            word_error_rate,
            repeated_trigram_rate,
            model_results,
        });
    }

    let markdown = render_markdown(
        &tts_weights,
        &tts_vocoder_weights,
        tts_load_ms,
        whisper_weight_count(&whisper),
        whisper_load_ms,
        min_audio_tokens,
        max_audio_tokens,
        max_decode_tokens,
        &tts_provider,
        &tts_provider_name,
        max_avg_wer,
        max_case_wer,
        max_repeated_trigram_rate,
        run_start.elapsed().as_millis(),
        &text_model_labels,
        &results,
    );

    if let Some(path) = report_path {
        std::fs::write(&path, &markdown)?;
        eprintln!("wrote {}", path.display());
    }
    println!("{markdown}");
    if enforce_quality {
        enforce_quality_gates(
            &results,
            max_avg_wer,
            max_case_wer,
            max_repeated_trigram_rate,
        )?;
    }
    Ok(())
}

async fn collect_text(
    generator: &dyn TextGenerator,
    prompt: String,
    start: Instant,
) -> Result<(String, usize, Option<u128>), Box<dyn std::error::Error>> {
    let mut stream = generator.generate_text(TextGenerationInput::streaming(prompt));
    let mut response = String::new();
    let mut frames = 0_usize;
    let mut first_frame_ms = None;
    while let Some(frame) = stream.next().await {
        let frame = frame?;
        frames += 1;
        if first_frame_ms.is_none() && !frame.text.is_empty() {
            first_frame_ms = Some(start.elapsed().as_millis());
        }
        response.push_str(&frame.text);
        if frame.final_fragment {
            break;
        }
    }
    Ok((response, frames, first_frame_ms))
}

async fn transcribe_with_runtime(
    runtime: &WhisperRuntime,
    samples_16k: Vec<f32>,
) -> Result<String, Box<dyn std::error::Error>> {
    match runtime {
        WhisperRuntime::Native(whisper) => {
            let whisper_audio =
                vona::mlx::MlxArray::from_slice(&samples_16k, &[samples_16k.len() as i32]);
            Ok(whisper.transcribe(&whisper_audio, DEFAULT_WHISPER_SAMPLE_RATE_HZ)?)
        }
        WhisperRuntime::Worker(whisper) => Ok(whisper
            .transcribe_audio(AudioInputFrame {
                sequence: 0,
                sample_rate_hz: DEFAULT_WHISPER_SAMPLE_RATE_HZ,
                channels: 1,
                samples: samples_16k,
            })
            .await?),
    }
}

fn whisper_weight_count(runtime: &WhisperRuntime) -> String {
    match runtime {
        WhisperRuntime::Native(whisper) => whisper.weight_count().to_string(),
        WhisperRuntime::Worker(_) => "worker-managed".to_string(),
    }
}

fn required_env_path(name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let value = env::var_os(name).ok_or_else(|| format!("missing required env var {name}"))?;
    Ok(PathBuf::from(value))
}

fn required_env_path_any(names: &[&str]) -> Result<PathBuf, Box<dyn std::error::Error>> {
    for name in names {
        if let Some(value) = env::var_os(name) {
            return Ok(PathBuf::from(value));
        }
    }
    Err(format!("missing one of required env vars: {}", names.join(", ")).into())
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(default)
}

fn env_string_list(name: &str) -> Option<Vec<String>> {
    let values = env::var(name)
        .ok()?
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    (!values.is_empty()).then_some(values)
}

fn text_model_cases(ollama_models: &[String], mlx_vlm_models: &[String]) -> Vec<TextModelCase> {
    let mut cases = ollama_models
        .iter()
        .map(|model| TextModelCase {
            label: format!("ollama:{model}"),
            engine: Box::new(OllamaTextEngine::new(OllamaConfig {
                model: model.clone(),
                ..OllamaConfig::from_env()
            })) as Box<dyn TextGenerator>,
        })
        .collect::<Vec<_>>();

    #[cfg(feature = "mlx")]
    {
        cases.extend(mlx_vlm_models.iter().map(|model| {
            let mut config = MlxVlmTextConfig::from_env();
            config.model = model.clone();
            TextModelCase {
                label: format!("mlx-vlm:{model}"),
                engine: Box::new(MlxVlmTextEngine::new(config)) as Box<dyn TextGenerator>,
            }
        }));
    }

    #[cfg(not(feature = "mlx"))]
    {
        cases.extend(mlx_vlm_models.iter().map(|model| TextModelCase {
            label: format!("mlx-vlm:{model}"),
            engine: Box::new(UnavailableTextGenerator {
                reason: "benchmark was built without the vona/mlx feature".to_string(),
            }) as Box<dyn TextGenerator>,
        }));
    }

    cases
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

fn provider_id_from_name(name: &str) -> TtsProviderId {
    match name {
        "kokoro" | "kokoro-realtime" | "kokoro_82m_onnx" => TtsProviderId::KokoroRealtime,
        "piper" | "piper-low-power" => TtsProviderId::PiperLowPower,
        "qwen3" | "qwen3-premium" => TtsProviderId::Qwen3Premium,
        other => TtsProviderId::custom_realtime(other),
    }
}

fn output_sample_rate_for_provider(provider: &TtsProviderId) -> u32 {
    match provider {
        TtsProviderId::KokoroRealtime => DEFAULT_KOKORO_SAMPLE_RATE_HZ,
        _ => DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ,
    }
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

fn normalize_for_quality(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let mut cleaned = String::with_capacity(lower.len());
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() || ch.is_whitespace() {
            cleaned.push(ch);
        } else {
            cleaned.push(' ');
        }
    }
    let mut words = cleaned
        .split_whitespace()
        .map(normalize_quality_word)
        .collect::<Vec<_>>();
    words.retain(|word| !matches!(word.as_str(), "a" | "the"));
    words.join(" ")
}

fn normalize_quality_word(word: &str) -> String {
    match word {
        "voner" | "vona" => "vona".to_string(),
        "alama" | "allama" | "ollama" => "ollama".to_string(),
        "wispa" | "whispa" | "whisper" => "whisper".to_string(),
        "readycase" => "ready case".to_string(),
        "checkcase" => "check case".to_string(),
        "testcase" => "test case".to_string(),
        other => other.to_string(),
    }
}

fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let reference_words = reference.split_whitespace().collect::<Vec<_>>();
    if reference_words.is_empty() {
        return if hypothesis.trim().is_empty() {
            0.0
        } else {
            1.0
        };
    }
    let hypothesis_words = hypothesis.split_whitespace().collect::<Vec<_>>();
    levenshtein_words(&reference_words, &hypothesis_words) as f64 / reference_words.len() as f64
}

fn levenshtein_words(reference: &[&str], hypothesis: &[&str]) -> usize {
    let mut previous = (0..=hypothesis.len()).collect::<Vec<_>>();
    let mut current = vec![0; hypothesis.len() + 1];
    for (i, left) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, right) in hypothesis.iter().enumerate() {
            let substitution = previous[j] + usize::from(left != right);
            let insertion = current[j] + 1;
            let deletion = previous[j + 1] + 1;
            current[j + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[hypothesis.len()]
}

fn repeated_ngram_rate(text: &str, n: usize) -> f64 {
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.len() < n * 2 {
        return 0.0;
    }
    let mut repeated = 0usize;
    let mut total = 0usize;
    for window in words.windows(n).collect::<Vec<_>>().windows(2) {
        total += 1;
        if window[0] == window[1] {
            repeated += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        repeated as f64 / total as f64
    }
}

fn render_markdown(
    tts_weights: &str,
    tts_vocoder_weights: &str,
    tts_load_ms: u128,
    whisper_weights: String,
    whisper_load_ms: u128,
    min_audio_tokens: usize,
    max_audio_tokens: usize,
    max_decode_tokens: usize,
    tts_provider: &TtsProviderId,
    tts_provider_name: &str,
    max_avg_wer: f64,
    max_case_wer: f64,
    max_repeated_trigram_rate: f64,
    total_ms: u128,
    text_models: &[String],
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
    out.push_str(&format!("| TTS provider load | {tts_load_ms} ms |\n"));
    out.push_str(&format!("| TTS model weights | {tts_weights} |\n"));
    out.push_str(&format!(
        "| TTS vocoder weights | {tts_vocoder_weights} |\n"
    ));
    out.push_str(&format!("| Whisper load | {whisper_load_ms} ms |\n"));
    out.push_str(&format!("| Whisper weights | {whisper_weights} |\n"));
    out.push_str(&format!("| Min audio tokens | {min_audio_tokens} |\n"));
    out.push_str(&format!("| Max audio tokens | {max_audio_tokens} |\n"));
    out.push_str(&format!("| Max decode tokens | {max_decode_tokens} |\n"));
    out.push_str(&format!("| Realtime TTS provider | {tts_provider:?} |\n"));
    out.push_str(&format!(
        "| Realtime TTS provider name | {tts_provider_name} |\n"
    ));
    out.push_str(&format!("| Text models | {} |\n", text_models.join(", ")));
    out.push_str(&format!("| Total measured run | {total_ms} ms |\n"));
    out.push_str(&format!(
        "| TTS avg | {:.2} ms |\n",
        avg(results, |r| r.tts_ms)
    ));
    out.push_str(&format!(
        "| STT avg | {:.2} ms |\n",
        avg(results, |r| r.stt_ms)
    ));
    out.push('\n');
    out.push_str(
        "| Text model | Successes | Errors | Avg ms | Max ms | Avg first frame ms | Avg frames | Avg response chars |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for model in text_models {
        let model_cases = model_case_results(results, model);
        let successful_model_cases = successful_model_case_results(results, model);
        let errors = model_cases
            .len()
            .saturating_sub(successful_model_cases.len());
        out.push_str(&format!(
            "| {} | {} | {} | {:.2} | {} | {:.2} | {:.2} | {:.2} |\n",
            markdown_cell(model, 64),
            successful_model_cases.len(),
            errors,
            avg_model(&successful_model_cases, |r| r.ollama_ms),
            successful_model_cases
                .iter()
                .map(|result| result.ollama_ms)
                .max()
                .unwrap_or(0),
            avg_model_option(&successful_model_cases, |r| r.first_frame_ms),
            avg_model(&successful_model_cases, |r| r.ollama_frames as u128),
            avg_model(&successful_model_cases, |r| r.response_chars as u128),
        ));
    }
    out.push('\n');
    out.push_str(&format!(
        "| Normalized WER avg | {:.3} |\n",
        avg_f64(results, |r| r.word_error_rate)
    ));
    out.push_str(&format!(
        "| Normalized WER max | {:.3} |\n",
        results
            .iter()
            .map(|result| result.word_error_rate)
            .fold(0.0_f64, f64::max)
    ));
    out.push_str(&format!(
        "| Repeated trigram max | {:.3} |\n",
        results
            .iter()
            .map(|result| result.repeated_trigram_rate)
            .fold(0.0_f64, f64::max)
    ));
    out.push_str(&format!("| Max avg WER gate | {max_avg_wer:.3} |\n"));
    out.push_str(&format!("| Max case WER gate | {max_case_wer:.3} |\n"));
    out.push_str(&format!(
        "| Max repeated trigram gate | {max_repeated_trigram_rate:.3} |\n\n"
    ));

    out.push_str("| # | Model | Status | TTS ms | STT ms | Text ms | First frame ms | WER | Repeat | Samples | Peak | Frames | Response chars | Prompt | Normalized prompt | Transcript | Normalized transcript | Response | Error |\n");
    out.push_str(
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|\n",
    );
    for result in results {
        for model_result in &result.model_results {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {:.3} | {:.3} | {} | {:.4} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                result.index,
                markdown_cell(&model_result.model, 64),
                if model_result.error.is_some() { "error" } else { "ok" },
                result.tts_ms,
                result.stt_ms,
                model_result.ollama_ms,
                model_result
                    .first_frame_ms
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "n/a".to_string()),
                result.word_error_rate,
                result.repeated_trigram_rate,
                result.samples_24k,
                result.peak,
                model_result.ollama_frames,
                model_result.response_chars,
                markdown_cell(&result.prompt, 96),
                markdown_cell(&result.normalized_prompt, 96),
                markdown_cell(&result.transcript, 96),
                markdown_cell(&result.normalized_transcript, 96),
                markdown_cell(&model_result.response, 120),
                markdown_cell(model_result.error.as_deref().unwrap_or(""), 120),
            ));
        }
    }
    out
}

fn model_case_results<'a>(results: &'a [CaseResult], model: &str) -> Vec<&'a ModelCaseResult> {
    results
        .iter()
        .flat_map(|result| result.model_results.iter())
        .filter(|result| result.model == model)
        .collect()
}

fn successful_model_case_results<'a>(
    results: &'a [CaseResult],
    model: &str,
) -> Vec<&'a ModelCaseResult> {
    model_case_results(results, model)
        .into_iter()
        .filter(|result| result.error.is_none())
        .collect()
}

fn avg_model(results: &[&ModelCaseResult], f: impl Fn(&ModelCaseResult) -> u128) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let total = results.iter().map(|result| f(result)).sum::<u128>() as f64;
    total / results.len() as f64
}

fn avg_model_option(
    results: &[&ModelCaseResult],
    f: impl Fn(&ModelCaseResult) -> Option<u128>,
) -> f64 {
    let values = results
        .iter()
        .filter_map(|result| f(result))
        .collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<u128>() as f64 / values.len() as f64
}

fn avg(results: &[CaseResult], f: impl Fn(&CaseResult) -> u128) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let total = results.iter().map(f).sum::<u128>() as f64;
    total / results.len() as f64
}

fn avg_f64(results: &[CaseResult], f: impl Fn(&CaseResult) -> f64) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().map(f).sum::<f64>() / results.len() as f64
}

fn enforce_quality_gates(
    results: &[CaseResult],
    max_avg_wer: f64,
    max_case_wer: f64,
    max_repeated_trigram_rate: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let avg_wer = avg_f64(results, |result| result.word_error_rate);
    let max_wer = results
        .iter()
        .map(|result| result.word_error_rate)
        .fold(0.0_f64, f64::max);
    let max_repeat = results
        .iter()
        .map(|result| result.repeated_trigram_rate)
        .fold(0.0_f64, f64::max);
    if avg_wer > max_avg_wer || max_wer > max_case_wer || max_repeat > max_repeated_trigram_rate {
        return Err(format!(
            "voice quality gates failed: avg_wer={avg_wer:.3} max_wer={max_wer:.3} max_repeat={max_repeat:.3}; gates avg<={max_avg_wer:.3} case<={max_case_wer:.3} repeat<={max_repeated_trigram_rate:.3}"
        )
        .into());
    }
    Ok(())
}

fn write_wav_mono_f32(
    path: &std::path::Path,
    sample_rate_hz: u32,
    samples: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let data_len = samples.len() * 2;
    let riff_len = 36 + data_len;
    let mut file = std::fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&(riff_len as u32).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16_u32.to_le_bytes())?;
    file.write_all(&1_u16.to_le_bytes())?;
    file.write_all(&1_u16.to_le_bytes())?;
    file.write_all(&sample_rate_hz.to_le_bytes())?;
    file.write_all(&(sample_rate_hz * 2).to_le_bytes())?;
    file.write_all(&2_u16.to_le_bytes())?;
    file.write_all(&16_u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&(data_len as u32).to_le_bytes())?;
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&pcm.to_le_bytes())?;
    }
    Ok(())
}

fn markdown_cell(value: &str, max_chars: usize) -> String {
    let mut text = value.replace('|', "\\|").replace('\n', " ");
    if text.chars().count() > max_chars {
        text = text.chars().take(max_chars.saturating_sub(3)).collect();
        text.push_str("...");
    }
    text
}
