#[cfg(feature = "native-mlx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;
    use std::path::PathBuf;
    use vona_mlx_qwen3_tts::{
        DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ, Qwen3TtsSpeechConfig, Qwen3TtsSpeechModel,
    };

    let mut args = std::env::args().skip(1);
    let model_path = args.next().ok_or(
        "usage: qwen3_tts_ab_harness <model-dir> [text]\n\
         set VONA_QWEN3_TTS_SEED for deterministic full-vs-stream codec generation",
    )?;
    let text = args
        .next()
        .unwrap_or_else(|| "Hello from Vona. This is a cached vocoder comparison.".to_string());
    let run_suite = text == "--suite" || std::env::var_os("VONA_QWEN3_TTS_AB_SUITE").is_some();
    let seed = std::env::var("VONA_QWEN3_TTS_SEED").ok();
    if seed.is_none() {
        eprintln!(
            "warning: VONA_QWEN3_TTS_SEED is unset; full and streaming runs may generate different codec paths"
        );
    }

    let mut config = Qwen3TtsSpeechConfig::new(PathBuf::from(model_path));
    if let Some(value) = env_usize("VONA_QWEN3_TTS_AB_MIN_AUDIO_TOKENS") {
        config.min_audio_tokens = value;
    }
    if let Some(value) = env_usize("VONA_QWEN3_TTS_AB_MAX_AUDIO_TOKENS") {
        config.max_audio_tokens = value.max(config.min_audio_tokens);
    }

    eprintln!("loading Qwen3 TTS model...");
    let model = Qwen3TtsSpeechModel::load(config)?;
    eprintln!(
        "loaded model weights={} vocoder_weights={}",
        model.weight_count(),
        model.vocoder_weight_count()
    );

    let chunk_tokens = std::env::var("VONA_MLX_TTS_STREAM_CHUNK_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let stream_vocoder_mode = std::env::var("VONA_MLX_QWEN3_TTS_STREAM_VOCODER_MODE")
        .unwrap_or_else(|_| "rolling".to_string());
    let cases = if run_suite {
        suite_cases()
    } else {
        vec![("single".to_string(), text)]
    };

    let mut case_reports = Vec::with_capacity(cases.len());
    let mut failed_cases = Vec::new();
    for (label, text) in cases {
        eprintln!("running case={label} text={text:?}");
        let case = run_case(&model, &label, &text, chunk_tokens, &stream_vocoder_mode)?;
        if std::env::var_os("VONA_QWEN3_TTS_AB_ENFORCE").is_some() {
            if let Err(error) = enforce_quality(&case.waveform, case.spectral_log_rmse) {
                failed_cases.push(format!("{label}: {error}"));
            }
        }
        case_reports.push(case.report);
    }

    let report = if run_suite {
        json!({
            "suite": "qwen3_tts_ab_small_text_suite",
            "seed": seed,
            "sample_rate_hz": DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ,
            "chunk_audio_tokens": chunk_tokens,
            "stream_vocoder_mode": stream_vocoder_mode,
            "case_count": case_reports.len(),
            "cases": case_reports,
        })
    } else {
        case_reports
            .into_iter()
            .next()
            .unwrap_or_else(|| json!({ "error": "no cases executed" }))
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(path) = std::env::var_os("VONA_QWEN3_TTS_AB_REPORT") {
        std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
        eprintln!("wrote {}", std::path::Path::new(&path).display());
    }

    if !failed_cases.is_empty() {
        return Err(format!(
            "{} quality gate failure(s): {}",
            failed_cases.len(),
            failed_cases.join("; ")
        )
        .into());
    }
    Ok(())
}

#[cfg(feature = "native-mlx")]
fn suite_cases() -> Vec<(String, String)> {
    vec![
        ("short_greeting", "Hello from Vona."),
        (
            "medium_reply",
            "I found the relevant files and I can walk you through the change now.",
        ),
        (
            "long_reply",
            "The wake pipeline is ready for a human test, but the release evidence still needs a balanced corpus with enrolled and unenrolled speakers across quiet, noisy, near-field, and far-field recordings.",
        ),
        (
            "punctuation",
            "Wait, really? Yes: commas, colons, parentheses, and pauses should not break the stream.",
        ),
        (
            "numbers",
            "Schedule the check for 7:45 PM, use threshold 0.62, and keep the sample rate at 24,000 hertz.",
        ),
        (
            "fragment",
            "Right, yes, that makes sense, let me try again.",
        ),
    ]
    .into_iter()
    .map(|(label, text)| (label.to_string(), text.to_string()))
    .collect()
}

#[cfg(feature = "native-mlx")]
struct CaseReport {
    report: serde_json::Value,
    waveform: WaveformDelta,
    spectral_log_rmse: f64,
}

#[cfg(feature = "native-mlx")]
fn run_case(
    model: &vona_mlx_qwen3_tts::Qwen3TtsSpeechModel,
    label: &str,
    text: &str,
    chunk_tokens: usize,
    stream_vocoder_mode: &str,
) -> Result<CaseReport, Box<dyn std::error::Error>> {
    use serde_json::json;
    use std::time::Instant;
    use vona_mlx::MlxSpeechModel;
    use vona_mlx_qwen3_tts::DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ;

    eprintln!("running full offline oracle for case={label}...");
    let offline_start = Instant::now();
    let offline = model.synthesize(text, DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ)?;
    let offline = offline.as_type::<f32>()?;
    offline.eval()?;
    let offline_ms = offline_start.elapsed().as_millis();
    let offline_samples = offline.as_slice::<f32>().to_vec();

    eprintln!("running streaming mode={stream_vocoder_mode} case={label}...");
    let streaming_start = Instant::now();
    let mut first_audio_ms = None;
    let mut stream_samples = Vec::new();
    model.synthesize_streaming(
        text,
        DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ,
        chunk_tokens,
        &mut |chunk| {
            if first_audio_ms.is_none() && !chunk.is_empty() {
                first_audio_ms = Some(streaming_start.elapsed().as_millis());
            }
            stream_samples.extend(chunk);
            Ok(())
        },
    )?;
    let streaming_ms = streaming_start.elapsed().as_millis();

    let waveform = waveform_delta(&offline_samples, &stream_samples);
    let spectral_log_rmse = log_spectral_rmse(&offline_samples, &stream_samples, 512, 256, 64);
    let report = json!({
        "label": label,
        "text": text,
        "offline_ms": offline_ms,
        "streaming_ms": streaming_ms,
        "first_audio_ms": first_audio_ms,
        "offline_samples": offline_samples.len(),
        "streaming_samples": stream_samples.len(),
        "waveform": {
            "compare_samples": waveform.compare_samples,
            "rmse": waveform.rmse,
            "mae": waveform.mae,
            "max_abs": waveform.max_abs,
            "correlation": waveform.correlation,
            "duration_delta_samples": waveform.duration_delta_samples,
        },
        "spectral": {
            "log_rmse": spectral_log_rmse,
        }
    });
    Ok(CaseReport {
        report,
        waveform,
        spectral_log_rmse,
    })
}

#[cfg(feature = "native-mlx")]
fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

#[cfg(feature = "native-mlx")]
#[derive(Debug)]
struct WaveformDelta {
    compare_samples: usize,
    rmse: f64,
    mae: f64,
    max_abs: f64,
    correlation: f64,
    duration_delta_samples: isize,
}

#[cfg(feature = "native-mlx")]
fn waveform_delta(left: &[f32], right: &[f32]) -> WaveformDelta {
    let len = left.len().min(right.len());
    let mut sum_sq = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut left_sq = 0.0_f64;
    let mut right_sq = 0.0_f64;
    let mut dot = 0.0_f64;
    for index in 0..len {
        let l = left[index] as f64;
        let r = right[index] as f64;
        let diff = l - r;
        sum_sq += diff * diff;
        sum_abs += diff.abs();
        max_abs = max_abs.max(diff.abs());
        left_sq += l * l;
        right_sq += r * r;
        dot += l * r;
    }
    let denom = len.max(1) as f64;
    let correlation = if left_sq > 0.0 && right_sq > 0.0 {
        dot / (left_sq.sqrt() * right_sq.sqrt())
    } else {
        0.0
    };
    WaveformDelta {
        compare_samples: len,
        rmse: (sum_sq / denom).sqrt(),
        mae: sum_abs / denom,
        max_abs,
        correlation,
        duration_delta_samples: right.len() as isize - left.len() as isize,
    }
}

#[cfg(feature = "native-mlx")]
fn log_spectral_rmse(left: &[f32], right: &[f32], window: usize, hop: usize, bins: usize) -> f64 {
    let len = left.len().min(right.len());
    if len < window || window == 0 || hop == 0 || bins == 0 {
        return 0.0;
    }
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;
    let mut start = 0usize;
    while start + window <= len {
        for bin in 0..bins {
            let l = dft_bin_log_mag(&left[start..start + window], bin);
            let r = dft_bin_log_mag(&right[start..start + window], bin);
            let diff = l - r;
            sum_sq += diff * diff;
            count += 1;
        }
        start += hop;
    }
    (sum_sq / count.max(1) as f64).sqrt()
}

#[cfg(feature = "native-mlx")]
fn dft_bin_log_mag(samples: &[f32], bin: usize) -> f64 {
    let n = samples.len() as f64;
    let mut real = 0.0_f64;
    let mut imag = 0.0_f64;
    for (index, sample) in samples.iter().enumerate() {
        let phase = -2.0 * std::f64::consts::PI * bin as f64 * index as f64 / n;
        let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * index as f64 / n).cos();
        real += *sample as f64 * window * phase.cos();
        imag += *sample as f64 * window * phase.sin();
    }
    (real.mul_add(real, imag * imag).sqrt() + 1.0e-9).ln()
}

#[cfg(feature = "native-mlx")]
fn enforce_quality(
    waveform: &WaveformDelta,
    spectral_log_rmse: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let max_rmse = std::env::var("VONA_QWEN3_TTS_AB_MAX_RMSE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.02);
    let min_corr = std::env::var("VONA_QWEN3_TTS_AB_MIN_CORRELATION")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.995);
    let max_spectral = std::env::var("VONA_QWEN3_TTS_AB_MAX_LOG_SPECTRAL_RMSE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.5);
    if waveform.rmse > max_rmse {
        return Err(format!("waveform RMSE {} exceeds {}", waveform.rmse, max_rmse).into());
    }
    if waveform.correlation < min_corr {
        return Err(format!(
            "waveform correlation {} is below {}",
            waveform.correlation, min_corr
        )
        .into());
    }
    if spectral_log_rmse > max_spectral {
        return Err(format!(
            "log spectral RMSE {} exceeds {}",
            spectral_log_rmse, max_spectral
        )
        .into());
    }
    Ok(())
}

#[cfg(not(feature = "native-mlx"))]
fn main() {
    println!("enable --features native-mlx to run the Qwen3 TTS A/B harness");
}
