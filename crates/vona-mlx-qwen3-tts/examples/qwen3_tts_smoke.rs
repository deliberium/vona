#[cfg(feature = "native-mlx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;
    use vona_mlx::MlxSpeechModel;
    use vona_mlx_qwen3_tts::{
        DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ, Qwen3TtsSpeechConfig, Qwen3TtsSpeechModel,
    };

    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .ok_or("usage: qwen3_tts_smoke <model-dir> [text]")?;
    let text = args
        .next()
        .unwrap_or_else(|| "Hello from Vona.".to_string());

    let mut config = Qwen3TtsSpeechConfig::new(PathBuf::from(model_path));
    config.min_audio_tokens = std::env::var("VONA_QWEN3_TTS_MIN_AUDIO_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    config.max_audio_tokens = std::env::var("VONA_QWEN3_TTS_MAX_AUDIO_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    eprintln!("loading Qwen3 TTS model...");
    let model = Qwen3TtsSpeechModel::load(config)?;
    eprintln!(
        "loaded model weights={} vocoder_weights={}",
        model.weight_count(),
        model.vocoder_weight_count()
    );
    eprintln!("synthesizing...");
    let audio = model.synthesize(&text, DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ)?;
    eprintln!("materializing output...");
    let audio = audio.as_type::<f32>()?;
    audio.eval()?;
    let samples = audio.as_slice::<f32>();
    let peak = samples
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    println!(
        "shape={:?} samples={} peak={peak:.6}",
        audio.shape(),
        samples.len()
    );
    if let Some(path) = std::env::var_os("VONA_QWEN3_TTS_WAV") {
        write_wav_mono_f32(
            std::path::Path::new(&path),
            DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ,
            samples,
        )?;
        eprintln!("wrote {}", std::path::Path::new(&path).display());
    }
    Ok(())
}

#[cfg(feature = "native-mlx")]
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

#[cfg(not(feature = "native-mlx"))]
fn main() {
    println!("enable --features native-mlx to run the Qwen3 TTS smoke example");
}
