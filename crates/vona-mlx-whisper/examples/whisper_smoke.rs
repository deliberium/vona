#[cfg(feature = "native-mlx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;
    use vona_mlx::MlxSpeechModel;
    use vona_mlx_whisper::{
        DEFAULT_WHISPER_SAMPLE_RATE_HZ, WhisperSpeechConfig, WhisperSpeechModel,
    };

    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .ok_or("usage: whisper_smoke <model-dir> [seconds-or-wav-path]")?;
    let input = args.next();
    let (audio_samples, description) = if let Some(input) = input {
        let path = PathBuf::from(&input);
        if path.is_file() {
            (read_pcm16_wav(&path)?, format!("{}", path.display()))
        } else {
            let seconds = input.parse::<usize>().unwrap_or(1).max(1);
            (
                vec![0.0_f32; seconds * DEFAULT_WHISPER_SAMPLE_RATE_HZ as usize],
                format!("{seconds}s silence"),
            )
        }
    } else {
        (
            vec![0.0_f32; DEFAULT_WHISPER_SAMPLE_RATE_HZ as usize],
            "1s silence".to_string(),
        )
    };
    let audio = mlx_rs::Array::from_slice(&audio_samples, &[audio_samples.len() as i32]);

    eprintln!("loading Whisper model...");
    let mut config = WhisperSpeechConfig::new(PathBuf::from(model_path));
    config.max_decode_tokens = std::env::var("VONA_WHISPER_MAX_DECODE_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(config.max_decode_tokens);
    let model = WhisperSpeechModel::load(config)?;
    eprintln!("loaded model weights={}", model.weight_count());
    eprintln!("transcribing {description}...");
    let transcript = model.transcribe(&audio, DEFAULT_WHISPER_SAMPLE_RATE_HZ)?;
    println!("{transcript}");
    Ok(())
}

#[cfg(feature = "native-mlx")]
fn read_pcm16_wav(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("expected a RIFF/WAVE file".into());
    }

    let mut offset = 12usize;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?) as usize;
        let start = offset + 8;
        let end = start.saturating_add(size).min(bytes.len());
        if id == b"fmt " {
            if size < 16 {
                return Err("invalid WAV fmt chunk".into());
            }
            let format = u16::from_le_bytes(bytes[start..start + 2].try_into()?);
            channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into()?);
            sample_rate = u32::from_le_bytes(bytes[start + 4..start + 8].try_into()?);
            bits_per_sample = u16::from_le_bytes(bytes[start + 14..start + 16].try_into()?);
            if format != 1 {
                return Err("expected PCM WAV format".into());
            }
        } else if id == b"data" {
            data = Some((start, end));
        }
        offset = end + (size % 2);
    }

    if channels != 1
        || sample_rate != vona_mlx_whisper::DEFAULT_WHISPER_SAMPLE_RATE_HZ
        || bits_per_sample != 16
    {
        return Err(format!(
            "expected mono {} Hz 16-bit PCM WAV, got channels={channels} sample_rate={sample_rate} bits={bits_per_sample}",
            vona_mlx_whisper::DEFAULT_WHISPER_SAMPLE_RATE_HZ
        )
        .into());
    }
    let (start, end) = data.ok_or("missing WAV data chunk")?;
    Ok(bytes[start..end]
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / i16::MAX as f32)
        .collect())
}

#[cfg(not(feature = "native-mlx"))]
fn main() {
    println!("enable --features native-mlx to run the Whisper smoke example");
}
