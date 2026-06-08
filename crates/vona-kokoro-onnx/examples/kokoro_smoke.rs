use std::{env, io::Write, path::Path};

use vona_core::{AudioSynthesisConfig, AudioSynthesizer};
use vona_kokoro_onnx::{DEFAULT_KOKORO_SAMPLE_RATE_HZ, KokoroOnnxConfig, KokoroOnnxSynthesizer};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .or_else(|| env::var("VONA_KOKORO_ONNX_MODEL").ok())
        .ok_or("usage: kokoro_smoke <model.onnx> <voices.bin> [text]")?;
    let voices_path = args
        .next()
        .or_else(|| env::var("VONA_KOKORO_VOICES").ok())
        .ok_or("usage: kokoro_smoke <model.onnx> <voices.bin> [text]")?;
    let text = args
        .next()
        .unwrap_or_else(|| "Hello from Vona Kokoro.".to_string());
    let mut config = KokoroOnnxConfig::new(model_path, voices_path);
    if let Ok(voice) = env::var("VONA_KOKORO_VOICE") {
        config.voice = voice;
    }

    let info = KokoroOnnxSynthesizer::inspect_model(&config)?;
    eprintln!("inputs: {:?}", info.inputs);
    eprintln!("outputs: {:?}", info.outputs);

    let synth = KokoroOnnxSynthesizer::load(config).await?;
    let frame = synth
        .synthesize_audio(
            text,
            AudioSynthesisConfig {
                sequence: 1,
                sample_rate_hz: DEFAULT_KOKORO_SAMPLE_RATE_HZ,
                channels: 1,
            },
        )
        .await?;
    let peak = frame
        .samples
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    println!(
        "samples={} sample_rate={} peak={peak:.6}",
        frame.samples.len(),
        frame.sample_rate_hz
    );
    if let Some(path) = env::var_os("VONA_KOKORO_WAV") {
        write_wav_mono_f32(Path::new(&path), frame.sample_rate_hz, &frame.samples)?;
        eprintln!("wrote {}", Path::new(&path).display());
    }
    Ok(())
}

fn write_wav_mono_f32(
    path: &Path,
    sample_rate_hz: u32,
    samples: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
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
