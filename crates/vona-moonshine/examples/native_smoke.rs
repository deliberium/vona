use std::{env, fs, path::PathBuf};

use vona_core::{AudioInputFrame, AudioTranscriber};
use vona_moonshine::{
    DEFAULT_MOONSHINE_ARCH, MoonshineTranscriberConfig, NativeMoonshineTranscriber,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio_path = env::var("VONA_MOONSHINE_SMOKE_PCM16LE")
        .map(PathBuf::from)
        .map_err(|_| "set VONA_MOONSHINE_SMOKE_PCM16LE to a 16 kHz mono pcm16le file")?;
    let library_path = env::var("VONA_MOONSHINE_LIBRARY_PATH")
        .map(PathBuf::from)
        .map_err(|_| "set VONA_MOONSHINE_LIBRARY_PATH")?;
    let model_path = env::var("VONA_MOONSHINE_MODEL_PATH")
        .map(PathBuf::from)
        .map_err(|_| "set VONA_MOONSHINE_MODEL_PATH")?;
    let model_arch =
        env::var("VONA_MOONSHINE_ARCH").unwrap_or_else(|_| DEFAULT_MOONSHINE_ARCH.to_string());

    let raw = fs::read(audio_path)?;
    let samples = raw
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
        .collect::<Vec<_>>();

    let mut config = MoonshineTranscriberConfig::from_env();
    config.native_library_path = Some(library_path);
    config.model_path = Some(model_path);
    config.model_arch = model_arch;

    let transcriber = NativeMoonshineTranscriber::load(config)?;
    let transcript = transcriber
        .transcribe_audio(AudioInputFrame {
            sequence: 1,
            sample_rate_hz: 16_000,
            channels: 1,
            samples,
        })
        .await?;
    println!("{transcript}");
    Ok(())
}
