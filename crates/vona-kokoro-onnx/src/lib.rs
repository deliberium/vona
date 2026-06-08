use async_trait::async_trait;
use kokoro_micro::TtsEngine;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};
use thiserror::Error;
use vona_core::{AudioOutputFrame, AudioProcessingError, AudioSynthesisConfig, AudioSynthesizer};

pub const DEFAULT_KOKORO_SAMPLE_RATE_HZ: u32 = 24_000;
pub const DEFAULT_KOKORO_VOICE: &str = "af_heart";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KokoroOnnxConfig {
    pub model_path: PathBuf,
    pub voices_path: PathBuf,
    pub voice: String,
    pub speed: f32,
    pub sample_rate_hz: u32,
    pub input_ids_name: String,
    pub style_name: String,
    pub speed_name: String,
    pub output_name: Option<String>,
}

impl KokoroOnnxConfig {
    pub fn new(model_path: impl Into<PathBuf>, voices_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            voices_path: voices_path.into(),
            voice: DEFAULT_KOKORO_VOICE.to_string(),
            speed: 1.0,
            sample_rate_hz: DEFAULT_KOKORO_SAMPLE_RATE_HZ,
            input_ids_name: "tokens".to_string(),
            style_name: "style".to_string(),
            speed_name: "speed".to_string(),
            output_name: None,
        }
    }

    pub fn from_env() -> Result<Self, KokoroOnnxError> {
        let model_path = std::env::var_os("VONA_KOKORO_ONNX_MODEL")
            .or_else(|| std::env::var_os("VONA_KOKORO_ONNX_MODEL_PATH"))
            .ok_or_else(|| KokoroOnnxError::Config("missing VONA_KOKORO_ONNX_MODEL".to_string()))?;
        let voices_path = std::env::var_os("VONA_KOKORO_VOICES")
            .or_else(|| std::env::var_os("VONA_KOKORO_VOICES_PATH"))
            .ok_or_else(|| KokoroOnnxError::Config("missing VONA_KOKORO_VOICES".to_string()))?;
        let mut config = Self::new(model_path, voices_path);
        if let Ok(voice) = std::env::var("VONA_KOKORO_VOICE") {
            config.voice = voice;
        }
        if let Ok(speed) = std::env::var("VONA_KOKORO_SPEED") {
            config.speed = speed.parse().map_err(|err| {
                KokoroOnnxError::Config(format!("invalid VONA_KOKORO_SPEED: {err}"))
            })?;
        }
        Ok(config)
    }
}

#[derive(Debug, Error)]
pub enum KokoroOnnxError {
    #[error("kokoro config error: {0}")]
    Config(String),
    #[error("kokoro model load failed: {0}")]
    Model(String),
    #[error("kokoro voice load failed: {0}")]
    Voice(String),
    #[error("kokoro tokenization failed: {0}")]
    Tokenization(String),
    #[error("kokoro inference failed: {0}")]
    Inference(String),
}

impl From<KokoroOnnxError> for AudioProcessingError {
    fn from(error: KokoroOnnxError) -> Self {
        match error {
            KokoroOnnxError::Config(message)
            | KokoroOnnxError::Model(message)
            | KokoroOnnxError::Voice(message) => AudioProcessingError::ModelUnavailable(message),
            KokoroOnnxError::Tokenization(message) => AudioProcessingError::InvalidInput(message),
            KokoroOnnxError::Inference(message) => AudioProcessingError::Inference(message),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KokoroVoice {
    embeddings: Vec<f32>,
    rows: usize,
    dims: usize,
}

impl KokoroVoice {
    pub fn load(voices_path: &Path, voice: &str) -> Result<Self, KokoroOnnxError> {
        let entry_name = format!("{voice}.npy");
        let bytes = read_stored_zip_entry(voices_path, &entry_name)?;
        let npy = parse_npy_f32(&bytes)?;
        let (rows, dims) = match npy.shape.as_slice() {
            [rows, dims] => (*rows, *dims),
            [rows, 1, dims] => (*rows, *dims),
            _ => {
                return Err(KokoroOnnxError::Voice(format!(
                    "expected 2D or [rows, 1, dims] Kokoro voice tensor for {voice}, got shape {:?}",
                    npy.shape
                )));
            }
        };
        if dims != 256 {
            return Err(KokoroOnnxError::Voice(format!(
                "expected Kokoro voice embedding dim 256 for {voice}, got {dims}"
            )));
        }
        Ok(Self {
            embeddings: npy.values,
            rows,
            dims,
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.dims)
    }
}

pub struct KokoroOnnxSynthesizer {
    config: KokoroOnnxConfig,
    engine: Mutex<TtsEngine>,
}

impl KokoroOnnxSynthesizer {
    pub async fn load(config: KokoroOnnxConfig) -> Result<Self, KokoroOnnxError> {
        let engine = TtsEngine::with_paths(
            &config.model_path.to_string_lossy(),
            &config.voices_path.to_string_lossy(),
        )
        .await
        .map_err(|err| KokoroOnnxError::Model(format!("kokoro engine load failed: {err}")))?;
        Ok(Self {
            config,
            engine: Mutex::new(engine),
        })
    }

    pub fn inspect_model(config: &KokoroOnnxConfig) -> Result<KokoroModelInfo, KokoroOnnxError> {
        let session = Session::builder()
            .map_err(|err| KokoroOnnxError::Model(format!("onnx session builder failed: {err}")))?
            .commit_from_file(&config.model_path)
            .map_err(|err| {
                KokoroOnnxError::Model(format!(
                    "failed to load ONNX model at {}: {err}",
                    config.model_path.display()
                ))
            })?;
        Ok(KokoroModelInfo {
            inputs: session
                .inputs()
                .iter()
                .map(|input| format!("{}: {}", input.name(), input.dtype()))
                .collect(),
            outputs: session
                .outputs()
                .iter()
                .map(|output| format!("{}: {}", output.name(), output.dtype()))
                .collect(),
        })
    }

    pub fn synthesize_samples(&self, text: &str) -> Result<Vec<f32>, KokoroOnnxError> {
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| KokoroOnnxError::Inference("kokoro engine lock poisoned".to_string()))?;
        engine
            .synthesize_with_options(
                text,
                Some(&self.config.voice),
                self.config.speed,
                1.0,
                Some("en"),
            )
            .map_err(|err| KokoroOnnxError::Inference(format!("kokoro synthesis failed: {err}")))
    }
}

#[async_trait]
impl AudioSynthesizer for KokoroOnnxSynthesizer {
    async fn synthesize_audio(
        &self,
        text: String,
        config: AudioSynthesisConfig,
    ) -> Result<AudioOutputFrame, AudioProcessingError> {
        if config.channels != 1 {
            return Err(AudioProcessingError::InvalidInput(format!(
                "Kokoro ONNX currently emits mono audio, requested {} channels",
                config.channels
            )));
        }
        let mut samples = self
            .synthesize_samples(&text)
            .map_err(AudioProcessingError::from)?;
        if config.sample_rate_hz != self.config.sample_rate_hz {
            samples = resample_mono(&samples, self.config.sample_rate_hz, config.sample_rate_hz);
        }
        Ok(AudioOutputFrame {
            sequence: config.sequence,
            sample_rate_hz: config.sample_rate_hz,
            channels: config.channels,
            samples,
            is_filler: false,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KokoroModelInfo {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

fn read_stored_zip_entry(path: &Path, entry_name: &str) -> Result<Vec<u8>, KokoroOnnxError> {
    let bytes = fs::read(path).map_err(|err| {
        KokoroOnnxError::Voice(format!("failed to read {}: {err}", path.display()))
    })?;
    let eocd = bytes
        .windows(4)
        .rposition(|window| window == b"PK\x05\x06")
        .ok_or_else(|| {
            KokoroOnnxError::Voice(format!("zip EOCD not found in {}", path.display()))
        })?;
    if eocd + 22 > bytes.len() {
        return Err(KokoroOnnxError::Voice(format!(
            "truncated zip EOCD in {}",
            path.display()
        )));
    }
    let central_dir_size = u32::from_le_bytes([
        bytes[eocd + 12],
        bytes[eocd + 13],
        bytes[eocd + 14],
        bytes[eocd + 15],
    ]) as usize;
    let central_dir_offset = u32::from_le_bytes([
        bytes[eocd + 16],
        bytes[eocd + 17],
        bytes[eocd + 18],
        bytes[eocd + 19],
    ]) as usize;
    let central_dir_end = central_dir_offset + central_dir_size;
    let mut offset = central_dir_offset;
    while offset + 46 <= central_dir_end && offset + 46 <= bytes.len() {
        if &bytes[offset..offset + 4] != b"PK\x01\x02" {
            return Err(KokoroOnnxError::Voice(format!(
                "invalid central directory entry in {}",
                path.display()
            )));
        }
        let method = u16::from_le_bytes([bytes[offset + 10], bytes[offset + 11]]);
        let compressed_size = u32::from_le_bytes([
            bytes[offset + 20],
            bytes[offset + 21],
            bytes[offset + 22],
            bytes[offset + 23],
        ]) as usize;
        let uncompressed_size = u32::from_le_bytes([
            bytes[offset + 24],
            bytes[offset + 25],
            bytes[offset + 26],
            bytes[offset + 27],
        ]) as usize;
        let name_len = u16::from_le_bytes([bytes[offset + 28], bytes[offset + 29]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[offset + 30], bytes[offset + 31]]) as usize;
        let comment_len = u16::from_le_bytes([bytes[offset + 32], bytes[offset + 33]]) as usize;
        let local_header_offset = u32::from_le_bytes([
            bytes[offset + 42],
            bytes[offset + 43],
            bytes[offset + 44],
            bytes[offset + 45],
        ]) as usize;
        let name_start = offset + 46;
        let name_end = name_start + name_len;
        if name_end > bytes.len() {
            return Err(KokoroOnnxError::Voice(format!(
                "truncated zip entry name in {}",
                path.display()
            )));
        }
        let name = std::str::from_utf8(&bytes[name_start..name_end]).map_err(|err| {
            KokoroOnnxError::Voice(format!(
                "invalid zip entry name in {}: {err}",
                path.display()
            ))
        })?;
        if name == entry_name {
            if method != 0 {
                return Err(KokoroOnnxError::Voice(format!(
                    "zip entry {entry_name} uses unsupported compression method {method}"
                )));
            }
            if compressed_size != uncompressed_size {
                return Err(KokoroOnnxError::Voice(format!(
                    "stored zip entry {entry_name} size mismatch"
                )));
            }
            if local_header_offset + 30 > bytes.len()
                || &bytes[local_header_offset..local_header_offset + 4] != b"PK\x03\x04"
            {
                return Err(KokoroOnnxError::Voice(format!(
                    "invalid local header for zip entry {entry_name}"
                )));
            }
            let local_name_len = u16::from_le_bytes([
                bytes[local_header_offset + 26],
                bytes[local_header_offset + 27],
            ]) as usize;
            let local_extra_len = u16::from_le_bytes([
                bytes[local_header_offset + 28],
                bytes[local_header_offset + 29],
            ]) as usize;
            let data_start = local_header_offset + 30 + local_name_len + local_extra_len;
            let data_end = data_start + compressed_size;
            if data_end > bytes.len() {
                return Err(KokoroOnnxError::Voice(format!(
                    "zip entry {entry_name} extends past end of {}",
                    path.display()
                )));
            }
            return Ok(bytes[data_start..data_end].to_vec());
        }
        offset = name_start + name_len + extra_len + comment_len;
    }
    Err(KokoroOnnxError::Voice(format!(
        "voice entry {entry_name} not found in {}",
        path.display()
    )))
}

struct NpyF32 {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn parse_npy_f32(bytes: &[u8]) -> Result<NpyF32, KokoroOnnxError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(KokoroOnnxError::Voice("invalid npy header".to_string()));
    }
    let major = bytes[6];
    let header_len_start = 8;
    let (header_len, data_start) = if major <= 1 {
        (
            u16::from_le_bytes([bytes[header_len_start], bytes[header_len_start + 1]]) as usize,
            10usize,
        )
    } else {
        (
            u32::from_le_bytes([
                bytes[header_len_start],
                bytes[header_len_start + 1],
                bytes[header_len_start + 2],
                bytes[header_len_start + 3],
            ]) as usize,
            12usize,
        )
    };
    let header_end = data_start + header_len;
    if header_end > bytes.len() {
        return Err(KokoroOnnxError::Voice("truncated npy header".to_string()));
    }
    let header = std::str::from_utf8(&bytes[data_start..header_end])
        .map_err(|err| KokoroOnnxError::Voice(format!("invalid npy header utf8: {err}")))?;
    if !header.contains("'descr': '<f4'") && !header.contains("\"descr\": \"<f4\"") {
        return Err(KokoroOnnxError::Voice(format!(
            "unsupported npy dtype in header: {header}"
        )));
    }
    if header.contains("True") {
        return Err(KokoroOnnxError::Voice(
            "fortran-order npy voice tensors are unsupported".to_string(),
        ));
    }
    let shape = parse_npy_shape(header)?;
    let expected_values = shape.iter().product::<usize>();
    let expected_bytes = expected_values * 4;
    if header_end + expected_bytes > bytes.len() {
        return Err(KokoroOnnxError::Voice("truncated npy data".to_string()));
    }
    let values = bytes[header_end..header_end + expected_bytes]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(NpyF32 { shape, values })
}

fn parse_npy_shape(header: &str) -> Result<Vec<usize>, KokoroOnnxError> {
    let shape_start = header
        .find('(')
        .ok_or_else(|| KokoroOnnxError::Voice(format!("missing npy shape in header: {header}")))?;
    let shape_end = header[shape_start..]
        .find(')')
        .map(|index| shape_start + index)
        .ok_or_else(|| KokoroOnnxError::Voice(format!("unterminated npy shape: {header}")))?;
    header[shape_start + 1..shape_end]
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>().map_err(|err| {
                KokoroOnnxError::Voice(format!("invalid npy shape dimension {part:?}: {err}"))
            })
        })
        .collect()
}

pub fn resample_mono(input: &[f32], src_hz: u32, dst_hz: u32) -> Vec<f32> {
    if input.is_empty() || src_hz == dst_hz {
        return input.to_vec();
    }
    let ratio = dst_hz as f64 / src_hz as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    (0..out_len)
        .map(|index| {
            let src_pos = index as f64 / ratio;
            let left = src_pos.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let frac = (src_pos - left as f64) as f32;
            input[left] * (1.0 - frac) + input[right] * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::resample_mono;

    #[test]
    fn resample_identity_returns_clone() {
        let samples = vec![0.0, 0.5, 1.0];
        assert_eq!(resample_mono(&samples, 24_000, 24_000), samples);
    }
}
