use serde::de::DeserializeOwned;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
    path::{Path, PathBuf},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpeechLoaderError {
    #[error("io error: {0}")]
    Io(String),
    #[error("invalid model metadata: {0}")]
    Metadata(String),
    #[error("model weights are missing: {0}")]
    MissingWeights(String),
    #[error("audio input is invalid: {0}")]
    InvalidAudio(String),
    #[error("MLX runtime failed: {0}")]
    Mlx(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechModelFiles {
    pub model_dir: PathBuf,
    pub config_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub safetensors_files: Vec<PathBuf>,
}

impl SpeechModelFiles {
    pub fn discover(model_dir: impl Into<PathBuf>) -> Result<Self, SpeechLoaderError> {
        let model_dir = model_dir.into();
        let config_path = model_dir.join("config.json");
        if !config_path.is_file() {
            return Err(SpeechLoaderError::Metadata(format!(
                "missing config.json in {}",
                model_dir.display()
            )));
        }

        let tokenizer_json = model_dir.join("tokenizer.json");
        let tokenizer_path = tokenizer_json.is_file().then_some(tokenizer_json);
        let safetensors_files = collect_safetensors_files(&model_dir)?;

        Ok(Self {
            model_dir,
            config_path,
            tokenizer_path,
            safetensors_files,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightMapIndex {
    pub weight_map: HashMap<String, String>,
}

impl serde::Serialize for WeightMapIndex {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut root = serde_json::Map::new();
        root.insert(
            "weight_map".to_string(),
            serde_json::to_value(&self.weight_map).map_err(serde::ser::Error::custom)?,
        );
        root.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for WeightMapIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let weight_map = value
            .get("weight_map")
            .ok_or_else(|| serde::de::Error::custom("missing weight_map"))?;
        Ok(Self {
            weight_map: serde_json::from_value(weight_map.clone())
                .map_err(serde::de::Error::custom)?,
        })
    }
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, SpeechLoaderError> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        SpeechLoaderError::Io(format!("failed to read {}: {error}", path.display()))
    })?;
    serde_json::from_str(&text).map_err(|error| {
        SpeechLoaderError::Metadata(format!("failed to parse {}: {error}", path.display()))
    })
}

pub fn collect_safetensors_files(model_dir: &Path) -> Result<Vec<PathBuf>, SpeechLoaderError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let index: WeightMapIndex = read_json(&index_path)?;
        let mut names: Vec<String> = index
            .weight_map
            .values()
            .collect::<HashSet<_>>()
            .into_iter()
            .cloned()
            .collect();
        names.sort();
        return Ok(names.into_iter().map(|name| model_dir.join(name)).collect());
    }

    let single_path = model_dir.join("model.safetensors");
    if single_path.is_file() {
        return Ok(vec![single_path]);
    }

    Err(SpeechLoaderError::MissingWeights(format!(
        "no model.safetensors or model.safetensors.index.json in {}",
        model_dir.display()
    )))
}

#[cfg(feature = "native-mlx")]
pub fn load_safetensors(
    files: &[PathBuf],
) -> Result<HashMap<String, mlx_rs::Array>, SpeechLoaderError> {
    use std::io::{Read, Seek};

    #[derive(Debug, serde::Deserialize)]
    struct SafeTensorHeaderEntry {
        dtype: String,
        shape: Vec<i32>,
        data_offsets: [u64; 2],
    }

    fn expected_len(shape: &[i32], name: &str, path: &Path) -> Result<usize, SpeechLoaderError> {
        shape
            .iter()
            .try_fold(1_usize, |total, dim| {
                usize::try_from(*dim)
                    .ok()
                    .and_then(|dim| total.checked_mul(dim))
            })
            .ok_or_else(|| {
                SpeechLoaderError::MissingWeights(format!(
                    "invalid safetensors shape for {name} in {}",
                    path.display()
                ))
            })
    }

    let mut weights = HashMap::new();
    for file in files {
        let mut reader = std::fs::File::open(file).map_err(|error| {
            SpeechLoaderError::Io(format!("failed to open {}: {error}", file.display()))
        })?;
        let mut len_bytes = [0_u8; 8];
        reader.read_exact(&mut len_bytes).map_err(|error| {
            SpeechLoaderError::Io(format!(
                "failed to read safetensors header length from {}: {error}",
                file.display()
            ))
        })?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len > 128 * 1024 * 1024 {
            return Err(SpeechLoaderError::Metadata(format!(
                "safetensors header in {} is unexpectedly large",
                file.display()
            )));
        }

        let mut header = vec![0_u8; header_len as usize];
        reader.read_exact(&mut header).map_err(|error| {
            SpeechLoaderError::Io(format!(
                "failed to read safetensors header from {}: {error}",
                file.display()
            ))
        })?;
        let header = serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(&header)
            .map_err(|error| {
                SpeechLoaderError::Metadata(format!(
                    "failed to parse safetensors header in {}: {error}",
                    file.display()
                ))
            })?;

        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let entry =
                serde_json::from_value::<SafeTensorHeaderEntry>(value).map_err(|error| {
                    SpeechLoaderError::Metadata(format!(
                        "failed to parse safetensors entry {name} in {}: {error}",
                        file.display()
                    ))
                })?;
            let byte_len = entry.data_offsets[1]
                .checked_sub(entry.data_offsets[0])
                .ok_or_else(|| {
                    SpeechLoaderError::MissingWeights(format!(
                        "invalid safetensors offsets for {name} in {}",
                        file.display()
                    ))
                })?;
            reader
                .seek(std::io::SeekFrom::Start(
                    8 + header_len + entry.data_offsets[0],
                ))
                .map_err(|error| {
                    SpeechLoaderError::Io(format!(
                        "failed to seek tensor {name} in {}: {error}",
                        file.display()
                    ))
                })?;
            let mut data = vec![0_u8; byte_len as usize];
            reader.read_exact(&mut data).map_err(|error| {
                SpeechLoaderError::Io(format!(
                    "failed to read tensor {name} in {}: {error}",
                    file.display()
                ))
            })?;
            let expected = expected_len(&entry.shape, &name, file)?;
            let array = match entry.dtype.as_str() {
                "F32" => {
                    let values = read_le_chunks::<4, f32>(&data, expected, &name, file, |bytes| {
                        f32::from_le_bytes(bytes)
                    })?;
                    mlx_rs::Array::from_slice(&values, &entry.shape)
                }
                "F16" => {
                    let values =
                        read_le_chunks::<2, half::f16>(&data, expected, &name, file, |bytes| {
                            half::f16::from_bits(u16::from_le_bytes(bytes))
                        })?;
                    mlx_rs::Array::from_slice(&values, &entry.shape)
                }
                "BF16" => {
                    let values =
                        read_le_chunks::<2, half::bf16>(&data, expected, &name, file, |bytes| {
                            half::bf16::from_bits(u16::from_le_bytes(bytes))
                        })?;
                    mlx_rs::Array::from_slice(&values, &entry.shape)
                }
                "I32" => {
                    let values = read_le_chunks::<4, i32>(&data, expected, &name, file, |bytes| {
                        i32::from_le_bytes(bytes)
                    })?;
                    mlx_rs::Array::from_slice(&values, &entry.shape)
                }
                "I64" => {
                    let values = read_le_chunks::<8, i64>(&data, expected, &name, file, |bytes| {
                        i64::from_le_bytes(bytes)
                    })?;
                    mlx_rs::Array::from_slice(&values, &entry.shape)
                }
                "U32" => {
                    let values = read_le_chunks::<4, u32>(&data, expected, &name, file, |bytes| {
                        u32::from_le_bytes(bytes)
                    })?;
                    mlx_rs::Array::from_slice(&values, &entry.shape)
                }
                "U8" => {
                    if data.len() != expected {
                        return Err(SpeechLoaderError::MissingWeights(format!(
                            "safetensors tensor {name} in {} has {} values, expected {expected}",
                            file.display(),
                            data.len()
                        )));
                    }
                    mlx_rs::Array::from_slice(&data, &entry.shape)
                }
                other => {
                    return Err(SpeechLoaderError::MissingWeights(format!(
                        "unsupported safetensors dtype {other} for {name} in {}",
                        file.display()
                    )));
                }
            };
            weights.insert(name, array);
        }
    }
    Ok(weights)
}

#[cfg(feature = "native-mlx")]
fn read_le_chunks<const N: usize, T>(
    data: &[u8],
    expected: usize,
    name: &str,
    path: &Path,
    convert: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>, SpeechLoaderError> {
    if data.len() != expected.checked_mul(N).unwrap_or(usize::MAX) {
        return Err(SpeechLoaderError::MissingWeights(format!(
            "safetensors tensor {name} in {} has {} bytes, expected {}",
            path.display(),
            data.len(),
            expected * N
        )));
    }
    Ok(data
        .chunks_exact(N)
        .map(|chunk| convert(chunk.try_into().expect("chunk size is fixed")))
        .collect())
}

#[cfg(feature = "native-mlx")]
pub fn safetensors_file_contains_dtype(
    path: &Path,
    dtype: &str,
) -> Result<bool, SpeechLoaderError> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|error| {
        SpeechLoaderError::Io(format!("failed to open {}: {error}", path.display()))
    })?;
    let mut len_bytes = [0_u8; 8];
    file.read_exact(&mut len_bytes).map_err(|error| {
        SpeechLoaderError::Io(format!(
            "failed to read safetensors header length from {}: {error}",
            path.display()
        ))
    })?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > 128 * 1024 * 1024 {
        return Err(SpeechLoaderError::Metadata(format!(
            "safetensors header in {} is unexpectedly large",
            path.display()
        )));
    }
    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header).map_err(|error| {
        SpeechLoaderError::Io(format!(
            "failed to read safetensors header from {}: {error}",
            path.display()
        ))
    })?;
    let header = serde_json::from_slice::<serde_json::Value>(&header).map_err(|error| {
        SpeechLoaderError::Metadata(format!(
            "failed to parse safetensors header in {}: {error}",
            path.display()
        ))
    })?;
    let Some(tensors) = header.as_object() else {
        return Ok(false);
    };
    Ok(tensors.iter().any(|(key, value)| {
        key != "__metadata__"
            && value
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| value == dtype)
    }))
}

#[cfg(feature = "native-mlx")]
pub struct MlxWeightView<'a> {
    weights: &'a HashMap<String, mlx_rs::Array>,
}

#[cfg(feature = "native-mlx")]
impl<'a> MlxWeightView<'a> {
    pub fn new(weights: &'a HashMap<String, mlx_rs::Array>) -> Self {
        Self { weights }
    }

    pub fn get(&self, key: &str) -> Result<&'a mlx_rs::Array, SpeechLoaderError> {
        self.weights
            .get(key)
            .ok_or_else(|| SpeechLoaderError::MissingWeights(format!("missing weight {key}")))
    }

    pub fn get_any(&self, keys: &[&str]) -> Result<&'a mlx_rs::Array, SpeechLoaderError> {
        keys.iter()
            .find_map(|key| self.weights.get(*key))
            .ok_or_else(|| {
                SpeechLoaderError::MissingWeights(format!(
                    "missing any of weights: {}",
                    keys.join(", ")
                ))
            })
    }

    pub fn optional(&self, key: &str) -> Option<&'a mlx_rs::Array> {
        self.weights.get(key)
    }
}

#[cfg(feature = "native-mlx")]
pub fn linear(
    input: &mlx_rs::Array,
    weight: &mlx_rs::Array,
    bias: Option<&mlx_rs::Array>,
) -> Result<mlx_rs::Array, SpeechLoaderError> {
    let mut output = mlx_rs::ops::matmul(input, weight.t())
        .map_err(|error| SpeechLoaderError::Mlx(error.to_string()))?;
    if let Some(bias) = bias {
        output += bias;
    }
    Ok(output)
}

#[cfg(feature = "native-mlx")]
pub fn embedding(
    weight: &mlx_rs::Array,
    token_ids: &[i32],
) -> Result<mlx_rs::Array, SpeechLoaderError> {
    let len = i32::try_from(token_ids.len()).map_err(|_| {
        SpeechLoaderError::InvalidAudio("token sequence exceeds MLX shape limits".to_string())
    })?;
    let indices = mlx_rs::Array::from_slice(token_ids, &[len]);
    weight
        .take_axis(&indices, 0)
        .map_err(|error| SpeechLoaderError::Mlx(error.to_string()))
}

#[cfg(feature = "native-mlx")]
pub fn conv1d_pytorch(
    input_nlc: &mlx_rs::Array,
    weight_okc: &mlx_rs::Array,
    bias: Option<&mlx_rs::Array>,
    stride: i32,
    padding: i32,
) -> Result<mlx_rs::Array, SpeechLoaderError> {
    let weight = weight_okc
        .transpose_axes(&[0, 2, 1])
        .map_err(|error| SpeechLoaderError::Mlx(error.to_string()))?;
    let mut output = mlx_rs::ops::conv1d(input_nlc, &weight, stride, padding, None, None)
        .map_err(|error| SpeechLoaderError::Mlx(error.to_string()))?;
    if let Some(bias) = bias {
        output += bias;
    }
    Ok(output)
}

#[cfg(feature = "native-mlx")]
pub fn layer_norm(
    input: &mlx_rs::Array,
    weight: Option<&mlx_rs::Array>,
    bias: Option<&mlx_rs::Array>,
    eps: f32,
) -> Result<mlx_rs::Array, SpeechLoaderError> {
    mlx_rs::fast::layer_norm(input, weight, bias, eps)
        .map_err(|error| SpeechLoaderError::Mlx(error.to_string()))
}

#[cfg(feature = "native-mlx")]
pub fn gelu(input: &mlx_rs::Array) -> Result<mlx_rs::Array, SpeechLoaderError> {
    mlx_rs::nn::gelu(input).map_err(|error| SpeechLoaderError::Mlx(error.to_string()))
}

#[cfg(feature = "native-mlx")]
pub fn scaled_dot_product_attention(
    queries: &mlx_rs::Array,
    keys: &mlx_rs::Array,
    values: &mlx_rs::Array,
    scale: f32,
) -> Result<mlx_rs::Array, SpeechLoaderError> {
    mlx_rs::fast::scaled_dot_product_attention(queries, keys, values, scale, None)
        .map_err(|error| SpeechLoaderError::Mlx(error.to_string()))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MelSpectrogramConfig {
    pub sample_rate_hz: u32,
    pub fft_size: usize,
    pub hop_length: usize,
    pub mel_bins: usize,
    pub min_frequency_hz: f32,
    pub max_frequency_hz: f32,
}

impl MelSpectrogramConfig {
    pub fn whisper() -> Self {
        Self {
            sample_rate_hz: 16_000,
            fft_size: 400,
            hop_length: 160,
            mel_bins: 80,
            min_frequency_hz: 0.0,
            max_frequency_hz: 8_000.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MelSpectrogram {
    pub frames: usize,
    pub bins: usize,
    pub values: Vec<f32>,
}

pub fn log_mel_spectrogram(
    samples: &[f32],
    config: MelSpectrogramConfig,
) -> Result<MelSpectrogram, SpeechLoaderError> {
    if samples.is_empty() {
        return Err(SpeechLoaderError::InvalidAudio(
            "audio sample buffer is empty".to_string(),
        ));
    }
    if config.fft_size == 0 || config.hop_length == 0 || config.mel_bins == 0 {
        return Err(SpeechLoaderError::InvalidAudio(
            "mel spectrogram dimensions must be non-zero".to_string(),
        ));
    }

    let padded = if samples.len() < config.fft_size {
        let mut padded = samples.to_vec();
        padded.resize(config.fft_size, 0.0);
        padded
    } else {
        samples.to_vec()
    };
    let frames = 1 + (padded.len() - config.fft_size) / config.hop_length;
    let window = hann_window(config.fft_size);
    let filters = mel_filterbank(config);
    let fft_bins = config.fft_size / 2 + 1;
    let mut values = Vec::with_capacity(frames * config.mel_bins);

    for frame_index in 0..frames {
        let offset = frame_index * config.hop_length;
        let mut power = vec![0.0_f32; fft_bins];
        for (bin, power_bin) in power.iter_mut().enumerate() {
            let mut real = 0.0_f32;
            let mut imag = 0.0_f32;
            for n in 0..config.fft_size {
                let angle = -2.0 * PI * bin as f32 * n as f32 / config.fft_size as f32;
                let sample = padded[offset + n] * window[n];
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            *power_bin = real.mul_add(real, imag * imag);
        }

        for mel in 0..config.mel_bins {
            let mut energy = 0.0_f32;
            for bin in 0..fft_bins {
                energy += power[bin] * filters[mel * fft_bins + bin];
            }
            values.push(energy.max(1.0e-10).log10());
        }
    }

    if let Some(max_log_mel) = values
        .iter()
        .copied()
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
    {
        let floor = max_log_mel - 8.0;
        for value in &mut values {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
    }

    Ok(MelSpectrogram {
        frames,
        bins: config.mel_bins,
        values,
    })
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / size as f32).cos())
        .collect()
}

fn mel_filterbank(config: MelSpectrogramConfig) -> Vec<f32> {
    let fft_bins = config.fft_size / 2 + 1;
    let min_mel = hz_to_mel(config.min_frequency_hz);
    let max_mel = hz_to_mel(config.max_frequency_hz);
    let mel_points: Vec<f32> = (0..config.mel_bins + 2)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (config.mel_bins + 1) as f32)
        .map(mel_to_hz)
        .collect();

    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|hz| ((config.fft_size as f32 + 1.0) * hz / config.sample_rate_hz as f32) as usize)
        .map(|bin| bin.min(fft_bins - 1))
        .collect();

    let mut filters = vec![0.0_f32; config.mel_bins * fft_bins];
    for mel in 0..config.mel_bins {
        let left = bin_points[mel];
        let center = bin_points[mel + 1].max(left + 1);
        let right = bin_points[mel + 2].max(center + 1).min(fft_bins - 1);

        for bin in left..center.min(fft_bins) {
            filters[mel * fft_bins + bin] = (bin - left) as f32 / (center - left) as f32;
        }
        for bin in center..=right {
            filters[mel * fft_bins + bin] = (right - bin) as f32 / (right - center) as f32;
        }
    }
    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_single_safetensors_file() {
        let root =
            std::env::temp_dir().join(format!("vona-mlx-speech-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), "{}").unwrap();
        std::fs::write(root.join("model.safetensors"), b"stub").unwrap();

        let files = SpeechModelFiles::discover(&root).unwrap();
        assert_eq!(
            files.safetensors_files,
            vec![root.join("model.safetensors")]
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn computes_log_mel_shape() {
        let samples = vec![0.0_f32; 480];
        let spec = log_mel_spectrogram(&samples, MelSpectrogramConfig::whisper()).unwrap();
        assert_eq!(spec.bins, 80);
        assert_eq!(spec.frames, 1);
        assert_eq!(spec.values.len(), 80);
    }
}
