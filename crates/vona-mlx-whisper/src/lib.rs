use std::{
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex as AsyncMutex,
};
use vona_core::{AudioInputFrame, AudioProcessingError, AudioTranscriber};
use vona_mlx::{LoadedMlxModel, MlxAudioError, MlxModelLoadRequest, MlxModelLoader};

#[cfg(feature = "native-mlx")]
use {
    std::{
        collections::{HashMap, HashSet},
        sync::Mutex,
    },
    vona_mlx::{MlxArray, MlxModelKind, MlxSpeechModel},
    vona_mlx_speech::{
        MelSpectrogramConfig, MlxWeightView, SpeechModelFiles, conv1d_pytorch, embedding, gelu,
        layer_norm, linear, log_mel_spectrogram, scaled_dot_product_attention,
    },
};

pub const DEFAULT_WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;
pub const DEFAULT_WHISPER_WORKER_BIN: &str = "vona_mlx_whisper_worker";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhisperSpeechConfig {
    pub model_path: PathBuf,
    pub language: Option<String>,
    pub task: WhisperTask,
    pub max_decode_tokens: usize,
    pub hotwords: Vec<TranscriptHotword>,
}

impl WhisperSpeechConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            language: None,
            task: WhisperTask::Transcribe,
            max_decode_tokens: 96,
            hotwords: default_transcript_hotwords(),
        }
    }

    pub fn with_hotwords(mut self, hotwords: Vec<TranscriptHotword>) -> Self {
        self.hotwords = hotwords;
        self
    }

    pub fn with_env_hotwords(mut self) -> Self {
        if let Some(hotwords) = transcript_hotwords_from_env("VONA_WHISPER_HOTWORDS") {
            self.hotwords = hotwords;
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptHotword {
    pub replacement: String,
    pub variants: Vec<String>,
}

impl TranscriptHotword {
    pub fn new(
        replacement: impl Into<String>,
        variants: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            replacement: replacement.into(),
            variants: variants.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectedWhisperConfig {
    pub worker_bin: PathBuf,
    pub model_path: PathBuf,
    pub language: Option<String>,
    pub task: WhisperTask,
    pub max_decode_tokens: usize,
    pub hotwords: Vec<TranscriptHotword>,
}

impl ProtectedWhisperConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        let speech = WhisperSpeechConfig::new(model_path);
        Self {
            worker_bin: std::env::var_os("VONA_WHISPER_WORKER_BIN")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_WHISPER_WORKER_BIN)),
            model_path: speech.model_path,
            language: speech.language,
            task: speech.task,
            max_decode_tokens: speech.max_decode_tokens,
            hotwords: speech.hotwords,
        }
    }

    pub fn from_env(model_path: impl Into<PathBuf>) -> Self {
        let mut config = Self::new(model_path);
        if let Some(value) = std::env::var_os("VONA_WHISPER_WORKER_BIN") {
            config.worker_bin = PathBuf::from(value);
        }
        if let Some(hotwords) = transcript_hotwords_from_env("VONA_WHISPER_HOTWORDS") {
            config.hotwords = hotwords;
        }
        config.max_decode_tokens = std::env::var("VONA_WHISPER_MAX_DECODE_TOKENS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(config.max_decode_tokens);
        config
    }

    pub fn speech_config(&self) -> WhisperSpeechConfig {
        WhisperSpeechConfig {
            model_path: self.model_path.clone(),
            language: self.language.clone(),
            task: self.task,
            max_decode_tokens: self.max_decode_tokens,
            hotwords: self.hotwords.clone(),
        }
    }
}

#[derive(Clone)]
pub struct ProtectedWhisperTranscriber {
    config: ProtectedWhisperConfig,
    worker: Arc<AsyncMutex<Option<WhisperWorker>>>,
    next_request_id: Arc<AtomicU64>,
}

impl ProtectedWhisperTranscriber {
    pub fn new(config: ProtectedWhisperConfig) -> Self {
        Self {
            config,
            worker: Arc::new(AsyncMutex::new(None)),
            next_request_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn from_env(model_path: impl Into<PathBuf>) -> Self {
        Self::new(ProtectedWhisperConfig::from_env(model_path))
    }

    pub fn config(&self) -> &ProtectedWhisperConfig {
        &self.config
    }

    pub async fn transcribe_samples(
        &self,
        samples: Vec<f32>,
        sample_rate_hz: u32,
        channels: u16,
    ) -> Result<String, MlxAudioError> {
        if sample_rate_hz != DEFAULT_WHISPER_SAMPLE_RATE_HZ {
            return Err(MlxAudioError::InvalidInput(format!(
                "protected Whisper expects {DEFAULT_WHISPER_SAMPLE_RATE_HZ} Hz audio, got {sample_rate_hz} Hz"
            )));
        }
        if channels != 1 {
            return Err(MlxAudioError::InvalidInput(format!(
                "protected Whisper expects mono audio, got {channels} channels"
            )));
        }

        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let mut guard = self.worker.lock().await;
        if guard.is_none() {
            *guard = Some(start_whisper_worker(&self.config).await?);
        }

        let Some(worker) = guard.as_mut() else {
            return Err(MlxAudioError::Runtime(
                "protected Whisper worker was not available after start".to_string(),
            ));
        };

        match send_whisper_worker_request(worker, request_id, sample_rate_hz, channels, &samples)
            .await
        {
            Ok(transcript) => Ok(transcript),
            Err(error) => {
                *guard = None;
                Err(error)
            }
        }
    }
}

#[async_trait]
impl AudioTranscriber for ProtectedWhisperTranscriber {
    async fn transcribe_audio(
        &self,
        input: AudioInputFrame,
    ) -> Result<String, AudioProcessingError> {
        self.transcribe_samples(input.samples, input.sample_rate_hz, input.channels)
            .await
            .map_err(Into::into)
    }
}

struct WhisperWorker {
    _child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WhisperWorkerReady {
    ready: bool,
    model: String,
    weights: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct WhisperWorkerRequest {
    id: u64,
    sample_rate_hz: u32,
    channels: u16,
    samples: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct WhisperWorkerResponse {
    id: u64,
    transcript: Option<String>,
    error: Option<String>,
}

async fn start_whisper_worker(
    config: &ProtectedWhisperConfig,
) -> Result<WhisperWorker, MlxAudioError> {
    let mut command = Command::new(&config.worker_bin);
    command
        .arg("--model")
        .arg(&config.model_path)
        .arg("--max-decode-tokens")
        .arg(config.max_decode_tokens.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    command.kill_on_drop(true);
    if let Some(language) = &config.language {
        command.arg("--language").arg(language);
    }
    if matches!(config.task, WhisperTask::Translate) {
        command.arg("--task").arg("translate");
    }
    if !config.hotwords.is_empty() {
        command
            .arg("--hotwords")
            .arg(format_hotwords(&config.hotwords));
    }

    let mut child = command.spawn().map_err(|error| {
        MlxAudioError::Runtime(format!(
            "failed to spawn protected Whisper worker {}: {error}",
            config.worker_bin.display()
        ))
    })?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| MlxAudioError::Runtime("Whisper worker stdin missing".to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| MlxAudioError::Runtime("Whisper worker stdout missing".to_string()))?;
    let mut worker = WhisperWorker {
        _child: child,
        stdin,
        stdout: BufReader::new(stdout).lines(),
    };
    let ready = worker
        .stdout
        .next_line()
        .await
        .map_err(|error| {
            MlxAudioError::Runtime(format!("Whisper worker ready read failed: {error}"))
        })?
        .ok_or_else(|| MlxAudioError::Runtime("Whisper worker exited before ready".to_string()))?;
    let ready: WhisperWorkerReady = serde_json::from_str(&ready).map_err(|error| {
        MlxAudioError::Runtime(format!("Whisper worker ready JSON invalid: {error}"))
    })?;
    if !ready.ready {
        return Err(MlxAudioError::Runtime(format!(
            "Whisper worker did not report ready for model {}",
            ready.model
        )));
    }
    Ok(worker)
}

async fn send_whisper_worker_request(
    worker: &mut WhisperWorker,
    request_id: u64,
    sample_rate_hz: u32,
    channels: u16,
    samples: &[f32],
) -> Result<String, MlxAudioError> {
    let header = serde_json::to_string(&WhisperWorkerRequest {
        id: request_id,
        sample_rate_hz,
        channels,
        samples: samples.len(),
    })
    .map_err(|error| MlxAudioError::Runtime(format!("Whisper request JSON failed: {error}")))?;
    worker
        .stdin
        .write_all(header.as_bytes())
        .await
        .map_err(|error| {
            MlxAudioError::Runtime(format!("Whisper worker header write failed: {error}"))
        })?;
    worker.stdin.write_all(b"\n").await.map_err(|error| {
        MlxAudioError::Runtime(format!("Whisper worker header write failed: {error}"))
    })?;
    write_f32_le(&mut worker.stdin, samples).await?;
    worker
        .stdin
        .flush()
        .await
        .map_err(|error| MlxAudioError::Runtime(format!("Whisper worker flush failed: {error}")))?;

    let response = worker
        .stdout
        .next_line()
        .await
        .map_err(|error| {
            MlxAudioError::Runtime(format!("Whisper worker response read failed: {error}"))
        })?
        .ok_or_else(|| {
            MlxAudioError::Runtime("Whisper worker exited during transcription".to_string())
        })?;
    let response: WhisperWorkerResponse = serde_json::from_str(&response).map_err(|error| {
        MlxAudioError::Runtime(format!("Whisper worker response JSON invalid: {error}"))
    })?;
    if response.id != request_id {
        return Err(MlxAudioError::Runtime(format!(
            "Whisper worker response id {} did not match request id {request_id}",
            response.id
        )));
    }
    if let Some(error) = response.error {
        return Err(MlxAudioError::Inference(error));
    }
    response.transcript.ok_or_else(|| {
        MlxAudioError::Runtime("Whisper worker response omitted transcript".to_string())
    })
}

async fn write_f32_le(stdin: &mut ChildStdin, samples: &[f32]) -> Result<(), MlxAudioError> {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    stdin.write_all(&bytes).await.map_err(|error| {
        MlxAudioError::Runtime(format!("Whisper worker PCM write failed: {error}"))
    })
}

fn format_hotwords(hotwords: &[TranscriptHotword]) -> String {
    hotwords
        .iter()
        .map(|hotword| format!("{}={}", hotword.replacement, hotword.variants.join("|")))
        .collect::<Vec<_>>()
        .join(",")
}

pub fn default_transcript_hotwords() -> Vec<TranscriptHotword> {
    vec![
        TranscriptHotword::new("Vona", ["vona", "voner", "vowna"]),
        TranscriptHotword::new("Qwen", ["qwen", "qn", "q-n"]),
        TranscriptHotword::new("Qwen speech", ["qnspeech", "q-n-speech", "qwenspeech"]),
        TranscriptHotword::new("Ollama", ["ollama", "alama", "allama"]),
        TranscriptHotword::new("Whisper", ["whisper", "wispa", "whispa"]),
        TranscriptHotword::new("Ready case", ["readycase"]),
        TranscriptHotword::new("Check case", ["checkcase"]),
        TranscriptHotword::new("Test case", ["testcase"]),
        TranscriptHotword::new("Speech check case", ["speechcheckcase"]),
    ]
}

pub fn transcript_hotwords_from_env(name: &str) -> Option<Vec<TranscriptHotword>> {
    let value = std::env::var(name).ok()?;
    parse_transcript_hotwords(&value).ok()
}

pub fn parse_transcript_hotwords(value: &str) -> Result<Vec<TranscriptHotword>, String> {
    let mut hotwords = Vec::new();
    for entry in value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let Some((replacement, variants)) = entry.split_once('=') else {
            return Err(format!(
                "invalid hotword entry {entry:?}; expected replacement=variant|variant"
            ));
        };
        let replacement = replacement.trim();
        if replacement.is_empty() {
            return Err("hotword replacement cannot be empty".to_string());
        }
        let variants = variants
            .split('|')
            .map(str::trim)
            .filter(|variant| !variant.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if variants.is_empty() {
            return Err(format!(
                "hotword {replacement:?} must include at least one variant"
            ));
        }
        hotwords.push(TranscriptHotword::new(replacement, variants));
    }
    Ok(hotwords)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhisperTask {
    Transcribe,
    Translate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhisperConfig {
    pub model_type: Option<String>,
    pub vocab_size: Option<u32>,
    pub num_mel_bins: Option<u32>,
    pub max_source_positions: Option<u32>,
    pub d_model: Option<u32>,
    pub encoder_layers: Option<u32>,
    pub decoder_layers: Option<u32>,
    pub encoder_attention_heads: Option<u32>,
    pub decoder_attention_heads: Option<u32>,
    pub max_target_positions: Option<u32>,
    pub decoder_start_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

#[cfg(feature = "native-mlx")]
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct WhisperGenerationConfig {
    pub suppress_tokens: Option<Vec<u32>>,
    pub begin_suppress_tokens: Option<Vec<u32>>,
    pub no_timestamps_token_id: Option<u32>,
    pub lang_to_id: Option<HashMap<String, u32>>,
    pub task_to_id: Option<HashMap<String, u32>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WhisperLoader;

impl MlxModelLoader for WhisperLoader {
    fn load_model(&self, request: MlxModelLoadRequest) -> Result<LoadedMlxModel, MlxAudioError> {
        #[cfg(not(feature = "native-mlx"))]
        {
            let _ = request;
            Err(MlxAudioError::Runtime(
                "enable the native-mlx feature to use native Whisper loading".to_string(),
            ))
        }

        #[cfg(feature = "native-mlx")]
        {
            if !matches!(
                request.kind,
                MlxModelKind::Speech | MlxModelKind::WhisperSpeech
            ) {
                return Err(MlxAudioError::InvalidInput(
                    "WhisperLoader only handles Whisper speech requests".to_string(),
                ));
            }
            let model_path = request.local_path.ok_or_else(|| {
                MlxAudioError::InvalidInput("WhisperLoader requires a local model path".to_string())
            })?;
            let model = WhisperSpeechModel::load(WhisperSpeechConfig::new(model_path))?;
            Ok(LoadedMlxModel::Speech(Arc::new(model)))
        }
    }
}

#[cfg(feature = "native-mlx")]
pub struct WhisperSpeechModel {
    files: SpeechModelFiles,
    config: WhisperConfig,
    generation_config: WhisperGenerationConfig,
    tokenizer: Option<tokenizers::Tokenizer>,
    weights: Mutex<HashMap<String, mlx_rs::Array>>,
    speech_config: WhisperSpeechConfig,
}

#[cfg(feature = "native-mlx")]
impl WhisperSpeechModel {
    pub fn load(speech_config: WhisperSpeechConfig) -> Result<Self, MlxAudioError> {
        let files = SpeechModelFiles::discover(&speech_config.model_path)
            .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?;
        let config = vona_mlx_speech::read_json::<WhisperConfig>(&files.config_path)
            .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?;
        let generation_config_path = files.model_dir.join("generation_config.json");
        let generation_config = if generation_config_path.is_file() {
            vona_mlx_speech::read_json::<WhisperGenerationConfig>(&generation_config_path)
                .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?
        } else {
            WhisperGenerationConfig::default()
        };
        let tokenizer = files
            .tokenizer_path
            .as_ref()
            .map(tokenizers::Tokenizer::from_file)
            .transpose()
            .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?;
        let weights = vona_mlx_speech::load_safetensors(&files.safetensors_files)
            .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?;

        Ok(Self {
            files,
            config,
            generation_config,
            tokenizer,
            weights: Mutex::new(weights),
            speech_config,
        })
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    pub fn model_dir(&self) -> &std::path::Path {
        &self.files.model_dir
    }

    pub fn weight_count(&self) -> usize {
        self.weights
            .lock()
            .map(|weights| weights.len())
            .unwrap_or(0)
    }

    fn encode_frontend(
        &self,
        mel_values: &[f32],
        frames: i32,
        bins: i32,
    ) -> Result<MlxArray, MlxAudioError> {
        let features = mlx_rs::Array::from_slice(mel_values, &[1, frames, bins]);
        let weights = self
            .weights
            .lock()
            .map_err(|_| MlxAudioError::Runtime("Whisper weight lock is poisoned".to_string()))?;
        let weights = MlxWeightView::new(&weights);

        let conv1 = conv1d_pytorch(
            &features,
            weights
                .get_any(&["model.encoder.conv1.weight", "encoder.conv1.weight"])
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights
                .optional("model.encoder.conv1.bias")
                .or_else(|| weights.optional("encoder.conv1.bias")),
            1,
            1,
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let conv1 = gelu(&conv1).map_err(|error| MlxAudioError::Inference(error.to_string()))?;

        let conv2 = conv1d_pytorch(
            &conv1,
            weights
                .get_any(&["model.encoder.conv2.weight", "encoder.conv2.weight"])
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights
                .optional("model.encoder.conv2.bias")
                .or_else(|| weights.optional("encoder.conv2.bias")),
            2,
            1,
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let conv2 = gelu(&conv2).map_err(|error| MlxAudioError::Inference(error.to_string()))?;

        let seq_len = conv2.shape().get(1).copied().unwrap_or(0);
        let hidden = if let Some(positional_weight) = weights
            .optional("model.encoder.embed_positions.weight")
            .or_else(|| weights.optional("encoder.embed_positions.weight"))
            .or_else(|| weights.optional("encoder.positional_embedding"))
        {
            let positions = (0..seq_len).collect::<Vec<i32>>();
            let pos = embedding(positional_weight, &positions)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?
                .reshape(&[1, seq_len, -1])
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
            conv2 + pos
        } else {
            conv2
        };

        hidden
            .eval()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        Ok(hidden)
    }

    fn encode_transformer(&self, frontend: &MlxArray) -> Result<MlxArray, MlxAudioError> {
        let weights = self
            .weights
            .lock()
            .map_err(|_| MlxAudioError::Runtime("Whisper weight lock is poisoned".to_string()))?;
        let weights = MlxWeightView::new(&weights);
        let layer_count = self.config.encoder_layers.unwrap_or(0);
        let head_count = self.config.encoder_attention_heads.unwrap_or(1).max(1) as i32;
        let mut hidden = frontend.clone();

        for layer in 0..layer_count {
            hidden = self.encoder_layer(&weights, layer, &hidden, head_count)?;
        }

        if let Some(norm_weight) = weights
            .optional("model.encoder.layer_norm.weight")
            .or_else(|| weights.optional("encoder.layer_norm.weight"))
            .or_else(|| weights.optional("encoder.ln_post.weight"))
        {
            hidden = layer_norm(
                &hidden,
                Some(norm_weight),
                weights
                    .optional("model.encoder.layer_norm.bias")
                    .or_else(|| weights.optional("encoder.layer_norm.bias"))
                    .or_else(|| weights.optional("encoder.ln_post.bias")),
                1.0e-5,
            )
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        }

        hidden
            .eval()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        Ok(hidden)
    }

    fn encoder_layer(
        &self,
        weights: &MlxWeightView<'_>,
        layer: u32,
        hidden: &MlxArray,
        head_count: i32,
    ) -> Result<MlxArray, MlxAudioError> {
        let prefixes = [
            format!("model.encoder.layers.{layer}"),
            format!("encoder.layers.{layer}"),
            format!("encoder.blocks.{layer}"),
        ];
        let residual = hidden.clone();
        let attn_input =
            self.layer_norm_with_prefix(weights, &prefixes, "self_attn_layer_norm", hidden)?;
        let attention = self.self_attention(weights, &prefixes, &attn_input, head_count, false)?;
        let hidden = residual + attention;

        let residual = hidden.clone();
        let mlp_input =
            self.layer_norm_with_prefix(weights, &prefixes, "final_layer_norm", &hidden)?;
        let fc1 = self.linear_with_prefix(weights, &prefixes, "fc1", &mlp_input)?;
        let fc1 = gelu(&fc1).map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let fc2 = self.linear_with_prefix(weights, &prefixes, "fc2", &fc1)?;
        Ok(residual + fc2)
    }

    fn self_attention(
        &self,
        weights: &MlxWeightView<'_>,
        prefixes: &[String; 3],
        hidden: &MlxArray,
        head_count: i32,
        causal: bool,
    ) -> Result<MlxArray, MlxAudioError> {
        let q = self.linear_with_prefix(weights, prefixes, "self_attn.q_proj", hidden)?;
        let k = self.linear_with_prefix(weights, prefixes, "self_attn.k_proj", hidden)?;
        let v = self.linear_with_prefix(weights, prefixes, "self_attn.v_proj", hidden)?;
        let shape = q.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let head_dim = hidden_size / head_count;

        let q = q
            .reshape(&[batch, seq_len, head_count, head_dim])
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let k = k
            .reshape(&[batch, seq_len, head_count, head_dim])
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let v = v
            .reshape(&[batch, seq_len, head_count, head_dim])
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;

        let attention = if causal {
            mlx_rs::fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                1.0 / (head_dim as f32).sqrt(),
                mlx_rs::fast::ScaledDotProductAttentionMask::Causal,
            )
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
        } else {
            scaled_dot_product_attention(&q, &k, &v, 1.0 / (head_dim as f32).sqrt())
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?
        };
        let attention = attention
            .transpose_axes(&[0, 2, 1, 3])
            .and_then(|array| array.reshape(&[batch, seq_len, hidden_size]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        self.linear_with_prefix(weights, prefixes, "self_attn.out_proj", &attention)
    }

    fn cross_attention(
        &self,
        weights: &MlxWeightView<'_>,
        prefixes: &[String; 3],
        hidden: &MlxArray,
        encoder_hidden: &MlxArray,
        head_count: i32,
    ) -> Result<MlxArray, MlxAudioError> {
        let q = self.linear_with_prefix(weights, prefixes, "encoder_attn.q_proj", hidden)?;
        let k =
            self.linear_with_prefix(weights, prefixes, "encoder_attn.k_proj", encoder_hidden)?;
        let v =
            self.linear_with_prefix(weights, prefixes, "encoder_attn.v_proj", encoder_hidden)?;
        let q_shape = q.shape();
        let batch = q_shape[0];
        let target_len = q_shape[1];
        let hidden_size = q_shape[2];
        let source_len = k.shape()[1];
        let head_dim = hidden_size / head_count;

        let q = q
            .reshape(&[batch, target_len, head_count, head_dim])
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let k = k
            .reshape(&[batch, source_len, head_count, head_dim])
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let v = v
            .reshape(&[batch, source_len, head_count, head_dim])
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;

        let attention = scaled_dot_product_attention(&q, &k, &v, 1.0 / (head_dim as f32).sqrt())
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let attention = attention
            .transpose_axes(&[0, 2, 1, 3])
            .and_then(|array| array.reshape(&[batch, target_len, hidden_size]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        self.linear_with_prefix(weights, prefixes, "encoder_attn.out_proj", &attention)
    }

    fn decode_greedy(&self, encoded: &MlxArray) -> Result<String, MlxAudioError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            MlxAudioError::ModelUnavailable("Whisper tokenizer.json is required".to_string())
        })?;
        let start_token = self
            .config
            .decoder_start_token_id
            .or(self.config.bos_token_id)
            .unwrap_or(50_258);
        let eos_token = self.config.eos_token_id.unwrap_or(50_257);
        let max_positions = self.config.max_target_positions.unwrap_or(448) as usize;
        let max_decode_tokens = self
            .speech_config
            .max_decode_tokens
            .min(max_positions.saturating_sub(1))
            .max(1);

        let prompt_tokens = self.decoder_prompt_tokens(start_token);
        let prompt_len = prompt_tokens.len();
        let mut tokens = prompt_tokens;
        for _ in 0..max_decode_tokens {
            let logits = self.decoder_logits(&tokens, encoded)?;
            let next_token =
                self.argmax_last_token(&logits, tokens.len(), tokens.len() == prompt_len)?;
            if next_token == eos_token {
                break;
            }
            tokens.push(next_token as i32);
        }

        let decoded_tokens = tokens
            .into_iter()
            .skip(prompt_len)
            .filter(|token| *token >= 0 && *token as u32 != eos_token)
            .map(|token| token as u32)
            .collect::<Vec<_>>();
        if std::env::var_os("VONA_WHISPER_DEBUG_TOKENS").is_some() {
            eprintln!("whisper decoded_tokens={decoded_tokens:?}");
        }
        tokenizer
            .decode(&decoded_tokens, true)
            .map(|text| {
                postprocess_whisper_transcript_with_hotwords(&text, &self.speech_config.hotwords)
            })
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }

    fn decoder_prompt_tokens(&self, start_token: u32) -> Vec<i32> {
        let language_token = self
            .speech_config
            .language
            .as_deref()
            .and_then(|language| self.language_token(language))
            .or_else(|| self.language_token("en"))
            .unwrap_or(50_259);
        let task_token = match self.speech_config.task {
            WhisperTask::Translate => self.task_token("translate").unwrap_or(50_358),
            WhisperTask::Transcribe => self.task_token("transcribe").unwrap_or(50_359),
        };
        let no_timestamps_token = self
            .generation_config
            .no_timestamps_token_id
            .unwrap_or(50_363) as i32;
        vec![
            start_token as i32,
            language_token,
            task_token,
            no_timestamps_token,
        ]
    }

    fn language_token(&self, language: &str) -> Option<i32> {
        let lang_to_id = self.generation_config.lang_to_id.as_ref()?;
        let normalized = language.trim().trim_matches('<').trim_matches('>');
        let key = if normalized.starts_with('|') && normalized.ends_with('|') {
            format!("<{normalized}>")
        } else {
            format!("<|{}|>", normalized.to_ascii_lowercase())
        };
        lang_to_id.get(&key).copied().map(|token| token as i32)
    }

    fn task_token(&self, task: &str) -> Option<i32> {
        self.generation_config
            .task_to_id
            .as_ref()
            .and_then(|task_to_id| task_to_id.get(task).copied())
            .map(|token| token as i32)
    }

    fn decoder_logits(
        &self,
        token_ids: &[i32],
        encoded: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let weights = self
            .weights
            .lock()
            .map_err(|_| MlxAudioError::Runtime("Whisper weight lock is poisoned".to_string()))?;
        let weights = MlxWeightView::new(&weights);
        let embed_weight = weights
            .get_any(&[
                "model.decoder.embed_tokens.weight",
                "decoder.embed_tokens.weight",
                "decoder.token_embedding.weight",
            ])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let mut hidden = embedding(embed_weight, token_ids)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
            .reshape(&[1, token_ids.len() as i32, -1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;

        if let Some(positional_weight) = weights
            .optional("model.decoder.embed_positions.weight")
            .or_else(|| weights.optional("decoder.embed_positions.weight"))
            .or_else(|| weights.optional("decoder.positional_embedding"))
        {
            let positions = (0..token_ids.len() as i32).collect::<Vec<_>>();
            let pos = embedding(positional_weight, &positions)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?
                .reshape(&[1, token_ids.len() as i32, -1])
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
            hidden += pos;
        }

        let layer_count = self.config.decoder_layers.unwrap_or(0);
        let head_count = self.config.decoder_attention_heads.unwrap_or(1).max(1) as i32;
        for layer in 0..layer_count {
            hidden = self.decoder_layer(&weights, layer, &hidden, encoded, head_count)?;
        }

        if let Some(norm_weight) = weights
            .optional("model.decoder.layer_norm.weight")
            .or_else(|| weights.optional("decoder.layer_norm.weight"))
            .or_else(|| weights.optional("decoder.ln.weight"))
        {
            hidden = layer_norm(
                &hidden,
                Some(norm_weight),
                weights
                    .optional("model.decoder.layer_norm.bias")
                    .or_else(|| weights.optional("decoder.layer_norm.bias"))
                    .or_else(|| weights.optional("decoder.ln.bias")),
                1.0e-5,
            )
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        }

        let logits = if let Some(weight) = weights
            .optional("proj_out.weight")
            .or_else(|| weights.optional("model.proj_out.weight"))
            .or_else(|| weights.optional("lm_head.weight"))
            .or_else(|| weights.optional("model.lm_head.weight"))
        {
            linear(
                &hidden,
                weight,
                weights
                    .optional("proj_out.bias")
                    .or_else(|| weights.optional("model.proj_out.bias"))
                    .or_else(|| weights.optional("lm_head.bias"))
                    .or_else(|| weights.optional("model.lm_head.bias")),
            )
        } else {
            linear(&hidden, embed_weight, None)
        }
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        logits
            .eval()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        Ok(logits)
    }

    fn decoder_layer(
        &self,
        weights: &MlxWeightView<'_>,
        layer: u32,
        hidden: &MlxArray,
        encoded: &MlxArray,
        head_count: i32,
    ) -> Result<MlxArray, MlxAudioError> {
        let prefixes = [
            format!("model.decoder.layers.{layer}"),
            format!("decoder.layers.{layer}"),
            format!("decoder.blocks.{layer}"),
        ];
        let residual = hidden.clone();
        let attn_input =
            self.layer_norm_with_prefix(weights, &prefixes, "self_attn_layer_norm", hidden)?;
        let attention = self.self_attention(weights, &prefixes, &attn_input, head_count, true)?;
        let hidden = residual + attention;

        let residual = hidden.clone();
        let cross_input =
            self.layer_norm_with_prefix(weights, &prefixes, "encoder_attn_layer_norm", &hidden)?;
        let cross_attention =
            self.cross_attention(weights, &prefixes, &cross_input, encoded, head_count)?;
        let hidden = residual + cross_attention;

        let residual = hidden.clone();
        let mlp_input =
            self.layer_norm_with_prefix(weights, &prefixes, "final_layer_norm", &hidden)?;
        let fc1 = self.linear_with_prefix(weights, &prefixes, "fc1", &mlp_input)?;
        let fc1 = gelu(&fc1).map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let fc2 = self.linear_with_prefix(weights, &prefixes, "fc2", &fc1)?;
        Ok(residual + fc2)
    }

    fn argmax_last_token(
        &self,
        logits: &MlxArray,
        sequence_len: usize,
        first_generated_token: bool,
    ) -> Result<u32, MlxAudioError> {
        let logits = logits
            .as_type::<f32>()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        logits
            .eval()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let shape = logits.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] as usize != sequence_len {
            return Err(MlxAudioError::Inference(format!(
                "unexpected Whisper decoder logits shape {:?}",
                shape
            )));
        }
        let vocab_size = shape[2] as usize;
        let values = logits.as_slice::<f32>();
        let offset = (sequence_len - 1) * vocab_size;
        let suppressed_tokens = self.suppressed_tokens(first_generated_token);
        values[offset..offset + vocab_size]
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                let token = *index as u32;
                !suppressed_tokens.contains(&token)
                    && (token <= 50_256 || (!first_generated_token && token == 50_257))
            })
            .max_by(|(_, left), (_, right)| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index as u32)
            .ok_or_else(|| MlxAudioError::Inference("empty Whisper logits".to_string()))
    }

    fn suppressed_tokens(&self, first_generated_token: bool) -> HashSet<u32> {
        let mut tokens = self
            .generation_config
            .suppress_tokens
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        if first_generated_token {
            tokens.extend(
                self.generation_config
                    .begin_suppress_tokens
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .copied(),
            );
        }
        tokens
    }

    fn linear_with_prefix(
        &self,
        weights: &MlxWeightView<'_>,
        prefixes: &[String; 3],
        module: &str,
        input: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let weight_keys = prefixed_weight_keys(prefixes, module);
        let bias_keys = prefixed_bias_keys(prefixes, module);
        let weight_key_refs = weight_keys.iter().map(String::as_str).collect::<Vec<_>>();
        let weight = weights
            .get_any(&weight_key_refs)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let bias = bias_keys.iter().find_map(|key| weights.optional(key));
        linear(input, weight, bias).map_err(|error| MlxAudioError::Inference(error.to_string()))
    }

    fn layer_norm_with_prefix(
        &self,
        weights: &MlxWeightView<'_>,
        prefixes: &[String; 3],
        module: &str,
        input: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let weight_keys = prefixed_weight_keys(prefixes, module);
        let bias_keys = prefixed_bias_keys(prefixes, module);
        let weight_key_refs = weight_keys.iter().map(String::as_str).collect::<Vec<_>>();
        let weight = weights
            .get_any(&weight_key_refs)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let bias = bias_keys.iter().find_map(|key| weights.optional(key));
        layer_norm(input, Some(weight), bias, 1.0e-5)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }
}

#[cfg(test)]
fn postprocess_whisper_transcript(text: &str) -> String {
    postprocess_whisper_transcript_with_hotwords(text, &default_transcript_hotwords())
}

#[cfg(any(feature = "native-mlx", test))]
fn postprocess_whisper_transcript_with_hotwords(
    text: &str,
    hotwords: &[TranscriptHotword],
) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut parts = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '?' | '!') {
            let sentence = current.trim();
            if !sentence.is_empty()
                && !parts
                    .iter()
                    .any(|existing: &String| existing.eq_ignore_ascii_case(sentence))
            {
                parts.push(sentence.to_string());
            }
            current.clear();
        }
    }
    let tail = current.trim();
    if !tail.is_empty()
        && !parts
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(tail))
        && !parts.iter().any(|existing| {
            existing
                .trim_end_matches(['.', '?', '!'])
                .to_ascii_lowercase()
                .starts_with(&tail.to_ascii_lowercase())
        })
    {
        parts.push(tail.to_string());
    }

    let collapsed = if parts.is_empty() {
        text
    } else {
        parts.join(" ")
    };
    collapse_repeated_word_runs(&collapsed, hotwords)
}

#[cfg(any(feature = "native-mlx", test))]
fn collapse_repeated_word_runs(text: &str, hotwords: &[TranscriptHotword]) -> String {
    let mut output = Vec::new();
    let mut previous_normalized = String::new();
    let mut repeat_count = 0usize;
    for word in text.split_whitespace() {
        let normalized = word
            .trim_matches(|ch: char| !ch.is_alphanumeric())
            .to_ascii_lowercase();
        if !normalized.is_empty() && normalized == previous_normalized {
            repeat_count += 1;
            if repeat_count >= 2 {
                continue;
            }
        } else {
            previous_normalized = normalized;
            repeat_count = 0;
        }
        output.push(word);
    }
    let output = collapse_repeated_expanded_restarts(&collapse_repeated_leading_phrase(
        &collapse_repeated_phrases(&output),
    ));
    finish_whisper_transcript(normalize_domain_transcript_terms(&output, hotwords))
}

#[cfg(any(feature = "native-mlx", test))]
fn collapse_repeated_phrases<'a>(words: &[&'a str]) -> Vec<&'a str> {
    let mut output = words.to_vec();
    let mut changed = true;

    while changed {
        changed = false;
        let mut index = 0usize;
        while index < output.len() {
            let remaining = output.len() - index;
            let max_phrase_len = (remaining / 2).min(8);
            let mut removed = false;

            for phrase_len in (2..=max_phrase_len).rev() {
                if normalized_word_slices_equal(
                    &output[index..index + phrase_len],
                    &output[index + phrase_len..index + (phrase_len * 2)],
                ) {
                    output.drain(index + phrase_len..index + (phrase_len * 2));
                    changed = true;
                    removed = true;
                    break;
                }
            }

            if !removed {
                index += 1;
            }
        }
    }

    output
}

#[cfg(any(feature = "native-mlx", test))]
fn collapse_repeated_expanded_restarts<'a>(words: &[&'a str]) -> Vec<&'a str> {
    let mut output = words.to_vec();
    let mut changed = true;

    while changed {
        changed = false;
        let mut index = 0usize;
        while index < output.len() {
            let mut removed = false;
            let remaining = output.len() - index;
            let max_left_len = remaining.min(8);

            'candidate: for left_len in (2..=max_left_len).rev() {
                for right_start in index + left_len..output.len() {
                    let max_right_len = (output.len() - right_start).min(8);
                    for right_len in (2..=max_right_len).rev() {
                        if expanded_word_slices_equal(
                            &output[index..index + left_len],
                            &output[right_start..right_start + right_len],
                        ) {
                            output.drain(right_start..right_start + right_len);
                            changed = true;
                            removed = true;
                            break 'candidate;
                        }
                    }
                }
            }

            if !removed {
                index += 1;
            }
        }
    }

    output
}

#[cfg(any(feature = "native-mlx", test))]
fn normalized_word_slices_equal(left: &[&str], right: &[&str]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right.iter()).all(|(left, right)| {
            let left = normalize_transcript_word(left);
            let right = normalize_transcript_word(right);
            !left.is_empty() && left == right
        })
}

#[cfg(any(feature = "native-mlx", test))]
fn expanded_word_slices_equal(left: &[&str], right: &[&str]) -> bool {
    let left = transcript_units(left);
    let right = transcript_units(right);
    left.len() >= 4 && left == right
}

#[cfg(any(feature = "native-mlx", test))]
fn transcript_units(words: &[&str]) -> Vec<String> {
    words
        .iter()
        .flat_map(|word| normalize_transcript_units(word))
        .collect()
}

#[cfg(any(feature = "native-mlx", test))]
fn normalize_transcript_units(word: &str) -> Vec<String> {
    let normalized = normalize_transcript_word(word);
    match normalized.as_str() {
        "readycase" => vec!["ready".to_string(), "case".to_string()],
        "checkcase" => vec!["check".to_string(), "case".to_string()],
        "testcase" => vec!["test".to_string(), "case".to_string()],
        "speechcheckcase" => vec![
            "speech".to_string(),
            "check".to_string(),
            "case".to_string(),
        ],
        "vowna" | "voner" => vec!["vona".to_string()],
        "" => Vec::new(),
        other => vec![other.to_string()],
    }
}

#[cfg(any(feature = "native-mlx", test))]
fn normalize_domain_transcript_terms(words: &[&str], hotwords: &[TranscriptHotword]) -> String {
    words
        .iter()
        .flat_map(|word| normalize_domain_transcript_word(word, hotwords))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(any(feature = "native-mlx", test))]
fn normalize_domain_transcript_word(word: &str, hotwords: &[TranscriptHotword]) -> Vec<String> {
    let leading = word
        .chars()
        .take_while(|ch| !ch.is_alphanumeric())
        .collect::<String>();
    let trailing = word
        .chars()
        .rev()
        .take_while(|ch| !ch.is_alphanumeric())
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    let core = word
        .trim_start_matches(|ch: char| !ch.is_alphanumeric())
        .trim_end_matches(|ch: char| !ch.is_alphanumeric());
    let normalized = core
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    let replacement = hotwords.iter().find_map(|hotword| {
        hotword
            .variants
            .iter()
            .any(|variant| normalize_hotword_key(variant) == normalized)
            .then(|| {
                hotword
                    .replacement
                    .split_whitespace()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
    });

    match replacement {
        Some(mut words) => {
            if let Some(first) = words.first_mut() {
                *first = format!("{leading}{first}");
            }
            if let Some(last) = words.last_mut() {
                *last = format!("{last}{trailing}");
            }
            words
        }
        None => vec![word.to_string()],
    }
}

#[cfg(any(feature = "native-mlx", test))]
fn collapse_repeated_leading_phrase<'a>(words: &[&'a str]) -> Vec<&'a str> {
    if words.len() < 10 {
        return words.to_vec();
    }

    let max_phrase_len = (words.len() / 2).min(8);
    for phrase_len in (5..=max_phrase_len).rev() {
        for repeat_index in phrase_len..=words.len().saturating_sub(phrase_len) {
            if normalized_word_slices_equal(
                &words[..phrase_len],
                &words[repeat_index..repeat_index + phrase_len],
            ) {
                return words[..repeat_index].to_vec();
            }
        }
    }

    words.to_vec()
}

#[cfg(any(feature = "native-mlx", test))]
fn normalize_transcript_word(word: &str) -> String {
    word.trim_matches(|ch: char| !ch.is_alphanumeric())
        .to_ascii_lowercase()
}

#[cfg(any(feature = "native-mlx", test))]
fn normalize_hotword_key(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

#[cfg(any(feature = "native-mlx", test))]
fn finish_whisper_transcript(text: String) -> String {
    let trimmed = text.trim();
    if trimmed.ends_with(',') || trimmed.ends_with(';') || trimmed.ends_with(':') {
        format!("{}.", trimmed.trim_end_matches([',', ';', ':']).trim_end())
    } else {
        trimmed.to_string()
    }
}

#[cfg(feature = "native-mlx")]
fn prefixed_weight_keys(prefixes: &[String; 3], module: &str) -> Vec<String> {
    prefixed_parameter_keys(prefixes, module, "weight")
}

#[cfg(feature = "native-mlx")]
fn prefixed_bias_keys(prefixes: &[String; 3], module: &str) -> Vec<String> {
    prefixed_parameter_keys(prefixes, module, "bias")
}

#[cfg(feature = "native-mlx")]
fn prefixed_parameter_keys(prefixes: &[String; 3], module: &str, suffix: &str) -> Vec<String> {
    let mut keys = prefixes
        .iter()
        .map(|prefix| format!("{prefix}.{module}.{suffix}"))
        .collect::<Vec<_>>();
    if let Some(alias) = mlx_whisper_module_alias(module) {
        keys.push(format!("{}.{}.{}", prefixes[2], alias, suffix));
    }
    keys
}

#[cfg(feature = "native-mlx")]
fn mlx_whisper_module_alias(module: &str) -> Option<&'static str> {
    match module {
        "self_attn_layer_norm" => Some("attn_ln"),
        "encoder_attn_layer_norm" => Some("cross_attn_ln"),
        "final_layer_norm" => Some("mlp_ln"),
        "self_attn.q_proj" => Some("attn.query"),
        "self_attn.k_proj" => Some("attn.key"),
        "self_attn.v_proj" => Some("attn.value"),
        "self_attn.out_proj" => Some("attn.out"),
        "encoder_attn.q_proj" => Some("cross_attn.query"),
        "encoder_attn.k_proj" => Some("cross_attn.key"),
        "encoder_attn.v_proj" => Some("cross_attn.value"),
        "encoder_attn.out_proj" => Some("cross_attn.out"),
        "fc1" => Some("mlp1"),
        "fc2" => Some("mlp2"),
        _ => None,
    }
}

#[cfg(feature = "native-mlx")]
impl MlxSpeechModel for WhisperSpeechModel {
    fn transcribe(&self, audio: &MlxArray, sample_rate_hz: u32) -> Result<String, MlxAudioError> {
        if sample_rate_hz != DEFAULT_WHISPER_SAMPLE_RATE_HZ {
            return Err(MlxAudioError::InvalidInput(format!(
                "Whisper expects {DEFAULT_WHISPER_SAMPLE_RATE_HZ} Hz audio, got {sample_rate_hz} Hz"
            )));
        }

        let mut mel_config = MelSpectrogramConfig::whisper();
        if let Some(num_mel_bins) = self.config.num_mel_bins {
            mel_config.mel_bins = num_mel_bins as usize;
        }
        let mel = log_mel_spectrogram(audio.as_slice::<f32>(), mel_config)
            .map_err(|error| MlxAudioError::InvalidInput(error.to_string()))?;
        let frames = i32::try_from(mel.frames).map_err(|_| {
            MlxAudioError::InvalidInput("mel frame count exceeds MLX shape limits".to_string())
        })?;
        let bins = i32::try_from(mel.bins).map_err(|_| {
            MlxAudioError::InvalidInput("mel bin count exceeds MLX shape limits".to_string())
        })?;
        let frontend = self.encode_frontend(&mel.values, frames, bins)?;
        let encoded = self.encode_transformer(&frontend)?;
        self.decode_greedy(&encoded)
    }

    fn synthesize(&self, _text: &str, _sample_rate_hz: u32) -> Result<MlxArray, MlxAudioError> {
        Err(MlxAudioError::ModelUnavailable(
            "Whisper speech model does not provide TTS".to_string(),
        ))
    }
}

#[cfg(not(feature = "native-mlx"))]
pub struct WhisperSpeechModel;

#[cfg(not(feature = "native-mlx"))]
impl WhisperSpeechModel {
    pub fn load(_speech_config: WhisperSpeechConfig) -> Result<Self, MlxAudioError> {
        Err(MlxAudioError::Runtime(
            "enable the native-mlx feature to use native Whisper loading".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn speech_config_sets_default_task() {
        let config = super::WhisperSpeechConfig::new("/tmp/model");
        assert_eq!(config.task, super::WhisperTask::Transcribe);
        assert_eq!(config.max_decode_tokens, 96);
    }

    #[test]
    fn non_native_loader_reports_feature_gate() {
        #[cfg(not(feature = "native-mlx"))]
        {
            let result =
                super::WhisperSpeechModel::load(super::WhisperSpeechConfig::new("/tmp/model"));
            assert!(matches!(result, Err(vona_mlx::MlxAudioError::Runtime(_))));
        }
    }

    #[test]
    fn postprocess_collapses_repeated_whisper_sentences() {
        let text = super::postprocess_whisper_transcript(
            " What should I focus on first today? What should I focus on first today? Good morning.",
        );

        assert_eq!(text, "What should I focus on first today? Good morning.");
    }

    #[test]
    fn postprocess_collapses_repeated_whisper_phrases() {
        let text = super::postprocess_whisper_transcript(
            "Hey Mera, can you tell me the time, the time, can you tell me the time, can you tell me the time.",
        );

        assert_eq!(text, "Hey Mera, can you tell me the time.");
    }

    #[test]
    fn postprocess_collapses_repeated_leading_phrase_restart() {
        let text = super::postprocess_whisper_transcript(
            "Hey Mera, can you tell me the time, hey Mera, can you tell me the assistant",
        );

        assert_eq!(text, "Hey Mera, can you tell me the time.");
    }

    #[test]
    fn postprocess_collapses_concatenated_case_restart() {
        let text = super::postprocess_whisper_transcript(
            "Rust Audio Pipeline Ready Case 13 Rust Audio Pipeline Readycase 13",
        );

        assert_eq!(text, "Rust Audio Pipeline Ready Case 13");
    }

    #[test]
    fn postprocess_collapses_vona_restart_variant() {
        let text = super::postprocess_whisper_transcript(
            "Vowna local inference check case 50 Vona local inference check case 50.",
        );

        assert_eq!(text, "Vona local inference check case 50");
    }

    #[test]
    fn postprocess_normalizes_domain_terms() {
        let text = super::postprocess_whisper_transcript(
            "A Q-N speech synthesis pass with Wispa and Alama on Vowna.",
        );

        assert_eq!(
            text,
            "A Qwen speech synthesis pass with Whisper and Ollama on Vona."
        );
    }

    #[test]
    fn postprocess_uses_configured_hotwords() {
        let hotwords = vec![super::TranscriptHotword::new(
            "Deliberium",
            ["delibrium", "deliberiam"],
        )];
        let text = super::postprocess_whisper_transcript_with_hotwords(
            "Ask delibrium about the plan.",
            &hotwords,
        );

        assert_eq!(text, "Ask Deliberium about the plan.");
    }

    #[test]
    fn parses_hotwords_from_env_format() {
        let hotwords =
            super::parse_transcript_hotwords("Deliberium=delibrium|deliberiam,Gemma 4=gemma for")
                .unwrap();

        assert_eq!(
            hotwords,
            vec![
                super::TranscriptHotword::new("Deliberium", ["delibrium", "deliberiam"]),
                super::TranscriptHotword::new("Gemma 4", ["gemma for"]),
            ]
        );
    }

    #[test]
    fn postprocess_keeps_non_repeated_phrases() {
        let text = super::postprocess_whisper_transcript(
            "Hey Mera, can you tell me the time in London and the weather in London?",
        );

        assert_eq!(
            text,
            "Hey Mera, can you tell me the time in London and the weather in London?"
        );
    }
}
