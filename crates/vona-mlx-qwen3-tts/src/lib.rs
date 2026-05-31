use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};
use vona_mlx::{LoadedMlxModel, MlxAudioError, MlxModelLoadRequest, MlxModelLoader};

#[cfg(feature = "native-mlx")]
use {
    std::collections::HashSet,
    std::sync::{Arc, Mutex},
    vona_mlx::{MlxArray, MlxModelKind, MlxSpeechModel},
    vona_mlx_speech::{
        MlxWeightView, SpeechModelFiles, embedding, gelu, linear, load_safetensors,
        safetensors_file_contains_dtype,
    },
};

#[cfg(feature = "native-mlx")]
use mlx_rs::ops::indexing::IndexOp;

pub const DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ: u32 = 24_000;
pub const DEFAULT_QWEN3_TTS_LANGUAGE: &str = "english";
pub const DEFAULT_QWEN3_TTS_SPEAKER: &str = "Vivian";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3TtsSpeechConfig {
    pub model_path: PathBuf,
    pub language: String,
    pub speaker: String,
    pub min_audio_tokens: usize,
    pub max_audio_tokens: usize,
    pub do_sample: bool,
    pub top_k: usize,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub subtalker_do_sample: bool,
    pub subtalker_top_k: usize,
    pub subtalker_temperature: f32,
}

impl Qwen3TtsSpeechConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        let min_audio_tokens = env_usize("VONA_MLX_QWEN3_TTS_MIN_AUDIO_TOKENS", 8);
        let max_audio_tokens =
            env_usize("VONA_MLX_QWEN3_TTS_MAX_AUDIO_TOKENS", 2048).max(min_audio_tokens);
        Self {
            model_path: model_path.into(),
            language: DEFAULT_QWEN3_TTS_LANGUAGE.to_string(),
            speaker: DEFAULT_QWEN3_TTS_SPEAKER.to_string(),
            min_audio_tokens,
            max_audio_tokens,
            do_sample: true,
            top_k: 50,
            temperature: 0.9,
            repetition_penalty: 1.05,
            subtalker_do_sample: true,
            subtalker_top_k: 50,
            subtalker_temperature: 0.9,
        }
    }
}

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(fallback)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3TtsConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub vocab_size: Option<u32>,
    #[serde(default)]
    pub hidden_size: Option<u32>,
    #[serde(default)]
    pub num_hidden_layers: Option<u32>,
    #[serde(default)]
    pub num_attention_heads: Option<u32>,
    #[serde(default)]
    pub num_key_value_heads: Option<u32>,
    #[serde(default)]
    pub tts_pad_token_id: Option<u32>,
    #[serde(default)]
    pub tts_bos_token_id: Option<u32>,
    #[serde(default)]
    pub tts_eos_token_id: Option<u32>,
    #[serde(default)]
    pub talker_config: TalkerConfig,
    #[serde(default)]
    pub vocoder_config: VocoderConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TalkerConfig {
    #[serde(default)]
    pub code_predictor_config: TalkerCodePredictorConfig,
    #[serde(default = "default_talker_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_talker_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_talker_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_talker_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_talker_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_talker_num_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_talker_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_codec_eos_token_id")]
    pub codec_eos_token_id: u32,
    #[serde(default = "default_codec_pad_id")]
    pub codec_pad_id: u32,
    #[serde(default = "default_codec_bos_id")]
    pub codec_bos_id: u32,
    #[serde(default = "default_codec_think_id")]
    pub codec_think_id: u32,
    #[serde(default = "default_codec_nothink_id")]
    pub codec_nothink_id: u32,
    #[serde(default = "default_codec_think_bos_id")]
    pub codec_think_bos_id: u32,
    #[serde(default = "default_codec_think_eos_id")]
    pub codec_think_eos_id: u32,
    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_id: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TalkerCodePredictorConfig {
    #[serde(default = "default_code_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_code_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_code_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_code_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_code_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_code_num_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_code_num_groups")]
    pub num_code_groups: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocoderConfig {
    #[serde(default = "default_vocoder_num_quantizers")]
    pub num_quantizers: usize,
    #[serde(default = "default_vocoder_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vocoder_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_vocoder_convnext_norm_eps")]
    pub convnext_norm_eps: f64,
    #[serde(default = "default_vocoder_stride_divisor")]
    pub transpose_stride_divisor: i32,
    #[serde(default = "default_vocoder_rope_theta")]
    pub rope_theta: f64,
}

fn default_talker_vocab_size() -> usize {
    3072
}

fn default_code_vocab_size() -> usize {
    2048
}

fn default_code_hidden_size() -> usize {
    1024
}

fn default_code_intermediate_size() -> usize {
    3072
}

fn default_code_num_layers() -> usize {
    5
}

fn default_code_num_heads() -> usize {
    16
}

fn default_code_num_kv_heads() -> usize {
    8
}

fn default_code_num_groups() -> usize {
    16
}

fn default_talker_hidden_size() -> usize {
    1024
}

fn default_talker_intermediate_size() -> usize {
    2048
}

fn default_talker_num_layers() -> usize {
    20
}

fn default_talker_num_heads() -> usize {
    16
}

fn default_talker_num_kv_heads() -> usize {
    2
}

fn default_rms_norm_eps() -> f64 {
    1.0e-6
}

fn default_talker_rope_theta() -> f64 {
    1_000_000.0
}

fn default_codec_eos_token_id() -> u32 {
    4198
}

fn default_codec_pad_id() -> u32 {
    4196
}

fn default_codec_bos_id() -> u32 {
    4197
}

fn default_codec_think_id() -> u32 {
    2154
}

fn default_codec_nothink_id() -> u32 {
    2155
}

fn default_codec_think_bos_id() -> u32 {
    2156
}

fn default_codec_think_eos_id() -> u32 {
    2157
}

fn default_vocoder_num_quantizers() -> usize {
    16
}

fn default_vocoder_attention_heads() -> usize {
    16
}

fn default_vocoder_norm_eps() -> f64 {
    1.0e-5
}

fn default_vocoder_convnext_norm_eps() -> f64 {
    1.0e-6
}

fn default_vocoder_stride_divisor() -> i32 {
    2
}

fn default_vocoder_rope_theta() -> f64 {
    10_000.0
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            code_predictor_config: TalkerCodePredictorConfig::default(),
            vocab_size: default_talker_vocab_size(),
            hidden_size: default_talker_hidden_size(),
            intermediate_size: default_talker_intermediate_size(),
            num_hidden_layers: default_talker_num_layers(),
            num_attention_heads: default_talker_num_heads(),
            num_key_value_heads: default_talker_num_kv_heads(),
            rms_norm_eps: default_rms_norm_eps(),
            rope_theta: default_talker_rope_theta(),
            codec_eos_token_id: default_codec_eos_token_id(),
            codec_pad_id: default_codec_pad_id(),
            codec_bos_id: default_codec_bos_id(),
            codec_think_id: default_codec_think_id(),
            codec_nothink_id: default_codec_nothink_id(),
            codec_think_bos_id: default_codec_think_bos_id(),
            codec_think_eos_id: default_codec_think_eos_id(),
            codec_language_id: HashMap::new(),
            spk_id: HashMap::new(),
        }
    }
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            num_quantizers: default_vocoder_num_quantizers(),
            num_attention_heads: default_vocoder_attention_heads(),
            norm_eps: default_vocoder_norm_eps(),
            convnext_norm_eps: default_vocoder_convnext_norm_eps(),
            transpose_stride_divisor: default_vocoder_stride_divisor(),
            rope_theta: default_vocoder_rope_theta(),
        }
    }
}

impl Default for TalkerCodePredictorConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_code_vocab_size(),
            hidden_size: default_code_hidden_size(),
            intermediate_size: default_code_intermediate_size(),
            num_hidden_layers: default_code_num_layers(),
            num_attention_heads: default_code_num_heads(),
            num_key_value_heads: default_code_num_kv_heads(),
            num_code_groups: default_code_num_groups(),
            rms_norm_eps: default_rms_norm_eps(),
        }
    }
}

#[cfg(feature = "native-mlx")]
impl Qwen3TtsConfig {
    fn language_id(&self, language: &str) -> Option<u32> {
        let language = language.trim().to_ascii_lowercase();
        if language.is_empty() || language == "auto" {
            return None;
        }
        self.talker_config.codec_language_id.get(&language).copied()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen3TtsLoader;

impl MlxModelLoader for Qwen3TtsLoader {
    fn load_model(&self, request: MlxModelLoadRequest) -> Result<LoadedMlxModel, MlxAudioError> {
        #[cfg(not(feature = "native-mlx"))]
        {
            let _ = request;
            Err(MlxAudioError::Runtime(
                "enable the native-mlx feature to use native Qwen3 TTS loading".to_string(),
            ))
        }

        #[cfg(feature = "native-mlx")]
        {
            if !matches!(
                request.kind,
                MlxModelKind::Speech | MlxModelKind::Qwen3TtsSpeech
            ) {
                return Err(MlxAudioError::InvalidInput(
                    "Qwen3TtsLoader only handles Qwen3 TTS speech requests".to_string(),
                ));
            }
            let model_path = request.local_path.ok_or_else(|| {
                MlxAudioError::InvalidInput(
                    "Qwen3TtsLoader requires a local model path".to_string(),
                )
            })?;
            let model = Qwen3TtsSpeechModel::load(Qwen3TtsSpeechConfig::new(model_path))?;
            Ok(LoadedMlxModel::Speech(Arc::new(model)))
        }
    }
}

#[cfg(feature = "native-mlx")]
pub struct Qwen3TtsSpeechModel {
    files: SpeechModelFiles,
    config: Qwen3TtsConfig,
    tokenizer: tokenizers::Tokenizer,
    weights: Mutex<HashMap<String, mlx_rs::Array>>,
    vocoder_weights: Mutex<Option<HashMap<String, mlx_rs::Array>>>,
    speech_config: Qwen3TtsSpeechConfig,
}

#[cfg(feature = "native-mlx")]
impl Qwen3TtsSpeechModel {
    pub fn load(speech_config: Qwen3TtsSpeechConfig) -> Result<Self, MlxAudioError> {
        let files = SpeechModelFiles::discover(&speech_config.model_path)
            .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?;
        let config = vona_mlx_speech::read_json::<Qwen3TtsConfig>(&files.config_path)
            .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?;
        let tokenizer = load_qwen_tokenizer(&files)?;
        for safetensors_file in &files.safetensors_files {
            if safetensors_file_contains_dtype(safetensors_file, "U32")
                .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?
            {
                return Err(MlxAudioError::ModelUnavailable(format!(
                    "quantized MLX safetensors are not yet supported by the native Qwen3 TTS loader; use a bf16 checkpoint instead of {}",
                    safetensors_file.display()
                )));
            }
        }
        let weights = load_safetensors(&files.safetensors_files)
            .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?;
        let vocoder_weights_path = files
            .model_dir
            .join("speech_tokenizer")
            .join("model.safetensors");
        let vocoder_weights = if vocoder_weights_path.is_file() {
            Some(
                load_safetensors(&[vocoder_weights_path])
                    .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?,
            )
        } else {
            None
        };

        Ok(Self {
            files,
            config,
            tokenizer,
            weights: Mutex::new(weights),
            vocoder_weights: Mutex::new(vocoder_weights),
            speech_config,
        })
    }

    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.config
    }

    pub fn weight_count(&self) -> usize {
        self.weights
            .lock()
            .map(|weights| weights.len())
            .unwrap_or(0)
    }

    pub fn vocoder_weight_count(&self) -> usize {
        self.vocoder_weights
            .lock()
            .ok()
            .and_then(|weights| weights.as_ref().map(HashMap::len))
            .unwrap_or(0)
    }

    fn build_generation_inputs(
        &self,
        prompt_token_ids: &[i32],
    ) -> Result<(MlxArray, MlxArray), MlxAudioError> {
        if prompt_token_ids.len() < 8 {
            return Err(MlxAudioError::InvalidInput(
                "Qwen3 TTS prompt did not tokenize into the expected ChatML structure".to_string(),
            ));
        }

        let weights = self
            .weights
            .lock()
            .map_err(|_| MlxAudioError::Runtime("Qwen3 TTS weight lock is poisoned".to_string()))?;
        let weights = MlxWeightView::new(&weights);
        let codec_embedding = weights
            .get("talker.model.codec_embedding.weight")
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let role_embeddings =
            self.project_text_token_ids_locked(&weights, &prompt_token_ids[..3])?;
        let first_text = self.project_text_token_ids_locked(&weights, &prompt_token_ids[3..4])?;
        let trailing_text = if prompt_token_ids.len() > 9 {
            self.project_text_token_ids_locked(
                &weights,
                &prompt_token_ids[4..prompt_token_ids.len() - 5],
            )?
        } else {
            self.empty_projected_text(&weights)?
        };
        let tts_bos =
            self.embed_tts_special(&weights, self.config.tts_bos_token_id.unwrap_or(151_672))?;
        let tts_eos =
            self.embed_tts_special(&weights, self.config.tts_eos_token_id.unwrap_or(151_673))?;
        let tts_pad = self.embed_tts_pad(&weights)?;

        let mut codec_prefill_ids =
            if let Some(language_id) = self.config.language_id(&self.speech_config.language) {
                vec![
                    self.config.talker_config.codec_think_id as i32,
                    self.config.talker_config.codec_think_bos_id as i32,
                    language_id as i32,
                    self.config.talker_config.codec_think_eos_id as i32,
                ]
            } else {
                vec![
                    self.config.talker_config.codec_nothink_id as i32,
                    self.config.talker_config.codec_think_bos_id as i32,
                    self.config.talker_config.codec_think_eos_id as i32,
                ]
            };
        codec_prefill_ids.push(self.config.talker_config.codec_pad_id as i32);
        codec_prefill_ids.push(self.config.talker_config.codec_bos_id as i32);

        let codec_prefill = embedding(codec_embedding, &codec_prefill_ids)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
            .reshape(&[1, codec_prefill_ids.len() as i32, -1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let prefix_len = codec_prefill.shape()[1] - 1;
        let pad_prefix =
            mlx_rs::ops::broadcast_to(&tts_pad, &[1, (prefix_len - 1).max(0), tts_pad.shape()[2]])
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let prefill_text_side = mlx_rs::ops::concatenate_axis(&[&pad_prefix, &tts_bos], 1)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let codec_side = codec_prefill.index((.., 0..prefix_len, ..));
        let codec_prefix = prefill_text_side + codec_side;
        let codec_bos = embedding(
            codec_embedding,
            &[self.config.talker_config.codec_bos_id as i32],
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?
        .reshape(&[1, 1, -1])
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let first_text_input = first_text + codec_bos.clone();
        let trailing_text = mlx_rs::ops::concatenate_axis(&[&trailing_text, &tts_eos], 1)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        if std::env::var_os("VONA_QWEN3_TTS_NON_STREAMING").is_some() {
            let codec_pad = embedding(
                codec_embedding,
                &[self.config.talker_config.codec_pad_id as i32],
            )
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
            .reshape(&[1, 1, -1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
            let codec_pad_repeated = mlx_rs::ops::broadcast_to(
                &codec_pad,
                &[1, trailing_text.shape()[1], codec_pad.shape()[2]],
            )
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
            let text_stream = trailing_text + codec_pad_repeated;
            let codec_bos_input = tts_pad.clone() + codec_bos;
            let prefix = mlx_rs::ops::concatenate_axis(
                &[
                    &role_embeddings,
                    &codec_prefix,
                    &text_stream,
                    &codec_bos_input,
                ],
                1,
            )
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
            return Ok((prefix, tts_pad));
        }
        let prefix =
            mlx_rs::ops::concatenate_axis(&[&role_embeddings, &codec_prefix, &first_text_input], 1)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let _ = tts_pad;
        Ok((prefix, trailing_text))
    }

    fn project_text_token_ids_locked(
        &self,
        weights: &MlxWeightView<'_>,
        token_ids: &[i32],
    ) -> Result<MlxArray, MlxAudioError> {
        let text_embedding = weights
            .get_any(&[
                "talker.model.text_embedding.weight",
                "talker.text_embedding.weight",
                "talker.model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "text_embedding.weight",
            ])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let embeddings = embedding(text_embedding, token_ids)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
            .reshape(&[1, token_ids.len() as i32, text_embedding.shape()[1]])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        self.project_text_embeddings_locked(weights, &embeddings)
    }

    fn embed_tts_special(
        &self,
        weights: &MlxWeightView<'_>,
        id: u32,
    ) -> Result<MlxArray, MlxAudioError> {
        let text_embedding = weights
            .get_any(&[
                "talker.model.text_embedding.weight",
                "talker.text_embedding.weight",
                "talker.model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "text_embedding.weight",
            ])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        embedding(text_embedding, &[id as i32])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
            .reshape(&[1, 1, text_embedding.shape()[1]])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
            .and_then(|embedding| self.project_text_embeddings_locked(weights, &embedding))
    }

    fn embed_tts_pad(&self, weights: &MlxWeightView<'_>) -> Result<MlxArray, MlxAudioError> {
        let text_embedding = weights
            .get_any(&[
                "talker.model.text_embedding.weight",
                "talker.text_embedding.weight",
                "talker.model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "text_embedding.weight",
            ])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let id = self.config.tts_pad_token_id.unwrap_or(151_671) as i32;
        embedding(text_embedding, &[id])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
            .reshape(&[1, 1, text_embedding.shape()[1]])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
            .and_then(|embedding| self.project_text_embeddings_locked(weights, &embedding))
    }

    fn empty_projected_text(&self, weights: &MlxWeightView<'_>) -> Result<MlxArray, MlxAudioError> {
        let text_embedding = weights
            .get_any(&[
                "talker.model.text_embedding.weight",
                "talker.text_embedding.weight",
                "talker.model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "text_embedding.weight",
            ])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let hidden = text_embedding.shape()[1];
        mlx_rs::Array::from_slice::<f32>(&[], &[1, 0, hidden])
            .as_type::<f32>()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }

    fn project_text_embeddings_locked(
        &self,
        weights: &MlxWeightView<'_>,
        embeddings: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let fc1 = linear(
            embeddings,
            weights
                .get("talker.text_projection.linear_fc1.weight")
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional("talker.text_projection.linear_fc1.bias"),
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let fc1 = gelu(&fc1).map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        linear(
            &fc1,
            weights
                .get("talker.text_projection.linear_fc2.weight")
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional("talker.text_projection.linear_fc2.bias"),
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }

    fn generate_codec_frames(
        &self,
        prefix: &MlxArray,
        trailing_text: &MlxArray,
    ) -> Result<Vec<Vec<u32>>, MlxAudioError> {
        let weights = self
            .weights
            .lock()
            .map_err(|_| MlxAudioError::Runtime("Qwen3 TTS weight lock is poisoned".to_string()))?;
        let weights = MlxWeightView::new(&weights);
        let tts_pad = self.embed_tts_pad(&weights)?;
        let mut sequence = prefix.clone();
        let mut frames = Vec::new();
        let mut generated_main_codes = HashSet::new();
        let mut rng = XorShift64::from_entropy();

        for _ in 0..self.speech_config.max_audio_tokens {
            let hidden = self.forward_talker(&weights, &sequence)?;
            let allow_eos = frames.len() >= self.speech_config.min_audio_tokens;
            let frame = self.predict_codec_frame_from_hidden(
                &weights,
                &hidden,
                allow_eos,
                &mut rng,
                &generated_main_codes,
            )?;
            if frame.first().copied() == Some(self.config.talker_config.codec_eos_token_id) {
                break;
            }
            let next_text = if (frames.len() as i32) < trailing_text.shape()[1] {
                trailing_text.index((.., frames.len() as i32..frames.len() as i32 + 1, ..))
            } else {
                tts_pad.clone()
            };
            let next_input = self.codec_frame_input(&weights, &frame, &next_text)?;
            if let Some(code) = frame.first().copied() {
                generated_main_codes.insert(code);
            }
            frames.push(frame);
            sequence = mlx_rs::ops::concatenate_axis(&[&sequence, &next_input], 1)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        }

        if std::env::var_os("VONA_QWEN3_TTS_DEBUG_CODES").is_some() {
            eprintln!(
                "qwen3_tts codec_frames={} first_frames={:?}",
                frames.len(),
                frames.iter().take(8).collect::<Vec<_>>()
            );
        }

        Ok(frames)
    }

    fn predict_codec_frame_from_hidden(
        &self,
        weights: &MlxWeightView<'_>,
        hidden: &MlxArray,
        allow_eos: bool,
        rng: &mut XorShift64,
        generated_main_codes: &HashSet<u32>,
    ) -> Result<Vec<u32>, MlxAudioError> {
        let codec_head = weights
            .get("talker.codec_head.weight")
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let logits = linear(
            hidden,
            codec_head,
            weights.optional("talker.codec_head.bias"),
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let suppressed_start = self.config.talker_config.vocab_size.saturating_sub(1024);
        let eos = self.config.talker_config.codec_eos_token_id;
        let code_0 = self.select_main_codec_token(
            &logits,
            rng,
            |index| {
                let id = index as u32;
                let is_reserved = index >= suppressed_start;
                (!is_reserved || id == eos) && (allow_eos || id != eos)
            },
            generated_main_codes,
        )?;
        if code_0 == self.config.talker_config.codec_eos_token_id {
            return Ok(vec![code_0]);
        }

        let seq_len = hidden.shape()[1];
        let main_hidden = hidden.index((.., seq_len - 1..seq_len, ..));
        let code_0_embedding = self.embed_main_codec(weights, code_0)?;
        let predictor_codes =
            self.generate_code_predictor_codes(weights, &main_hidden, &code_0_embedding, rng)?;
        let mut frame = Vec::with_capacity(1 + predictor_codes.len());
        frame.push(code_0);
        frame.extend(predictor_codes);
        Ok(frame)
    }

    fn codec_frame_input(
        &self,
        weights: &MlxWeightView<'_>,
        frame: &[u32],
        tts_pad: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        if frame.len()
            != self
                .config
                .talker_config
                .code_predictor_config
                .num_code_groups
        {
            return Err(MlxAudioError::Inference(format!(
                "Qwen3 TTS generated codec frame has {} groups, expected {}",
                frame.len(),
                self.config
                    .talker_config
                    .code_predictor_config
                    .num_code_groups
            )));
        }

        let mut codec_sum = self.embed_main_codec(weights, frame[0])?;
        for (index, code) in frame.iter().copied().enumerate().skip(1) {
            codec_sum += self.embed_predictor_codec(weights, index - 1, code)?;
        }
        Ok(codec_sum + tts_pad.clone())
    }

    fn embed_main_codec(
        &self,
        weights: &MlxWeightView<'_>,
        code: u32,
    ) -> Result<MlxArray, MlxAudioError> {
        let codec_embedding = weights
            .get("talker.model.codec_embedding.weight")
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        embedding(codec_embedding, &[code as i32])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
            .reshape(&[1, 1, -1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }

    fn embed_predictor_codec(
        &self,
        weights: &MlxWeightView<'_>,
        index: usize,
        code: u32,
    ) -> Result<MlxArray, MlxAudioError> {
        let embedding_key = format!("talker.code_predictor.model.codec_embedding.{index}.weight");
        weights
            .get(&embedding_key)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
            .and_then(|embedding_weight| {
                embedding(embedding_weight, &[code as i32])
                    .map_err(|error| MlxAudioError::Inference(error.to_string()))
            })?
            .reshape(&[1, 1, -1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }

    fn forward_talker(
        &self,
        weights: &MlxWeightView<'_>,
        input: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let mut hidden = input.clone();
        let layer_count = self.config.talker_config.num_hidden_layers as u32;
        let head_count = self.config.talker_config.num_attention_heads as i32;
        let kv_head_count = self.config.talker_config.num_key_value_heads as i32;
        for layer in 0..layer_count {
            hidden = self.talker_layer(weights, layer, &hidden, head_count, kv_head_count)?;
        }
        let norm_weight = weights
            .get("talker.model.norm.weight")
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        self.rms_norm(&hidden, norm_weight)
    }

    fn generate_code_predictor_codes(
        &self,
        weights: &MlxWeightView<'_>,
        main_hidden: &MlxArray,
        code_0_embedding: &MlxArray,
        rng: &mut XorShift64,
    ) -> Result<Vec<u32>, MlxAudioError> {
        let config = &self.config.talker_config.code_predictor_config;
        let count = config.num_code_groups.saturating_sub(1);
        let mut sequence = mlx_rs::ops::concatenate_axis(&[main_hidden, code_0_embedding], 1)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let mut codes = Vec::with_capacity(count);

        for step in 0..count {
            let projected_sequence = if weights
                .optional("talker.code_predictor.small_to_mtp_projection.weight")
                .is_some()
            {
                self.linear_key(
                    weights,
                    "talker.code_predictor.small_to_mtp_projection",
                    &sequence,
                )?
            } else {
                sequence.clone()
            };
            let mut hidden = projected_sequence;
            for layer in 0..config.num_hidden_layers as u32 {
                hidden = self.transformer_layer(
                    weights,
                    &format!("talker.code_predictor.model.layers.{layer}"),
                    &hidden,
                    config.num_attention_heads as i32,
                    config.num_key_value_heads as i32,
                    config.rms_norm_eps as f32,
                )?;
            }
            let norm_weight = weights
                .get("talker.code_predictor.model.norm.weight")
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
            let hidden = self.rms_norm_eps(&hidden, norm_weight, config.rms_norm_eps as f32)?;
            let last = hidden.index((.., hidden.shape()[1] - 1..hidden.shape()[1], ..));
            let logits = self.linear_key(
                weights,
                &format!("talker.code_predictor.lm_head.{step}"),
                &last,
            )?;
            let code = self.select_subtalker_last_token(&logits, rng)?;
            codes.push(code);

            let code_embedding = self.embed_predictor_codec(weights, step, code)?;
            sequence = mlx_rs::ops::concatenate_axis(&[&sequence, &code_embedding], 1)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        }

        Ok(codes)
    }

    fn talker_layer(
        &self,
        weights: &MlxWeightView<'_>,
        layer: u32,
        hidden: &MlxArray,
        head_count: i32,
        kv_head_count: i32,
    ) -> Result<MlxArray, MlxAudioError> {
        let prefix = format!("talker.model.layers.{layer}");
        self.transformer_layer(
            weights,
            &prefix,
            hidden,
            head_count,
            kv_head_count,
            self.config.talker_config.rms_norm_eps as f32,
        )
    }

    fn transformer_layer(
        &self,
        weights: &MlxWeightView<'_>,
        prefix: &str,
        hidden: &MlxArray,
        head_count: i32,
        kv_head_count: i32,
        rms_norm_eps: f32,
    ) -> Result<MlxArray, MlxAudioError> {
        let residual = hidden.clone();
        let norm = self.rms_norm_eps(
            hidden,
            weights
                .get(&format!("{prefix}.input_layernorm.weight"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            rms_norm_eps,
        )?;
        let attention =
            self.talker_attention(weights, &prefix, &norm, head_count, kv_head_count)?;
        let hidden = residual + attention;

        let residual = hidden.clone();
        let norm = self.rms_norm_eps(
            &hidden,
            weights
                .get(&format!("{prefix}.post_attention_layernorm.weight"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            rms_norm_eps,
        )?;
        let gate = self.linear_key(weights, &format!("{prefix}.mlp.gate_proj"), &norm)?;
        let gate =
            mlx_rs::nn::silu(&gate).map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let up = self.linear_key(weights, &format!("{prefix}.mlp.up_proj"), &norm)?;
        let gated = gate * up;
        let down = self.linear_key(weights, &format!("{prefix}.mlp.down_proj"), &gated)?;
        Ok(residual + down)
    }

    fn talker_attention(
        &self,
        weights: &MlxWeightView<'_>,
        prefix: &str,
        hidden: &MlxArray,
        head_count: i32,
        kv_head_count: i32,
    ) -> Result<MlxArray, MlxAudioError> {
        let q = self.linear_key(weights, &format!("{prefix}.self_attn.q_proj"), hidden)?;
        let k = self.linear_key(weights, &format!("{prefix}.self_attn.k_proj"), hidden)?;
        let v = self.linear_key(weights, &format!("{prefix}.self_attn.v_proj"), hidden)?;
        let shape = q.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let head_dim = hidden_size / head_count;
        let q = q
            .reshape(&[batch, seq_len, head_count, head_dim])
            .and_then(|array| {
                self.apply_attention_norm(
                    weights,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    &array,
                    self.config.talker_config.rms_norm_eps as f32,
                )
            })
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .and_then(|array| {
                mlx_rs::fast::rope(
                    array,
                    head_dim,
                    false,
                    self.config.talker_config.rope_theta as f32,
                    1.0,
                    0,
                    None,
                )
            })
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let k = k
            .reshape(&[batch, seq_len, kv_head_count, head_dim])
            .and_then(|array| {
                self.apply_attention_norm(
                    weights,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    &array,
                    self.config.talker_config.rms_norm_eps as f32,
                )
            })
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .and_then(|array| {
                mlx_rs::fast::rope(
                    array,
                    head_dim,
                    false,
                    self.config.talker_config.rope_theta as f32,
                    1.0,
                    0,
                    None,
                )
            })
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let v = v
            .reshape(&[batch, seq_len, kv_head_count, head_dim])
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let (k, v) = repeat_kv_heads(k, v, head_count, kv_head_count)?;

        let attention = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0 / (head_dim as f32).sqrt(),
            mlx_rs::fast::ScaledDotProductAttentionMask::Causal,
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let attention = attention
            .transpose_axes(&[0, 2, 1, 3])
            .and_then(|array| array.reshape(&[batch, seq_len, hidden_size]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        self.linear_key(weights, &format!("{prefix}.self_attn.o_proj"), &attention)
    }

    fn apply_attention_norm(
        &self,
        weights: &MlxWeightView<'_>,
        key: &str,
        input: &MlxArray,
        eps: f32,
    ) -> std::result::Result<MlxArray, mlx_rs::error::Exception> {
        if let Some(weight) = weights.optional(key) {
            self.rms_norm_eps(input, weight, eps)
                .map_err(|error| mlx_rs::error::Exception::custom(error.to_string()))
        } else {
            Ok(input.clone())
        }
    }

    fn linear_key(
        &self,
        weights: &MlxWeightView<'_>,
        prefix: &str,
        input: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let weight_key = format!("{prefix}.weight");
        let bias_key = format!("{prefix}.bias");
        linear(
            input,
            weights
                .get(&weight_key)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional(&bias_key),
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }

    fn rms_norm(&self, input: &MlxArray, weight: &MlxArray) -> Result<MlxArray, MlxAudioError> {
        self.rms_norm_eps(input, weight, self.config.talker_config.rms_norm_eps as f32)
    }

    fn rms_norm_eps(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        eps: f32,
    ) -> Result<MlxArray, MlxAudioError> {
        let input_f32 = input
            .as_type::<f32>()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let square = mlx_rs::ops::square(&input_f32)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let variance = mlx_rs::ops::mean_axis(&square, -1, true)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let scale = mlx_rs::ops::rsqrt(variance + eps)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        Ok(input_f32 * scale * weight)
    }

    fn select_subtalker_last_token(
        &self,
        logits: &MlxArray,
        rng: &mut XorShift64,
    ) -> Result<u32, MlxAudioError> {
        self.select_last_token_with_options(
            logits,
            rng,
            |_| true,
            self.speech_config.subtalker_do_sample,
            self.speech_config.subtalker_top_k,
            self.speech_config.subtalker_temperature,
            1.0,
            &HashSet::new(),
        )
    }

    fn select_main_codec_token(
        &self,
        logits: &MlxArray,
        rng: &mut XorShift64,
        include: impl Fn(usize) -> bool,
        generated_main_codes: &HashSet<u32>,
    ) -> Result<u32, MlxAudioError> {
        self.select_last_token_with_options(
            logits,
            rng,
            include,
            self.speech_config.do_sample,
            self.speech_config.top_k,
            self.speech_config.temperature,
            self.speech_config.repetition_penalty,
            generated_main_codes,
        )
    }

    fn select_last_token_with_options(
        &self,
        logits: &MlxArray,
        rng: &mut XorShift64,
        include: impl Fn(usize) -> bool,
        do_sample: bool,
        top_k: usize,
        temperature: f32,
        repetition_penalty: f32,
        generated_tokens: &HashSet<u32>,
    ) -> Result<u32, MlxAudioError> {
        let logits = logits
            .as_type::<f32>()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        logits
            .eval()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let shape = logits.shape();
        if shape.len() != 3 || shape[0] != 1 {
            return Err(MlxAudioError::Inference(format!(
                "unexpected Qwen3 TTS logits shape {:?}",
                shape
            )));
        }
        let seq_len = shape[1] as usize;
        let vocab_size = shape[2] as usize;
        let values = logits.as_slice::<f32>();
        let offset = (seq_len.saturating_sub(1)) * vocab_size;
        let mut candidates = values[offset..offset + vocab_size]
            .iter()
            .enumerate()
            .filter(|(index, _)| include(*index))
            .filter_map(|(index, value)| {
                value.is_finite().then_some((
                    index,
                    apply_repetition_penalty(
                        *value,
                        generated_tokens.contains(&(index as u32)),
                        repetition_penalty,
                    ),
                ))
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(MlxAudioError::Inference(
                "empty Qwen3 TTS logits after filtering".to_string(),
            ));
        }
        candidates.sort_by(|(_, left), (_, right)| {
            right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal)
        });
        if !do_sample {
            return Ok(candidates[0].0 as u32);
        }
        let top_k = top_k.max(1).min(candidates.len());
        candidates.truncate(top_k);
        let temperature = temperature.max(1.0e-5);
        let max_logit = candidates[0].1;
        let mut total = 0.0_f64;
        let weights = candidates
            .iter()
            .map(|(_, logit)| {
                let weight = ((*logit - max_logit) / temperature).exp() as f64;
                total += weight;
                weight
            })
            .collect::<Vec<_>>();
        if total <= 0.0 || !total.is_finite() {
            return Ok(candidates[0].0 as u32);
        }
        let mut threshold = rng.next_f64() * total;
        for ((index, _), weight) in candidates.iter().zip(weights) {
            threshold -= weight;
            if threshold <= 0.0 {
                return Ok(*index as u32);
            }
        }
        Ok(candidates[0].0 as u32)
    }

    fn decode_vocoder_latents(&self, codec_frames: &[Vec<u32>]) -> Result<MlxArray, MlxAudioError> {
        if codec_frames.is_empty() {
            return Err(MlxAudioError::InvalidInput(
                "Qwen3 TTS codec frame list is empty".to_string(),
            ));
        }
        let weights = self.vocoder_weights.lock().map_err(|_| {
            MlxAudioError::Runtime("Qwen3 vocoder weight lock is poisoned".to_string())
        })?;
        let weights = weights.as_ref().ok_or_else(|| {
            MlxAudioError::ModelUnavailable(format!(
                "missing Qwen3 TTS vocoder weights at {}/speech_tokenizer/model.safetensors",
                self.files.model_dir.display()
            ))
        })?;
        let weights = MlxWeightView::new(weights);
        let quantizer_count = self.config.vocoder_config.num_quantizers;
        if codec_frames
            .iter()
            .any(|frame| frame.len() != quantizer_count)
        {
            return Err(MlxAudioError::InvalidInput(format!(
                "Qwen3 TTS vocoder expects {quantizer_count} codec IDs per frame"
            )));
        }

        let semantic =
            self.decode_vq_group(&weights, "decoder.quantizer.rvq_first.", 0, 1, codec_frames)?;
        let acoustic = self.decode_vq_group(
            &weights,
            "decoder.quantizer.rvq_rest.",
            1,
            quantizer_count,
            codec_frames,
        )?;
        let decoded = semantic + acoustic;
        decoded
            .eval()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        Ok(decoded)
    }

    fn run_vocoder_frontend(&self, latents: &MlxArray) -> Result<MlxArray, MlxAudioError> {
        let weights = self.vocoder_weights.lock().map_err(|_| {
            MlxAudioError::Runtime("Qwen3 vocoder weight lock is poisoned".to_string())
        })?;
        let weights = weights.as_ref().ok_or_else(|| {
            MlxAudioError::ModelUnavailable(format!(
                "missing Qwen3 TTS vocoder weights at {}/speech_tokenizer/model.safetensors",
                self.files.model_dir.display()
            ))
        })?;
        let weights = MlxWeightView::new(weights);

        let pre_conv = self.causal_conv1d_ncl(
            latents,
            weights
                .get("decoder.pre_conv.conv.weight")
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional("decoder.pre_conv.conv.bias"),
            1,
            1,
            1,
        )?;
        let hidden = self.vocoder_pre_transformer(&weights, &pre_conv)?;
        self.vocoder_decode_waveform(&weights, &hidden)
    }

    fn vocoder_pre_transformer(
        &self,
        weights: &MlxWeightView<'_>,
        hidden_ncl: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let hidden = hidden_ncl
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let mut hidden = linear(
            &hidden,
            weights
                .get("decoder.pre_transformer.input_proj.weight")
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional("decoder.pre_transformer.input_proj.bias"),
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;

        let layer_count = count_indexed_layers(weights, "decoder.pre_transformer.layers.");
        for layer in 0..layer_count {
            hidden = self.vocoder_transformer_layer(
                weights,
                &format!("decoder.pre_transformer.layers.{layer}"),
                &hidden,
            )?;
        }

        hidden = self.rms_norm_eps(
            &hidden,
            weights
                .get("decoder.pre_transformer.norm.weight")
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            self.config.vocoder_config.norm_eps as f32,
        )?;
        let hidden = linear(
            &hidden,
            weights
                .get("decoder.pre_transformer.output_proj.weight")
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional("decoder.pre_transformer.output_proj.bias"),
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let hidden = hidden
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        hidden
            .eval()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        Ok(hidden)
    }

    fn vocoder_transformer_layer(
        &self,
        weights: &MlxWeightView<'_>,
        prefix: &str,
        hidden: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let residual = hidden.clone();
        let norm = self.rms_norm_eps(
            hidden,
            weights
                .get(&format!("{prefix}.input_layernorm.weight"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            self.config.vocoder_config.norm_eps as f32,
        )?;
        let attention = self.vocoder_attention(weights, prefix, &norm)?;
        let attention_scale = weights
            .get(&format!("{prefix}.self_attn_layer_scale.scale"))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let hidden = residual + attention * attention_scale;

        let residual = hidden.clone();
        let norm = self.rms_norm_eps(
            &hidden,
            weights
                .get(&format!("{prefix}.post_attention_layernorm.weight"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            self.config.vocoder_config.norm_eps as f32,
        )?;
        let gate = self.linear_key(weights, &format!("{prefix}.mlp.gate_proj"), &norm)?;
        let gate =
            mlx_rs::nn::silu(&gate).map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let up = self.linear_key(weights, &format!("{prefix}.mlp.up_proj"), &norm)?;
        let down = self.linear_key(weights, &format!("{prefix}.mlp.down_proj"), &(gate * up))?;
        let mlp_scale = weights
            .get(&format!("{prefix}.mlp_layer_scale.scale"))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        Ok(residual + down * mlp_scale)
    }

    fn vocoder_attention(
        &self,
        weights: &MlxWeightView<'_>,
        prefix: &str,
        hidden: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let q = self.linear_key(weights, &format!("{prefix}.self_attn.q_proj"), hidden)?;
        let k = self.linear_key(weights, &format!("{prefix}.self_attn.k_proj"), hidden)?;
        let v = self.linear_key(weights, &format!("{prefix}.self_attn.v_proj"), hidden)?;
        let batch = q.shape()[0];
        let seq_len = q.shape()[1];
        let q_dim = q.shape()[2];
        let head_count = (self.config.vocoder_config.num_attention_heads as i32).max(1);
        let head_dim = (q_dim / head_count).max(1);
        let kv_head_count = (k.shape()[2] / head_dim).max(1);
        let q = q
            .reshape(&[batch, seq_len, head_count, head_dim])
            .and_then(|array| {
                self.apply_attention_norm(
                    weights,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    &array,
                    self.config.vocoder_config.norm_eps as f32,
                )
            })
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .and_then(|array| {
                mlx_rs::fast::rope(
                    array,
                    head_dim,
                    false,
                    self.config.vocoder_config.rope_theta as f32,
                    1.0,
                    0,
                    None,
                )
            })
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let k = k
            .reshape(&[batch, seq_len, kv_head_count, head_dim])
            .and_then(|array| {
                self.apply_attention_norm(
                    weights,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    &array,
                    self.config.vocoder_config.norm_eps as f32,
                )
            })
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .and_then(|array| {
                mlx_rs::fast::rope(
                    array,
                    head_dim,
                    false,
                    self.config.vocoder_config.rope_theta as f32,
                    1.0,
                    0,
                    None,
                )
            })
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let v = v
            .reshape(&[batch, seq_len, kv_head_count, head_dim])
            .and_then(|array| array.transpose_axes(&[0, 2, 1, 3]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let (k, v) = repeat_kv_heads(k, v, head_count, kv_head_count)?;

        let attention = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0 / (head_dim as f32).sqrt(),
            mlx_rs::fast::ScaledDotProductAttentionMask::Causal,
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let attention = attention
            .transpose_axes(&[0, 2, 1, 3])
            .and_then(|array| array.reshape(&[batch, seq_len, q_dim]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        self.linear_key(weights, &format!("{prefix}.self_attn.o_proj"), &attention)
    }

    fn causal_conv1d_ncl(
        &self,
        input_ncl: &MlxArray,
        weight_ock: &MlxArray,
        bias: Option<&MlxArray>,
        stride: i32,
        dilation: i32,
        groups: i32,
    ) -> Result<MlxArray, MlxAudioError> {
        if groups > 1 {
            let groups = groups as usize;
            let input_channels = input_ncl.shape()[1] as usize;
            let output_channels = weight_ock.shape()[0] as usize;
            if input_channels % groups != 0 || output_channels % groups != 0 {
                return Err(MlxAudioError::Inference(format!(
                    "invalid grouped convolution shape input_channels={input_channels}, output_channels={output_channels}, groups={groups}"
                )));
            }
            let input_per_group = input_channels / groups;
            let output_per_group = output_channels / groups;
            let mut outputs = Vec::with_capacity(groups);
            for group in 0..groups {
                let input_start = (group * input_per_group) as i32;
                let input_end = input_start + input_per_group as i32;
                let output_start = (group * output_per_group) as i32;
                let output_end = output_start + output_per_group as i32;
                let input_slice = input_ncl.index((.., input_start..input_end, ..));
                let weight_slice = weight_ock.index((output_start..output_end, .., ..));
                let bias_slice = bias.map(|bias| bias.index(output_start..output_end));
                outputs.push(self.causal_conv1d_ncl(
                    &input_slice,
                    &weight_slice,
                    bias_slice.as_ref(),
                    stride,
                    dilation,
                    1,
                )?);
            }
            let output_refs = outputs.iter().collect::<Vec<_>>();
            return mlx_rs::ops::concatenate_axis(&output_refs, 1)
                .map_err(|error| MlxAudioError::Inference(error.to_string()));
        }

        let input_nlc = input_ncl
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let kernel = weight_ock.shape()[2];
        let effective_kernel = (kernel - 1) * dilation + 1;
        let left_padding = (effective_kernel - stride).max(0);
        let padded = if left_padding > 0 {
            mlx_rs::ops::pad(&input_nlc, &[(0, 0), (left_padding, 0), (0, 0)], None, None)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?
        } else {
            input_nlc
        };
        let weight_okc = weight_ock
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let mut output = mlx_rs::ops::conv1d(&padded, &weight_okc, stride, 0, dilation, groups)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        if let Some(bias) = bias {
            output += bias;
        }
        output
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }

    fn causal_transpose_conv1d_ncl(
        &self,
        input_ncl: &MlxArray,
        weight_iok: &MlxArray,
        bias: Option<&MlxArray>,
        stride: i32,
    ) -> Result<MlxArray, MlxAudioError> {
        let input_nlc = input_ncl
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let weight_oki = weight_iok
            .transpose_axes(&[1, 2, 0])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let mut output = mlx_rs::ops::conv_transpose1d(&input_nlc, &weight_oki, stride, 0, 1, 0, 1)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        if let Some(bias) = bias {
            output += bias;
        }
        let right_pad = (weight_iok.shape()[2] - stride).max(0);
        if right_pad > 0 && output.shape()[1] > right_pad {
            output = output.index((.., 0..output.shape()[1] - right_pad, ..));
        }
        output
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }

    fn vocoder_decode_waveform(
        &self,
        weights: &MlxWeightView<'_>,
        hidden: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let mut hidden = hidden.clone();
        let upsample_count = count_upsample_blocks(weights);
        for index in 0..upsample_count {
            hidden = self.causal_transpose_conv1d_ncl(
                &hidden,
                weights
                    .get(&format!("decoder.upsample.{index}.0.conv.weight"))
                    .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
                weights.optional(&format!("decoder.upsample.{index}.0.conv.bias")),
                inferred_transpose_stride(
                    weights
                        .get(&format!("decoder.upsample.{index}.0.conv.weight"))
                        .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
                    self.config.vocoder_config.transpose_stride_divisor,
                ),
            )?;
            hidden =
                self.convnext_block(weights, &format!("decoder.upsample.{index}.1."), &hidden)?;
        }

        hidden = self.causal_conv1d_ncl(
            &hidden,
            weights
                .get("decoder.decoder.0.conv.weight")
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional("decoder.decoder.0.conv.bias"),
            1,
            1,
            1,
        )?;

        let decoder_block_count = count_decoder_blocks(weights);
        for block in 0..decoder_block_count {
            hidden =
                self.decoder_block(weights, &format!("decoder.decoder.{}.", block + 1), &hidden)?;
        }

        let final_snake_index = decoder_block_count + 1;
        hidden = self.snake_beta(
            &hidden,
            weights
                .get(&format!("decoder.decoder.{final_snake_index}.alpha"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights
                .get(&format!("decoder.decoder.{final_snake_index}.beta"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
        )?;
        let final_conv_index = final_snake_index + 1;
        hidden = self.causal_conv1d_ncl(
            &hidden,
            weights
                .get(&format!("decoder.decoder.{final_conv_index}.conv.weight"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional(&format!("decoder.decoder.{final_conv_index}.conv.bias")),
            1,
            1,
            1,
        )?;
        let hidden = mlx_rs::ops::clip(&hidden, (-1.0f32, 1.0f32))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let hidden = if hidden.shape().get(1).copied() == Some(1) {
            hidden
                .reshape(&[hidden.shape()[0], hidden.shape()[2]])
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?
        } else {
            hidden
        };
        hidden
            .eval()
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        Ok(hidden)
    }

    fn convnext_block(
        &self,
        weights: &MlxWeightView<'_>,
        prefix: &str,
        input: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let residual = input.clone();
        let hidden = self.causal_conv1d_ncl(
            input,
            weights
                .get(&format!("{prefix}dwconv.conv.weight"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional(&format!("{prefix}dwconv.conv.bias")),
            1,
            1,
            input.shape()[1],
        )?;
        let hidden = hidden
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let hidden = mlx_rs::fast::layer_norm(
            &hidden,
            Some(
                weights
                    .get(&format!("{prefix}norm.weight"))
                    .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            ),
            Some(
                weights
                    .get(&format!("{prefix}norm.bias"))
                    .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            ),
            self.config.vocoder_config.convnext_norm_eps as f32,
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let hidden = linear(
            &hidden,
            weights
                .get(&format!("{prefix}pwconv1.weight"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional(&format!("{prefix}pwconv1.bias")),
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let hidden = gelu(&hidden).map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let hidden = linear(
            &hidden,
            weights
                .get(&format!("{prefix}pwconv2.weight"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional(&format!("{prefix}pwconv2.bias")),
        )
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let hidden = hidden
            * weights
                .get(&format!("{prefix}gamma"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let hidden = hidden
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        Ok(residual + hidden)
    }

    fn decoder_block(
        &self,
        weights: &MlxWeightView<'_>,
        prefix: &str,
        input: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let mut hidden = self.snake_beta(
            input,
            weights
                .get(&format!("{prefix}block.0.alpha"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights
                .get(&format!("{prefix}block.0.beta"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
        )?;
        let trans_weight = weights
            .get(&format!("{prefix}block.1.conv.weight"))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        hidden = self.causal_transpose_conv1d_ncl(
            &hidden,
            trans_weight,
            weights.optional(&format!("{prefix}block.1.conv.bias")),
            inferred_transpose_stride(
                trans_weight,
                self.config.vocoder_config.transpose_stride_divisor,
            ),
        )?;
        for (unit, dilation) in [1, 3, 9].into_iter().enumerate() {
            hidden = self.decoder_residual_unit(
                weights,
                &format!("{prefix}block.{}.", unit + 2),
                &hidden,
                dilation,
            )?;
        }
        Ok(hidden)
    }

    fn decoder_residual_unit(
        &self,
        weights: &MlxWeightView<'_>,
        prefix: &str,
        input: &MlxArray,
        dilation: i32,
    ) -> Result<MlxArray, MlxAudioError> {
        let residual = input.clone();
        let hidden = self.snake_beta(
            input,
            weights
                .get(&format!("{prefix}act1.alpha"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights
                .get(&format!("{prefix}act1.beta"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
        )?;
        let hidden = self.causal_conv1d_ncl(
            &hidden,
            weights
                .get(&format!("{prefix}conv1.conv.weight"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional(&format!("{prefix}conv1.conv.bias")),
            1,
            dilation,
            1,
        )?;
        let hidden = self.snake_beta(
            &hidden,
            weights
                .get(&format!("{prefix}act2.alpha"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights
                .get(&format!("{prefix}act2.beta"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
        )?;
        let hidden = self.causal_conv1d_ncl(
            &hidden,
            weights
                .get(&format!("{prefix}conv2.conv.weight"))
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
            weights.optional(&format!("{prefix}conv2.conv.bias")),
            1,
            1,
            1,
        )?;
        Ok(residual + hidden)
    }

    fn snake_beta(
        &self,
        input: &MlxArray,
        alpha: &MlxArray,
        beta: &MlxArray,
    ) -> Result<MlxArray, MlxAudioError> {
        let alpha = mlx_rs::ops::exp(alpha)
            .and_then(|array| array.reshape(&[1, -1, 1]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let beta = mlx_rs::ops::exp(beta)
            .and_then(|array| array.reshape(&[1, -1, 1]))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let sin_term = mlx_rs::ops::sin(input.clone() * alpha)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        Ok(input.clone() + (sin_term.clone() * sin_term) / (beta + 1.0e-9f32))
    }

    fn decode_vq_group(
        &self,
        weights: &MlxWeightView<'_>,
        group_prefix: &str,
        start_quantizer: usize,
        end_quantizer: usize,
        codec_frames: &[Vec<u32>],
    ) -> Result<MlxArray, MlxAudioError> {
        let mut decoded_layers = Vec::new();
        for quantizer in start_quantizer..end_quantizer {
            let local_index = quantizer - start_quantizer;
            let layer_prefix = format!("{group_prefix}vq.layers.{local_index}.");
            decoded_layers.push(self.decode_vq_layer(
                weights,
                &layer_prefix,
                quantizer,
                codec_frames,
            )?);
        }
        let mut decoded = decoded_layers
            .into_iter()
            .reduce(|left, right| left + right)
            .ok_or_else(|| MlxAudioError::Inference("empty Qwen3 VQ group".to_string()))?;

        let output_projection_key = format!("{group_prefix}output_proj.weight");
        if let Some(projection) = weights.optional(&output_projection_key) {
            let projection = squeeze_last_singleton(projection)?;
            decoded = linear(
                &decoded
                    .transpose_axes(&[0, 2, 1])
                    .map_err(|error| MlxAudioError::Inference(error.to_string()))?,
                &projection,
                None,
            )
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        }
        Ok(decoded)
    }

    fn decode_vq_layer(
        &self,
        weights: &MlxWeightView<'_>,
        layer_prefix: &str,
        quantizer_index: usize,
        codec_frames: &[Vec<u32>],
    ) -> Result<MlxArray, MlxAudioError> {
        let embedding_sum = weights
            .get(&format!("{layer_prefix}_codebook.embedding_sum"))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let cluster_usage = weights
            .get(&format!("{layer_prefix}_codebook.cluster_usage"))
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        let usage = cluster_usage
            .reshape(&[-1, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
            + 1.0e-5f32;
        let codebook = embedding_sum.clone() / usage;
        let token_ids = codec_frames
            .iter()
            .map(|frame| i32::try_from(frame[quantizer_index]).unwrap_or(i32::MAX))
            .collect::<Vec<_>>();
        let mut decoded = embedding(&codebook, &token_ids)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?
            .reshape(&[1, token_ids.len() as i32, -1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;

        let projection_key = format!("{layer_prefix}project_out.weight");
        if let Some(projection) = weights.optional(&projection_key) {
            let projection = squeeze_last_singleton(projection)?;
            decoded = linear(&decoded, &projection, None)
                .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        }

        decoded
            .transpose_axes(&[0, 2, 1])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }
}

#[cfg(feature = "native-mlx")]
fn load_qwen_tokenizer(files: &SpeechModelFiles) -> Result<tokenizers::Tokenizer, MlxAudioError> {
    if let Some(tokenizer_path) = files.tokenizer_path.as_ref() {
        return tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()));
    }

    let vocab_path = files.model_dir.join("vocab.json");
    let merges_path = files.model_dir.join("merges.txt");
    let tokenizer_config_path = files.model_dir.join("tokenizer_config.json");
    if !vocab_path.is_file() || !merges_path.is_file() || !tokenizer_config_path.is_file() {
        return Err(MlxAudioError::ModelUnavailable(format!(
            "missing tokenizer.json or vocab.json/merges.txt/tokenizer_config.json in {}",
            files.model_dir.display()
        )));
    }

    let vocab_path_str = vocab_path.to_str().ok_or_else(|| {
        MlxAudioError::ModelUnavailable(format!(
            "non-utf8 tokenizer vocab path {}",
            vocab_path.display()
        ))
    })?;
    let merges_path_str = merges_path.to_str().ok_or_else(|| {
        MlxAudioError::ModelUnavailable(format!(
            "non-utf8 tokenizer merges path {}",
            merges_path.display()
        ))
    })?;
    let bpe = tokenizers::models::bpe::BPE::from_file(vocab_path_str, merges_path_str)
        .build()
        .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?;
    let mut tokenizer = tokenizers::Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::default().add_prefix_space(false),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer.with_post_processor(Some(
        tokenizers::processors::byte_level::ByteLevel::default().trim_offsets(false),
    ));

    let tokenizer_config = std::fs::read_to_string(&tokenizer_config_path)
        .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?;
    let tokenizer_config = serde_json::from_str::<serde_json::Value>(&tokenizer_config)
        .map_err(|error| MlxAudioError::ModelUnavailable(error.to_string()))?;
    let added = tokenizer_config
        .get("added_tokens_decoder")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            MlxAudioError::ModelUnavailable(format!(
                "tokenizer_config.json in {} has no added_tokens_decoder",
                files.model_dir.display()
            ))
        })?;
    let mut added = added.iter().collect::<Vec<_>>();
    added.sort_by_key(|(id, _)| id.parse::<u32>().unwrap_or(u32::MAX));
    for (_, token) in added {
        let Some(content) = token.get("content").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let special = token
            .get("special")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let mut added_token = tokenizers::AddedToken::from(content.to_string(), special);
        added_token = added_token
            .single_word(
                token
                    .get("single_word")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            )
            .lstrip(
                token
                    .get("lstrip")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            )
            .rstrip(
                token
                    .get("rstrip")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            )
            .normalized(
                token
                    .get("normalized")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            );
        if special {
            tokenizer.add_special_tokens(&[added_token]);
        } else {
            tokenizer.add_tokens(&[added_token]);
        }
    }

    Ok(tokenizer)
}

#[cfg(feature = "native-mlx")]
fn squeeze_last_singleton(array: &MlxArray) -> Result<MlxArray, MlxAudioError> {
    if array.shape().last().copied() == Some(1) {
        array
            .reshape(&[array.shape()[0], array.shape()[1]])
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
    } else {
        Ok(array.clone())
    }
}

#[cfg(feature = "native-mlx")]
fn count_indexed_layers(weights: &MlxWeightView<'_>, prefix: &str) -> usize {
    (0..)
        .take_while(|index| {
            weights
                .optional(&format!("{prefix}{index}.input_layernorm.weight"))
                .is_some()
        })
        .count()
}

#[cfg(feature = "native-mlx")]
fn count_decoder_blocks(weights: &MlxWeightView<'_>) -> usize {
    (1..)
        .take_while(|index| {
            weights
                .optional(&format!("decoder.decoder.{index}.block.0.alpha"))
                .is_some()
        })
        .count()
}

#[cfg(feature = "native-mlx")]
fn count_upsample_blocks(weights: &MlxWeightView<'_>) -> usize {
    (0..)
        .take_while(|index| {
            weights
                .optional(&format!("decoder.upsample.{index}.0.conv.weight"))
                .is_some()
        })
        .count()
}

#[cfg(feature = "native-mlx")]
fn inferred_transpose_stride(weight_iok: &MlxArray, divisor: i32) -> i32 {
    let divisor = divisor.max(1);
    let kernel = weight_iok.shape()[2];
    if kernel <= divisor {
        kernel.max(1)
    } else {
        (kernel / divisor).max(1)
    }
}

#[cfg(feature = "native-mlx")]
fn repeat_kv_heads(
    key: MlxArray,
    value: MlxArray,
    head_count: i32,
    kv_head_count: i32,
) -> Result<(MlxArray, MlxArray), MlxAudioError> {
    if kv_head_count == head_count {
        return Ok((key, value));
    }
    if kv_head_count <= 0 || head_count % kv_head_count != 0 {
        return Err(MlxAudioError::Inference(format!(
            "invalid grouped-query attention heads: heads={head_count}, kv_heads={kv_head_count}"
        )));
    }

    let repeat = head_count / kv_head_count;
    let key_shape = key.shape();
    let value_shape = value.shape();
    let key = key
        .reshape(&[key_shape[0], kv_head_count, 1, key_shape[2], key_shape[3]])
        .and_then(|array| {
            mlx_rs::ops::broadcast_to(
                &array,
                &[
                    key_shape[0],
                    kv_head_count,
                    repeat,
                    key_shape[2],
                    key_shape[3],
                ],
            )
        })
        .and_then(|array| array.reshape(&[key_shape[0], head_count, key_shape[2], key_shape[3]]))
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
    let value = value
        .reshape(&[
            value_shape[0],
            kv_head_count,
            1,
            value_shape[2],
            value_shape[3],
        ])
        .and_then(|array| {
            mlx_rs::ops::broadcast_to(
                &array,
                &[
                    value_shape[0],
                    kv_head_count,
                    repeat,
                    value_shape[2],
                    value_shape[3],
                ],
            )
        })
        .and_then(|array| {
            array.reshape(&[value_shape[0], head_count, value_shape[2], value_shape[3]])
        })
        .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
    Ok((key, value))
}

#[cfg(feature = "native-mlx")]
fn apply_repetition_penalty(logit: f32, seen: bool, penalty: f32) -> f32 {
    if !seen || (penalty - 1.0).abs() < f32::EPSILON {
        logit
    } else if logit < 0.0 {
        logit * penalty
    } else {
        logit / penalty
    }
}

#[cfg(feature = "native-mlx")]
struct XorShift64 {
    state: u64,
}

#[cfg(feature = "native-mlx")]
impl XorShift64 {
    fn from_entropy() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0x9e37_79b9_7f4a_7c15);
        let seed = std::env::var("VONA_QWEN3_TTS_SEED")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(nanos);
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x.max(1);
        x
    }

    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }
}

#[cfg(feature = "native-mlx")]
impl MlxSpeechModel for Qwen3TtsSpeechModel {
    fn transcribe(&self, _audio: &MlxArray, _sample_rate_hz: u32) -> Result<String, MlxAudioError> {
        Err(MlxAudioError::ModelUnavailable(
            "Qwen3 TTS speech model does not provide ASR".to_string(),
        ))
    }

    fn synthesize(&self, text: &str, sample_rate_hz: u32) -> Result<MlxArray, MlxAudioError> {
        if sample_rate_hz != DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ {
            return Err(MlxAudioError::InvalidInput(format!(
                "Qwen3 TTS currently expects {DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ} Hz output, got {sample_rate_hz} Hz"
            )));
        }

        let prompt = format!(
            "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
            text.trim()
        );
        let token_ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| MlxAudioError::InvalidInput(error.to_string()))?
            .get_ids()
            .iter()
            .map(|id| i32::try_from(*id).unwrap_or(i32::MAX))
            .collect::<Vec<_>>();
        if std::env::var_os("VONA_QWEN3_TTS_DEBUG_PROMPT").is_some() {
            eprintln!(
                "qwen3_tts prompt_tokens={} ids={:?}",
                token_ids.len(),
                token_ids
            );
        }
        let _len = i32::try_from(token_ids.len()).map_err(|_| {
            MlxAudioError::InvalidInput("tokenized prompt is too long for MLX shape".to_string())
        })?;
        let (generation_prefix, trailing_text) = self.build_generation_inputs(&token_ids)?;
        let codec_frames = self.generate_codec_frames(&generation_prefix, &trailing_text)?;
        if codec_frames.is_empty() {
            return Err(MlxAudioError::Inference(
                "Qwen3 TTS generated no codec frames before EOS".to_string(),
            ));
        }
        if self.vocoder_weight_count() == 0 {
            return Err(MlxAudioError::ModelUnavailable(format!(
                "missing Qwen3 TTS vocoder weights at {}/speech_tokenizer/model.safetensors",
                self.files.model_dir.display()
            )));
        }

        let latents = self.decode_vocoder_latents(&codec_frames)?;
        let audio = self.run_vocoder_frontend(&latents)?;
        let lower = mlx_rs::Array::from_slice(&[-1.0_f32], &[1]);
        let upper = mlx_rs::Array::from_slice(&[1.0_f32], &[1]);
        let audio = mlx_rs::ops::maximum(&audio, &lower)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))?;
        mlx_rs::ops::minimum(&audio, &upper)
            .map_err(|error| MlxAudioError::Inference(error.to_string()))
    }
}

#[cfg(not(feature = "native-mlx"))]
pub struct Qwen3TtsSpeechModel;

#[cfg(not(feature = "native-mlx"))]
impl Qwen3TtsSpeechModel {
    pub fn load(_speech_config: Qwen3TtsSpeechConfig) -> Result<Self, MlxAudioError> {
        Err(MlxAudioError::Runtime(
            "enable the native-mlx feature to use native Qwen3 TTS loading".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn speech_config_sets_defaults() {
        let config = super::Qwen3TtsSpeechConfig::new("/tmp/model");
        assert_eq!(config.language, super::DEFAULT_QWEN3_TTS_LANGUAGE);
        assert_eq!(config.speaker, super::DEFAULT_QWEN3_TTS_SPEAKER);
    }

    #[test]
    fn qwen3_config_sets_vocoder_defaults() {
        let config = super::Qwen3TtsConfig {
            model_type: None,
            vocab_size: None,
            hidden_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            tts_pad_token_id: None,
            tts_bos_token_id: None,
            tts_eos_token_id: None,
            talker_config: super::TalkerConfig::default(),
            vocoder_config: super::VocoderConfig::default(),
        };
        assert_eq!(config.vocoder_config.num_quantizers, 16);
        assert_eq!(config.vocoder_config.num_attention_heads, 16);
        assert_eq!(config.vocoder_config.transpose_stride_divisor, 2);
    }

    #[test]
    #[cfg(feature = "native-mlx")]
    fn infers_stride_for_pre_decoder_and_decoder_transposed_convs() {
        let pre_decoder = mlx_rs::Array::zeros::<f32>(&[1024, 1024, 2]).unwrap();
        let decoder = mlx_rs::Array::zeros::<f32>(&[1536, 768, 16]).unwrap();

        assert_eq!(super::inferred_transpose_stride(&pre_decoder, 2), 2);
        assert_eq!(super::inferred_transpose_stride(&decoder, 2), 8);
    }

    #[test]
    fn non_native_loader_reports_feature_gate() {
        #[cfg(not(feature = "native-mlx"))]
        {
            let result =
                super::Qwen3TtsSpeechModel::load(super::Qwen3TtsSpeechConfig::new("/tmp/model"));
            assert!(matches!(result, Err(vona_mlx::MlxAudioError::Runtime(_))));
        }
    }
}
