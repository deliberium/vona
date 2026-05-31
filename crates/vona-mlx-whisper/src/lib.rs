use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use vona_mlx::{LoadedMlxModel, MlxAudioError, MlxModelLoadRequest, MlxModelLoader};

#[cfg(feature = "native-mlx")]
use {
    std::{
        collections::{HashMap, HashSet},
        sync::{Arc, Mutex},
    },
    vona_mlx::{MlxArray, MlxModelKind, MlxSpeechModel},
    vona_mlx_speech::{
        MelSpectrogramConfig, MlxWeightView, SpeechModelFiles, conv1d_pytorch, embedding, gelu,
        layer_norm, linear, log_mel_spectrogram, scaled_dot_product_attention,
    },
};

pub const DEFAULT_WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhisperSpeechConfig {
    pub model_path: PathBuf,
    pub language: Option<String>,
    pub task: WhisperTask,
    pub max_decode_tokens: usize,
}

impl WhisperSpeechConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            language: None,
            task: WhisperTask::Transcribe,
            max_decode_tokens: 96,
        }
    }
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
            .map(|text| postprocess_whisper_transcript(&text))
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

#[cfg(any(feature = "native-mlx", test))]
fn postprocess_whisper_transcript(text: &str) -> String {
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
    collapse_repeated_word_runs(&collapsed)
}

#[cfg(any(feature = "native-mlx", test))]
fn collapse_repeated_word_runs(text: &str) -> String {
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
    output.join(" ")
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
}
