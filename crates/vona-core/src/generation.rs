use crate::types::{AudioInputFrame, AudioOutputFrame};
use async_trait::async_trait;
use futures_core::Stream;
use futures_util::stream;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, pin::Pin, sync::Arc};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextGenerationInput {
    pub prompt: String,
    pub stream: bool,
}

impl TextGenerationInput {
    pub fn streaming(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            stream: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextGenerationFrame {
    pub text: String,
    pub final_fragment: bool,
}

#[derive(Debug, Error)]
pub enum TextGenerationError {
    #[error("text generation start failed: {0}")]
    Start(String),
    #[error("text generation stream failed: {0}")]
    Stream(String),
}

pub type TokenStream =
    Pin<Box<dyn Stream<Item = Result<TextGenerationFrame, TextGenerationError>> + Send + 'static>>;

pub trait TextGenerator: Send + Sync {
    fn generate_text(&self, input: TextGenerationInput) -> TokenStream;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextBackendId {
    OllamaPhi4Mini,
    MlxVlmGemma4,
    Custom { name: String },
}

impl TextBackendId {
    pub fn custom(name: impl Into<String>) -> Self {
        Self::Custom { name: name.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextTurnKind {
    ShortInteractive,
    LongOrComplex,
    ForcedLowLatency,
    ForcedReasoning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextRoutingRequest {
    pub prompt: String,
    pub stream: bool,
    pub prefer_low_latency: bool,
    pub prefer_reasoning_quality: bool,
    pub expect_long_answer: bool,
    pub allow_reasoning_backend: bool,
    pub override_backend: Option<TextBackendId>,
}

impl TextRoutingRequest {
    pub fn interactive(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            stream: true,
            prefer_low_latency: true,
            prefer_reasoning_quality: false,
            expect_long_answer: false,
            allow_reasoning_backend: true,
            override_backend: None,
        }
    }

    pub fn reasoning(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            stream: true,
            prefer_low_latency: false,
            prefer_reasoning_quality: true,
            expect_long_answer: true,
            allow_reasoning_backend: true,
            override_backend: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextBackendSelection {
    pub backend: TextBackendId,
    pub turn_kind: TextTurnKind,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextRoutingPolicy {
    pub low_latency_backend: TextBackendId,
    pub reasoning_backend: TextBackendId,
    pub max_low_latency_prompt_chars: usize,
    pub complex_request_terms: Vec<String>,
}

impl Default for TextRoutingPolicy {
    fn default() -> Self {
        Self {
            low_latency_backend: TextBackendId::OllamaPhi4Mini,
            reasoning_backend: TextBackendId::MlxVlmGemma4,
            max_low_latency_prompt_chars: 420,
            complex_request_terms: [
                "analyze",
                "architecture",
                "benchmark",
                "compare",
                "debug",
                "design",
                "diagnose",
                "explain",
                "implement",
                "optimize",
                "plan",
                "reason",
                "tradeoff",
                "why",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        }
    }
}

impl TextRoutingPolicy {
    pub fn with_low_latency_backend(mut self, backend: TextBackendId) -> Self {
        self.low_latency_backend = backend;
        self
    }

    pub fn with_reasoning_backend(mut self, backend: TextBackendId) -> Self {
        self.reasoning_backend = backend;
        self
    }

    pub fn select_backend(&self, request: &TextRoutingRequest) -> TextBackendSelection {
        if let Some(backend) = &request.override_backend {
            return TextBackendSelection {
                backend: backend.clone(),
                turn_kind: match backend == &self.reasoning_backend {
                    true => TextTurnKind::ForcedReasoning,
                    false => TextTurnKind::ForcedLowLatency,
                },
                reason: "downstream application explicitly selected the text backend".to_string(),
            };
        }

        if !request.allow_reasoning_backend {
            return TextBackendSelection {
                backend: self.low_latency_backend.clone(),
                turn_kind: TextTurnKind::ForcedLowLatency,
                reason: "reasoning backend is disabled for this turn".to_string(),
            };
        }

        let prompt = normalize_policy_text(&request.prompt);
        let route_to_reasoning = request.prefer_reasoning_quality
            || request.expect_long_answer
            || prompt.chars().count() > self.max_low_latency_prompt_chars
            || is_complex_text_request(&prompt, &self.complex_request_terms);

        if route_to_reasoning {
            return TextBackendSelection {
                backend: self.reasoning_backend.clone(),
                turn_kind: TextTurnKind::LongOrComplex,
                reason: "long, complex, or quality-weighted turn routes to the reasoning backend"
                    .to_string(),
            };
        }

        TextBackendSelection {
            backend: self.low_latency_backend.clone(),
            turn_kind: TextTurnKind::ShortInteractive,
            reason: "short interactive turn routes to the low-latency backend".to_string(),
        }
    }
}

#[derive(Clone, Default)]
pub struct PolicyTextGenerator {
    policy: TextRoutingPolicy,
    low_latency: Option<Arc<dyn TextGenerator>>,
    reasoning: Option<Arc<dyn TextGenerator>>,
    custom: HashMap<String, Arc<dyn TextGenerator>>,
}

impl PolicyTextGenerator {
    pub fn new(policy: TextRoutingPolicy) -> Self {
        Self {
            policy,
            low_latency: None,
            reasoning: None,
            custom: HashMap::new(),
        }
    }

    pub fn with_low_latency(mut self, generator: Arc<dyn TextGenerator>) -> Self {
        self.low_latency = Some(generator);
        self
    }

    pub fn with_reasoning(mut self, generator: Arc<dyn TextGenerator>) -> Self {
        self.reasoning = Some(generator);
        self
    }

    pub fn with_custom_backend(
        mut self,
        name: impl Into<String>,
        generator: Arc<dyn TextGenerator>,
    ) -> Self {
        self.custom.insert(name.into(), generator);
        self
    }

    pub fn policy(&self) -> &TextRoutingPolicy {
        &self.policy
    }

    pub fn select_backend(&self, request: &TextRoutingRequest) -> TextBackendSelection {
        self.policy.select_backend(request)
    }

    pub fn generate_with_policy(&self, request: TextRoutingRequest) -> TokenStream {
        let selection = self.select_backend(&request);
        let input = TextGenerationInput {
            prompt: request.prompt,
            stream: request.stream,
        };
        match self.generator_for_backend(&selection.backend) {
            Some(generator) => generator.generate_text(input),
            None => Box::pin(stream::once(async move {
                Err(TextGenerationError::Start(format!(
                    "selected text backend {:?} is not configured: {}",
                    selection.backend, selection.reason
                )))
            })),
        }
    }

    fn generator_for_backend(&self, backend: &TextBackendId) -> Option<Arc<dyn TextGenerator>> {
        match backend {
            TextBackendId::OllamaPhi4Mini => self.low_latency.clone(),
            TextBackendId::MlxVlmGemma4 => self.reasoning.clone(),
            TextBackendId::Custom { name } => self.custom.get(name).cloned(),
        }
    }
}

impl TextGenerator for PolicyTextGenerator {
    fn generate_text(&self, input: TextGenerationInput) -> TokenStream {
        self.generate_with_policy(TextRoutingRequest {
            prompt: input.prompt,
            stream: input.stream,
            ..TextRoutingRequest::interactive("")
        })
    }
}

fn normalize_policy_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_complex_text_request(text: &str, terms: &[String]) -> bool {
    let lower = text.to_ascii_lowercase();
    let question_count = lower.chars().filter(|ch| *ch == '?').count();
    question_count > 1
        || terms.iter().any(|term| {
            lower
                .split(|ch: char| !ch.is_ascii_alphanumeric())
                .any(|word| word == term)
        })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioSynthesisConfig {
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
}

#[derive(Debug, Error)]
pub enum AudioProcessingError {
    #[error("audio runtime failed: {0}")]
    Runtime(String),
    #[error("audio model is unavailable: {0}")]
    ModelUnavailable(String),
    #[error("audio input is invalid: {0}")]
    InvalidInput(String),
    #[error("audio inference failed: {0}")]
    Inference(String),
}

#[async_trait]
pub trait AudioTranscriber: Send + Sync {
    async fn transcribe_audio(
        &self,
        input: AudioInputFrame,
    ) -> Result<String, AudioProcessingError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamingTranscriptionConfig {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub step_ms: u32,
    pub min_buffer_ms: u32,
    pub max_buffer_ms: u32,
    pub stability_passes: u32,
}

impl StreamingTranscriptionConfig {
    pub fn new(sample_rate_hz: u32, channels: u16) -> Self {
        Self {
            sample_rate_hz,
            channels,
            step_ms: 600,
            min_buffer_ms: 1_200,
            max_buffer_ms: 30_000,
            stability_passes: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamingTranscriptKind {
    Partial,
    Final,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamingTranscriptUpdate {
    pub kind: StreamingTranscriptKind,
    pub text: String,
    pub stability_passes: u32,
    pub total_audio_ms: u64,
}

#[async_trait]
pub trait StreamingTranscriptionSession: Send {
    async fn push_audio(
        &mut self,
        input: AudioInputFrame,
    ) -> Result<Option<StreamingTranscriptUpdate>, AudioProcessingError>;

    async fn finish(&mut self) -> Result<Option<StreamingTranscriptUpdate>, AudioProcessingError>;
}

#[async_trait]
pub trait AudioStreamingTranscriber: Send + Sync {
    async fn start_streaming_transcription(
        &self,
        config: StreamingTranscriptionConfig,
    ) -> Result<Box<dyn StreamingTranscriptionSession>, AudioProcessingError>;
}

#[async_trait]
pub trait AudioSynthesizer: Send + Sync {
    async fn synthesize_audio(
        &self,
        text: String,
        config: AudioSynthesisConfig,
    ) -> Result<AudioOutputFrame, AudioProcessingError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsProviderId {
    CachedAck,
    KokoroRealtime,
    PiperLowPower,
    Qwen3Premium,
    CustomRealtime { name: String },
}

impl TtsProviderId {
    pub fn custom_realtime(name: impl Into<String>) -> Self {
        Self::CustomRealtime { name: name.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsTurnKind {
    Acknowledgement,
    ShortRealtimeReply,
    LongOrPremiumReply,
    LowPowerReply,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsPolicyRequest {
    pub text: String,
    pub prefer_low_latency: bool,
    pub prefer_premium_quality: bool,
    pub low_power_mode: bool,
    pub allow_cached_ack: bool,
}

impl TtsPolicyRequest {
    pub fn realtime(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            prefer_low_latency: true,
            prefer_premium_quality: false,
            low_power_mode: false,
            allow_cached_ack: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsProviderSelection {
    pub provider: TtsProviderId,
    pub turn_kind: TtsTurnKind,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealtimeTtsPolicy {
    pub cached_ack_enabled: bool,
    pub max_cached_ack_chars: usize,
    pub max_realtime_chars: usize,
    pub realtime_provider: TtsProviderId,
    pub premium_provider: TtsProviderId,
    pub low_power_provider: TtsProviderId,
    pub fallback_provider: TtsProviderId,
}

impl Default for RealtimeTtsPolicy {
    fn default() -> Self {
        Self {
            cached_ack_enabled: true,
            max_cached_ack_chars: 28,
            max_realtime_chars: 260,
            realtime_provider: TtsProviderId::KokoroRealtime,
            premium_provider: TtsProviderId::Qwen3Premium,
            low_power_provider: TtsProviderId::PiperLowPower,
            fallback_provider: TtsProviderId::PiperLowPower,
        }
    }
}

impl RealtimeTtsPolicy {
    pub fn with_realtime_provider(mut self, provider: TtsProviderId) -> Self {
        self.realtime_provider = provider;
        self
    }

    pub fn with_fallback_provider(mut self, provider: TtsProviderId) -> Self {
        self.fallback_provider = provider;
        self
    }

    pub fn select_provider(&self, request: &TtsPolicyRequest) -> TtsProviderSelection {
        let normalized = normalize_tts_text(&request.text);
        if request.low_power_mode {
            return TtsProviderSelection {
                provider: self.low_power_provider.clone(),
                turn_kind: TtsTurnKind::LowPowerReply,
                reason: "low power mode routes speech to the lightweight provider".to_string(),
            };
        }
        if self.cached_ack_enabled
            && request.allow_cached_ack
            && normalized.len() <= self.max_cached_ack_chars
            && is_acknowledgement_text(&normalized)
        {
            return TtsProviderSelection {
                provider: TtsProviderId::CachedAck,
                turn_kind: TtsTurnKind::Acknowledgement,
                reason: "short acknowledgement can be served from cached audio".to_string(),
            };
        }
        if request.prefer_premium_quality || normalized.len() > self.max_realtime_chars {
            return TtsProviderSelection {
                provider: self.premium_provider.clone(),
                turn_kind: TtsTurnKind::LongOrPremiumReply,
                reason: "long or premium reply routes to the high-quality provider".to_string(),
            };
        }
        TtsProviderSelection {
            provider: self.realtime_provider.clone(),
            turn_kind: TtsTurnKind::ShortRealtimeReply,
            reason: "short realtime reply routes to the low-latency provider".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedAudioClip {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl CachedAudioClip {
    pub fn into_output(self, sequence: u64, is_filler: bool) -> AudioOutputFrame {
        AudioOutputFrame {
            sequence,
            sample_rate_hz: self.sample_rate_hz,
            channels: self.channels,
            samples: self.samples,
            is_filler,
        }
    }
}

#[derive(Clone, Default)]
pub struct PolicyAudioSynthesizer {
    policy: RealtimeTtsPolicy,
    cached_ack: Option<CachedAudioClip>,
    kokoro_realtime: Option<Arc<dyn AudioSynthesizer>>,
    piper_low_power: Option<Arc<dyn AudioSynthesizer>>,
    qwen3_premium: Option<Arc<dyn AudioSynthesizer>>,
    custom_realtime: HashMap<String, Arc<dyn AudioSynthesizer>>,
}

impl PolicyAudioSynthesizer {
    pub fn new(policy: RealtimeTtsPolicy) -> Self {
        Self {
            policy,
            cached_ack: None,
            kokoro_realtime: None,
            piper_low_power: None,
            qwen3_premium: None,
            custom_realtime: HashMap::new(),
        }
    }

    pub fn with_cached_ack(mut self, clip: CachedAudioClip) -> Self {
        self.cached_ack = Some(clip);
        self
    }

    pub fn with_kokoro_realtime(mut self, synthesizer: Arc<dyn AudioSynthesizer>) -> Self {
        self.kokoro_realtime = Some(synthesizer);
        self
    }

    pub fn with_piper_low_power(mut self, synthesizer: Arc<dyn AudioSynthesizer>) -> Self {
        self.piper_low_power = Some(synthesizer);
        self
    }

    pub fn with_qwen3_premium(mut self, synthesizer: Arc<dyn AudioSynthesizer>) -> Self {
        self.qwen3_premium = Some(synthesizer);
        self
    }

    pub fn with_custom_realtime_provider(
        mut self,
        name: impl Into<String>,
        synthesizer: Arc<dyn AudioSynthesizer>,
    ) -> Self {
        self.custom_realtime.insert(name.into(), synthesizer);
        self
    }

    pub fn policy(&self) -> &RealtimeTtsPolicy {
        &self.policy
    }

    pub fn select_provider(&self, request: &TtsPolicyRequest) -> TtsProviderSelection {
        self.policy.select_provider(request)
    }

    pub async fn synthesize_with_policy(
        &self,
        request: TtsPolicyRequest,
        config: AudioSynthesisConfig,
    ) -> Result<AudioOutputFrame, AudioProcessingError> {
        let selection = self.select_provider(&request);
        match self
            .synthesize_with_provider(selection.provider.clone(), request.text.clone(), config.clone())
            .await
        {
            Ok(frame) => Ok(frame),
            Err(error) if selection.provider != self.policy.fallback_provider => self
                .synthesize_with_provider(self.policy.fallback_provider.clone(), request.text, config)
                .await
                .map_err(|fallback_error| {
                    AudioProcessingError::Runtime(format!(
                        "selected TTS provider {:?} failed ({error}); fallback {:?} failed ({fallback_error})",
                        selection.provider, self.policy.fallback_provider
                    ))
                }),
            Err(error) => Err(error),
        }
    }

    async fn synthesize_with_provider(
        &self,
        provider: TtsProviderId,
        text: String,
        config: AudioSynthesisConfig,
    ) -> Result<AudioOutputFrame, AudioProcessingError> {
        match provider {
            TtsProviderId::CachedAck => {
                let clip = self.cached_ack.clone().ok_or_else(|| {
                    AudioProcessingError::ModelUnavailable(
                        "cached acknowledgement audio is not configured".to_string(),
                    )
                })?;
                if clip.sample_rate_hz != config.sample_rate_hz || clip.channels != config.channels
                {
                    return Err(AudioProcessingError::InvalidInput(format!(
                        "cached acknowledgement clip is {} Hz/{} ch, requested {} Hz/{} ch",
                        clip.sample_rate_hz, clip.channels, config.sample_rate_hz, config.channels
                    )));
                }
                Ok(clip.into_output(config.sequence, true))
            }
            TtsProviderId::KokoroRealtime => {
                synthesize_from_slot(&self.kokoro_realtime, provider, text, config).await
            }
            TtsProviderId::PiperLowPower => {
                synthesize_from_slot(&self.piper_low_power, provider, text, config).await
            }
            TtsProviderId::Qwen3Premium => {
                synthesize_from_slot(&self.qwen3_premium, provider, text, config).await
            }
            TtsProviderId::CustomRealtime { name } => {
                let synthesizer = self.custom_realtime.get(&name).ok_or_else(|| {
                    AudioProcessingError::ModelUnavailable(format!(
                        "custom realtime TTS provider {name:?} is not configured"
                    ))
                })?;
                synthesizer.synthesize_audio(text, config).await
            }
        }
    }
}

#[async_trait]
impl AudioSynthesizer for PolicyAudioSynthesizer {
    async fn synthesize_audio(
        &self,
        text: String,
        config: AudioSynthesisConfig,
    ) -> Result<AudioOutputFrame, AudioProcessingError> {
        self.synthesize_with_policy(TtsPolicyRequest::realtime(text), config)
            .await
    }
}

async fn synthesize_from_slot(
    slot: &Option<Arc<dyn AudioSynthesizer>>,
    provider: TtsProviderId,
    text: String,
    config: AudioSynthesisConfig,
) -> Result<AudioOutputFrame, AudioProcessingError> {
    let synthesizer = slot.as_ref().ok_or_else(|| {
        AudioProcessingError::ModelUnavailable(format!(
            "{provider:?} synthesizer is not configured"
        ))
    })?;
    synthesizer.synthesize_audio(text, config).await
}

fn normalize_tts_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_acknowledgement_text(text: &str) -> bool {
    matches!(
        text.trim()
            .trim_matches(|ch: char| ch.is_ascii_punctuation())
            .to_ascii_lowercase()
            .as_str(),
        "ok" | "okay" | "sure" | "one moment" | "checking" | "i'm checking" | "let me check"
    )
}

#[cfg(test)]
mod tests {
    use super::{
        AudioProcessingError, AudioSynthesisConfig, AudioSynthesizer, CachedAudioClip,
        PolicyAudioSynthesizer, PolicyTextGenerator, RealtimeTtsPolicy, TextBackendId,
        TextGenerationFrame, TextGenerationInput, TextGenerator, TextRoutingPolicy,
        TextRoutingRequest, TextTurnKind, TokenStream, TtsPolicyRequest, TtsProviderId,
        TtsTurnKind,
    };
    use crate::types::AudioOutputFrame;
    use async_trait::async_trait;
    use futures_util::{StreamExt, stream};
    use std::sync::Arc;

    #[test]
    fn realtime_policy_routes_acknowledgement_to_cached_audio() {
        let policy = RealtimeTtsPolicy::default();
        let selection = policy.select_provider(&TtsPolicyRequest::realtime("Okay."));

        assert_eq!(selection.provider, TtsProviderId::CachedAck);
        assert_eq!(selection.turn_kind, TtsTurnKind::Acknowledgement);
    }

    #[test]
    fn realtime_policy_routes_short_reply_to_kokoro() {
        let policy = RealtimeTtsPolicy::default();
        let selection =
            policy.select_provider(&TtsPolicyRequest::realtime("Here is the short answer."));

        assert_eq!(selection.provider, TtsProviderId::KokoroRealtime);
        assert_eq!(selection.turn_kind, TtsTurnKind::ShortRealtimeReply);
    }

    #[test]
    fn realtime_policy_can_route_short_reply_to_custom_provider() {
        let policy = RealtimeTtsPolicy::default()
            .with_realtime_provider(TtsProviderId::custom_realtime("client-streaming-tts"));
        let selection =
            policy.select_provider(&TtsPolicyRequest::realtime("Here is the short answer."));

        assert_eq!(
            selection.provider,
            TtsProviderId::custom_realtime("client-streaming-tts")
        );
        assert_eq!(selection.turn_kind, TtsTurnKind::ShortRealtimeReply);
    }

    #[test]
    fn realtime_policy_routes_long_or_premium_reply_to_qwen3() {
        let policy = RealtimeTtsPolicy::default();
        let mut request = TtsPolicyRequest::realtime("Detailed answer.");
        request.prefer_premium_quality = true;
        let selection = policy.select_provider(&request);

        assert_eq!(selection.provider, TtsProviderId::Qwen3Premium);
        assert_eq!(selection.turn_kind, TtsTurnKind::LongOrPremiumReply);
    }

    #[test]
    fn realtime_policy_routes_low_power_to_piper() {
        let policy = RealtimeTtsPolicy::default();
        let mut request = TtsPolicyRequest::realtime("Any reply.");
        request.low_power_mode = true;
        let selection = policy.select_provider(&request);

        assert_eq!(selection.provider, TtsProviderId::PiperLowPower);
        assert_eq!(selection.turn_kind, TtsTurnKind::LowPowerReply);
    }

    #[tokio::test]
    async fn policy_synthesizer_serves_cached_ack_clip() {
        let router = PolicyAudioSynthesizer::new(RealtimeTtsPolicy::default()).with_cached_ack(
            CachedAudioClip {
                sample_rate_hz: 24_000,
                channels: 1,
                samples: vec![0.1, 0.2],
            },
        );

        let frame = router
            .synthesize_audio(
                "Okay.".to_string(),
                AudioSynthesisConfig {
                    sequence: 9,
                    sample_rate_hz: 24_000,
                    channels: 1,
                },
            )
            .await
            .unwrap();

        assert_eq!(frame.sequence, 9);
        assert!(frame.is_filler);
        assert_eq!(frame.samples, vec![0.1, 0.2]);
    }

    #[tokio::test]
    async fn policy_synthesizer_falls_back_when_selected_provider_missing() {
        let router = PolicyAudioSynthesizer::new(RealtimeTtsPolicy::default())
            .with_piper_low_power(Arc::new(MockSynthesizer {
                label: "piper",
                sample: 0.25,
            }));

        let frame = router
            .synthesize_audio(
                "This short reply would normally route to Kokoro.".to_string(),
                AudioSynthesisConfig {
                    sequence: 3,
                    sample_rate_hz: 24_000,
                    channels: 1,
                },
            )
            .await
            .unwrap();

        assert_eq!(frame.sequence, 3);
        assert_eq!(frame.samples, vec![0.25]);
    }

    #[tokio::test]
    async fn policy_synthesizer_routes_to_configured_custom_realtime_provider() {
        let policy = RealtimeTtsPolicy::default()
            .with_realtime_provider(TtsProviderId::custom_realtime("client-streaming-tts"));
        let router = PolicyAudioSynthesizer::new(policy).with_custom_realtime_provider(
            "client-streaming-tts",
            Arc::new(MockSynthesizer {
                label: "client-streaming-tts",
                sample: 0.75,
            }),
        );

        let frame = router
            .synthesize_audio(
                "This should route to the downstream realtime synthesizer.".to_string(),
                AudioSynthesisConfig {
                    sequence: 4,
                    sample_rate_hz: 24_000,
                    channels: 1,
                },
            )
            .await
            .unwrap();

        assert_eq!(frame.sequence, 4);
        assert_eq!(frame.samples, vec![0.75]);
    }

    #[test]
    fn text_policy_routes_short_turn_to_phi4() {
        let selection = TextRoutingPolicy::default().select_backend(
            &TextRoutingRequest::interactive("Tell me the current status."),
        );

        assert_eq!(selection.backend, TextBackendId::OllamaPhi4Mini);
        assert_eq!(selection.turn_kind, TextTurnKind::ShortInteractive);
    }

    #[test]
    fn text_policy_routes_reasoning_turn_to_gemma() {
        let selection =
            TextRoutingPolicy::default().select_backend(&TextRoutingRequest::reasoning(
                "Compare the tradeoffs and explain why one design is better.",
            ));

        assert_eq!(selection.backend, TextBackendId::MlxVlmGemma4);
        assert_eq!(selection.turn_kind, TextTurnKind::LongOrComplex);
    }

    #[test]
    fn text_policy_honors_downstream_override() {
        let mut request = TextRoutingRequest::reasoning("Analyze this carefully.");
        request.override_backend = Some(TextBackendId::custom("customer-text-runtime"));

        let selection = TextRoutingPolicy::default().select_backend(&request);

        assert_eq!(
            selection.backend,
            TextBackendId::custom("customer-text-runtime")
        );
        assert_eq!(selection.turn_kind, TextTurnKind::ForcedLowLatency);
    }

    #[tokio::test]
    async fn policy_text_generator_routes_to_configured_reasoning_engine() {
        let router = PolicyTextGenerator::new(TextRoutingPolicy::default())
            .with_low_latency(Arc::new(MockTextGenerator { label: "phi4" }))
            .with_reasoning(Arc::new(MockTextGenerator { label: "gemma" }));

        let mut stream = router.generate_with_policy(TextRoutingRequest::reasoning(
            "Explain the architecture tradeoffs.",
        ));
        let frame = stream.next().await.unwrap().unwrap();

        assert_eq!(frame.text, "gemma");
        assert!(frame.final_fragment);
    }

    struct MockSynthesizer {
        label: &'static str,
        sample: f32,
    }

    #[async_trait]
    impl AudioSynthesizer for MockSynthesizer {
        async fn synthesize_audio(
            &self,
            text: String,
            config: AudioSynthesisConfig,
        ) -> Result<AudioOutputFrame, AudioProcessingError> {
            assert!(!text.is_empty(), "{} received empty text", self.label);
            Ok(AudioOutputFrame {
                sequence: config.sequence,
                sample_rate_hz: config.sample_rate_hz,
                channels: config.channels,
                samples: vec![self.sample],
                is_filler: false,
            })
        }
    }

    struct MockTextGenerator {
        label: &'static str,
    }

    impl TextGenerator for MockTextGenerator {
        fn generate_text(&self, _input: TextGenerationInput) -> TokenStream {
            let frame = TextGenerationFrame {
                text: self.label.to_string(),
                final_fragment: true,
            };
            Box::pin(stream::once(async move { Ok(frame) }))
        }
    }
}
