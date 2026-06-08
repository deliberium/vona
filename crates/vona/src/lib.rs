pub mod backend {
    pub use vona_core::backend::*;
}

pub mod backends {
    pub use vona_core::backends::*;
}

pub mod realtime {
    pub use vona_core::realtime::*;
}

pub mod runtime {
    pub use vona_core::runtime::*;
}

pub mod session {
    pub use vona_core::session::*;
}

pub mod skills {
    pub use vona_core::skills::*;
}

pub mod transport {
    pub use vona_core::transport::*;
}

pub mod types {
    pub use vona_core::types::*;
}

pub use vona_core::{
    AudioInputFrame, AudioOutputFrame, AudioProcessingError, AudioStreamingTranscriber,
    AudioSynthesisConfig, AudioSynthesizer, AudioTranscriber, AudioTransport, AuditEvent,
    AuditEventKind, AuditSink, BackendCapabilities, BackendError, BackendStep, CachedAudioClip,
    ControlEvent, ExternalContextEvent, FallbackReason, FillerStrategy, NoOpAuditSink,
    PassthroughStsBackend, PolicyAudioSynthesizer, PolicyTextGenerator, RealtimeLatencyMark,
    RealtimeLatencyStage, RealtimeTtsPolicy, RealtimeVoiceBackend, RealtimeVoiceCapabilities,
    RealtimeVoiceControl, RealtimeVoiceError, RealtimeVoiceInput, RealtimeVoiceModelFamily,
    RealtimeVoiceOutput, RealtimeVoiceSessionConfig, SessionCloseReason, SessionConfig,
    SessionError, SessionMetrics, SessionPolicy, SessionState, SessionSummary, Skill, SkillCall,
    SkillContext, SkillError, SkillExecutor, SkillOutput, SkillRegistry, SpeechStyleProfile,
    SpeechToSpeechBackend, StreamingTranscriptKind, StreamingTranscriptUpdate,
    StreamingTranscriptionConfig, StreamingTranscriptionSession, TextBackendId,
    TextBackendSelection, TextGenerationError, TextGenerationFrame, TextGenerationInput,
    TextGenerator, TextRoutingPolicy, TextRoutingRequest, TextTurnKind, TokenStream,
    TransportError, TtsPolicyRequest, TtsProviderId, TtsProviderSelection, TtsTurnKind,
    VonaRuntime, run_session,
};

#[cfg(feature = "azure-speech")]
pub use vona_azure_speech as azure_speech;
#[cfg(feature = "azure-speech")]
pub use vona_azure_speech::{
    AzureSpeechConfig, AzureSpeechMappingError, AzureSpeechMessage, AzureVoiceLiveConfig,
};

#[cfg(feature = "deepgram")]
pub use vona_deepgram as deepgram;
#[cfg(feature = "deepgram")]
pub use vona_deepgram::{
    DeepgramConfig, DeepgramMappingError, DeepgramSttConfig, DeepgramTtsConfig, DeepgramTtsMessage,
};

#[cfg(feature = "elevenlabs")]
pub use vona_elevenlabs as elevenlabs;
#[cfg(feature = "elevenlabs")]
pub use vona_elevenlabs::{
    ElevenLabsMappingError, ElevenLabsTtsConfig, ElevenLabsWebSocketMessage,
};

#[cfg(feature = "gemini-live")]
pub use vona_gemini_live as gemini_live;
#[cfg(feature = "gemini-live")]
pub use vona_gemini_live::{GeminiLiveClientMessage, GeminiLiveConfig, GeminiLiveMappingError};

#[cfg(feature = "model-provisioning")]
pub use vona_model_provisioning as model_provisioning;
#[cfg(feature = "model-provisioning")]
pub use vona_model_provisioning::{
    HttpModelProvisioner, LocalModelProvider, ModelArtifact, ModelCache, ModelManifest,
    PlannedArtifact, ProvisionPlan, ProvisioningError, distil_whisper_large_v3_manifest,
    kokoro_82m_onnx_realtime_manifest, mlx_speech_model_manifests, piper_low_power_tts_manifest,
    qwen3_tts_12hz_0_6b_base_bf16_manifest, realtime_tts_model_manifests,
};

#[cfg(feature = "kokoro-onnx")]
pub use vona_kokoro_onnx as kokoro_onnx;
#[cfg(feature = "kokoro-onnx")]
pub use vona_kokoro_onnx::{
    DEFAULT_KOKORO_SAMPLE_RATE_HZ, DEFAULT_KOKORO_VOICE, KokoroModelInfo, KokoroOnnxConfig,
    KokoroOnnxError, KokoroOnnxSynthesizer, KokoroVoice,
};

#[cfg(feature = "moonshine")]
pub use vona_moonshine as moonshine;
#[cfg(feature = "moonshine")]
pub use vona_moonshine::{
    DEFAULT_MOONSHINE_ARCH, DEFAULT_MOONSHINE_SAMPLE_RATE_HZ, MoonshineTranscriberConfig,
    NativeMoonshineTranscriber, ProtectedMoonshineTranscriber,
    TranscriptHotword as MoonshineTranscriptHotword,
    default_transcript_hotwords as default_moonshine_transcript_hotwords,
    parse_transcript_hotwords as parse_moonshine_transcript_hotwords,
    postprocess_transcript as postprocess_moonshine_transcript,
    transcript_hotwords_from_env as moonshine_transcript_hotwords_from_env,
};

#[cfg(feature = "moshi")]
pub use vona_moshi as moshi;
#[cfg(feature = "moshi")]
pub use vona_moshi::{MoshiBackend, MoshiConfig, MoshiSession};

#[cfg(feature = "mlx")]
pub use vona_mlx as mlx;
#[cfg(feature = "mlx")]
pub use vona_mlx::{
    DEFAULT_VLM_TEXT_MODEL_ID, LoadedMlxModel, MlxAudioConfig, MlxAudioEngine, MlxAudioError,
    MlxAudioSession, MlxModelKind, MlxModelLoadRequest, MlxModelLoader, MlxModelsLoader,
    MlxSpeechModel, MlxVlmTextConfig, MlxVlmTextEngine,
};

#[cfg(feature = "mlx-qwen3-tts")]
pub use vona_mlx_qwen3_tts as mlx_qwen3_tts;
#[cfg(feature = "mlx-qwen3-tts")]
pub use vona_mlx_qwen3_tts::{
    DEFAULT_QWEN3_TTS_SAMPLE_RATE_HZ, Qwen3TtsConfig, Qwen3TtsLoader, Qwen3TtsSpeechConfig,
    Qwen3TtsSpeechModel,
};

#[cfg(feature = "mlx-whisper")]
pub use vona_mlx_whisper as mlx_whisper;
#[cfg(feature = "mlx-whisper")]
pub use vona_mlx_whisper::{
    DEFAULT_WHISPER_SAMPLE_RATE_HZ, DEFAULT_WHISPER_WORKER_BIN, ProtectedWhisperConfig,
    ProtectedWhisperTranscriber, TranscriptHotword, WhisperConfig, WhisperLoader,
    WhisperSpeechConfig, WhisperSpeechModel, WhisperTask, default_transcript_hotwords,
    parse_transcript_hotwords, transcript_hotwords_from_env,
};

#[cfg(feature = "ollama")]
pub use vona_ollama as ollama;
#[cfg(feature = "ollama")]
pub use vona_ollama::{OllamaConfig, OllamaError, OllamaSession, OllamaTextEngine};

#[cfg(feature = "openai-realtime")]
pub use vona_openai_realtime as openai_realtime;
#[cfg(feature = "openai-realtime")]
pub use vona_openai_realtime::{
    OpenAiClientEvent, OpenAiRealtimeConfig, OpenAiRealtimeMappingError,
};

#[cfg(feature = "qwen")]
pub use vona_qwen as qwen;
#[cfg(feature = "qwen")]
pub use vona_qwen::{
    QwenAsrRealtimeConfig, QwenClientEvent, QwenRealtimeMappingError, QwenRealtimeServerOutput,
    QwenTtsRealtimeConfig, qwen_audio_append_event, qwen_audio_commit_event,
    qwen_input_text_append_event, qwen_input_text_commit_event, qwen_response_create_event,
    qwen_server_event_to_output, qwen_session_finish_event, qwen_session_update_event,
    qwen_session_update_event_for_asr, qwen_session_update_event_for_tts,
};

#[cfg(feature = "seamless")]
pub use vona_seamless as seamless;
#[cfg(feature = "seamless")]
pub use vona_seamless::{
    SeamlessM4tHttpBackend, SeamlessM4tHttpConfig, SeamlessM4tHttpSession, SeamlessM4tLocalBackend,
    SeamlessM4tLocalConfig, SeamlessM4tLocalSession, SeamlessM4tRemoteBackend,
    SeamlessM4tRemoteConfig, SeamlessM4tRemoteSession, SeamlessM4tRemoteStepRequest,
    SeamlessM4tRemoteStepResponse, SeamlessM4tRemoteTransport, SeamlessM4tRemoteTransportError,
};

#[cfg(feature = "test-harness")]
pub use vona_test_harness as test_harness;
#[cfg(feature = "test-harness")]
pub use vona_test_harness::{
    AllowAllPolicy, EchoSkillExecutor, MockBackend, ScriptedRealtimeBackend, ScriptedTransport,
};

#[cfg(feature = "transport-local")]
pub use vona_transport_local as transport_local;
#[cfg(feature = "transport-local")]
pub use vona_transport_local::{
    LocalIpcSeamlessM4tBackend, LocalIpcSeamlessM4tTransport, LocalIpcTransportConfig,
    LocalIpcTransportInitError,
};

#[cfg(feature = "wake")]
pub use vona_wake as wake;
#[cfg(feature = "wake")]
pub use vona_wake::{
    EmbeddingSpeakerVerifier, EnergyWakeDetector, NoopSpeakerVerifier, SpeakerMatch,
    SpeakerProfile, SpeakerVerification, SpeakerVerifier, TemplateWakeDetector, WakeCandidate,
    WakeContext, WakeDecision, WakeDetector, WakeGate, WakeGatedTransport, WakeMetrics, WakePolicy,
    WakeRejectReason, WakeState, WakeTemplate, WakeTransportError, simple_audio_embedding,
};

pub const CRATE_NAME: &str = "vona";
