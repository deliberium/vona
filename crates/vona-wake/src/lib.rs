use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::Mutex;
use thiserror::Error;
use vona_core::transport::{AudioTransport, TransportError};
use vona_core::types::AudioInputFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeState {
    Dormant,
    Candidate,
    Awake,
    Suppressed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WakePolicy {
    pub wake_phrases: Vec<String>,
    pub candidate_threshold: f32,
    pub accept_threshold: f32,
    pub speaker_threshold: f32,
    pub preroll_ms: u32,
    pub cooldown_ms: u32,
    pub followup_window_ms: u32,
    pub require_near_field: bool,
    pub allow_barge_in: bool,
    pub require_speaker_verification: bool,
}

impl Default for WakePolicy {
    fn default() -> Self {
        Self {
            wake_phrases: vec!["vona".to_string(), "hey vona".to_string()],
            candidate_threshold: 0.55,
            accept_threshold: 0.72,
            speaker_threshold: 0.78,
            preroll_ms: 800,
            cooldown_ms: 1_200,
            followup_window_ms: 10_000,
            require_near_field: false,
            allow_barge_in: true,
            require_speaker_verification: false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WakeContext {
    pub playback_active: bool,
    pub privacy_mode: bool,
    pub followup_eligible: bool,
    pub near_field: Option<bool>,
    pub allowed_speakers: Vec<SpeakerProfile>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerProfile {
    pub speaker_id: String,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WakeTemplate {
    pub phrase: String,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub frame_embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    pub sample_count: usize,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerMatch {
    pub speaker_id: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WakeCandidate {
    pub phrase: Option<String>,
    pub confidence: f32,
    pub start_sequence: u64,
    pub end_sequence: u64,
    #[serde(default)]
    pub evidence: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeRejectReason {
    BelowThreshold,
    PrivacyMode,
    PlaybackSuppressed,
    NotNearField,
    SpeakerVerificationRequired,
    SpeakerRejected,
    Cooldown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeDecision {
    Idle,
    Candidate(WakeCandidate),
    Accepted {
        phrase: Option<String>,
        confidence: f32,
        speaker: Option<SpeakerMatch>,
        preroll: Vec<AudioInputFrame>,
        evidence: Value,
    },
    Rejected {
        reason: WakeRejectReason,
        confidence: f32,
    },
    Suppressed {
        reason: WakeRejectReason,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerVerification {
    pub best_match: Option<SpeakerMatch>,
    pub confidence: f32,
    pub accepted: bool,
}

impl SpeakerVerification {
    pub fn skipped() -> Self {
        Self {
            best_match: None,
            confidence: 0.0,
            accepted: true,
        }
    }
}

pub trait WakeDetector: Send {
    fn analyze(
        &mut self,
        frame: &AudioInputFrame,
        context: &WakeContext,
        recent_audio: &[AudioInputFrame],
    ) -> WakeDecision;
}

pub trait SpeakerVerifier: Send {
    fn verify(
        &mut self,
        recent_audio: &[AudioInputFrame],
        allowed_speakers: &[SpeakerProfile],
        policy: &WakePolicy,
    ) -> SpeakerVerification;
}

#[derive(Debug, Clone, Default)]
pub struct NoopSpeakerVerifier;

impl SpeakerVerifier for NoopSpeakerVerifier {
    fn verify(
        &mut self,
        _recent_audio: &[AudioInputFrame],
        _allowed_speakers: &[SpeakerProfile],
        _policy: &WakePolicy,
    ) -> SpeakerVerification {
        SpeakerVerification::skipped()
    }
}

#[derive(Debug, Clone)]
pub struct EnergyWakeDetector {
    pub phrase: Option<String>,
    pub average_abs_threshold: f32,
    pub peak_abs_threshold: f32,
}

impl Default for EnergyWakeDetector {
    fn default() -> Self {
        Self {
            phrase: Some("vona".to_string()),
            average_abs_threshold: 0.04,
            peak_abs_threshold: 0.18,
        }
    }
}

impl WakeDetector for EnergyWakeDetector {
    fn analyze(
        &mut self,
        frame: &AudioInputFrame,
        _context: &WakeContext,
        _recent_audio: &[AudioInputFrame],
    ) -> WakeDecision {
        let avg = average_abs(&frame.samples);
        let peak = peak_abs(&frame.samples);
        let avg_score = if self.average_abs_threshold <= f32::EPSILON {
            1.0
        } else {
            avg / self.average_abs_threshold
        };
        let peak_score = if self.peak_abs_threshold <= f32::EPSILON {
            1.0
        } else {
            peak / self.peak_abs_threshold
        };
        let confidence = ((avg_score + peak_score) * 0.5).clamp(0.0, 1.0);

        if confidence > 0.0 {
            WakeDecision::Candidate(WakeCandidate {
                phrase: self.phrase.clone(),
                confidence,
                start_sequence: frame.sequence,
                end_sequence: frame.sequence,
                evidence: json!({
                    "average_abs": avg,
                    "peak_abs": peak,
                    "detector": "energy",
                }),
            })
        } else {
            WakeDecision::Idle
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemplateWakeDetector {
    pub templates: Vec<WakeTemplate>,
    pub min_energy: f32,
}

impl TemplateWakeDetector {
    pub fn new(templates: Vec<WakeTemplate>) -> Self {
        Self {
            templates,
            min_energy: 0.01,
        }
    }

    pub fn enroll(phrase: impl Into<String>, frames: &[AudioInputFrame]) -> WakeTemplate {
        WakeTemplate {
            phrase: phrase.into(),
            embedding: simple_audio_embedding(frames),
            frame_embeddings: frames
                .iter()
                .filter(|frame| average_abs(&frame.samples) > 0.002)
                .map(|frame| simple_audio_embedding(std::slice::from_ref(frame)))
                .collect(),
            sample_count: frames.iter().map(|frame| frame.samples.len()).sum(),
            metadata: Value::Null,
        }
    }
}

impl WakeDetector for TemplateWakeDetector {
    fn analyze(
        &mut self,
        frame: &AudioInputFrame,
        _context: &WakeContext,
        recent_audio: &[AudioInputFrame],
    ) -> WakeDecision {
        if self.templates.is_empty() || average_abs(&frame.samples) < self.min_energy {
            return WakeDecision::Idle;
        }

        let recent_samples = recent_audio
            .iter()
            .map(|frame| frame.samples.len())
            .sum::<usize>();
        let Some((template, confidence)) = self
            .templates
            .iter()
            .filter(|template| {
                !template.embedding.is_empty()
                    && (template.sample_count == 0
                        || recent_samples >= template.sample_count.saturating_mul(4) / 5)
            })
            .map(|template| {
                let template_samples = template.sample_count.max(1);
                let start = recent_samples.saturating_sub(template_samples);
                let mut skipped = 0usize;
                let mut window = Vec::new();
                for frame in recent_audio {
                    let next = skipped + frame.samples.len();
                    if next > start {
                        let offset = start.saturating_sub(skipped);
                        window.push(AudioInputFrame {
                            sequence: frame.sequence + offset as u64,
                            sample_rate_hz: frame.sample_rate_hz,
                            channels: frame.channels,
                            samples: frame.samples[offset..].to_vec(),
                        });
                    }
                    skipped = next;
                }
                let live = simple_audio_embedding(&window);
                let sequence_score = template_sequence_score(&window, template);
                let duration_score = if template.sample_count == 0 {
                    1.0
                } else {
                    let ratio = recent_samples.min(template.sample_count) as f32
                        / recent_samples.max(template.sample_count) as f32;
                    ratio.clamp(0.0, 1.0)
                };
                (
                    template,
                    ((feature_similarity(&live, &template.embedding) * 0.35)
                        + (sequence_score * 0.65))
                        * duration_score,
                )
            })
            .max_by(|left, right| left.1.total_cmp(&right.1))
        else {
            return WakeDecision::Idle;
        };

        WakeDecision::Candidate(WakeCandidate {
            phrase: Some(template.phrase.clone()),
            confidence: confidence.clamp(0.0, 1.0),
            start_sequence: recent_audio
                .first()
                .map(|frame| frame.sequence)
                .unwrap_or(frame.sequence),
            end_sequence: frame.sequence,
            evidence: json!({
                "detector": "template",
                "template": template.phrase,
                "min_energy": self.min_energy,
            }),
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct EmbeddingSpeakerVerifier;

impl SpeakerVerifier for EmbeddingSpeakerVerifier {
    fn verify(
        &mut self,
        recent_audio: &[AudioInputFrame],
        allowed_speakers: &[SpeakerProfile],
        policy: &WakePolicy,
    ) -> SpeakerVerification {
        if allowed_speakers.is_empty() {
            return SpeakerVerification {
                best_match: None,
                confidence: 0.0,
                accepted: !policy.require_speaker_verification,
            };
        }

        let live = simple_audio_embedding(recent_audio);
        let best_match = allowed_speakers
            .iter()
            .filter(|profile| !profile.embedding.is_empty())
            .map(|profile| SpeakerMatch {
                speaker_id: profile.speaker_id.clone(),
                confidence: cosine_similarity(&live, &profile.embedding).clamp(0.0, 1.0),
            })
            .max_by(|left, right| left.confidence.total_cmp(&right.confidence));
        let confidence = best_match
            .as_ref()
            .map(|speaker_match| speaker_match.confidence)
            .unwrap_or(0.0);

        SpeakerVerification {
            best_match,
            confidence,
            accepted: confidence >= policy.speaker_threshold,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WakeMetrics {
    pub frames_seen: u64,
    pub candidates: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub suppressed: u64,
}

#[derive(Debug)]
pub struct WakeGate<D, V = NoopSpeakerVerifier> {
    detector: D,
    verifier: V,
    policy: WakePolicy,
    state: WakeState,
    ring: VecDeque<AudioInputFrame>,
    ring_samples: usize,
    accepted_until_sequence: Option<u64>,
    cooldown_until_sequence: Option<u64>,
    metrics: WakeMetrics,
}

impl<D> WakeGate<D, NoopSpeakerVerifier>
where
    D: WakeDetector,
{
    pub fn new(detector: D, policy: WakePolicy) -> Self {
        Self::with_verifier(detector, NoopSpeakerVerifier, policy)
    }
}

impl<D, V> WakeGate<D, V>
where
    D: WakeDetector,
    V: SpeakerVerifier,
{
    pub fn with_verifier(detector: D, verifier: V, policy: WakePolicy) -> Self {
        Self {
            detector,
            verifier,
            policy,
            state: WakeState::Dormant,
            ring: VecDeque::new(),
            ring_samples: 0,
            accepted_until_sequence: None,
            cooldown_until_sequence: None,
            metrics: WakeMetrics::default(),
        }
    }

    pub fn state(&self) -> WakeState {
        self.state
    }

    pub fn policy(&self) -> &WakePolicy {
        &self.policy
    }

    pub fn metrics(&self) -> &WakeMetrics {
        &self.metrics
    }

    pub fn sleep(&mut self) {
        self.state = WakeState::Dormant;
    }

    pub fn reset(&mut self) {
        self.state = WakeState::Dormant;
        self.ring.clear();
        self.ring_samples = 0;
        self.accepted_until_sequence = None;
        self.cooldown_until_sequence = None;
        self.metrics = WakeMetrics::default();
    }

    pub fn push_frame(&mut self, frame: AudioInputFrame, context: &WakeContext) -> WakeDecision {
        self.metrics.frames_seen += 1;
        self.push_ring(frame.clone());

        if context.privacy_mode {
            self.state = WakeState::Suppressed;
            self.metrics.suppressed += 1;
            return WakeDecision::Suppressed {
                reason: WakeRejectReason::PrivacyMode,
            };
        }

        if self.policy.require_near_field && context.near_field == Some(false) {
            self.state = WakeState::Suppressed;
            self.metrics.suppressed += 1;
            return WakeDecision::Suppressed {
                reason: WakeRejectReason::NotNearField,
            };
        }

        if context.playback_active && !self.policy.allow_barge_in {
            self.state = WakeState::Suppressed;
            self.metrics.suppressed += 1;
            return WakeDecision::Suppressed {
                reason: WakeRejectReason::PlaybackSuppressed,
            };
        }

        if self.state == WakeState::Awake {
            return WakeDecision::Accepted {
                phrase: None,
                confidence: 1.0,
                speaker: None,
                preroll: vec![frame],
                evidence: json!({ "mode": "already_awake" }),
            };
        }

        if let Some(until) = self.cooldown_until_sequence
            && frame.sequence <= until
        {
            self.metrics.suppressed += 1;
            return WakeDecision::Suppressed {
                reason: WakeRejectReason::Cooldown,
            };
        }

        if let Some(until) = self.accepted_until_sequence
            && context.followup_eligible
            && frame.sequence <= until
        {
            self.state = WakeState::Awake;
            return WakeDecision::Accepted {
                phrase: None,
                confidence: 1.0,
                speaker: None,
                preroll: vec![frame],
                evidence: json!({ "mode": "followup" }),
            };
        }

        let recent: Vec<_> = self.ring.iter().cloned().collect();
        let candidate = match self.detector.analyze(&frame, context, &recent) {
            WakeDecision::Candidate(candidate) => candidate,
            WakeDecision::Idle => {
                self.state = WakeState::Dormant;
                return WakeDecision::Idle;
            }
            WakeDecision::Accepted { .. } => {
                self.metrics.accepted += 1;
                self.state = WakeState::Awake;
                return self.accept(frame, None, 1.0, None, json!({ "detector": "accepted" }));
            }
            WakeDecision::Rejected { reason, confidence } => {
                self.metrics.rejected += 1;
                self.state = WakeState::Dormant;
                return WakeDecision::Rejected { reason, confidence };
            }
            WakeDecision::Suppressed { reason } => {
                self.metrics.suppressed += 1;
                self.state = WakeState::Suppressed;
                return WakeDecision::Suppressed { reason };
            }
        };

        self.metrics.candidates += 1;
        self.state = WakeState::Candidate;

        if candidate.confidence < self.policy.candidate_threshold {
            return WakeDecision::Candidate(candidate);
        }

        if candidate.confidence < self.policy.accept_threshold {
            self.metrics.rejected += 1;
            self.state = WakeState::Dormant;
            return WakeDecision::Rejected {
                reason: WakeRejectReason::BelowThreshold,
                confidence: candidate.confidence,
            };
        }

        let verification =
            if self.policy.require_speaker_verification || !context.allowed_speakers.is_empty() {
                self.verifier
                    .verify(&recent, &context.allowed_speakers, &self.policy)
            } else {
                SpeakerVerification::skipped()
            };

        if self.policy.require_speaker_verification && context.allowed_speakers.is_empty() {
            self.metrics.rejected += 1;
            self.state = WakeState::Dormant;
            return WakeDecision::Rejected {
                reason: WakeRejectReason::SpeakerVerificationRequired,
                confidence: candidate.confidence,
            };
        }

        if !verification.accepted {
            self.metrics.rejected += 1;
            self.state = WakeState::Dormant;
            return WakeDecision::Rejected {
                reason: WakeRejectReason::SpeakerRejected,
                confidence: candidate.confidence,
            };
        }

        self.metrics.accepted += 1;
        self.accept(
            frame,
            candidate.phrase,
            candidate.confidence,
            verification.best_match,
            candidate.evidence,
        )
    }

    fn accept(
        &mut self,
        frame: AudioInputFrame,
        phrase: Option<String>,
        confidence: f32,
        speaker: Option<SpeakerMatch>,
        evidence: Value,
    ) -> WakeDecision {
        self.state = WakeState::Awake;
        let samples_per_ms = frame.sample_rate_hz as u64 / 1_000;
        let cooldown_samples = samples_per_ms.saturating_mul(self.policy.cooldown_ms as u64);
        let followup_samples = samples_per_ms.saturating_mul(self.policy.followup_window_ms as u64);
        self.cooldown_until_sequence = Some(frame.sequence.saturating_add(cooldown_samples));
        self.accepted_until_sequence = Some(frame.sequence.saturating_add(followup_samples));
        WakeDecision::Accepted {
            phrase,
            confidence,
            speaker,
            preroll: self.preroll(),
            evidence,
        }
    }

    fn push_ring(&mut self, frame: AudioInputFrame) {
        self.ring_samples = self.ring_samples.saturating_add(frame.samples.len());
        self.ring.push_back(frame);
        let max_samples = self
            .ring
            .back()
            .map(|last| (last.sample_rate_hz as usize * self.policy.preroll_ms as usize) / 1_000)
            .unwrap_or(0);
        while max_samples > 0 && self.ring_samples > max_samples {
            if let Some(front) = self.ring.pop_front() {
                self.ring_samples = self.ring_samples.saturating_sub(front.samples.len());
            } else {
                break;
            }
        }
    }

    fn preroll(&self) -> Vec<AudioInputFrame> {
        self.ring.iter().cloned().collect()
    }
}

#[derive(Debug, Error)]
pub enum WakeTransportError {
    #[error("wake gate mutex poisoned")]
    GatePoisoned,
}

#[derive(Debug)]
pub struct WakeGatedTransport<T, D, V = NoopSpeakerVerifier> {
    inner: T,
    gate: Mutex<WakeGate<D, V>>,
    context: Mutex<WakeContext>,
    admitted: Mutex<VecDeque<AudioInputFrame>>,
    awake: Mutex<bool>,
}

impl<T, D> WakeGatedTransport<T, D, NoopSpeakerVerifier>
where
    D: WakeDetector,
{
    pub fn new(inner: T, gate: WakeGate<D, NoopSpeakerVerifier>, context: WakeContext) -> Self {
        Self::with_context(inner, gate, context)
    }
}

impl<T, D, V> WakeGatedTransport<T, D, V>
where
    D: WakeDetector,
    V: SpeakerVerifier,
{
    pub fn with_context(inner: T, gate: WakeGate<D, V>, context: WakeContext) -> Self {
        Self {
            inner,
            gate: Mutex::new(gate),
            context: Mutex::new(context),
            admitted: Mutex::new(VecDeque::new()),
            awake: Mutex::new(false),
        }
    }

    pub fn update_context(&self, context: WakeContext) -> Result<(), WakeTransportError> {
        *self
            .context
            .lock()
            .map_err(|_| WakeTransportError::GatePoisoned)? = context;
        Ok(())
    }

    pub fn sleep(&self) -> Result<(), WakeTransportError> {
        self.gate
            .lock()
            .map_err(|_| WakeTransportError::GatePoisoned)?
            .sleep();
        *self
            .awake
            .lock()
            .map_err(|_| WakeTransportError::GatePoisoned)? = false;
        self.admitted
            .lock()
            .map_err(|_| WakeTransportError::GatePoisoned)?
            .clear();
        Ok(())
    }
}

#[async_trait]
impl<T, D, V> AudioTransport for WakeGatedTransport<T, D, V>
where
    T: AudioTransport,
    D: WakeDetector + Send,
    V: SpeakerVerifier + Send,
{
    fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate_hz()
    }

    fn channels(&self) -> u16 {
        self.inner.channels()
    }

    async fn recv_frame(&self) -> Result<Option<AudioInputFrame>, TransportError> {
        if let Some(frame) = self
            .admitted
            .lock()
            .map_err(|_| TransportError::Receive("wake admitted queue poisoned".to_string()))?
            .pop_front()
        {
            return Ok(Some(frame));
        }

        loop {
            let Some(frame) = self.inner.recv_frame().await? else {
                return Ok(None);
            };
            if *self
                .awake
                .lock()
                .map_err(|_| TransportError::Receive("wake awake flag poisoned".to_string()))?
            {
                return Ok(Some(frame));
            }
            let context = self
                .context
                .lock()
                .map_err(|_| TransportError::Receive("wake context poisoned".to_string()))?
                .clone();
            let decision = self
                .gate
                .lock()
                .map_err(|_| TransportError::Receive("wake gate poisoned".to_string()))?
                .push_frame(frame.clone(), &context);

            match decision {
                WakeDecision::Accepted { preroll, .. } => {
                    *self.awake.lock().map_err(|_| {
                        TransportError::Receive("wake awake flag poisoned".to_string())
                    })? = true;
                    let mut admitted = self.admitted.lock().map_err(|_| {
                        TransportError::Receive("wake admitted queue poisoned".to_string())
                    })?;
                    admitted.extend(preroll);
                    if let Some(next) = admitted.pop_front() {
                        return Ok(Some(next));
                    }
                    return Ok(Some(frame));
                }
                WakeDecision::Idle
                | WakeDecision::Candidate(_)
                | WakeDecision::Rejected { .. }
                | WakeDecision::Suppressed { .. } => {}
            }
        }
    }

    async fn send_frame(
        &self,
        frame: vona_core::types::AudioOutputFrame,
    ) -> Result<(), TransportError> {
        self.inner.send_frame(frame).await
    }

    async fn clear_output(&self) -> Result<(), TransportError> {
        self.inner.clear_output().await
    }
}

pub fn simple_audio_embedding(frames: &[AudioInputFrame]) -> Vec<f32> {
    let raw_samples = frames
        .iter()
        .flat_map(|frame| frame.samples.iter().copied())
        .collect::<Vec<_>>();
    let mut count = 0usize;
    let mut sum_abs = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut peak = 0.0f32;

    for sample in raw_samples.iter().copied() {
        count += 1;
        let abs = sample.abs();
        sum_abs += abs;
        sum_sq += sample * sample;
        peak = peak.max(abs);
    }

    if count == 0 {
        return vec![0.0; 12];
    }

    let normalizer = peak.max(1e-6);
    let samples = raw_samples
        .iter()
        .map(|sample| *sample / normalizer)
        .collect::<Vec<_>>();
    let mut embedding = vec![
        sum_abs / count as f32,
        (sum_sq / count as f32).sqrt(),
        peak,
        samples.iter().filter(|sample| **sample >= 0.0).count() as f32 / count as f32,
        zero_crossing_rate(&samples),
    ];

    let segment_count = 7usize;
    let segment_len = count.div_ceil(segment_count);
    for segment in samples.chunks(segment_len.max(1)).take(segment_count) {
        let energy = segment.iter().map(|sample| sample.abs()).sum::<f32>() / segment.len() as f32;
        embedding.push(energy);
    }
    embedding.extend(spectral_fingerprint(&samples, 16_000));
    while embedding.len() < 24 {
        embedding.push(0.0);
    }

    embedding
}

fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
        .count();
    crossings as f32 / samples.len() as f32
}

fn spectral_fingerprint(samples: &[f32], sample_rate_hz: u32) -> Vec<f32> {
    const BINS: [f32; 12] = [
        120.0, 160.0, 220.0, 280.0, 360.0, 480.0, 620.0, 760.0, 900.0, 1_080.0, 1_260.0, 1_480.0,
    ];
    if samples.is_empty() {
        return vec![0.0; BINS.len()];
    }
    let normalizer = samples
        .iter()
        .map(|sample| sample.abs())
        .sum::<f32>()
        .max(1e-6);
    BINS.iter()
        .map(|freq| {
            let omega = std::f32::consts::TAU * *freq / sample_rate_hz as f32;
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for (index, sample) in samples.iter().copied().enumerate() {
                let phase = omega * index as f32;
                real += sample * phase.cos();
                imag -= sample * phase.sin();
            }
            (real.mul_add(real, imag * imag).sqrt() / normalizer).clamp(0.0, 1.0)
        })
        .collect()
}

fn template_sequence_score(window: &[AudioInputFrame], template: &WakeTemplate) -> f32 {
    if window.is_empty() || template.frame_embeddings.is_empty() {
        return feature_similarity(&simple_audio_embedding(window), &template.embedding).max(0.0);
    }

    let live = window
        .iter()
        .filter(|frame| average_abs(&frame.samples) > 0.002)
        .map(|frame| simple_audio_embedding(std::slice::from_ref(frame)))
        .collect::<Vec<_>>();
    if live.is_empty() {
        return 0.0;
    }
    let template_len = template.frame_embeddings.len();
    let live_len = live.len();
    let mut total = 0.0f32;
    for (index, expected) in template.frame_embeddings.iter().enumerate() {
        let live_index = if template_len <= 1 {
            0
        } else {
            index.saturating_mul(live_len.saturating_sub(1)) / template_len.saturating_sub(1)
        };
        total += feature_similarity(&live[live_index], expected).clamp(0.0, 1.0);
    }
    total / template_len as f32
}

fn feature_similarity(left: &[f32], right: &[f32]) -> f32 {
    let len = left.len().min(right.len());
    let start = if len > 6 { 3 } else { 0 };
    if len <= start {
        return 0.0;
    }
    let cosine = cosine_similarity(&left[start..len], &right[start..len]).clamp(0.0, 1.0);
    let mean_abs_delta = (start..len)
        .map(|index| (left[index] - right[index]).abs())
        .sum::<f32>()
        / (len - start) as f32;
    let distance_score = (1.0 - (mean_abs_delta * 4.0)).clamp(0.0, 1.0);
    (cosine * 0.45) + (distance_score * 0.55)
}

fn average_abs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(|sample| sample.abs()).sum::<f32>() / samples.len() as f32
}

fn peak_abs(samples: &[f32]) -> f32 {
    samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let len = left.len().min(right.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for index in 0..len {
        dot += left[index] * right[index];
        left_norm += left[index] * left[index];
        right_norm += right[index] * right[index];
    }

    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        return 0.0;
    }

    dot / (left_norm.sqrt() * right_norm.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use vona_core::types::AudioOutputFrame;

    fn frame(sequence: u64, samples: Vec<f32>) -> AudioInputFrame {
        AudioInputFrame {
            sequence,
            sample_rate_hz: 16_000,
            channels: 1,
            samples,
        }
    }

    #[test]
    fn gate_accepts_candidate_above_threshold_with_preroll() {
        let detector = EnergyWakeDetector {
            average_abs_threshold: 0.05,
            peak_abs_threshold: 0.2,
            ..EnergyWakeDetector::default()
        };
        let mut gate = WakeGate::new(
            detector,
            WakePolicy {
                candidate_threshold: 0.3,
                accept_threshold: 0.7,
                preroll_ms: 100,
                ..WakePolicy::default()
            },
        );

        assert_eq!(
            gate.push_frame(frame(0, vec![0.0; 160]), &WakeContext::default()),
            WakeDecision::Idle
        );

        let decision = gate.push_frame(frame(160, vec![0.25; 160]), &WakeContext::default());
        match decision {
            WakeDecision::Accepted {
                phrase,
                confidence,
                preroll,
                ..
            } => {
                assert_eq!(phrase.as_deref(), Some("vona"));
                assert!(confidence >= 0.7);
                assert_eq!(preroll.len(), 2);
            }
            other => panic!("expected accepted, got {other:?}"),
        }
        assert_eq!(gate.state(), WakeState::Awake);
        assert_eq!(gate.metrics().accepted, 1);
    }

    #[test]
    fn gate_suppresses_privacy_mode_before_detector() {
        let mut gate = WakeGate::new(EnergyWakeDetector::default(), WakePolicy::default());
        let decision = gate.push_frame(
            frame(0, vec![1.0; 160]),
            &WakeContext {
                privacy_mode: true,
                ..WakeContext::default()
            },
        );
        assert_eq!(
            decision,
            WakeDecision::Suppressed {
                reason: WakeRejectReason::PrivacyMode,
            }
        );
        assert_eq!(gate.metrics().suppressed, 1);
    }

    #[test]
    fn gate_can_sleep_then_enforces_cooldown_before_rearming() {
        let mut gate = WakeGate::new(
            EnergyWakeDetector {
                average_abs_threshold: 0.01,
                peak_abs_threshold: 0.01,
                ..EnergyWakeDetector::default()
            },
            WakePolicy {
                candidate_threshold: 0.1,
                accept_threshold: 0.2,
                cooldown_ms: 100,
                ..WakePolicy::default()
            },
        );

        assert!(matches!(
            gate.push_frame(frame(0, vec![0.5; 160]), &WakeContext::default()),
            WakeDecision::Accepted { .. }
        ));
        gate.sleep();
        assert_eq!(
            gate.push_frame(frame(320, vec![0.5; 160]), &WakeContext::default()),
            WakeDecision::Suppressed {
                reason: WakeRejectReason::Cooldown,
            }
        );
        assert!(matches!(
            gate.push_frame(frame(2_000, vec![0.5; 160]), &WakeContext::default()),
            WakeDecision::Accepted { .. }
        ));
    }

    #[test]
    fn followup_eligible_frame_is_admitted_after_sleep() {
        let mut gate = WakeGate::new(
            EnergyWakeDetector {
                average_abs_threshold: 0.01,
                peak_abs_threshold: 0.01,
                ..EnergyWakeDetector::default()
            },
            WakePolicy {
                candidate_threshold: 0.1,
                accept_threshold: 0.2,
                followup_window_ms: 500,
                cooldown_ms: 0,
                ..WakePolicy::default()
            },
        );

        assert!(matches!(
            gate.push_frame(frame(0, vec![0.5; 160]), &WakeContext::default()),
            WakeDecision::Accepted { .. }
        ));
        gate.sleep();
        assert_eq!(
            gate.push_frame(
                frame(320, vec![0.0; 160]),
                &WakeContext {
                    followup_eligible: true,
                    ..WakeContext::default()
                },
            ),
            WakeDecision::Accepted {
                phrase: None,
                confidence: 1.0,
                speaker: None,
                preroll: vec![frame(320, vec![0.0; 160])],
                evidence: json!({ "mode": "followup" }),
            }
        );
    }

    #[test]
    fn near_field_policy_suppresses_far_field_audio() {
        let mut gate = WakeGate::new(
            EnergyWakeDetector {
                average_abs_threshold: 0.01,
                peak_abs_threshold: 0.01,
                ..EnergyWakeDetector::default()
            },
            WakePolicy {
                require_near_field: true,
                ..WakePolicy::default()
            },
        );

        assert_eq!(
            gate.push_frame(
                frame(0, vec![0.5; 160]),
                &WakeContext {
                    near_field: Some(false),
                    ..WakeContext::default()
                },
            ),
            WakeDecision::Suppressed {
                reason: WakeRejectReason::NotNearField,
            }
        );
    }

    #[test]
    fn template_detector_accepts_enrolled_phrase_shape() {
        let enrollment = vec![
            frame(0, ramp_frame(0.02, 160)),
            frame(160, ramp_frame(0.12, 160)),
            frame(320, ramp_frame(0.20, 160)),
        ];
        let template = TemplateWakeDetector::enroll("hey vona", &enrollment);
        let mut gate = WakeGate::new(
            TemplateWakeDetector::new(vec![template]),
            WakePolicy {
                candidate_threshold: 0.4,
                accept_threshold: 0.8,
                cooldown_ms: 0,
                ..WakePolicy::default()
            },
        );

        for enrolled_frame in enrollment {
            let decision = gate.push_frame(enrolled_frame, &WakeContext::default());
            if let WakeDecision::Accepted { phrase, .. } = decision {
                assert_eq!(phrase.as_deref(), Some("hey vona"));
                return;
            }
        }
        panic!("template detector did not accept enrolled phrase shape");
    }

    #[test]
    fn voice_verification_rejects_unknown_required_speaker() {
        let detector = EnergyWakeDetector {
            average_abs_threshold: 0.01,
            peak_abs_threshold: 0.01,
            ..EnergyWakeDetector::default()
        };
        let mut gate = WakeGate::with_verifier(
            detector,
            EmbeddingSpeakerVerifier,
            WakePolicy {
                candidate_threshold: 0.1,
                accept_threshold: 0.2,
                require_speaker_verification: true,
                speaker_threshold: 0.99,
                ..WakePolicy::default()
            },
        );

        let decision = gate.push_frame(frame(0, vec![0.5; 160]), &WakeContext::default());
        assert_eq!(
            decision,
            WakeDecision::Rejected {
                reason: WakeRejectReason::SpeakerVerificationRequired,
                confidence: 1.0,
            }
        );
    }

    #[test]
    fn voice_verification_accepts_matching_profile() {
        let live_frame = frame(0, vec![0.5; 160]);
        let profile = SpeakerProfile {
            speaker_id: "owner".to_string(),
            embedding: simple_audio_embedding(std::slice::from_ref(&live_frame)),
            metadata: Value::Null,
        };
        let detector = EnergyWakeDetector {
            average_abs_threshold: 0.01,
            peak_abs_threshold: 0.01,
            ..EnergyWakeDetector::default()
        };
        let mut gate = WakeGate::with_verifier(
            detector,
            EmbeddingSpeakerVerifier,
            WakePolicy {
                candidate_threshold: 0.1,
                accept_threshold: 0.2,
                require_speaker_verification: true,
                speaker_threshold: 0.95,
                ..WakePolicy::default()
            },
        );

        let decision = gate.push_frame(
            live_frame,
            &WakeContext {
                allowed_speakers: vec![profile],
                ..WakeContext::default()
            },
        );
        match decision {
            WakeDecision::Accepted {
                speaker: Some(speaker),
                ..
            } => assert_eq!(speaker.speaker_id, "owner"),
            other => panic!("expected speaker accepted, got {other:?}"),
        }
    }

    #[test]
    fn voice_verification_rejects_non_matching_profile() {
        let live_frame = frame(0, vec![0.5; 160]);
        let profile = SpeakerProfile {
            speaker_id: "other-user".to_string(),
            embedding: vec![0.0, 1.0, 0.0, 1.0],
            metadata: Value::Null,
        };
        let detector = EnergyWakeDetector {
            average_abs_threshold: 0.01,
            peak_abs_threshold: 0.01,
            ..EnergyWakeDetector::default()
        };
        let mut gate = WakeGate::with_verifier(
            detector,
            EmbeddingSpeakerVerifier,
            WakePolicy {
                candidate_threshold: 0.1,
                accept_threshold: 0.2,
                require_speaker_verification: true,
                speaker_threshold: 0.95,
                ..WakePolicy::default()
            },
        );

        let decision = gate.push_frame(
            live_frame,
            &WakeContext {
                allowed_speakers: vec![profile],
                ..WakeContext::default()
            },
        );
        assert_eq!(
            decision,
            WakeDecision::Rejected {
                reason: WakeRejectReason::SpeakerRejected,
                confidence: 1.0,
            }
        );
    }

    #[test]
    fn playback_suppression_blocks_barge_in_when_policy_disallows_it() {
        let mut gate = WakeGate::new(
            EnergyWakeDetector {
                average_abs_threshold: 0.01,
                peak_abs_threshold: 0.01,
                ..EnergyWakeDetector::default()
            },
            WakePolicy {
                allow_barge_in: false,
                ..WakePolicy::default()
            },
        );

        let decision = gate.push_frame(
            frame(0, vec![0.5; 160]),
            &WakeContext {
                playback_active: true,
                ..WakeContext::default()
            },
        );
        assert_eq!(
            decision,
            WakeDecision::Suppressed {
                reason: WakeRejectReason::PlaybackSuppressed,
            }
        );
    }

    fn ramp_frame(peak: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let carrier = ((index as f32) / 8.0).sin();
                let envelope = index as f32 / len as f32;
                carrier * envelope * peak
            })
            .collect()
    }

    #[derive(Clone, Default)]
    struct TestTransport {
        incoming: Arc<Mutex<VecDeque<AudioInputFrame>>>,
        sent: Arc<Mutex<Vec<AudioOutputFrame>>>,
    }

    #[async_trait]
    impl AudioTransport for TestTransport {
        fn sample_rate_hz(&self) -> u32 {
            16_000
        }

        fn channels(&self) -> u16 {
            1
        }

        async fn recv_frame(&self) -> Result<Option<AudioInputFrame>, TransportError> {
            Ok(self.incoming.lock().expect("incoming").pop_front())
        }

        async fn send_frame(&self, frame: AudioOutputFrame) -> Result<(), TransportError> {
            self.sent.lock().expect("sent").push(frame);
            Ok(())
        }

        async fn clear_output(&self) -> Result<(), TransportError> {
            self.sent.lock().expect("sent").clear();
            Ok(())
        }
    }

    #[tokio::test]
    async fn wake_gated_transport_withholds_until_acceptance_then_releases_preroll() {
        let transport = TestTransport::default();
        transport
            .incoming
            .lock()
            .expect("incoming")
            .extend([frame(0, vec![0.0; 160]), frame(160, vec![0.4; 160])]);
        let gate = WakeGate::new(
            EnergyWakeDetector {
                average_abs_threshold: 0.05,
                peak_abs_threshold: 0.05,
                ..EnergyWakeDetector::default()
            },
            WakePolicy {
                candidate_threshold: 0.1,
                accept_threshold: 0.5,
                preroll_ms: 100,
                ..WakePolicy::default()
            },
        );
        let gated = WakeGatedTransport::new(transport, gate, WakeContext::default());

        let first = gated.recv_frame().await.expect("recv").expect("frame");
        let second = gated.recv_frame().await.expect("recv").expect("frame");
        assert_eq!(first.sequence, 0);
        assert_eq!(second.sequence, 160);
    }
}
