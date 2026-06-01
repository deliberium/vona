use ring::digest;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use vona_core::types::AudioInputFrame;
use vona_wake::{
    EmbeddingSpeakerVerifier, SpeakerProfile, TemplateWakeDetector, WakeContext, WakeDecision,
    WakeGate, WakePolicy, simple_audio_embedding,
};

#[derive(Debug, Clone, Deserialize)]
struct EvalManifest {
    #[serde(default)]
    corpus: Option<CorpusMetadata>,
    templates: Vec<TemplateInput>,
    cases: Vec<CaseInput>,
    #[serde(default)]
    policy: ManifestPolicy,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CorpusMetadata {
    id: String,
    version: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    created_by: Option<String>,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    collection_ledger_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TemplateInput {
    phrase: String,
    path: PathBuf,
    #[serde(default)]
    speaker_id: Option<String>,
    #[serde(default)]
    source_type: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    split: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct CaseInput {
    id: String,
    path: PathBuf,
    should_wake: bool,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    expected_phrase: Option<String>,
    #[serde(default)]
    wake_start_ms: Option<u64>,
    #[serde(default)]
    speaker_id: Option<String>,
    #[serde(default)]
    environment: Option<String>,
    #[serde(default)]
    distance: Option<String>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    source_type: Option<String>,
    #[serde(default)]
    split: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ManifestPolicy {
    candidate_threshold: Option<f32>,
    accept_threshold: Option<f32>,
    speaker_threshold: Option<f32>,
    min_energy: Option<f32>,
    preroll_ms: Option<u32>,
    rearm_ms: Option<u64>,
    require_speaker_verification: Option<bool>,
}

#[derive(Debug, Serialize)]
struct WakeEvent {
    at_ms: u64,
    confidence: f32,
    phrase: Option<String>,
    speaker_id: Option<String>,
    speaker_confidence: Option<f32>,
}

#[derive(Debug, Serialize)]
struct CaseResult {
    id: String,
    path: String,
    text: Option<String>,
    speaker_id: Option<String>,
    environment: Option<String>,
    distance: Option<String>,
    device: Option<String>,
    session_id: Option<String>,
    category: Option<String>,
    source_type: Option<String>,
    split: Option<String>,
    should_wake: bool,
    woke: bool,
    confidence: Option<f32>,
    phrase: Option<String>,
    expected_phrase: Option<String>,
    wake_start_ms: Option<u64>,
    phrase_matched: bool,
    frames: usize,
    duration_ms: u64,
    first_wake_ms: Option<u64>,
    detection_latency_ms: Option<i64>,
    wake_event_count: usize,
    wake_events: Vec<WakeEvent>,
}

#[derive(Debug, Serialize)]
struct EvalReport {
    manifest_path: String,
    manifest_sha256: String,
    corpus: Option<CorpusMetadata>,
    templates: usize,
    cases: usize,
    positives: usize,
    negatives: usize,
    true_positives: usize,
    false_positives: usize,
    true_negatives: usize,
    false_negatives: usize,
    repeated_positive_wake_events: usize,
    phrase_mismatches: usize,
    precision: f32,
    recall: f32,
    positive_audio_seconds: f32,
    negative_audio_seconds: f32,
    false_wakes_per_hour: f32,
    mean_first_wake_ms: Option<f32>,
    max_first_wake_ms: Option<u64>,
    mean_detection_latency_ms: Option<f32>,
    max_detection_latency_ms: Option<i64>,
    confidence_intervals: ConfidenceIntervals,
    coverage: CoverageReport,
    subgroups: SubgroupReport,
    leakage: LeakageReport,
    threshold_sweep: Vec<ThresholdSweepPoint>,
    cases_detail: Vec<CaseResult>,
}

#[derive(Debug, Serialize)]
struct ConfidenceIntervals {
    confidence_level: f32,
    precision_lower_bound: Option<f32>,
    recall_lower_bound: Option<f32>,
    false_wakes_per_hour_upper_bound: Option<f32>,
}

#[derive(Debug, Serialize)]
struct ThresholdSweepPoint {
    candidate_threshold: f32,
    accept_threshold: f32,
    true_positives: usize,
    false_positives: usize,
    true_negatives: usize,
    false_negatives: usize,
    repeated_positive_wake_events: usize,
    phrase_mismatches: usize,
    precision: f32,
    recall: f32,
    false_wakes_per_hour: f32,
    max_detection_latency_ms: Option<i64>,
    splits: Vec<GroupMetrics>,
}

#[derive(Debug, Default, Serialize)]
struct CoverageReport {
    speakers: usize,
    environments: usize,
    distances: usize,
    devices: usize,
    sessions: usize,
    categories: usize,
    speaker_ids: Vec<String>,
    environment_ids: Vec<String>,
    distance_ids: Vec<String>,
    device_ids: Vec<String>,
    session_ids: Vec<String>,
    category_ids: Vec<String>,
}

#[derive(Debug, Default, Serialize)]
struct SubgroupReport {
    speakers: Vec<GroupMetrics>,
    environments: Vec<GroupMetrics>,
    distances: Vec<GroupMetrics>,
    devices: Vec<GroupMetrics>,
    sessions: Vec<GroupMetrics>,
    categories: Vec<GroupMetrics>,
    splits: Vec<GroupMetrics>,
}

#[derive(Debug, Serialize)]
struct GroupMetrics {
    id: String,
    cases: usize,
    positives: usize,
    negatives: usize,
    true_positives: usize,
    false_positives: usize,
    true_negatives: usize,
    false_negatives: usize,
    repeated_positive_wake_events: usize,
    phrase_mismatches: usize,
    precision: f32,
    recall: f32,
    negative_audio_seconds: f32,
    false_wakes_per_hour: f32,
    max_detection_latency_ms: Option<i64>,
}

#[derive(Debug, Default, Serialize)]
struct LeakageReport {
    template_case_path_overlaps: usize,
    template_case_audio_overlaps: usize,
    template_case_fingerprint_overlaps: usize,
    duplicate_case_paths: usize,
    duplicate_case_audio: usize,
    duplicate_case_fingerprints: usize,
    template_case_paths: Vec<String>,
    template_case_audio_hashes: Vec<String>,
    template_case_audio_fingerprints: Vec<String>,
    duplicate_case_path_groups: Vec<DuplicatePathGroup>,
    duplicate_case_audio_groups: Vec<DuplicateAudioGroup>,
    duplicate_case_fingerprint_groups: Vec<DuplicateFingerprintGroup>,
}

#[derive(Debug, Serialize)]
struct DuplicatePathGroup {
    path: String,
    case_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DuplicateAudioGroup {
    hash: String,
    case_ids: Vec<String>,
    paths: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DuplicateFingerprintGroup {
    fingerprint: String,
    case_ids: Vec<String>,
    paths: Vec<String>,
}

fn main() {
    let manifest_path = env::var("VONA_WAKE_REAL_EVAL_MANIFEST")
        .map(PathBuf::from)
        .or_else(|_| env::args().nth(1).map(PathBuf::from).ok_or(()))
        .expect("pass a manifest path as argv[1] or VONA_WAKE_REAL_EVAL_MANIFEST");
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let manifest = load_manifest(&manifest_path);
    assert!(
        !manifest.templates.is_empty(),
        "real voice eval manifest must include at least one enrollment template"
    );
    assert!(
        !manifest.cases.is_empty(),
        "real voice eval manifest must include at least one labeled case"
    );

    let enrolled = manifest
        .templates
        .iter()
        .map(|template| {
            let path = resolve_path(manifest_dir, &template.path);
            let frames = wav_to_frames(&path, 320);
            assert!(
                !frames.is_empty(),
                "template '{}' produced no frames from {}",
                template.phrase,
                path.display()
            );
            let wake_template = TemplateWakeDetector::enroll(template.phrase.clone(), &frames);
            let speaker_profile = template
                .speaker_id
                .as_ref()
                .map(|speaker_id| SpeakerProfile {
                    speaker_id: speaker_id.clone(),
                    embedding: simple_audio_embedding(&frames),
                    metadata: json!({
                        "template_phrase": template.phrase,
                        "template_path": template.path,
                        "source_type": template.source_type,
                        "session_id": template.session_id,
                        "split": template.split,
                    }),
                });
            (wake_template, speaker_profile)
        })
        .collect::<Vec<_>>();
    let templates = enrolled
        .iter()
        .map(|(template, _)| template.clone())
        .collect::<Vec<_>>();
    let speaker_profiles = enrolled
        .iter()
        .filter_map(|(_, profile)| profile.clone())
        .collect::<Vec<_>>();

    let mut results = Vec::new();
    for case in &manifest.cases {
        let path = resolve_path(manifest_dir, &case.path);
        let frames = wav_to_frames(&path, 320);
        assert!(
            !frames.is_empty(),
            "case '{}' produced no frames from {}",
            case.id,
            path.display()
        );
        let wake_events = run_case(
            &frames,
            &manifest.policy,
            templates.clone(),
            speaker_profiles.clone(),
        );
        results.push(case_result(
            case,
            path,
            frames.len(),
            audio_duration_ms(&frames),
            wake_events,
        ));
    }

    let template_paths = manifest
        .templates
        .iter()
        .map(|template| resolve_path(manifest_dir, &template.path))
        .collect::<Vec<_>>();
    let report = build_report(
        &manifest_path,
        manifest.corpus.clone(),
        manifest.templates.len(),
        &template_paths,
        build_threshold_sweep(
            manifest_dir,
            &manifest,
            templates.clone(),
            speaker_profiles.clone(),
        ),
        results,
    );
    let json = serde_json::to_string_pretty(&report).expect("serialize real voice eval report");
    if let Ok(report_path) = env::var("VONA_WAKE_REAL_EVAL_REPORT") {
        fs::write(report_path, &json).expect("write real voice eval report");
    }
    println!("{json}");

    if env::var("VONA_WAKE_REAL_EVAL_ENFORCE").ok().as_deref() == Some("1") {
        enforce_report(&report);
    }
}

fn load_manifest(path: &Path) -> EvalManifest {
    let manifest = fs::read_to_string(path).expect("read real voice eval manifest");
    serde_json::from_str(&manifest).expect("parse real voice eval manifest")
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn run_case(
    frames: &[AudioInputFrame],
    policy: &ManifestPolicy,
    templates: Vec<vona_wake::WakeTemplate>,
    speaker_profiles: Vec<SpeakerProfile>,
) -> Vec<WakeEvent> {
    let require_speaker_verification = policy.require_speaker_verification.unwrap_or(false);
    let mut gate = WakeGate::with_verifier(
        TemplateWakeDetector {
            templates,
            min_energy: policy.min_energy.unwrap_or(0.0005),
        },
        EmbeddingSpeakerVerifier,
        WakePolicy {
            candidate_threshold: policy.candidate_threshold.unwrap_or(0.88),
            accept_threshold: policy.accept_threshold.unwrap_or(0.92),
            speaker_threshold: policy.speaker_threshold.unwrap_or(0.78),
            preroll_ms: policy.preroll_ms.unwrap_or(1_200),
            cooldown_ms: 0,
            require_speaker_verification,
            ..WakePolicy::default()
        },
    );
    let context = WakeContext {
        allowed_speakers: speaker_profiles,
        ..WakeContext::default()
    };
    let rearm_ms = policy.rearm_ms.unwrap_or(1_200);
    let mut ignore_until_ms = None;
    let mut events = Vec::new();
    for frame in frames.iter().cloned() {
        let frame_start_ms = frame.sequence.saturating_mul(1_000) / frame.sample_rate_hz as u64;
        if ignore_until_ms.is_some_and(|until| frame_start_ms < until) {
            continue;
        }
        if let WakeDecision::Accepted {
            confidence,
            phrase,
            speaker,
            ..
        } = gate.push_frame(frame, &context)
        {
            events.push(WakeEvent {
                at_ms: frame_start_ms,
                confidence,
                phrase,
                speaker_id: speaker.as_ref().map(|speaker| speaker.speaker_id.clone()),
                speaker_confidence: speaker.map(|speaker| speaker.confidence),
            });
            gate.reset();
            ignore_until_ms = Some(frame_start_ms.saturating_add(rearm_ms));
        }
    }
    events
}

fn build_threshold_sweep(
    manifest_dir: &Path,
    manifest: &EvalManifest,
    templates: Vec<vona_wake::WakeTemplate>,
    speaker_profiles: Vec<SpeakerProfile>,
) -> Vec<ThresholdSweepPoint> {
    threshold_sweep_values(&manifest.policy)
        .into_iter()
        .map(|accept_threshold| {
            let candidate_threshold = round_threshold(
                (accept_threshold - threshold_gap(&manifest.policy)).clamp(0.0, 1.0),
            );
            let policy = ManifestPolicy {
                candidate_threshold: Some(candidate_threshold),
                accept_threshold: Some(accept_threshold),
                speaker_threshold: manifest.policy.speaker_threshold,
                min_energy: manifest.policy.min_energy,
                preroll_ms: manifest.policy.preroll_ms,
                rearm_ms: manifest.policy.rearm_ms,
                require_speaker_verification: manifest.policy.require_speaker_verification,
            };
            let results = manifest
                .cases
                .iter()
                .map(|case| {
                    let path = resolve_path(manifest_dir, &case.path);
                    let frames = wav_to_frames(&path, 320);
                    let wake_events = run_case(
                        &frames,
                        &policy,
                        templates.clone(),
                        speaker_profiles.clone(),
                    );
                    case_result(
                        case,
                        path,
                        frames.len(),
                        audio_duration_ms(&frames),
                        wake_events,
                    )
                })
                .collect::<Vec<_>>();
            summarize_threshold_point(candidate_threshold, accept_threshold, &results)
        })
        .collect()
}

fn threshold_sweep_values(policy: &ManifestPolicy) -> Vec<f32> {
    let configured = policy.accept_threshold.unwrap_or(0.92);
    let mut values = [
        configured - 0.06,
        configured - 0.03,
        configured,
        configured + 0.03,
        configured + 0.06,
    ]
    .into_iter()
    .map(round_threshold)
    .filter(|value| (0.0..=1.0).contains(value))
    .collect::<Vec<_>>();
    values.sort_by(|left, right| left.total_cmp(right));
    values.dedup_by(|left, right| (*left - *right).abs() < f32::EPSILON);
    values
}

fn round_threshold(value: f32) -> f32 {
    (value * 100.0).round() / 100.0
}

fn threshold_gap(policy: &ManifestPolicy) -> f32 {
    policy
        .accept_threshold
        .zip(policy.candidate_threshold)
        .map(|(accept, candidate)| (accept - candidate).max(0.0))
        .unwrap_or(0.04)
}

fn case_result(
    case: &CaseInput,
    path: PathBuf,
    frames: usize,
    duration_ms: u64,
    wake_events: Vec<WakeEvent>,
) -> CaseResult {
    let first_event = wake_events.first();
    let woke = first_event.is_some();
    let confidence = first_event.map(|event| event.confidence);
    let phrase = first_event.and_then(|event| event.phrase.clone());
    let first_wake_ms = first_event.map(|event| event.at_ms);
    let phrase_matched = case
        .expected_phrase
        .as_ref()
        .is_none_or(|expected| phrase.as_deref() == Some(expected.as_str()));

    CaseResult {
        id: case.id.clone(),
        path: path.display().to_string(),
        text: case.text.clone(),
        speaker_id: case.speaker_id.clone(),
        environment: case.environment.clone(),
        distance: case.distance.clone(),
        device: case.device.clone(),
        session_id: case.session_id.clone(),
        category: case.category.clone(),
        source_type: case.source_type.clone(),
        split: case.split.clone(),
        should_wake: case.should_wake,
        woke,
        confidence,
        phrase,
        expected_phrase: case.expected_phrase.clone(),
        wake_start_ms: case.wake_start_ms,
        phrase_matched,
        frames,
        duration_ms,
        first_wake_ms,
        detection_latency_ms: first_wake_ms
            .map(|wake_ms| wake_ms as i64 - case.wake_start_ms.unwrap_or_default() as i64),
        wake_event_count: wake_events.len(),
        wake_events,
    }
}

fn build_report(
    manifest_path: &Path,
    corpus: Option<CorpusMetadata>,
    template_count: usize,
    template_paths: &[PathBuf],
    threshold_sweep: Vec<ThresholdSweepPoint>,
    cases_detail: Vec<CaseResult>,
) -> EvalReport {
    let positives = cases_detail.iter().filter(|case| case.should_wake).count();
    let negatives = cases_detail.len().saturating_sub(positives);
    let true_positives = cases_detail
        .iter()
        .filter(|case| case.should_wake && case.woke)
        .count();
    let false_positives = cases_detail
        .iter()
        .filter(|case| !case.should_wake)
        .map(|case| case.wake_event_count)
        .sum::<usize>();
    let true_negatives = cases_detail
        .iter()
        .filter(|case| !case.should_wake && !case.woke)
        .count();
    let false_negatives = cases_detail
        .iter()
        .filter(|case| case.should_wake && !case.woke)
        .count();
    let phrase_mismatches = cases_detail
        .iter()
        .filter(|case| case.should_wake && case.woke && !case.phrase_matched)
        .count();
    let repeated_positive_wake_events = cases_detail
        .iter()
        .filter(|case| case.should_wake)
        .map(|case| case.wake_event_count.saturating_sub(1))
        .sum::<usize>();
    let precision = if true_positives + false_positives == 0 {
        0.0
    } else {
        true_positives as f32 / (true_positives + false_positives) as f32
    };
    let recall = if positives == 0 {
        0.0
    } else {
        true_positives as f32 / positives as f32
    };
    let positive_audio_seconds = cases_detail
        .iter()
        .filter(|case| case.should_wake)
        .map(|case| case.duration_ms as f32 / 1_000.0)
        .sum::<f32>();
    let negative_audio_seconds = cases_detail
        .iter()
        .filter(|case| !case.should_wake)
        .map(|case| case.duration_ms as f32 / 1_000.0)
        .sum::<f32>();
    let false_wakes_per_hour = if negative_audio_seconds <= f32::EPSILON {
        0.0
    } else {
        false_positives as f32 / (negative_audio_seconds / 3_600.0)
    };
    let wake_latencies = cases_detail
        .iter()
        .filter(|case| case.should_wake)
        .filter_map(|case| case.first_wake_ms)
        .collect::<Vec<_>>();
    let mean_first_wake_ms = (!wake_latencies.is_empty())
        .then(|| wake_latencies.iter().sum::<u64>() as f32 / wake_latencies.len() as f32);
    let max_first_wake_ms = wake_latencies.iter().copied().max();
    let detection_latencies = cases_detail
        .iter()
        .filter(|case| case.should_wake)
        .filter_map(|case| case.detection_latency_ms)
        .collect::<Vec<_>>();
    let mean_detection_latency_ms = (!detection_latencies.is_empty())
        .then(|| detection_latencies.iter().sum::<i64>() as f32 / detection_latencies.len() as f32);
    let max_detection_latency_ms = detection_latencies.iter().copied().max();
    let coverage = build_coverage(&cases_detail);
    let subgroups = build_subgroups(&cases_detail);
    let leakage = build_leakage(template_paths, &cases_detail);
    let confidence_intervals = build_confidence_intervals(
        true_positives,
        false_positives,
        positives,
        negative_audio_seconds,
    );

    EvalReport {
        manifest_path: manifest_path.display().to_string(),
        manifest_sha256: sha256_file(manifest_path),
        corpus,
        templates: template_count,
        cases: cases_detail.len(),
        positives,
        negatives,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        repeated_positive_wake_events,
        phrase_mismatches,
        precision,
        recall,
        positive_audio_seconds,
        negative_audio_seconds,
        false_wakes_per_hour,
        mean_first_wake_ms,
        max_first_wake_ms,
        mean_detection_latency_ms,
        max_detection_latency_ms,
        confidence_intervals,
        coverage,
        subgroups,
        leakage,
        threshold_sweep,
        cases_detail,
    }
}

fn sha256_file(path: &Path) -> String {
    let bytes = fs::read(path).expect("read manifest for sha256");
    let digest = digest::digest(&digest::SHA256, &bytes);
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn build_confidence_intervals(
    true_positives: usize,
    false_positives: usize,
    positives: usize,
    negative_audio_seconds: f32,
) -> ConfidenceIntervals {
    ConfidenceIntervals {
        confidence_level: 0.95,
        precision_lower_bound: wilson_lower_bound(true_positives, true_positives + false_positives),
        recall_lower_bound: wilson_lower_bound(true_positives, positives),
        false_wakes_per_hour_upper_bound: poisson_rate_upper_95(
            false_positives,
            negative_audio_seconds,
        ),
    }
}

fn wilson_lower_bound(successes: usize, trials: usize) -> Option<f32> {
    if trials == 0 {
        return None;
    }
    let z = 1.959_963_984_540_054_f64;
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let center = p + z2 / (2.0 * n);
    let margin = z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt();
    let denominator = 1.0 + z2 / n;
    Some(((center - margin) / denominator).max(0.0) as f32)
}

fn poisson_rate_upper_95(events: usize, exposure_seconds: f32) -> Option<f32> {
    if exposure_seconds <= f32::EPSILON {
        return None;
    }
    let exposure_hours = exposure_seconds as f64 / 3_600.0;
    let upper_events = poisson_count_upper_95(events);
    Some((upper_events / exposure_hours) as f32)
}

fn poisson_count_upper_95(events: usize) -> f64 {
    if events == 0 {
        return -0.05_f64.ln();
    }
    let z = 1.644_853_626_951_472_2_f64;
    let degrees_of_freedom = 2.0 * (events as f64 + 1.0);
    let chi_square_upper = degrees_of_freedom
        * (1.0 - 2.0 / (9.0 * degrees_of_freedom) + z * (2.0 / (9.0 * degrees_of_freedom)).sqrt())
            .powi(3);
    0.5 * chi_square_upper
}

fn summarize_threshold_point(
    candidate_threshold: f32,
    accept_threshold: f32,
    cases_detail: &[CaseResult],
) -> ThresholdSweepPoint {
    let positives = cases_detail.iter().filter(|case| case.should_wake).count();
    let true_positives = cases_detail
        .iter()
        .filter(|case| case.should_wake && case.woke)
        .count();
    let false_positives = cases_detail
        .iter()
        .filter(|case| !case.should_wake)
        .map(|case| case.wake_event_count)
        .sum::<usize>();
    let true_negatives = cases_detail
        .iter()
        .filter(|case| !case.should_wake && !case.woke)
        .count();
    let false_negatives = cases_detail
        .iter()
        .filter(|case| case.should_wake && !case.woke)
        .count();
    let phrase_mismatches = cases_detail
        .iter()
        .filter(|case| case.should_wake && case.woke && !case.phrase_matched)
        .count();
    let repeated_positive_wake_events = cases_detail
        .iter()
        .filter(|case| case.should_wake)
        .map(|case| case.wake_event_count.saturating_sub(1))
        .sum::<usize>();
    let precision = if true_positives + false_positives == 0 {
        0.0
    } else {
        true_positives as f32 / (true_positives + false_positives) as f32
    };
    let recall = if positives == 0 {
        0.0
    } else {
        true_positives as f32 / positives as f32
    };
    let negative_audio_seconds = cases_detail
        .iter()
        .filter(|case| !case.should_wake)
        .map(|case| case.duration_ms as f32 / 1_000.0)
        .sum::<f32>();
    let false_wakes_per_hour = if negative_audio_seconds <= f32::EPSILON {
        0.0
    } else {
        false_positives as f32 / (negative_audio_seconds / 3_600.0)
    };
    let max_detection_latency_ms = cases_detail
        .iter()
        .filter(|case| case.should_wake)
        .filter_map(|case| case.detection_latency_ms)
        .max();

    ThresholdSweepPoint {
        candidate_threshold,
        accept_threshold,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        repeated_positive_wake_events,
        phrase_mismatches,
        precision,
        recall,
        false_wakes_per_hour,
        max_detection_latency_ms,
        splits: build_subgroups(cases_detail).splits,
    }
}

fn build_leakage(template_paths: &[PathBuf], cases_detail: &[CaseResult]) -> LeakageReport {
    let template_path_set = template_paths
        .iter()
        .map(|path| normalize_path_string(path))
        .collect::<BTreeSet<_>>();
    let template_case_paths = cases_detail
        .iter()
        .map(|case| normalize_path_string(Path::new(&case.path)))
        .filter(|case_path| template_path_set.contains(case_path))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let mut by_case_path = BTreeMap::<String, Vec<String>>::new();
    for case in cases_detail {
        by_case_path
            .entry(normalize_path_string(Path::new(&case.path)))
            .or_default()
            .push(case.id.clone());
    }
    let duplicate_case_path_groups = by_case_path
        .into_iter()
        .filter_map(|(path, case_ids)| {
            (case_ids.len() > 1).then_some(DuplicatePathGroup { path, case_ids })
        })
        .collect::<Vec<_>>();

    let template_hashes = template_paths
        .iter()
        .filter_map(|path| audio_hash(path).ok())
        .collect::<BTreeSet<_>>();
    let template_fingerprints = template_paths
        .iter()
        .filter_map(|path| audio_fingerprint(path).ok())
        .collect::<BTreeSet<_>>();
    let mut template_case_audio_hashes = BTreeSet::<String>::new();
    let mut template_case_audio_fingerprints = BTreeSet::<String>::new();
    let mut by_case_hash = BTreeMap::<String, Vec<(String, String)>>::new();
    let mut by_case_fingerprint = BTreeMap::<String, Vec<(String, String)>>::new();
    for case in cases_detail {
        let path = Path::new(&case.path);
        if let Ok(hash) = audio_hash(path) {
            if template_hashes.contains(&hash) {
                template_case_audio_hashes.insert(hash.clone());
            }
            by_case_hash
                .entry(hash)
                .or_default()
                .push((case.id.clone(), normalize_path_string(path)));
        }
        if let Ok(fingerprint) = audio_fingerprint(path) {
            if template_fingerprints.contains(&fingerprint) {
                template_case_audio_fingerprints.insert(fingerprint.clone());
            }
            by_case_fingerprint
                .entry(fingerprint)
                .or_default()
                .push((case.id.clone(), normalize_path_string(path)));
        }
    }
    let duplicate_case_audio_groups = by_case_hash
        .into_iter()
        .filter_map(|(hash, cases)| {
            if cases.len() <= 1 {
                return None;
            }
            let mut case_ids = Vec::new();
            let mut paths = BTreeSet::new();
            for (case_id, path) in cases {
                case_ids.push(case_id);
                paths.insert(path);
            }
            Some(DuplicateAudioGroup {
                hash,
                case_ids,
                paths: paths.into_iter().collect(),
            })
        })
        .collect::<Vec<_>>();
    let duplicate_case_fingerprint_groups = by_case_fingerprint
        .into_iter()
        .filter_map(|(fingerprint, cases)| {
            if cases.len() <= 1 {
                return None;
            }
            let mut case_ids = Vec::new();
            let mut paths = BTreeSet::new();
            for (case_id, path) in cases {
                case_ids.push(case_id);
                paths.insert(path);
            }
            Some(DuplicateFingerprintGroup {
                fingerprint,
                case_ids,
                paths: paths.into_iter().collect(),
            })
        })
        .collect::<Vec<_>>();

    LeakageReport {
        template_case_path_overlaps: template_case_paths.len(),
        template_case_audio_overlaps: template_case_audio_hashes.len(),
        template_case_fingerprint_overlaps: template_case_audio_fingerprints.len(),
        duplicate_case_paths: duplicate_case_path_groups.len(),
        duplicate_case_audio: duplicate_case_audio_groups.len(),
        duplicate_case_fingerprints: duplicate_case_fingerprint_groups.len(),
        template_case_paths,
        template_case_audio_hashes: template_case_audio_hashes.into_iter().collect(),
        template_case_audio_fingerprints: template_case_audio_fingerprints.into_iter().collect(),
        duplicate_case_path_groups,
        duplicate_case_audio_groups,
        duplicate_case_fingerprint_groups,
    }
}

fn normalize_path_string(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn audio_hash(path: &Path) -> std::io::Result<String> {
    let bytes = fs::read(path)?;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(format!("{hash:016x}"))
}

fn audio_fingerprint(path: &Path) -> std::io::Result<String> {
    let samples = wav_samples_i16(path)?;
    if samples.is_empty() {
        return Ok(String::new());
    }
    let peak = samples
        .iter()
        .map(|sample| sample.unsigned_abs())
        .max()
        .unwrap_or_default();
    if peak == 0 {
        return Ok("silence".to_string());
    }
    let bins = 2_048usize;
    let step = samples.len().div_ceil(bins).max(1);
    let mut quantized = Vec::with_capacity(samples.len().div_ceil(step));
    for chunk in samples.chunks(step) {
        let sum = chunk.iter().map(|sample| *sample as f64).sum::<f64>();
        let average = sum / chunk.len() as f64;
        let scaled = ((average / peak as f64) * 127.0).round() as i16;
        quantized.push((scaled.clamp(-127, 127) + 127) as u8);
    }
    let digest = digest::digest(&digest::SHA256, &quantized);
    Ok(digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn build_coverage(cases_detail: &[CaseResult]) -> CoverageReport {
    let speaker_ids = unique_values(
        cases_detail
            .iter()
            .filter_map(|case| case.speaker_id.clone()),
    );
    let environment_ids = unique_values(
        cases_detail
            .iter()
            .filter_map(|case| case.environment.clone()),
    );
    let distance_ids = unique_values(cases_detail.iter().filter_map(|case| case.distance.clone()));
    let device_ids = unique_values(cases_detail.iter().filter_map(|case| case.device.clone()));
    let session_ids = unique_values(
        cases_detail
            .iter()
            .filter_map(|case| case.session_id.clone()),
    );
    let category_ids = unique_values(cases_detail.iter().filter_map(|case| case.category.clone()));
    CoverageReport {
        speakers: speaker_ids.len(),
        environments: environment_ids.len(),
        distances: distance_ids.len(),
        devices: device_ids.len(),
        sessions: session_ids.len(),
        categories: category_ids.len(),
        speaker_ids,
        environment_ids,
        distance_ids,
        device_ids,
        session_ids,
        category_ids,
    }
}

fn unique_values(values: impl Iterator<Item = String>) -> Vec<String> {
    values
        .filter(|value| !value.trim().is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn build_subgroups(cases_detail: &[CaseResult]) -> SubgroupReport {
    SubgroupReport {
        speakers: group_by(cases_detail, |case| case.speaker_id.as_deref()),
        environments: group_by(cases_detail, |case| case.environment.as_deref()),
        distances: group_by(cases_detail, |case| case.distance.as_deref()),
        devices: group_by(cases_detail, |case| case.device.as_deref()),
        sessions: group_by(cases_detail, |case| case.session_id.as_deref()),
        categories: group_by(cases_detail, |case| case.category.as_deref()),
        splits: group_by(cases_detail, |case| case.split.as_deref()),
    }
}

fn group_by<'a>(
    cases_detail: &'a [CaseResult],
    key: impl Fn(&'a CaseResult) -> Option<&'a str>,
) -> Vec<GroupMetrics> {
    let mut groups: BTreeMap<String, Vec<&CaseResult>> = BTreeMap::new();
    for case in cases_detail {
        if let Some(value) = key(case).filter(|value| !value.trim().is_empty()) {
            groups.entry(value.to_string()).or_default().push(case);
        }
    }
    groups
        .into_iter()
        .map(|(id, cases)| group_metrics(id, &cases))
        .collect()
}

fn group_metrics(id: String, cases: &[&CaseResult]) -> GroupMetrics {
    let positives = cases.iter().filter(|case| case.should_wake).count();
    let negatives = cases.len().saturating_sub(positives);
    let true_positives = cases
        .iter()
        .filter(|case| case.should_wake && case.woke)
        .count();
    let false_positives = cases
        .iter()
        .filter(|case| !case.should_wake)
        .map(|case| case.wake_event_count)
        .sum::<usize>();
    let true_negatives = cases
        .iter()
        .filter(|case| !case.should_wake && !case.woke)
        .count();
    let false_negatives = cases
        .iter()
        .filter(|case| case.should_wake && !case.woke)
        .count();
    let phrase_mismatches = cases
        .iter()
        .filter(|case| case.should_wake && case.woke && !case.phrase_matched)
        .count();
    let repeated_positive_wake_events = cases
        .iter()
        .filter(|case| case.should_wake)
        .map(|case| case.wake_event_count.saturating_sub(1))
        .sum::<usize>();
    let precision = if true_positives + false_positives == 0 {
        0.0
    } else {
        true_positives as f32 / (true_positives + false_positives) as f32
    };
    let recall = if positives == 0 {
        0.0
    } else {
        true_positives as f32 / positives as f32
    };
    let negative_audio_seconds = cases
        .iter()
        .filter(|case| !case.should_wake)
        .map(|case| case.duration_ms as f32 / 1_000.0)
        .sum::<f32>();
    let false_wakes_per_hour = if negative_audio_seconds <= f32::EPSILON {
        0.0
    } else {
        false_positives as f32 / (negative_audio_seconds / 3_600.0)
    };
    let max_detection_latency_ms = cases
        .iter()
        .filter(|case| case.should_wake)
        .filter_map(|case| case.detection_latency_ms)
        .max();

    GroupMetrics {
        id,
        cases: cases.len(),
        positives,
        negatives,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        repeated_positive_wake_events,
        phrase_mismatches,
        precision,
        recall,
        negative_audio_seconds,
        false_wakes_per_hour,
        max_detection_latency_ms,
    }
}

fn enforce_report(report: &EvalReport) {
    let min_precision = env_float("VONA_WAKE_REAL_EVAL_MIN_PRECISION", 0.98);
    let min_recall = env_float("VONA_WAKE_REAL_EVAL_MIN_RECALL", 0.98);
    let min_precision_lower_bound =
        env_optional_float("VONA_WAKE_REAL_EVAL_MIN_PRECISION_LOWER_BOUND");
    let min_recall_lower_bound = env_optional_float("VONA_WAKE_REAL_EVAL_MIN_RECALL_LOWER_BOUND");
    let min_cases = env_usize("VONA_WAKE_REAL_EVAL_MIN_CASES", 1);
    let min_positive_cases = env_usize("VONA_WAKE_REAL_EVAL_MIN_POSITIVE_CASES", 1);
    let min_negative_cases = env_usize("VONA_WAKE_REAL_EVAL_MIN_NEGATIVE_CASES", 1);
    let min_negative_audio_seconds =
        env_float("VONA_WAKE_REAL_EVAL_MIN_NEGATIVE_AUDIO_SECONDS", 0.0);
    let min_speakers = env_usize("VONA_WAKE_REAL_EVAL_MIN_SPEAKERS", 0);
    let min_environments = env_usize("VONA_WAKE_REAL_EVAL_MIN_ENVIRONMENTS", 0);
    let min_distances = env_usize("VONA_WAKE_REAL_EVAL_MIN_DISTANCES", 0);
    let min_devices = env_usize("VONA_WAKE_REAL_EVAL_MIN_DEVICES", 0);
    let min_sessions = env_usize("VONA_WAKE_REAL_EVAL_MIN_SESSIONS", 0);
    let min_categories = env_usize("VONA_WAKE_REAL_EVAL_MIN_CATEGORIES", 0);
    let max_false_positives = env_usize("VONA_WAKE_REAL_EVAL_MAX_FALSE_POSITIVES", 0);
    let max_false_negatives = env_usize("VONA_WAKE_REAL_EVAL_MAX_FALSE_NEGATIVES", 0);
    let max_repeated_positive_wake_events =
        env_usize("VONA_WAKE_REAL_EVAL_MAX_REPEATED_POSITIVE_WAKE_EVENTS", 0);
    let max_phrase_mismatches = env_usize("VONA_WAKE_REAL_EVAL_MAX_PHRASE_MISMATCHES", 0);
    let max_false_wakes_per_hour = env_float("VONA_WAKE_REAL_EVAL_MAX_FALSE_WAKES_PER_HOUR", 0.05);
    let max_false_wakes_per_hour_upper_bound =
        env_optional_float("VONA_WAKE_REAL_EVAL_MAX_FALSE_WAKES_PER_HOUR_UPPER_BOUND");
    let max_first_wake_ms = env_u64("VONA_WAKE_REAL_EVAL_MAX_FIRST_WAKE_MS", 1_500);
    let max_detection_latency_ms = env_i64(
        "VONA_WAKE_REAL_EVAL_MAX_DETECTION_LATENCY_MS",
        max_first_wake_ms as i64,
    );
    let min_subgroup_precision =
        env_float("VONA_WAKE_REAL_EVAL_MIN_SUBGROUP_PRECISION", min_precision);
    let min_subgroup_recall = env_float("VONA_WAKE_REAL_EVAL_MIN_SUBGROUP_RECALL", min_recall);
    let max_subgroup_repeated_positive_wake_events = env_usize(
        "VONA_WAKE_REAL_EVAL_MAX_SUBGROUP_REPEATED_POSITIVE_WAKE_EVENTS",
        max_repeated_positive_wake_events,
    );
    let max_subgroup_false_wakes_per_hour = env_float(
        "VONA_WAKE_REAL_EVAL_MAX_SUBGROUP_FALSE_WAKES_PER_HOUR",
        max_false_wakes_per_hour,
    );
    let max_subgroup_detection_latency_ms = env_i64(
        "VONA_WAKE_REAL_EVAL_MAX_SUBGROUP_DETECTION_LATENCY_MS",
        max_detection_latency_ms,
    );
    let max_template_case_path_overlaps =
        env_usize("VONA_WAKE_REAL_EVAL_MAX_TEMPLATE_CASE_PATH_OVERLAPS", 0);
    let max_template_case_audio_overlaps =
        env_usize("VONA_WAKE_REAL_EVAL_MAX_TEMPLATE_CASE_AUDIO_OVERLAPS", 0);
    let max_template_case_fingerprint_overlaps = env_usize(
        "VONA_WAKE_REAL_EVAL_MAX_TEMPLATE_CASE_FINGERPRINT_OVERLAPS",
        0,
    );
    let max_duplicate_case_paths = env_usize("VONA_WAKE_REAL_EVAL_MAX_DUPLICATE_CASE_PATHS", 0);
    let max_duplicate_case_audio = env_usize("VONA_WAKE_REAL_EVAL_MAX_DUPLICATE_CASE_AUDIO", 0);
    let max_duplicate_case_fingerprints =
        env_usize("VONA_WAKE_REAL_EVAL_MAX_DUPLICATE_CASE_FINGERPRINTS", 0);

    assert!(
        report.cases >= min_cases,
        "real voice cases {} below required {}",
        report.cases,
        min_cases
    );
    assert!(
        report.positives >= min_positive_cases,
        "real voice positive cases {} below required {}",
        report.positives,
        min_positive_cases
    );
    assert!(
        report.negatives >= min_negative_cases,
        "real voice negative cases {} below required {}",
        report.negatives,
        min_negative_cases
    );
    assert!(
        report.negative_audio_seconds >= min_negative_audio_seconds,
        "real voice negative audio seconds {} below required {}",
        report.negative_audio_seconds,
        min_negative_audio_seconds
    );
    assert!(
        report.coverage.speakers >= min_speakers,
        "real voice speaker coverage {} below required {}",
        report.coverage.speakers,
        min_speakers
    );
    assert!(
        report.coverage.environments >= min_environments,
        "real voice environment coverage {} below required {}",
        report.coverage.environments,
        min_environments
    );
    assert!(
        report.coverage.distances >= min_distances,
        "real voice distance coverage {} below required {}",
        report.coverage.distances,
        min_distances
    );
    assert!(
        report.coverage.devices >= min_devices,
        "real voice device coverage {} below required {}",
        report.coverage.devices,
        min_devices
    );
    assert!(
        report.coverage.sessions >= min_sessions,
        "real voice session coverage {} below required {}",
        report.coverage.sessions,
        min_sessions
    );
    assert!(
        report.coverage.categories >= min_categories,
        "real voice category coverage {} below required {}",
        report.coverage.categories,
        min_categories
    );
    assert!(
        report.leakage.template_case_path_overlaps <= max_template_case_path_overlaps,
        "real voice template/case path overlaps {} exceeded allowed {}",
        report.leakage.template_case_path_overlaps,
        max_template_case_path_overlaps
    );
    assert!(
        report.leakage.duplicate_case_paths <= max_duplicate_case_paths,
        "real voice duplicate case paths {} exceeded allowed {}",
        report.leakage.duplicate_case_paths,
        max_duplicate_case_paths
    );
    assert!(
        report.leakage.template_case_audio_overlaps <= max_template_case_audio_overlaps,
        "real voice template/case audio overlaps {} exceeded allowed {}",
        report.leakage.template_case_audio_overlaps,
        max_template_case_audio_overlaps
    );
    assert!(
        report.leakage.template_case_fingerprint_overlaps <= max_template_case_fingerprint_overlaps,
        "real voice template/case normalized audio fingerprint overlaps {} exceeded allowed {}",
        report.leakage.template_case_fingerprint_overlaps,
        max_template_case_fingerprint_overlaps
    );
    assert!(
        report.leakage.duplicate_case_audio <= max_duplicate_case_audio,
        "real voice duplicate case audio groups {} exceeded allowed {}",
        report.leakage.duplicate_case_audio,
        max_duplicate_case_audio
    );
    assert!(
        report.leakage.duplicate_case_fingerprints <= max_duplicate_case_fingerprints,
        "real voice duplicate case normalized audio fingerprint groups {} exceeded allowed {}",
        report.leakage.duplicate_case_fingerprints,
        max_duplicate_case_fingerprints
    );
    assert!(
        report.precision >= min_precision,
        "real voice precision {} is below required {}",
        report.precision,
        min_precision
    );
    assert!(
        report.recall >= min_recall,
        "real voice recall {} is below required {}",
        report.recall,
        min_recall
    );
    assert!(
        report.false_positives <= max_false_positives,
        "real voice false positives {} exceeded allowed {}",
        report.false_positives,
        max_false_positives
    );
    assert!(
        report.false_negatives <= max_false_negatives,
        "real voice false negatives {} exceeded allowed {}",
        report.false_negatives,
        max_false_negatives
    );
    assert!(
        report.repeated_positive_wake_events <= max_repeated_positive_wake_events,
        "real voice repeated positive wake events {} exceeded allowed {}",
        report.repeated_positive_wake_events,
        max_repeated_positive_wake_events
    );
    assert!(
        report.phrase_mismatches <= max_phrase_mismatches,
        "real voice phrase mismatches {} exceeded allowed {}",
        report.phrase_mismatches,
        max_phrase_mismatches
    );
    assert!(
        report.false_wakes_per_hour <= max_false_wakes_per_hour,
        "real voice false wakes/hour {} exceeded allowed {}",
        report.false_wakes_per_hour,
        max_false_wakes_per_hour
    );
    if let Some(required_precision_lower_bound) = min_precision_lower_bound {
        let observed = report
            .confidence_intervals
            .precision_lower_bound
            .expect("precision lower confidence bound requires at least one precision trial");
        assert!(
            observed >= required_precision_lower_bound,
            "real voice precision lower confidence bound {} is below required {}",
            observed,
            required_precision_lower_bound
        );
    }
    if let Some(required_recall_lower_bound) = min_recall_lower_bound {
        let observed = report
            .confidence_intervals
            .recall_lower_bound
            .expect("recall lower confidence bound requires at least one positive case");
        assert!(
            observed >= required_recall_lower_bound,
            "real voice recall lower confidence bound {} is below required {}",
            observed,
            required_recall_lower_bound
        );
    }
    if let Some(allowed_false_wake_upper_bound) = max_false_wakes_per_hour_upper_bound {
        let observed = report
            .confidence_intervals
            .false_wakes_per_hour_upper_bound
            .expect("false wake upper confidence bound requires negative audio exposure");
        assert!(
            observed <= allowed_false_wake_upper_bound,
            "real voice false wakes/hour upper confidence bound {} exceeded allowed {}",
            observed,
            allowed_false_wake_upper_bound
        );
    }
    if let Some(observed_max_first_wake_ms) = report.max_first_wake_ms {
        assert!(
            observed_max_first_wake_ms <= max_first_wake_ms,
            "real voice first wake latency {} ms exceeded allowed {} ms",
            observed_max_first_wake_ms,
            max_first_wake_ms
        );
    }
    if let Some(observed_max_detection_latency_ms) = report.max_detection_latency_ms {
        assert!(
            observed_max_detection_latency_ms <= max_detection_latency_ms,
            "real voice detection latency {} ms exceeded allowed {} ms",
            observed_max_detection_latency_ms,
            max_detection_latency_ms
        );
    }
    enforce_subgroup_report(
        "speaker",
        &report.subgroups.speakers,
        min_subgroup_precision,
        min_subgroup_recall,
        max_subgroup_repeated_positive_wake_events,
        max_subgroup_false_wakes_per_hour,
        max_subgroup_detection_latency_ms,
    );
    enforce_subgroup_report(
        "environment",
        &report.subgroups.environments,
        min_subgroup_precision,
        min_subgroup_recall,
        max_subgroup_repeated_positive_wake_events,
        max_subgroup_false_wakes_per_hour,
        max_subgroup_detection_latency_ms,
    );
    enforce_subgroup_report(
        "distance",
        &report.subgroups.distances,
        min_subgroup_precision,
        min_subgroup_recall,
        max_subgroup_repeated_positive_wake_events,
        max_subgroup_false_wakes_per_hour,
        max_subgroup_detection_latency_ms,
    );
    enforce_subgroup_report(
        "device",
        &report.subgroups.devices,
        min_subgroup_precision,
        min_subgroup_recall,
        max_subgroup_repeated_positive_wake_events,
        max_subgroup_false_wakes_per_hour,
        max_subgroup_detection_latency_ms,
    );
    enforce_subgroup_report(
        "session",
        &report.subgroups.sessions,
        min_subgroup_precision,
        min_subgroup_recall,
        max_subgroup_repeated_positive_wake_events,
        max_subgroup_false_wakes_per_hour,
        max_subgroup_detection_latency_ms,
    );
    enforce_subgroup_report(
        "category",
        &report.subgroups.categories,
        min_subgroup_precision,
        min_subgroup_recall,
        max_subgroup_repeated_positive_wake_events,
        max_subgroup_false_wakes_per_hour,
        max_subgroup_detection_latency_ms,
    );
}

fn enforce_subgroup_report(
    dimension: &str,
    groups: &[GroupMetrics],
    min_precision: f32,
    min_recall: f32,
    max_repeated_positive_wake_events: usize,
    max_false_wakes_per_hour: f32,
    max_detection_latency_ms: i64,
) {
    for group in groups {
        if group.true_positives + group.false_positives > 0 {
            assert!(
                group.precision >= min_precision,
                "real voice {dimension} subgroup '{}' precision {} is below required {}",
                group.id,
                group.precision,
                min_precision
            );
        }
        if group.positives > 0 {
            assert!(
                group.recall >= min_recall,
                "real voice {dimension} subgroup '{}' recall {} is below required {}",
                group.id,
                group.recall,
                min_recall
            );
            assert!(
                group.repeated_positive_wake_events <= max_repeated_positive_wake_events,
                "real voice {dimension} subgroup '{}' repeated positive wake events {} exceeded allowed {}",
                group.id,
                group.repeated_positive_wake_events,
                max_repeated_positive_wake_events
            );
        }
        if group.negative_audio_seconds > f32::EPSILON {
            assert!(
                group.false_wakes_per_hour <= max_false_wakes_per_hour,
                "real voice {dimension} subgroup '{}' false wakes/hour {} exceeded allowed {}",
                group.id,
                group.false_wakes_per_hour,
                max_false_wakes_per_hour
            );
        }
        if let Some(observed_max_detection_latency_ms) = group.max_detection_latency_ms {
            assert!(
                observed_max_detection_latency_ms <= max_detection_latency_ms,
                "real voice {dimension} subgroup '{}' detection latency {} ms exceeded allowed {} ms",
                group.id,
                observed_max_detection_latency_ms,
                max_detection_latency_ms
            );
        }
    }
}

fn env_float(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_optional_float(name: &str) -> Option<f32> {
    env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .and_then(|value| value.parse::<f32>().ok())
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_i64(name: &str, default: i64) -> i64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(default)
}

fn audio_duration_ms(frames: &[AudioInputFrame]) -> u64 {
    frames
        .last()
        .map(|frame| {
            let end_sample = frame.sequence.saturating_add(frame.samples.len() as u64);
            end_sample.saturating_mul(1_000) / frame.sample_rate_hz as u64
        })
        .unwrap_or_default()
}

fn wav_samples_i16(path: &Path) -> std::io::Result<Vec<i16>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 44 {
        return Err(invalid_wav("wav file is too small"));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(invalid_wav("wav must be RIFF/WAVE"));
    }

    let mut offset = 12usize;
    let mut channels = 1u16;
    let mut sample_rate_hz = 16_000u32;
    let mut bits_per_sample = 16u16;
    let mut data = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let start = offset + 8;
        let end = start.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " => {
                if start + 16 > bytes.len() {
                    return Err(invalid_wav("wav fmt chunk is too small"));
                }
                let audio_format = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
                channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap());
                sample_rate_hz =
                    u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
                bits_per_sample =
                    u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap());
                if audio_format != 1 {
                    return Err(invalid_wav("expected PCM wav"));
                }
            }
            b"data" => data = Some(bytes[start..end].to_vec()),
            _ => {}
        }
        offset = end + (size % 2);
    }

    if channels != 1 || sample_rate_hz != 16_000 || bits_per_sample != 16 {
        return Err(invalid_wav("expected 16 kHz mono 16-bit PCM wav"));
    }
    Ok(data
        .ok_or_else(|| invalid_wav("wav data chunk"))?
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn invalid_wav(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

fn wav_to_frames(path: &Path, frame_samples: usize) -> Vec<AudioInputFrame> {
    let bytes = fs::read(path).expect("read wav");
    assert!(
        bytes.len() >= 44,
        "wav file is too small: {}",
        path.display()
    );
    assert_eq!(&bytes[0..4], b"RIFF", "wav must be RIFF");
    assert_eq!(&bytes[8..12], b"WAVE", "wav must be WAVE");

    let mut offset = 12usize;
    let mut channels = 1u16;
    let mut sample_rate_hz = 16_000u32;
    let mut bits_per_sample = 16u16;
    let mut data = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let start = offset + 8;
        let end = start.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " => {
                let audio_format = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
                channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap());
                sample_rate_hz =
                    u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
                bits_per_sample =
                    u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap());
                assert_eq!(audio_format, 1, "expected PCM wav");
            }
            b"data" => data = Some(bytes[start..end].to_vec()),
            _ => {}
        }
        offset = end + (size % 2);
    }

    assert_eq!(channels, 1, "expected mono wav");
    assert_eq!(sample_rate_hz, 16_000, "expected 16 kHz wav");
    assert_eq!(bits_per_sample, 16, "expected 16-bit wav");
    let data = data.expect("wav data chunk");
    let samples = data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32)
        .collect::<Vec<_>>();

    samples
        .chunks(frame_samples)
        .enumerate()
        .map(|(index, chunk)| AudioInputFrame {
            sequence: (index * frame_samples) as u64,
            sample_rate_hz,
            channels,
            samples: chunk.to_vec(),
        })
        .collect()
}
