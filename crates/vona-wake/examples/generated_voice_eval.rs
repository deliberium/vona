use serde::Serialize;
use serde_json::{Value, json};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use vona_core::types::AudioInputFrame;
use vona_wake::{
    EmbeddingSpeakerVerifier, SpeakerProfile, TemplateWakeDetector, WakeContext, WakeDecision,
    WakeGate, WakePolicy, WakeTemplate,
};

#[derive(Debug, Clone)]
struct EvalCase {
    id: &'static str,
    text: &'static str,
    should_wake: bool,
}

#[derive(Debug, Serialize)]
struct CaseResult {
    id: String,
    text: String,
    should_wake: bool,
    expected_phrase: Option<String>,
    wake_start_ms: Option<u64>,
    category: String,
    source_type: String,
    split: String,
    woke: bool,
    confidence: Option<f32>,
    phrase: Option<String>,
    frames: usize,
    audio_path: String,
    audio_source: String,
}

#[derive(Debug, Serialize)]
struct EvalReport {
    generated_dir: String,
    manifest_path: String,
    positives: usize,
    negatives: usize,
    true_positives: usize,
    false_positives: usize,
    true_negatives: usize,
    false_negatives: usize,
    precision: f32,
    recall: f32,
    unauthorized_rejected: bool,
    privacy_suppressed: bool,
    cases: Vec<CaseResult>,
}

fn main() {
    let generated_dir = env::var("VONA_WAKE_EVAL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/private/tmp"));
    fs::create_dir_all(&generated_dir).expect("create eval dir");

    let voice = env::var("VONA_WAKE_EVAL_VOICE").ok();
    let cases = eval_cases();
    let enrollment = template_variant(
        &generated_dir,
        "enroll_hey_vona",
        synthesize_case(
            &generated_dir,
            "enroll_hey_vona",
            "hey vona",
            voice.as_deref(),
        ),
    );
    let vona_enrollment = template_variant(
        &generated_dir,
        "enroll_vona",
        synthesize_case(&generated_dir, "enroll_vona", "vona", voice.as_deref()),
    );
    let templates = vec![
        TemplateWakeDetector::enroll("hey vona", &enrollment.frames),
        TemplateWakeDetector::enroll("vona", &vona_enrollment.frames),
    ];
    let mut results = Vec::new();
    for case in &cases {
        let audio = synthesize_case(&generated_dir, case.id, case.text, voice.as_deref());
        for variant in eval_variants(&generated_dir, case.id, &audio) {
            results.push(run_case(case, &variant, templates.clone()));
        }
    }

    let positives = results.iter().filter(|case| case.should_wake).count();
    let negatives = results.len().saturating_sub(positives);
    let true_positives = results
        .iter()
        .filter(|case| case.should_wake && case.woke)
        .count();
    let false_positives = results
        .iter()
        .filter(|case| !case.should_wake && case.woke)
        .count();
    let true_negatives = results
        .iter()
        .filter(|case| !case.should_wake && !case.woke)
        .count();
    let false_negatives = results
        .iter()
        .filter(|case| case.should_wake && !case.woke)
        .count();
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

    let unauthorized_rejected = run_unauthorized_check(&enrollment.frames, templates.clone());
    let privacy_suppressed = run_privacy_check(&enrollment.frames, templates);
    let manifest_path = env::var("VONA_WAKE_EVAL_MANIFEST")
        .map(PathBuf::from)
        .unwrap_or_else(|_| generated_dir.join("vona-wake-generated-manifest.json"));
    write_generated_manifest(&manifest_path, &enrollment, &vona_enrollment, &results);
    let report = EvalReport {
        generated_dir: generated_dir.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
        positives,
        negatives,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        precision,
        recall,
        unauthorized_rejected,
        privacy_suppressed,
        cases: results,
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    if let Ok(report_path) = env::var("VONA_WAKE_EVAL_REPORT") {
        fs::write(report_path, &json).expect("write generated voice eval report");
    }
    println!("{json}");

    if env::var("VONA_WAKE_EVAL_ENFORCE").ok().as_deref() == Some("1") {
        assert_eq!(
            report.false_negatives, 0,
            "generated wake phrases should be accepted"
        );
        assert_eq!(
            report.false_positives, 0,
            "generated non-wake phrases should be rejected"
        );
        assert!(
            report.unauthorized_rejected,
            "generated unauthorized speaker check did not reject"
        );
        assert!(
            report.privacy_suppressed,
            "generated privacy-mode check did not suppress"
        );
    }
}

fn eval_cases() -> Vec<EvalCase> {
    vec![
        EvalCase {
            id: "positive_hey_vona",
            text: "hey vona",
            should_wake: true,
        },
        EvalCase {
            id: "positive_vona",
            text: "vona",
            should_wake: true,
        },
        EvalCase {
            id: "positive_hey_vona_request",
            text: "hey vona can you help",
            should_wake: true,
        },
        EvalCase {
            id: "positive_vona_request",
            text: "vona start listening",
            should_wake: true,
        },
        EvalCase {
            id: "negative_hey_luna",
            text: "hey luna",
            should_wake: false,
        },
        EvalCase {
            id: "negative_hey_mona",
            text: "hey mona",
            should_wake: false,
        },
        EvalCase {
            id: "negative_vocal_note",
            text: "make a vocal note",
            should_wake: false,
        },
        EvalCase {
            id: "negative_weather",
            text: "what is the weather today",
            should_wake: false,
        },
        EvalCase {
            id: "negative_background",
            text: "the meeting starts in ten minutes",
            should_wake: false,
        },
    ]
}

struct EvalVariant {
    id: String,
    frames: Vec<AudioInputFrame>,
    path: PathBuf,
    source: String,
}

fn eval_variants(dir: &Path, case_id: &str, audio: &SynthesizedAudio) -> Vec<EvalVariant> {
    let quiet_path = dir.join(format!("vona-wake-eval-{case_id}-quiet.wav"));
    let quiet_frames = transform_frames(&audio.frames, 0.45, 0.0);
    write_wav_frames(&quiet_path, &quiet_frames);
    let loud_path = dir.join(format!("vona-wake-eval-{case_id}-loud.wav"));
    let loud_frames = transform_frames(&audio.frames, 1.35, 0.0);
    write_wav_frames(&loud_path, &loud_frames);
    let noisy_path = dir.join(format!("vona-wake-eval-{case_id}-noisy.wav"));
    let noisy_frames = transform_frames(&audio.frames, 1.0, 0.003);
    write_wav_frames(&noisy_path, &noisy_frames);

    vec![
        EvalVariant {
            id: "clean".to_string(),
            frames: audio.frames.clone(),
            path: audio.path.clone(),
            source: audio.source.clone(),
        },
        EvalVariant {
            id: "quiet".to_string(),
            frames: quiet_frames,
            path: quiet_path,
            source: format!("{}+quiet-gain", audio.source),
        },
        EvalVariant {
            id: "loud".to_string(),
            frames: loud_frames,
            path: loud_path,
            source: format!("{}+loud-gain", audio.source),
        },
        EvalVariant {
            id: "noisy".to_string(),
            frames: noisy_frames,
            path: noisy_path,
            source: format!("{}+deterministic-noise", audio.source),
        },
    ]
}

fn run_case(case: &EvalCase, variant: &EvalVariant, templates: Vec<WakeTemplate>) -> CaseResult {
    let mut gate = WakeGate::new(
        TemplateWakeDetector {
            templates,
            min_energy: 0.0005,
        },
        WakePolicy {
            candidate_threshold: 0.88,
            accept_threshold: 0.92,
            preroll_ms: 1_200,
            cooldown_ms: 0,
            ..WakePolicy::default()
        },
    );
    let context = WakeContext::default();
    let mut woke = false;
    let mut confidence = None;
    let mut phrase = None;
    for frame in variant.frames.iter().cloned() {
        if let WakeDecision::Accepted {
            confidence: accepted_confidence,
            phrase: accepted_phrase,
            ..
        } = gate.push_frame(frame, &context)
        {
            woke = true;
            confidence = Some(accepted_confidence);
            phrase = accepted_phrase;
            break;
        }
    }

    CaseResult {
        id: format!("{}:{}", case.id, variant.id),
        text: case.text.to_string(),
        should_wake: case.should_wake,
        expected_phrase: expected_phrase(case).map(str::to_string),
        wake_start_ms: case.should_wake.then_some(0),
        category: category(case).to_string(),
        source_type: variant.source.clone(),
        split: "evaluation".to_string(),
        woke,
        confidence,
        phrase,
        frames: variant.frames.len(),
        audio_path: variant.path.display().to_string(),
        audio_source: variant.source.clone(),
    }
}

fn write_generated_manifest(
    path: &Path,
    hey_vona_enrollment: &SynthesizedAudio,
    vona_enrollment: &SynthesizedAudio,
    results: &[CaseResult],
) {
    let manifest = json!({
        "corpus": {
            "id": "vona-wake-generated-regression",
            "version": "1",
            "source": "synthetic-generated",
            "created_by": "vona-wake generated_voice_eval",
            "notes": "Generated TTS/pseudo-voice regression corpus. Useful for deterministic CI, not a real-world reliability corpus."
        },
        "templates": [
            {
                "phrase": "hey vona",
                "path": hey_vona_enrollment.path,
                "speaker_id": "generated-speaker",
                "source_type": hey_vona_enrollment.source,
                "split": "enrollment"
            },
            {
                "phrase": "vona",
                "path": vona_enrollment.path,
                "speaker_id": "generated-speaker",
                "source_type": vona_enrollment.source,
                "split": "enrollment"
            }
        ],
        "cases": results.iter().map(|case| {
            json!({
                "id": case.id,
                "path": case.audio_path,
                "should_wake": case.should_wake,
                "text": case.text,
                "expected_phrase": case.expected_phrase,
                "wake_start_ms": case.wake_start_ms,
                "speaker_id": "generated-speaker",
                "environment": "synthetic",
                "distance": "synthetic",
                "device": "generated-audio",
                "category": case.category,
                "source_type": case.source_type,
                "split": case.split,
            })
        }).collect::<Vec<_>>(),
        "policy": {
            "candidate_threshold": 0.88,
            "accept_threshold": 0.92,
            "min_energy": 0.0005,
            "preroll_ms": 1200,
            "rearm_ms": 1200,
            "require_speaker_verification": false
        }
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create generated manifest dir");
    }
    let json = serde_json::to_string_pretty(&manifest).expect("serialize generated manifest");
    fs::write(path, json).expect("write generated voice manifest");
}

fn expected_phrase(case: &EvalCase) -> Option<&'static str> {
    if !case.should_wake {
        return None;
    }
    if case.text.starts_with("hey vona") {
        Some("hey vona")
    } else if case.text.starts_with("vona") {
        Some("vona")
    } else {
        None
    }
}

fn category(case: &EvalCase) -> &'static str {
    if case.should_wake {
        "wake-positive"
    } else if case.text.contains("luna") || case.text.contains("mona") {
        "near-miss"
    } else {
        "ordinary-speech"
    }
}

fn transform_frames(frames: &[AudioInputFrame], gain: f32, noise: f32) -> Vec<AudioInputFrame> {
    let mut seed = 0x5EED_1234_u32;
    frames
        .iter()
        .map(|frame| AudioInputFrame {
            sequence: frame.sequence,
            sample_rate_hz: frame.sample_rate_hz,
            channels: frame.channels,
            samples: frame
                .samples
                .iter()
                .map(|sample| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let random = ((seed >> 8) as f32 / 0x00FF_FFFF as f32) * 2.0 - 1.0;
                    (sample * gain + random * noise).clamp(-1.0, 1.0)
                })
                .collect(),
        })
        .collect()
}

fn run_unauthorized_check(frames: &[AudioInputFrame], templates: Vec<WakeTemplate>) -> bool {
    let mut gate = WakeGate::with_verifier(
        TemplateWakeDetector {
            templates,
            min_energy: 0.0005,
        },
        EmbeddingSpeakerVerifier,
        WakePolicy {
            candidate_threshold: 0.88,
            accept_threshold: 0.92,
            speaker_threshold: 0.99,
            require_speaker_verification: true,
            preroll_ms: 1_200,
            cooldown_ms: 0,
            ..WakePolicy::default()
        },
    );
    let context = WakeContext {
        allowed_speakers: vec![SpeakerProfile {
            speaker_id: "other-speaker".to_string(),
            embedding: vec![0.0; 24],
            metadata: Value::Null,
        }],
        ..WakeContext::default()
    };
    frames.iter().cloned().any(|frame| {
        matches!(
            gate.push_frame(frame, &context),
            WakeDecision::Rejected { .. }
        )
    })
}

fn run_privacy_check(frames: &[AudioInputFrame], templates: Vec<WakeTemplate>) -> bool {
    let mut gate = WakeGate::new(
        TemplateWakeDetector {
            templates,
            min_energy: 0.0005,
        },
        WakePolicy {
            preroll_ms: 1_200,
            ..WakePolicy::default()
        },
    );
    let context = WakeContext {
        privacy_mode: true,
        ..WakeContext::default()
    };
    frames.iter().cloned().any(|frame| {
        matches!(
            gate.push_frame(frame, &context),
            WakeDecision::Suppressed { .. }
        )
    })
}

struct SynthesizedAudio {
    frames: Vec<AudioInputFrame>,
    path: PathBuf,
    source: String,
}

fn synthesize_case(dir: &Path, id: &str, text: &str, voice: Option<&str>) -> SynthesizedAudio {
    let aiff = dir.join(format!("vona-wake-eval-{id}.aiff"));
    let wav = dir.join(format!("vona-wake-eval-{id}.wav"));
    let _ = fs::remove_file(&aiff);
    let _ = fs::remove_file(&wav);

    if let Some(frames) = synthesize_with_say(&aiff, &wav, text, voice) {
        return SynthesizedAudio {
            frames,
            path: wav,
            source: "macos-say".to_string(),
        };
    }

    let frames = pseudo_voice_frames(text, 320);
    write_wav_frames(&wav, &frames);
    SynthesizedAudio {
        frames,
        path: wav,
        source: "deterministic-pseudo-voice".to_string(),
    }
}

fn template_variant(dir: &Path, id: &str, audio: SynthesizedAudio) -> SynthesizedAudio {
    let path = dir.join(format!("vona-wake-eval-{id}-template.wav"));
    let frames = transform_frames(&audio.frames, 0.92, 0.0002);
    write_wav_frames(&path, &frames);
    SynthesizedAudio {
        frames,
        path,
        source: format!("{}+template-jitter", audio.source),
    }
}

fn synthesize_with_say(
    aiff: &Path,
    wav: &Path,
    text: &str,
    voice: Option<&str>,
) -> Option<Vec<AudioInputFrame>> {
    let mut say = Command::new("say");
    if let Some(voice) = voice.filter(|voice| !voice.trim().is_empty()) {
        say.arg("-v").arg(voice);
    }
    let say_status = say.arg("-o").arg(aiff).arg(text).status().ok()?;
    if !say_status.success() {
        return None;
    }

    let convert_status = Command::new("afconvert")
        .arg("-f")
        .arg("WAVE")
        .arg("-d")
        .arg("LEI16@16000")
        .arg(aiff)
        .arg(wav)
        .status()
        .ok()?;
    if !convert_status.success() {
        return None;
    }

    let frames = wav_to_frames(wav, 320);
    (!frames.is_empty()).then_some(frames)
}

fn wav_to_frames(path: &Path, frame_samples: usize) -> Vec<AudioInputFrame> {
    let bytes = fs::read(path).unwrap_or_default();
    if bytes.len() < 44 {
        return Vec::new();
    }
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

fn write_wav_frames(path: &Path, frames: &[AudioInputFrame]) {
    let sample_rate_hz = frames
        .first()
        .map(|frame| frame.sample_rate_hz)
        .unwrap_or(16_000);
    let channels = frames.first().map(|frame| frame.channels).unwrap_or(1);
    let samples = frames
        .iter()
        .flat_map(|frame| frame.samples.iter().copied())
        .collect::<Vec<_>>();
    let data_len = samples.len() * 2;
    let riff_len = 36 + data_len;
    let mut wav = Vec::with_capacity(44 + data_len);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(riff_len as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate_hz.to_le_bytes());
    let byte_rate = sample_rate_hz * channels as u32 * 2;
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels * 2;
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_len as u32).to_le_bytes());
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        wav.extend_from_slice(&pcm.to_le_bytes());
    }
    let mut file = fs::File::create(path).expect("create generated wav");
    file.write_all(&wav).expect("write generated wav");
}

fn pseudo_voice_frames(text: &str, frame_samples: usize) -> Vec<AudioInputFrame> {
    let sample_rate_hz = 16_000u32;
    let mut samples = Vec::new();
    for symbol in text.chars().filter(|symbol| symbol.is_ascii()) {
        if symbol.is_whitespace() {
            samples.extend(std::iter::repeat_n(0.0, sample_rate_hz as usize / 12));
            continue;
        }

        let code = symbol.to_ascii_lowercase() as u32;
        let base = 140.0 + ((code.wrapping_mul(37)) % 360) as f32;
        let formant = 520.0 + ((code.wrapping_mul(97)) % 1_200) as f32;
        let overtone = 1_600.0 + ((code.wrapping_mul(53)) % 1_100) as f32;
        let duration = 0.085 + ((code % 5) as f32 * 0.012);
        let token_samples = (duration * sample_rate_hz as f32) as usize;
        for index in 0..token_samples {
            let t = index as f32 / sample_rate_hz as f32;
            let attack = (index as f32 / (0.025 * sample_rate_hz as f32)).clamp(0.0, 1.0);
            let release = ((token_samples.saturating_sub(index)) as f32
                / (0.045 * sample_rate_hz as f32))
                .clamp(0.0, 1.0);
            let envelope = attack.min(release);
            let carrier = (std::f32::consts::TAU * base * t).sin() * 0.32
                + (std::f32::consts::TAU * formant * t).sin() * 0.11
                + (std::f32::consts::TAU * overtone * t).sin() * 0.04;
            samples.push((carrier * envelope).clamp(-0.8, 0.8));
        }
        samples.extend(std::iter::repeat_n(0.0, sample_rate_hz as usize / 100));
    }

    samples
        .chunks(frame_samples)
        .enumerate()
        .map(|(index, chunk)| AudioInputFrame {
            sequence: (index * frame_samples) as u64,
            sample_rate_hz,
            channels: 1,
            samples: chunk.to_vec(),
        })
        .collect()
}
