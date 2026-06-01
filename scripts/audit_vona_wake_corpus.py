#!/usr/bin/env python3
"""Audit a vona-wake real-voice corpus manifest before running evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import wave
from array import array
from collections import Counter, defaultdict
from pathlib import Path

from plan_vona_wake_corpus import required_negative_audio_hours, required_wilson_trials


DEFAULT_REQUIRED_CATEGORIES = (
    "wake-positive,"
    "unauthorized-wake,"
    "near-miss,"
    "ordinary-speech,"
    "ordinary-command,"
    "background-speech"
)
DEFAULT_REQUIRED_POSITIVE_CATEGORIES = "wake-positive"
DEFAULT_REQUIRED_NEGATIVE_CATEGORIES = (
    "unauthorized-wake,"
    "near-miss,"
    "ordinary-speech,"
    "ordinary-command,"
    "background-speech"
)
DEFAULT_REQUIRED_WAKE_START_BUCKETS = "early,mid,late"
DEFAULT_REQUIRED_WAKE_PHRASES = "hey vona,vona"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="real_voice_eval manifest path")
    parser.add_argument("--min-speakers", type=int, default=5)
    parser.add_argument("--min-environments", type=int, default=3)
    parser.add_argument("--min-distances", type=int, default=3)
    parser.add_argument("--min-devices", type=int, default=2)
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--min-categories", type=int, default=4)
    parser.add_argument("--min-template-speakers", type=int, default=5)
    parser.add_argument("--required-wake-phrases", default=DEFAULT_REQUIRED_WAKE_PHRASES)
    parser.add_argument("--min-templates-per-speaker-phrase", type=int, default=1)
    parser.add_argument("--min-unauthorized-wake-cases", type=int, default=1)
    parser.add_argument("--min-unauthorized-wake-speakers", type=int, default=1)
    parser.add_argument("--min-calibration-positive-cases", type=int, default=1)
    parser.add_argument("--min-calibration-negative-audio-seconds", type=float, default=1.0)
    parser.add_argument("--min-calibration-speaker-positive-cases", type=int, default=2)
    parser.add_argument("--min-calibration-phrase-positive-cases", type=int, default=1)
    parser.add_argument("--min-calibration-subgroup-positive-cases", type=int, default=1)
    parser.add_argument("--min-calibration-subgroup-negative-audio-seconds", type=float, default=1.0)
    parser.add_argument("--min-evaluation-positive-cases", type=int)
    parser.add_argument("--min-evaluation-negative-audio-seconds", type=float)
    parser.add_argument("--min-evaluation-speaker-positive-cases", type=int, default=10)
    parser.add_argument(
        "--min-evaluation-heldout-session-positive-cases-per-template-speaker",
        type=int,
        default=1,
        help="Require evaluation positives for each enrolled speaker from sessions not used by that speaker's enrollment templates",
    )
    parser.add_argument("--min-evaluation-phrase-positive-cases", type=int, default=10)
    parser.add_argument("--min-evaluation-subgroup-positive-cases", type=int, default=1)
    parser.add_argument("--min-evaluation-subgroup-negative-audio-seconds", type=float, default=600.0)
    parser.add_argument("--min-evaluation-speakers", type=int)
    parser.add_argument("--min-evaluation-environments", type=int)
    parser.add_argument("--min-evaluation-distances", type=int)
    parser.add_argument("--min-evaluation-devices", type=int)
    parser.add_argument("--min-evaluation-sessions", type=int)
    parser.add_argument("--min-evaluation-categories", type=int)
    parser.add_argument("--min-positive-speakers", type=int)
    parser.add_argument("--min-positive-environments", type=int)
    parser.add_argument("--min-positive-distances", type=int)
    parser.add_argument("--min-positive-devices", type=int)
    parser.add_argument("--min-positive-sessions", type=int)
    parser.add_argument("--min-positive-categories", type=int, default=1)
    parser.add_argument("--min-negative-speakers", type=int)
    parser.add_argument("--min-negative-environments", type=int)
    parser.add_argument("--min-negative-distances", type=int)
    parser.add_argument("--min-negative-devices", type=int)
    parser.add_argument("--min-negative-sessions", type=int)
    parser.add_argument("--min-negative-categories", type=int)
    parser.add_argument("--required-categories", default=DEFAULT_REQUIRED_CATEGORIES)
    parser.add_argument("--required-calibration-categories", default=DEFAULT_REQUIRED_CATEGORIES)
    parser.add_argument("--required-evaluation-categories", default=DEFAULT_REQUIRED_CATEGORIES)
    parser.add_argument("--required-positive-categories", default=DEFAULT_REQUIRED_POSITIVE_CATEGORIES)
    parser.add_argument("--required-negative-categories", default=DEFAULT_REQUIRED_NEGATIVE_CATEGORIES)
    parser.add_argument("--min-calibration-category-positive-cases", type=int, default=1)
    parser.add_argument("--min-evaluation-category-positive-cases", type=int, default=1)
    parser.add_argument("--min-calibration-category-negative-audio-seconds", type=float, default=1.0)
    parser.add_argument("--min-evaluation-category-negative-audio-seconds", type=float, default=600.0)
    parser.add_argument("--required-wake-start-buckets", default=DEFAULT_REQUIRED_WAKE_START_BUCKETS)
    parser.add_argument(
        "--required-calibration-wake-start-buckets",
        default=DEFAULT_REQUIRED_WAKE_START_BUCKETS,
    )
    parser.add_argument(
        "--required-evaluation-wake-start-buckets",
        default=DEFAULT_REQUIRED_WAKE_START_BUCKETS,
    )
    parser.add_argument("--observed-precision", type=float, default=0.98)
    parser.add_argument("--precision-lower-bound", type=float, default=0.95)
    parser.add_argument("--observed-recall", type=float, default=0.98)
    parser.add_argument("--recall-lower-bound", type=float, default=0.95)
    parser.add_argument("--false-wake-events", type=int, default=0)
    parser.add_argument("--false-wakes-per-hour-upper-bound", type=float, default=0.05)
    parser.add_argument("--min-template-rms", type=float, default=0.001)
    parser.add_argument("--min-positive-rms", type=float, default=0.001)
    parser.add_argument("--min-negative-rms", type=float, default=0.0001)
    parser.add_argument("--max-clipped-ratio", type=float, default=0.01)
    parser.add_argument(
        "--allow-missing-corpus-identity",
        action="store_true",
        help="Do not require top-level corpus.id and corpus.version metadata",
    )
    parser.add_argument(
        "--allow-non-human-source",
        action="store_true",
        help="Do not require template/case source_type=human-recorded",
    )
    parser.add_argument(
        "--allow-missing-collection-ledger",
        action="store_true",
        help="Do not require top-level collection_ledger consent/provenance metadata",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero if audit fails")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    corpus = corpus_metadata(manifest)
    collection_ledger = collection_ledger_metadata(manifest)
    manifest_dir = args.manifest.parent
    cases = manifest.get("cases", [])
    templates = manifest.get("templates", [])
    case_ids = [str(case.get("id", "")).strip() for case in cases]
    template_paths = [
        resolve_path(manifest_dir, Path(template["path"])) for template in templates
    ]
    case_paths = [
        resolve_path(manifest_dir, Path(case["path"])) for case in cases
    ]

    template_stats = [wav_stats(path) for path in template_paths]
    case_stats = [
        wav_stats(resolve_path(manifest_dir, Path(case["path"])))
        for case in cases
    ]
    template_audio_seconds = sum(stats["duration_seconds"] for stats in template_stats)
    durations = [stats["duration_seconds"] for stats in case_stats]
    positives = [case for case in cases if case.get("should_wake") is True]
    negatives = [case for case in cases if case.get("should_wake") is False]
    template_speakers = values(templates, "speaker_id")
    unauthorized_wake_cases = [
        case
        for case in negatives
        if str(case.get("category", "")).strip() == "unauthorized-wake"
    ]
    unauthorized_wake_speakers = values(unauthorized_wake_cases, "speaker_id")
    unauthorized_template_speaker_overlaps = sorted(
        unauthorized_wake_speakers & template_speakers
    )
    negative_audio_seconds = sum(
        durations[index]
        for index, case in enumerate(cases)
        if case.get("should_wake") is False
    )
    split_metrics = split_summary(cases, durations)
    split_coverage = {
        "calibration": metadata_for_split(cases, "calibration"),
        "evaluation": metadata_for_split(cases, "evaluation"),
    }
    metadata = {
        "speakers": values(cases, "speaker_id"),
        "environments": values(cases, "environment"),
        "distances": values(cases, "distance"),
        "devices": values(cases, "device"),
        "sessions": values(cases, "session_id"),
        "categories": values(cases, "category"),
    }
    missing_metadata = missing_metadata_counts(cases)
    positive_annotation_gaps = missing_positive_annotations(positives)
    positive_annotation_errors = invalid_positive_annotations(positives, cases, durations)
    subgroups = subgroup_metrics(cases, durations)
    calibration_category_balance = dimension_metrics_for_split(
        cases, durations, "calibration", "category"
    )
    evaluation_category_balance = dimension_metrics_for_split(
        cases, durations, "evaluation", "category"
    )
    calibration_speaker_positive_counts = speaker_positive_counts_for_split(
        cases, "calibration"
    )
    evaluation_speaker_positive_counts = speaker_positive_counts_for_split(
        cases, "evaluation"
    )
    calibration_phrase_positive_counts = phrase_positive_counts_for_split(
        cases, "calibration"
    )
    evaluation_phrase_positive_counts = phrase_positive_counts_for_split(
        cases, "evaluation"
    )
    template_phrase_counts = template_phrase_balance(templates)
    calibration_subgroup_balance = {
        key: dimension_metrics_for_split(cases, durations, "calibration", key)
        for key in ["environment", "distance", "device", "session_id"]
    }
    evaluation_subgroup_balance = {
        key: dimension_metrics_for_split(cases, durations, "evaluation", key)
        for key in ["environment", "distance", "device", "session_id"]
    }
    leakage = leakage_report(template_paths, case_paths, cases)
    duplicate_case_ids = duplicate_values(case_ids)
    source_provenance = {
        "template_source_counts": source_counts(templates),
        "case_source_counts": source_counts(cases),
        "non_human_templates": non_human_sources(templates),
        "non_human_cases": non_human_sources(cases),
    }
    audio_quality = {
        "templates": quality_entries(templates, template_stats),
        "cases": quality_entries(cases, case_stats),
    }
    audio_quality_failures = quality_failures(
        templates,
        template_stats,
        cases,
        case_stats,
        args,
    )
    plan = {
        "minimum_precision_trials": required_wilson_trials(
            args.observed_precision, args.precision_lower_bound
        ),
        "minimum_positive_cases": required_wilson_trials(
            args.observed_recall, args.recall_lower_bound
        ),
        "minimum_negative_audio_seconds": math.ceil(
            required_negative_audio_hours(
                args.false_wake_events, args.false_wakes_per_hour_upper_bound
            )
            * 3600
        ),
    }
    failures = audit_failures(
        templates=templates,
        cases=cases,
        positives=positives,
        negatives=negatives,
        negative_audio_seconds=negative_audio_seconds,
        metadata=metadata,
        template_speakers=template_speakers,
        unauthorized_wake_cases=unauthorized_wake_cases,
        unauthorized_wake_speakers=unauthorized_wake_speakers,
        unauthorized_template_speaker_overlaps=unauthorized_template_speaker_overlaps,
        missing_metadata=missing_metadata,
        positive_annotation_gaps=positive_annotation_gaps,
        positive_annotation_errors=positive_annotation_errors,
        subgroups=subgroups,
        calibration_category_balance=calibration_category_balance,
        evaluation_category_balance=evaluation_category_balance,
        calibration_speaker_positive_counts=calibration_speaker_positive_counts,
        evaluation_speaker_positive_counts=evaluation_speaker_positive_counts,
        calibration_phrase_positive_counts=calibration_phrase_positive_counts,
        evaluation_phrase_positive_counts=evaluation_phrase_positive_counts,
        template_phrase_counts=template_phrase_counts,
        calibration_subgroup_balance=calibration_subgroup_balance,
        evaluation_subgroup_balance=evaluation_subgroup_balance,
        split_metrics=split_metrics,
        split_coverage=split_coverage,
        leakage=leakage,
        duplicate_case_ids=duplicate_case_ids,
        source_provenance=source_provenance,
        corpus=corpus,
        collection_ledger=collection_ledger,
        audio_quality_failures=audio_quality_failures,
        allow_non_human_source=args.allow_non_human_source,
        allow_missing_corpus_identity=args.allow_missing_corpus_identity,
        allow_missing_collection_ledger=args.allow_missing_collection_ledger,
        plan=plan,
        min_speakers=args.min_speakers,
        min_environments=args.min_environments,
        min_distances=args.min_distances,
        min_devices=args.min_devices,
        min_sessions=args.min_sessions,
        min_categories=args.min_categories,
        min_template_speakers=args.min_template_speakers,
        required_wake_phrases=csv_set(args.required_wake_phrases),
        min_templates_per_speaker_phrase=args.min_templates_per_speaker_phrase,
        min_unauthorized_wake_cases=args.min_unauthorized_wake_cases,
        min_unauthorized_wake_speakers=args.min_unauthorized_wake_speakers,
        min_calibration_positive_cases=args.min_calibration_positive_cases,
        min_calibration_negative_audio_seconds=args.min_calibration_negative_audio_seconds,
        min_calibration_speaker_positive_cases=args.min_calibration_speaker_positive_cases,
        min_calibration_phrase_positive_cases=args.min_calibration_phrase_positive_cases,
        min_calibration_subgroup_positive_cases=args.min_calibration_subgroup_positive_cases,
        min_calibration_subgroup_negative_audio_seconds=args.min_calibration_subgroup_negative_audio_seconds,
        min_evaluation_positive_cases=args.min_evaluation_positive_cases
        or max(plan["minimum_precision_trials"], plan["minimum_positive_cases"]),
        min_evaluation_negative_audio_seconds=args.min_evaluation_negative_audio_seconds
        or plan["minimum_negative_audio_seconds"],
        min_evaluation_speaker_positive_cases=args.min_evaluation_speaker_positive_cases,
        min_evaluation_heldout_session_positive_cases_per_template_speaker=args.min_evaluation_heldout_session_positive_cases_per_template_speaker,
        min_evaluation_phrase_positive_cases=args.min_evaluation_phrase_positive_cases,
        min_evaluation_subgroup_positive_cases=args.min_evaluation_subgroup_positive_cases,
        min_evaluation_subgroup_negative_audio_seconds=args.min_evaluation_subgroup_negative_audio_seconds,
        min_evaluation_speakers=args.min_evaluation_speakers or args.min_speakers,
        min_evaluation_environments=args.min_evaluation_environments or args.min_environments,
        min_evaluation_distances=args.min_evaluation_distances or args.min_distances,
        min_evaluation_devices=args.min_evaluation_devices or args.min_devices,
        min_evaluation_sessions=args.min_evaluation_sessions or args.min_sessions,
        min_evaluation_categories=args.min_evaluation_categories or args.min_categories,
        min_positive_speakers=args.min_positive_speakers or args.min_speakers,
        min_positive_environments=args.min_positive_environments or args.min_environments,
        min_positive_distances=args.min_positive_distances or args.min_distances,
        min_positive_devices=args.min_positive_devices or args.min_devices,
        min_positive_sessions=args.min_positive_sessions or args.min_sessions,
        min_positive_categories=args.min_positive_categories,
        min_negative_speakers=args.min_negative_speakers or args.min_speakers,
        min_negative_environments=args.min_negative_environments or args.min_environments,
        min_negative_distances=args.min_negative_distances or args.min_distances,
        min_negative_devices=args.min_negative_devices or args.min_devices,
        min_negative_sessions=args.min_negative_sessions or args.min_sessions,
        min_negative_categories=args.min_negative_categories or args.min_categories,
        required_categories=csv_set(args.required_categories),
        required_calibration_categories=csv_set(args.required_calibration_categories),
        required_evaluation_categories=csv_set(args.required_evaluation_categories),
        required_positive_categories=csv_set(args.required_positive_categories),
        required_negative_categories=csv_set(args.required_negative_categories),
        min_calibration_category_positive_cases=args.min_calibration_category_positive_cases,
        min_evaluation_category_positive_cases=args.min_evaluation_category_positive_cases,
        min_calibration_category_negative_audio_seconds=args.min_calibration_category_negative_audio_seconds,
        min_evaluation_category_negative_audio_seconds=args.min_evaluation_category_negative_audio_seconds,
        required_wake_start_buckets=csv_set(args.required_wake_start_buckets),
        required_calibration_wake_start_buckets=csv_set(
            args.required_calibration_wake_start_buckets
        ),
        required_evaluation_wake_start_buckets=csv_set(
            args.required_evaluation_wake_start_buckets
        ),
    )
    report = {
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "corpus": corpus,
        "collection_ledger": collection_ledger,
        "ready": not failures,
        "templates": len(templates),
        "template_audio_seconds": template_audio_seconds,
        "cases": len(cases),
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "negative_audio_seconds": negative_audio_seconds,
        "negative_audio_hours": negative_audio_seconds / 3600,
        "metadata": {
            key: {"count": len(found), "values": sorted(found)}
            for key, found in metadata.items()
        },
        "required_categories": {
            "aggregate": sorted(csv_set(args.required_categories)),
            "calibration": sorted(csv_set(args.required_calibration_categories)),
            "evaluation": sorted(csv_set(args.required_evaluation_categories)),
        },
        "speaker_gating": {
            "template_speakers": len(template_speakers),
            "template_speaker_ids": sorted(template_speakers),
            "unauthorized_wake_cases": len(unauthorized_wake_cases),
            "unauthorized_wake_speakers": len(unauthorized_wake_speakers),
            "unauthorized_wake_speaker_ids": sorted(unauthorized_wake_speakers),
            "unauthorized_template_speaker_overlaps": unauthorized_template_speaker_overlaps,
        },
        "source_provenance": source_provenance,
        "audio_quality": audio_quality,
        "splits": split_metrics,
        "split_coverage": {
            split: {
                key: {"count": len(found), "values": sorted(found)}
                for key, found in coverage.items()
            }
            for split, coverage in split_coverage.items()
        },
        "subgroups": subgroups,
        "category_balance": {
            "calibration": calibration_category_balance,
            "evaluation": evaluation_category_balance,
        },
        "speaker_positive_balance": {
            "calibration": calibration_speaker_positive_counts,
            "evaluation": evaluation_speaker_positive_counts,
            "evaluation_heldout_session": heldout_session_positive_counts(
                sessions_by_speaker(templates),
                [
                    case
                    for case in positives
                    if str(case.get("split", "")).strip() == "evaluation"
                ],
            ),
        },
        "phrase_positive_balance": {
            "calibration": calibration_phrase_positive_counts,
            "evaluation": evaluation_phrase_positive_counts,
        },
        "template_phrase_balance": template_phrase_counts,
        "subgroup_balance": {
            "calibration": calibration_subgroup_balance,
            "evaluation": evaluation_subgroup_balance,
        },
        "wake_start_buckets": {
            "aggregate": sorted(wake_start_buckets(positives)),
            "calibration": sorted(
                wake_start_buckets(
                    [case for case in positives if str(case.get("split", "")).strip() == "calibration"]
                )
            ),
            "evaluation": sorted(
                wake_start_buckets(
                    [case for case in positives if str(case.get("split", "")).strip() == "evaluation"]
                )
            ),
        },
        "missing_metadata": missing_metadata,
        "missing_positive_annotations": positive_annotation_gaps,
        "invalid_positive_annotations": positive_annotation_errors,
        "leakage": leakage,
        "duplicate_case_ids": duplicate_case_ids,
        "planning_targets": plan,
        "failures": failures,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report)

    if args.enforce and failures:
        return 1
    return 0


def resolve_path(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else base / path


def wav_duration_seconds(path: Path) -> float:
    return wav_stats(path)["duration_seconds"]


def wav_stats(path: Path) -> dict[str, float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        if channels != 1 or sample_rate != 16_000 or sample_width != 2:
            raise SystemExit(
                f"{path} must be 16 kHz mono 16-bit PCM WAV; "
                f"got {channels} channel(s), {sample_rate} Hz, {sample_width * 8}-bit"
            )
        if frames == 0:
            raise SystemExit(f"{path} has no audio frames")
        total = 0
        square_sum = 0
        peak = 0
        clipped = 0
        while data := wav.readframes(65_536):
            samples = array("h")
            samples.frombytes(data)
            if sys.byteorder != "little":
                samples.byteswap()
            for sample in samples:
                absolute = abs(sample)
                total += 1
                square_sum += sample * sample
                peak = max(peak, absolute)
                if absolute >= 32_760:
                    clipped += 1
    rms = (square_sum / max(1, total)) ** 0.5 / 32_768.0
    return {
        "duration_seconds": frames / sample_rate,
        "rms": rms,
        "peak": peak / 32_768.0,
        "clipped_ratio": clipped / max(1, total),
    }


def quality_entries(
    items: list[dict[str, object]], stats: list[dict[str, float]]
) -> list[dict[str, object]]:
    return [
        {
            "id": str(item.get("id") or item.get("path") or index),
            "rms": stat["rms"],
            "peak": stat["peak"],
            "clipped_ratio": stat["clipped_ratio"],
        }
        for index, (item, stat) in enumerate(zip(items, stats))
    ]


def quality_failures(
    templates: list[dict[str, object]],
    template_stats: list[dict[str, float]],
    cases: list[dict[str, object]],
    case_stats: list[dict[str, float]],
    args,
) -> list[str]:
    failures = []
    for index, (template, stats) in enumerate(zip(templates, template_stats)):
        label = str(template.get("path") or f"template-{index}")
        failures.extend(quality_errors(label, stats, args.min_template_rms, args.max_clipped_ratio))
    for index, (case, stats) in enumerate(zip(cases, case_stats)):
        label = str(case.get("id") or case.get("path") or f"case-{index}")
        min_rms = args.min_positive_rms if case.get("should_wake") is True else args.min_negative_rms
        failures.extend(quality_errors(label, stats, min_rms, args.max_clipped_ratio))
    return failures


def quality_errors(
    label: str,
    stats: dict[str, float],
    min_rms: float,
    max_clipped_ratio: float,
) -> list[str]:
    failures = []
    if stats["rms"] < min_rms:
        failures.append(f"{label} rms {stats['rms']:.6f} below required {min_rms:.6f}")
    if stats["clipped_ratio"] > max_clipped_ratio:
        failures.append(
            f"{label} clipped ratio {stats['clipped_ratio']:.6f} exceeded allowed "
            f"{max_clipped_ratio:.6f}"
        )
    return failures


def values(cases: list[dict[str, object]], key: str) -> set[str]:
    return {
        str(case[key]).strip()
        for case in cases
        if str(case.get(key, "")).strip()
    }


def csv_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def wake_start_buckets(cases: list[dict[str, object]]) -> set[str]:
    buckets = set()
    for case in cases:
        try:
            wake_start_ms = int(case.get("wake_start_ms", ""))
        except (TypeError, ValueError):
            continue
        buckets.add(wake_start_bucket(wake_start_ms))
    return buckets


def wake_start_bucket(wake_start_ms: int) -> str:
    if wake_start_ms <= 250:
        return "early"
    if wake_start_ms <= 1500:
        return "mid"
    return "late"


def missing_metadata_counts(cases: list[dict[str, object]]) -> dict[str, int]:
    counts = {}
    for key in ["speaker_id", "environment", "distance", "device", "session_id", "category"]:
        counts[key] = sum(1 for case in cases if not str(case.get(key, "")).strip())
    return counts


def source_counts(items: list[dict[str, object]]) -> dict[str, int]:
    return dict(Counter(source_type(item) for item in items))


def source_type(item: dict[str, object]) -> str:
    return str(item.get("source_type", "")).strip().lower() or "missing"


def non_human_sources(items: list[dict[str, object]]) -> list[str]:
    return sorted(
        str(item.get("id") or item.get("path") or f"item-{index}")
        for index, item in enumerate(items)
        if source_type(item) != "human-recorded"
    )


def corpus_metadata(manifest: dict[str, object]) -> dict[str, object]:
    raw = manifest.get("corpus")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "id": str(raw.get("id", "")).strip(),
        "version": str(raw.get("version", "")).strip(),
        "source": str(raw.get("source", "")).strip(),
        "created_by": str(raw.get("created_by", "")).strip(),
        "notes": str(raw.get("notes", "")).strip(),
        "collection_ledger_sha256": str(raw.get("collection_ledger_sha256", "")).strip(),
    }


def collection_ledger_metadata(manifest: dict[str, object]) -> dict[str, object]:
    raw = manifest.get("collection_ledger")
    if not isinstance(raw, dict):
        return {}
    speakers = raw.get("speakers")
    if not isinstance(speakers, list):
        speakers = []
    devices = raw.get("devices")
    if not isinstance(devices, list):
        devices = []
    sessions = raw.get("sessions")
    if not isinstance(sessions, list):
        sessions = []
    speaker_ids = sorted(
        {
            str(speaker.get("speaker_id", "")).strip()
            for speaker in speakers
            if isinstance(speaker, dict) and str(speaker.get("speaker_id", "")).strip()
        }
    )
    device_ids = sorted(
        {
            str(device.get("device_id", "")).strip()
            for device in devices
            if isinstance(device, dict) and str(device.get("device_id", "")).strip()
        }
    )
    session_ids = sorted(
        {
            str(session.get("session_id", "")).strip()
            for session in sessions
            if isinstance(session, dict) and str(session.get("session_id", "")).strip()
        }
    )
    return {
        "present": True,
        "consent_protocol": str(raw.get("consent_protocol", "")).strip(),
        "collection_protocol": str(raw.get("collection_protocol", "")).strip(),
        "collected_by": sorted(string_list(raw.get("collected_by"))),
        "collection_started_at": str(raw.get("collection_started_at", "")).strip(),
        "collection_completed_at": str(raw.get("collection_completed_at", "")).strip(),
        "speaker_ids": speaker_ids,
        "speakers": speakers,
        "device_ids": device_ids,
        "devices": devices,
        "session_ids": session_ids,
        "sessions": sessions,
        "notes": str(raw.get("notes", "")).strip(),
    }


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def collection_ledger_failures(
    ledger: dict[str, object],
    templates: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> list[str]:
    failures: list[str] = []
    if not ledger:
        return ["manifest collection_ledger is required"]
    if not ledger.get("consent_protocol"):
        failures.append("collection_ledger.consent_protocol is required")
    if not ledger.get("collection_protocol"):
        failures.append("collection_ledger.collection_protocol is required")
    if not ledger.get("collected_by"):
        failures.append("collection_ledger.collected_by must include at least one operator")
    if not ledger.get("collection_started_at"):
        failures.append("collection_ledger.collection_started_at is required")
    if not ledger.get("collection_completed_at"):
        failures.append("collection_ledger.collection_completed_at is required")
    ledger_speakers = set(ledger.get("speaker_ids", []))
    manifest_speakers = values(templates + cases, "speaker_id")
    missing_speakers = sorted(manifest_speakers - ledger_speakers)
    if missing_speakers:
        failures.append(
            "collection_ledger missing speaker consent/provenance entries for: "
            + ", ".join(missing_speakers)
        )
    ledger_devices = set(ledger.get("device_ids", []))
    manifest_devices = values(cases, "device")
    missing_devices = sorted(manifest_devices - ledger_devices)
    if missing_devices:
        failures.append(
            "collection_ledger missing device provenance entries for: "
            + ", ".join(missing_devices)
        )
    ledger_sessions = set(ledger.get("session_ids", []))
    manifest_sessions = values(templates + cases, "session_id")
    missing_sessions = sorted(manifest_sessions - ledger_sessions)
    if missing_sessions:
        failures.append(
            "collection_ledger missing session provenance entries for: "
            + ", ".join(missing_sessions)
        )
    speaker_entries = ledger.get("speakers", [])
    for index, speaker in enumerate(speaker_entries, start=1):
        if not isinstance(speaker, dict):
            failures.append(f"collection_ledger.speakers[{index}] must be an object")
            continue
        speaker_id = str(speaker.get("speaker_id", "")).strip()
        if not speaker_id:
            failures.append(f"collection_ledger.speakers[{index}].speaker_id is required")
        if not str(speaker.get("consent_record", "")).strip():
            failures.append(
                f"collection_ledger speaker {speaker_id or index} missing consent_record"
            )
        if not str(speaker.get("consent_obtained_at", "")).strip():
            failures.append(
                f"collection_ledger speaker {speaker_id or index} missing consent_obtained_at"
            )
    device_entries = ledger.get("devices", [])
    for index, device in enumerate(device_entries, start=1):
        if not isinstance(device, dict):
            failures.append(f"collection_ledger.devices[{index}] must be an object")
            continue
        device_id = str(device.get("device_id", "")).strip()
        if not device_id:
            failures.append(f"collection_ledger.devices[{index}].device_id is required")
        if not str(device.get("recorder", "")).strip():
            failures.append(f"collection_ledger device {device_id or index} missing recorder")
        if not str(device.get("sample_rate_hz", "")).strip():
            failures.append(
                f"collection_ledger device {device_id or index} missing sample_rate_hz"
            )
    session_entries = ledger.get("sessions", [])
    for index, session in enumerate(session_entries, start=1):
        if not isinstance(session, dict):
            failures.append(f"collection_ledger.sessions[{index}] must be an object")
            continue
        session_id = str(session.get("session_id", "")).strip()
        if not session_id:
            failures.append(f"collection_ledger.sessions[{index}].session_id is required")
        if not str(session.get("collected_at", "")).strip():
            failures.append(
                f"collection_ledger session {session_id or index} missing collected_at"
            )
        if not str(session.get("operator", "")).strip():
            failures.append(
                f"collection_ledger session {session_id or index} missing operator"
            )
    return failures


def missing_positive_annotations(cases: list[dict[str, object]]) -> dict[str, int]:
    return {
        "expected_phrase": sum(
            1 for case in cases if not str(case.get("expected_phrase", "")).strip()
        ),
        "wake_start_ms": sum(1 for case in cases if "wake_start_ms" not in case),
    }


def invalid_positive_annotations(
    positives: list[dict[str, object]],
    cases: list[dict[str, object]],
    durations: list[float],
) -> list[dict[str, object]]:
    duration_by_id = {
        str(case.get("id", "")): durations[index]
        for index, case in enumerate(cases)
    }
    errors = []
    for case in positives:
        case_id = str(case.get("id", ""))
        error = positive_annotation_error(case, duration_by_id.get(case_id, 0.0))
        if error:
            errors.append({"id": case_id, "error": error})
    return errors


def positive_annotation_error(case: dict[str, object], duration_seconds: float) -> str | None:
    expected = str(case.get("expected_phrase", "")).strip()
    text = str(case.get("text", "")).strip()
    if expected and text and expected not in text:
        return "expected_phrase is not contained in text"
    try:
        wake_start_ms = int(case.get("wake_start_ms", ""))
    except (TypeError, ValueError):
        return "wake_start_ms is not an integer"
    if wake_start_ms < 0:
        return "wake_start_ms is negative"
    if duration_seconds > 0.0 and wake_start_ms >= duration_seconds * 1000.0:
        return "wake_start_ms is outside audio duration"
    return None


def subgroup_metrics(
    cases: list[dict[str, object]], durations: list[float]
) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        "speakers": dimension_metrics(cases, durations, "speaker_id"),
        "environments": dimension_metrics(cases, durations, "environment"),
        "distances": dimension_metrics(cases, durations, "distance"),
        "devices": dimension_metrics(cases, durations, "device"),
        "sessions": dimension_metrics(cases, durations, "session_id"),
        "categories": dimension_metrics(cases, durations, "category"),
    }


def split_summary(
    cases: list[dict[str, object]], durations: list[float]
) -> dict[str, dict[str, float | int]]:
    normalized_cases = []
    for case in cases:
        clone = dict(case)
        clone["split"] = str(case.get("split", "")).strip() or "missing"
        normalized_cases.append(clone)
    return dimension_metrics(normalized_cases, durations, "split")


def metadata_for_split(
    cases: list[dict[str, object]], split: str
) -> dict[str, set[str]]:
    split_cases = [case for case in cases if str(case.get("split", "")).strip() == split]
    return {
        "speakers": values(split_cases, "speaker_id"),
        "environments": values(split_cases, "environment"),
        "distances": values(split_cases, "distance"),
        "devices": values(split_cases, "device"),
        "sessions": values(split_cases, "session_id"),
        "categories": values(split_cases, "category"),
    }


def dimension_metrics(
    cases: list[dict[str, object]], durations: list[float], key: str
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for index, case in enumerate(cases):
        value = str(case.get(key, "")).strip()
        if not value:
            continue
        group = metrics.setdefault(
            value,
            {
                "cases": 0,
                "positive_cases": 0,
                "negative_cases": 0,
                "positive_audio_seconds": 0.0,
                "negative_audio_seconds": 0.0,
            },
        )
        duration = durations[index]
        group["cases"] += 1
        if case.get("should_wake") is True:
            group["positive_cases"] += 1
            group["positive_audio_seconds"] += duration
        elif case.get("should_wake") is False:
            group["negative_cases"] += 1
            group["negative_audio_seconds"] += duration
    return dict(sorted(metrics.items()))


def dimension_metrics_for_split(
    cases: list[dict[str, object]],
    durations: list[float],
    split: str,
    key: str,
) -> dict[str, dict[str, float | int]]:
    split_cases = []
    split_durations = []
    for index, case in enumerate(cases):
        if str(case.get("split", "")).strip() == split:
            split_cases.append(case)
            split_durations.append(durations[index])
    return dimension_metrics(split_cases, split_durations, key)


def speaker_positive_counts_for_split(
    cases: list[dict[str, object]], split: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case in cases:
        if str(case.get("split", "")).strip() != split:
            continue
        if case.get("should_wake") is not True:
            continue
        speaker = str(case.get("speaker_id", "")).strip()
        if speaker:
            counts[speaker] += 1
    return dict(sorted(counts.items()))


def phrase_positive_counts_for_split(
    cases: list[dict[str, object]], split: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case in cases:
        if str(case.get("split", "")).strip() != split:
            continue
        if case.get("should_wake") is not True:
            continue
        phrase = str(case.get("expected_phrase", "")).strip()
        if phrase:
            counts[phrase] += 1
    return dict(sorted(counts.items()))


def template_phrase_balance(
    templates: list[dict[str, object]]
) -> dict[str, dict[str, int]]:
    balance: dict[str, Counter[str]] = {}
    for template in templates:
        speaker = str(template.get("speaker_id", "")).strip()
        phrase = str(template.get("phrase", "")).strip()
        if speaker and phrase:
            balance.setdefault(speaker, Counter())[phrase] += 1
    return {speaker: dict(sorted(counts.items())) for speaker, counts in sorted(balance.items())}


def leakage_report(
    template_paths: list[Path],
    case_paths: list[Path],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    normalized_template_paths = {canonical_path(path) for path in template_paths}
    normalized_case_paths = [canonical_path(path) for path in case_paths]
    template_hashes = {audio_hash(path) for path in template_paths}
    case_hashes = [audio_hash(path) for path in case_paths]
    template_fingerprints = {audio_fingerprint(path) for path in template_paths}
    case_fingerprints = [audio_fingerprint(path) for path in case_paths]
    by_case_path = duplicate_groups(
        (normalized_case_paths[index], str(cases[index].get("id", f"case-{index}")))
        for index in range(len(cases))
    )
    by_case_hash = duplicate_groups(
        (case_hashes[index], str(cases[index].get("id", f"case-{index}")))
        for index in range(len(cases))
    )
    by_case_fingerprint = duplicate_groups(
        (case_fingerprints[index], str(cases[index].get("id", f"case-{index}")))
        for index in range(len(cases))
    )
    return {
        "template_case_path_overlaps": sorted(
            set(normalized_case_paths) & normalized_template_paths
        ),
        "template_case_audio_overlaps": sorted(set(case_hashes) & template_hashes),
        "template_case_fingerprint_overlaps": sorted(
            set(case_fingerprints) & template_fingerprints
        ),
        "duplicate_case_paths": by_case_path,
        "duplicate_case_audio": by_case_hash,
        "duplicate_case_fingerprints": by_case_fingerprint,
    }


def canonical_path(path: Path) -> str:
    return str(path.resolve())


def audio_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_fingerprint(path: Path, bins: int = 2048) -> str:
    with wave.open(str(path), "rb") as wav:
        samples = array("h")
        while data := wav.readframes(65_536):
            chunk = array("h")
            chunk.frombytes(data)
            if sys.byteorder != "little":
                chunk.byteswap()
            samples.extend(chunk)
    if not samples:
        return ""
    peak = max(abs(sample) for sample in samples)
    if peak == 0:
        return "silence"
    step = max(1, math.ceil(len(samples) / bins))
    quantized = bytearray()
    for start in range(0, len(samples), step):
        window = samples[start : start + step]
        average = sum(window) / len(window)
        scaled = int(round((average / peak) * 127))
        quantized.append(max(-127, min(127, scaled)) + 127)
    return hashlib.sha256(bytes(quantized)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def duplicate_groups(pairs) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for key, case_id in pairs:
        groups[key].append(case_id)
    return {
        key: sorted(case_ids)
        for key, case_ids in sorted(groups.items())
        if len(case_ids) > 1
    }


def audit_failures(
    *,
    templates: list[dict[str, object]],
    cases: list[dict[str, object]],
    positives: list[dict[str, object]],
    negatives: list[dict[str, object]],
    negative_audio_seconds: float,
    metadata: dict[str, set[str]],
    template_speakers: set[str],
    unauthorized_wake_cases: list[dict[str, object]],
    unauthorized_wake_speakers: set[str],
    unauthorized_template_speaker_overlaps: list[str],
    missing_metadata: dict[str, int],
    positive_annotation_gaps: dict[str, int],
    positive_annotation_errors: list[dict[str, object]],
    subgroups: dict[str, dict[str, dict[str, float | int]]],
    calibration_category_balance: dict[str, dict[str, float | int]],
    evaluation_category_balance: dict[str, dict[str, float | int]],
    calibration_speaker_positive_counts: dict[str, int],
    evaluation_speaker_positive_counts: dict[str, int],
    calibration_phrase_positive_counts: dict[str, int],
    evaluation_phrase_positive_counts: dict[str, int],
    template_phrase_counts: dict[str, dict[str, int]],
    calibration_subgroup_balance: dict[str, dict[str, dict[str, float | int]]],
    evaluation_subgroup_balance: dict[str, dict[str, dict[str, float | int]]],
    split_metrics: dict[str, dict[str, float | int]],
    split_coverage: dict[str, dict[str, set[str]]],
    leakage: dict[str, object],
    duplicate_case_ids: list[str],
    source_provenance: dict[str, object],
    corpus: dict[str, object],
    collection_ledger: dict[str, object],
    audio_quality_failures: list[str],
    allow_non_human_source: bool,
    allow_missing_corpus_identity: bool,
    allow_missing_collection_ledger: bool,
    plan: dict[str, int],
    min_speakers: int,
    min_environments: int,
    min_distances: int,
    min_devices: int,
    min_sessions: int,
    min_categories: int,
    min_template_speakers: int,
    required_wake_phrases: set[str],
    min_templates_per_speaker_phrase: int,
    min_unauthorized_wake_cases: int,
    min_unauthorized_wake_speakers: int,
    min_calibration_positive_cases: int,
    min_calibration_negative_audio_seconds: float,
    min_calibration_speaker_positive_cases: int,
    min_calibration_phrase_positive_cases: int,
    min_calibration_subgroup_positive_cases: int,
    min_calibration_subgroup_negative_audio_seconds: float,
    min_evaluation_positive_cases: int,
    min_evaluation_negative_audio_seconds: float,
    min_evaluation_speaker_positive_cases: int,
    min_evaluation_heldout_session_positive_cases_per_template_speaker: int,
    min_evaluation_phrase_positive_cases: int,
    min_evaluation_subgroup_positive_cases: int,
    min_evaluation_subgroup_negative_audio_seconds: float,
    min_evaluation_speakers: int,
    min_evaluation_environments: int,
    min_evaluation_distances: int,
    min_evaluation_devices: int,
    min_evaluation_sessions: int,
    min_evaluation_categories: int,
    min_positive_speakers: int,
    min_positive_environments: int,
    min_positive_distances: int,
    min_positive_devices: int,
    min_positive_sessions: int,
    min_positive_categories: int,
    min_negative_speakers: int,
    min_negative_environments: int,
    min_negative_distances: int,
    min_negative_devices: int,
    min_negative_sessions: int,
    min_negative_categories: int,
    required_categories: set[str],
    required_calibration_categories: set[str],
    required_evaluation_categories: set[str],
    required_positive_categories: set[str],
    required_negative_categories: set[str],
    min_calibration_category_positive_cases: int,
    min_evaluation_category_positive_cases: int,
    min_calibration_category_negative_audio_seconds: float,
    min_evaluation_category_negative_audio_seconds: float,
    required_wake_start_buckets: set[str],
    required_calibration_wake_start_buckets: set[str],
    required_evaluation_wake_start_buckets: set[str],
) -> list[str]:
    failures = []
    if not templates:
        failures.append("manifest has no enrollment templates")
    if duplicate_case_ids:
        failures.append(f"duplicate case ids: {', '.join(duplicate_case_ids)}")
    if not allow_missing_corpus_identity:
        if not corpus["id"]:
            failures.append("manifest corpus.id is required")
        if not corpus["version"]:
            failures.append("manifest corpus.version is required")
        if str(corpus.get("source", "")).strip().lower() != "human-recorded":
            failures.append("manifest corpus.source must be human-recorded")
    if not allow_missing_collection_ledger:
        failures.extend(collection_ledger_failures(collection_ledger, templates, cases))
    failures.extend(audio_quality_failures)
    if not allow_non_human_source:
        non_human_templates = source_provenance["non_human_templates"]
        non_human_cases = source_provenance["non_human_cases"]
        if non_human_templates:
            failures.append(
                f"{len(non_human_templates)} template(s) missing human-recorded source_type"
            )
        if non_human_cases:
            failures.append(f"{len(non_human_cases)} case(s) missing human-recorded source_type")
    positive_target = max(
        plan["minimum_precision_trials"], plan["minimum_positive_cases"]
    )
    if len(positives) < positive_target:
        failures.append(
            f"positive cases {len(positives)} below planned minimum {positive_target} "
            f"(precision trials {plan['minimum_precision_trials']}, "
            f"recall cases {plan['minimum_positive_cases']})"
        )
    if len(negatives) == 0:
        failures.append("manifest has no negative cases")
    if negative_audio_seconds < plan["minimum_negative_audio_seconds"]:
        failures.append(
            f"negative audio seconds {negative_audio_seconds:.3f} below planned minimum "
            f"{plan['minimum_negative_audio_seconds']}"
        )
    coverage_expectations = {
        "speakers": min_speakers,
        "environments": min_environments,
        "distances": min_distances,
        "devices": min_devices,
        "sessions": min_sessions,
        "categories": min_categories,
    }
    for key, minimum in coverage_expectations.items():
        if len(metadata[key]) < minimum:
            failures.append(f"{key} coverage {len(metadata[key])} below required {minimum}")
    missing_required_categories = sorted(required_categories - metadata["categories"])
    if missing_required_categories:
        failures.append(
            "missing required categories: " + ", ".join(missing_required_categories)
        )
    if len(template_speakers) < min_template_speakers:
        failures.append(
            f"template speaker coverage {len(template_speakers)} below required "
            f"{min_template_speakers}"
        )
    failures.extend(
        template_phrase_balance_failures(
            template_phrase_counts,
            template_speakers,
            required_wake_phrases,
            min_templates_per_speaker_phrase,
        )
    )
    failures.extend(label_semantic_failures(cases, required_wake_phrases))
    if len(unauthorized_wake_cases) < min_unauthorized_wake_cases:
        failures.append(
            f"unauthorized wake cases {len(unauthorized_wake_cases)} below required "
            f"{min_unauthorized_wake_cases}"
        )
    if len(unauthorized_wake_speakers) < min_unauthorized_wake_speakers:
        failures.append(
            f"unauthorized wake speakers {len(unauthorized_wake_speakers)} below required "
            f"{min_unauthorized_wake_speakers}"
        )
    if unauthorized_template_speaker_overlaps:
        failures.append(
            "unauthorized wake speakers overlap enrolled template speakers: "
            + ", ".join(unauthorized_template_speaker_overlaps)
        )
    calibration = split_metrics.get("calibration", {})
    evaluation = split_metrics.get("evaluation", {})
    missing_split_cases = split_metrics.get("missing", {}).get("cases", 0)
    if missing_split_cases:
        failures.append(f"{missing_split_cases} case(s) missing split")
    if calibration.get("positive_cases", 0) < min_calibration_positive_cases:
        failures.append(
            f"calibration positive cases {calibration.get('positive_cases', 0)} "
            f"below required {min_calibration_positive_cases}"
        )
    failures.extend(
        speaker_positive_balance_failures(
            "calibration",
            calibration_speaker_positive_counts,
            template_speakers,
            min_calibration_speaker_positive_cases,
        )
    )
    failures.extend(
        phrase_positive_balance_failures(
            "calibration",
            calibration_phrase_positive_counts,
            required_wake_phrases,
            min_calibration_phrase_positive_cases,
        )
    )
    if calibration.get("negative_audio_seconds", 0.0) < min_calibration_negative_audio_seconds:
        failures.append(
            f"calibration negative audio seconds "
            f"{calibration.get('negative_audio_seconds', 0.0):.3f} below required "
            f"{min_calibration_negative_audio_seconds}"
        )
    if evaluation.get("positive_cases", 0) < min_evaluation_positive_cases:
        failures.append(
            f"evaluation positive cases {evaluation.get('positive_cases', 0)} "
            f"below required {min_evaluation_positive_cases}"
        )
    failures.extend(
        speaker_positive_balance_failures(
            "evaluation",
            evaluation_speaker_positive_counts,
            template_speakers,
            min_evaluation_speaker_positive_cases,
        )
    )
    failures.extend(
        heldout_session_positive_failures(
            templates,
            [case for case in positives if str(case.get("split", "")).strip() == "evaluation"],
            min_evaluation_heldout_session_positive_cases_per_template_speaker,
        )
    )
    failures.extend(
        phrase_positive_balance_failures(
            "evaluation",
            evaluation_phrase_positive_counts,
            required_wake_phrases,
            min_evaluation_phrase_positive_cases,
        )
    )
    if evaluation.get("negative_audio_seconds", 0.0) < min_evaluation_negative_audio_seconds:
        failures.append(
            f"evaluation negative audio seconds "
            f"{evaluation.get('negative_audio_seconds', 0.0):.3f} below required "
            f"{min_evaluation_negative_audio_seconds}"
        )
    for key, minimum in {
        "speakers": min_evaluation_speakers,
        "environments": min_evaluation_environments,
        "distances": min_evaluation_distances,
        "devices": min_evaluation_devices,
        "sessions": min_evaluation_sessions,
        "categories": min_evaluation_categories,
    }.items():
        observed = len(split_coverage.get("evaluation", {}).get(key, set()))
        if observed < minimum:
            failures.append(f"evaluation {key} coverage {observed} below required {minimum}")
    missing_calibration_categories = sorted(
        required_calibration_categories
        - split_coverage.get("calibration", {}).get("categories", set())
    )
    if missing_calibration_categories:
        failures.append(
            "calibration split missing required categories: "
            + ", ".join(missing_calibration_categories)
        )
    missing_evaluation_categories = sorted(
        required_evaluation_categories
        - split_coverage.get("evaluation", {}).get("categories", set())
    )
    if missing_evaluation_categories:
        failures.append(
            "evaluation split missing required categories: "
            + ", ".join(missing_evaluation_categories)
        )
    failures.extend(
        category_balance_failures(
            "calibration",
            calibration_category_balance,
            required_positive_categories,
            required_negative_categories,
            min_calibration_category_positive_cases,
            min_calibration_category_negative_audio_seconds,
        )
    )
    failures.extend(
        category_balance_failures(
            "evaluation",
            evaluation_category_balance,
            required_positive_categories,
            required_negative_categories,
            min_evaluation_category_positive_cases,
            min_evaluation_category_negative_audio_seconds,
        )
    )
    aggregate_wake_start_buckets = wake_start_buckets(positives)
    missing_wake_start_buckets = sorted(required_wake_start_buckets - aggregate_wake_start_buckets)
    if missing_wake_start_buckets:
        failures.append(
            "missing required wake_start_ms buckets: " + ", ".join(missing_wake_start_buckets)
        )
    calibration_wake_start_buckets = wake_start_buckets(
        [case for case in positives if str(case.get("split", "")).strip() == "calibration"]
    )
    missing_calibration_wake_start_buckets = sorted(
        required_calibration_wake_start_buckets - calibration_wake_start_buckets
    )
    if missing_calibration_wake_start_buckets:
        failures.append(
            "calibration split missing required wake_start_ms buckets: "
            + ", ".join(missing_calibration_wake_start_buckets)
        )
    evaluation_wake_start_buckets = wake_start_buckets(
        [case for case in positives if str(case.get("split", "")).strip() == "evaluation"]
    )
    missing_evaluation_wake_start_buckets = sorted(
        required_evaluation_wake_start_buckets - evaluation_wake_start_buckets
    )
    if missing_evaluation_wake_start_buckets:
        failures.append(
            "evaluation split missing required wake_start_ms buckets: "
            + ", ".join(missing_evaluation_wake_start_buckets)
        )
    for key, coverage_key in [
        ("environment", "environments"),
        ("distance", "distances"),
        ("device", "devices"),
        ("session_id", "sessions"),
    ]:
        failures.extend(
            subgroup_balance_failures(
                "calibration",
                key,
                calibration_subgroup_balance[key],
                split_coverage.get("calibration", {}).get(coverage_key, set()),
                min_calibration_subgroup_positive_cases,
                min_calibration_subgroup_negative_audio_seconds,
            )
        )
        failures.extend(
            subgroup_balance_failures(
                "evaluation",
                key,
                evaluation_subgroup_balance[key],
                split_coverage.get("evaluation", {}).get(coverage_key, set()),
                min_evaluation_subgroup_positive_cases,
                min_evaluation_subgroup_negative_audio_seconds,
            )
        )
    positive_coverage_expectations = {
        "speakers": min_positive_speakers,
        "environments": min_positive_environments,
        "distances": min_positive_distances,
        "devices": min_positive_devices,
        "sessions": min_positive_sessions,
        "categories": min_positive_categories,
    }
    negative_coverage_expectations = {
        "speakers": min_negative_speakers,
        "environments": min_negative_environments,
        "distances": min_negative_distances,
        "devices": min_negative_devices,
        "sessions": min_negative_sessions,
        "categories": min_negative_categories,
    }
    for key, minimum in positive_coverage_expectations.items():
        observed = groups_with_positive_cases(subgroups[key])
        if observed < minimum:
            failures.append(f"{key} with positive cases {observed} below required {minimum}")
    for key, minimum in negative_coverage_expectations.items():
        observed = groups_with_negative_audio(subgroups[key])
        if observed < minimum:
            failures.append(f"{key} with negative audio {observed} below required {minimum}")
    for key, count in missing_metadata.items():
        if count:
            failures.append(f"{count} case(s) missing {key}")
    for key, count in positive_annotation_gaps.items():
        if count:
            failures.append(f"{count} positive case(s) missing {key}")
    if positive_annotation_errors:
        failures.append(
            f"{len(positive_annotation_errors)} positive case(s) have invalid expected_phrase or wake_start_ms"
        )
    if leakage["template_case_path_overlaps"]:
        failures.append(
            f"{len(leakage['template_case_path_overlaps'])} template/case path overlap(s)"
        )
    if leakage["template_case_audio_overlaps"]:
        failures.append(
            f"{len(leakage['template_case_audio_overlaps'])} template/case audio overlap(s)"
        )
    if leakage["template_case_fingerprint_overlaps"]:
        failures.append(
            f"{len(leakage['template_case_fingerprint_overlaps'])} template/case "
            "normalized audio fingerprint overlap(s)"
        )
    if leakage["duplicate_case_paths"]:
        failures.append(f"{len(leakage['duplicate_case_paths'])} duplicate case path group(s)")
    if leakage["duplicate_case_audio"]:
        failures.append(f"{len(leakage['duplicate_case_audio'])} duplicate case audio group(s)")
    if leakage["duplicate_case_fingerprints"]:
        failures.append(
            f"{len(leakage['duplicate_case_fingerprints'])} duplicate case normalized audio "
            "fingerprint group(s)"
        )
    return failures


def groups_with_positive_cases(groups: dict[str, dict[str, float | int]]) -> int:
    return sum(1 for group in groups.values() if group["positive_cases"] > 0)


def groups_with_negative_audio(groups: dict[str, dict[str, float | int]]) -> int:
    return sum(1 for group in groups.values() if group["negative_audio_seconds"] > 0.0)


def heldout_session_positive_failures(
    templates: list[dict[str, object]],
    evaluation_positives: list[dict[str, object]],
    minimum: int,
) -> list[str]:
    if minimum <= 0:
        return []
    template_sessions = sessions_by_speaker(templates)
    counts = heldout_session_positive_counts(template_sessions, evaluation_positives)
    failures = []
    for speaker in sorted(template_sessions):
        observed = counts.get(speaker, 0)
        if observed < minimum:
            sessions = ", ".join(sorted(template_sessions[speaker])) or "missing"
            failures.append(
                f"evaluation speaker {speaker} held-out-session positive cases {observed} "
                f"below required {minimum} (template sessions: {sessions})"
            )
    return failures


def heldout_session_positive_counts(
    template_sessions: dict[str, set[str]],
    evaluation_positives: list[dict[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for case in evaluation_positives:
        speaker = str(case.get("speaker_id", "")).strip()
        session = str(case.get("session_id", "")).strip()
        if not speaker or not session:
            continue
        if session not in template_sessions.get(speaker, set()):
            counts[speaker] += 1
    return dict(counts)


def sessions_by_speaker(rows: list[dict[str, object]]) -> dict[str, set[str]]:
    sessions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        speaker = str(row.get("speaker_id", "")).strip()
        session = str(row.get("session_id", "")).strip()
        if speaker and session:
            sessions[speaker].add(session)
    return dict(sessions)


def category_balance_failures(
    split: str,
    balance: dict[str, dict[str, float | int]],
    positive_categories: set[str],
    negative_categories: set[str],
    min_positive_cases: int,
    min_negative_audio_seconds: float,
) -> list[str]:
    failures = []
    for category in sorted(positive_categories):
        observed = int(balance.get(category, {}).get("positive_cases", 0))
        if observed < min_positive_cases:
            failures.append(
                f"{split} category {category} positive cases {observed} below required "
                f"{min_positive_cases}"
            )
    for category in sorted(negative_categories):
        observed = float(balance.get(category, {}).get("negative_audio_seconds", 0.0))
        if observed < min_negative_audio_seconds:
            failures.append(
                f"{split} category {category} negative audio seconds {observed:.3f} "
                f"below required {min_negative_audio_seconds}"
            )
    return failures


def speaker_positive_balance_failures(
    split: str,
    counts: dict[str, int],
    template_speakers: set[str],
    minimum: int,
) -> list[str]:
    failures = []
    for speaker in sorted(template_speakers):
        observed = counts.get(speaker, 0)
        if observed < minimum:
            failures.append(
                f"{split} speaker {speaker} positive cases {observed} below required {minimum}"
            )
    return failures


def phrase_positive_balance_failures(
    split: str,
    counts: dict[str, int],
    required_phrases: set[str],
    minimum: int,
) -> list[str]:
    failures = []
    for phrase in sorted(required_phrases):
        observed = counts.get(phrase, 0)
        if observed < minimum:
            failures.append(
                f"{split} phrase {phrase!r} positive cases {observed} below required {minimum}"
            )
    return failures


def template_phrase_balance_failures(
    balance: dict[str, dict[str, int]],
    template_speakers: set[str],
    required_phrases: set[str],
    minimum: int,
) -> list[str]:
    failures = []
    for speaker in sorted(template_speakers):
        for phrase in sorted(required_phrases):
            observed = balance.get(speaker, {}).get(phrase, 0)
            if observed < minimum:
                failures.append(
                    f"template speaker {speaker} phrase {phrase!r} templates {observed} "
                    f"below required {minimum}"
                )
    return failures


def label_semantic_failures(
    cases: list[dict[str, object]], required_wake_phrases: set[str]
) -> list[str]:
    failures = []
    required_phrases = {phrase.lower() for phrase in required_wake_phrases}
    positive_bad_expected = []
    unauthorized_missing_phrase = []
    negative_with_expected_phrase = []
    negative_wake_phrase_leaks = []
    for case in cases:
        case_id = str(case.get("id") or case.get("path") or "case")
        text = normalized_text(case.get("text"))
        expected = normalized_text(case.get("expected_phrase"))
        category = normalized_text(case.get("category"))
        contains_required_phrase = contains_phrase(text, required_phrases)
        if case.get("should_wake") is True:
            if expected and expected not in required_phrases:
                positive_bad_expected.append(case_id)
        elif case.get("should_wake") is False:
            if expected:
                negative_with_expected_phrase.append(case_id)
            if category == "unauthorized-wake":
                if not contains_required_phrase:
                    unauthorized_missing_phrase.append(case_id)
            elif contains_required_phrase:
                negative_wake_phrase_leaks.append(case_id)
    if positive_bad_expected:
        failures.append(
            f"{len(positive_bad_expected)} positive case(s) use expected_phrase outside required wake phrases"
        )
    if unauthorized_missing_phrase:
        failures.append(
            f"{len(unauthorized_missing_phrase)} unauthorized-wake negative case(s) do not contain a required wake phrase"
        )
    if negative_with_expected_phrase:
        failures.append(
            f"{len(negative_with_expected_phrase)} negative case(s) should not set expected_phrase"
        )
    if negative_wake_phrase_leaks:
        failures.append(
            f"{len(negative_wake_phrase_leaks)} non-unauthorized negative case(s) contain a required wake phrase"
        )
    return failures


def normalized_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def contains_phrase(text: str, phrases: set[str]) -> bool:
    text_tokens = text.split()
    for phrase in phrases:
        phrase_tokens = phrase.split()
        if phrase_tokens and contains_token_sequence(text_tokens, phrase_tokens):
            return True
    return False


def contains_token_sequence(tokens: list[str], phrase_tokens: list[str]) -> bool:
    if len(phrase_tokens) > len(tokens):
        return False
    return any(
        tokens[index : index + len(phrase_tokens)] == phrase_tokens
        for index in range(len(tokens) - len(phrase_tokens) + 1)
    )


def subgroup_balance_failures(
    split: str,
    key: str,
    balance: dict[str, dict[str, float | int]],
    required_values: set[str],
    min_positive_cases: int,
    min_negative_audio_seconds: float,
) -> list[str]:
    failures = []
    for value in sorted(required_values):
        group = balance.get(value, {})
        positive_cases = int(group.get("positive_cases", 0))
        negative_audio_seconds = float(group.get("negative_audio_seconds", 0.0))
        if positive_cases < min_positive_cases:
            failures.append(
                f"{split} {key} {value} positive cases {positive_cases} below required "
                f"{min_positive_cases}"
            )
        if negative_audio_seconds < min_negative_audio_seconds:
            failures.append(
                f"{split} {key} {value} negative audio seconds {negative_audio_seconds:.3f} "
                f"below required {min_negative_audio_seconds}"
            )
    return failures


def print_text_report(report: dict[str, object]) -> None:
    status = "ready" if report["ready"] else "not_ready"
    print(f"status={status}")
    print(f"templates={report['templates']}")
    print(f"template_audio_seconds={report['template_audio_seconds']:.3f}")
    print(f"cases={report['cases']}")
    print(f"positive_cases={report['positive_cases']}")
    print(f"negative_cases={report['negative_cases']}")
    print(f"negative_audio_seconds={report['negative_audio_seconds']:.3f}")
    print(f"negative_audio_hours={report['negative_audio_hours']:.3f}")
    metadata = report["metadata"]
    for key in ["speakers", "environments", "distances", "devices", "sessions", "categories"]:
        print(f"{key}={metadata[key]['count']}")
    speaker_gating = report["speaker_gating"]
    print(f"template_speakers={speaker_gating['template_speakers']}")
    print(f"unauthorized_wake_cases={speaker_gating['unauthorized_wake_cases']}")
    print(f"unauthorized_wake_speakers={speaker_gating['unauthorized_wake_speakers']}")
    print(
        "unauthorized_template_speaker_overlaps="
        f"{len(speaker_gating['unauthorized_template_speaker_overlaps'])}"
    )
    source_provenance = report["source_provenance"]
    print(f"template_source_counts={source_provenance['template_source_counts']}")
    print(f"case_source_counts={source_provenance['case_source_counts']}")
    template_quality = report["audio_quality"]["templates"]
    case_quality = report["audio_quality"]["cases"]
    if template_quality:
        print(f"template_min_rms={min(item['rms'] for item in template_quality):.6f}")
        print(f"template_max_clipped_ratio={max(item['clipped_ratio'] for item in template_quality):.6f}")
    if case_quality:
        print(f"case_min_rms={min(item['rms'] for item in case_quality):.6f}")
        print(f"case_max_clipped_ratio={max(item['clipped_ratio'] for item in case_quality):.6f}")
    for split, metrics in report["splits"].items():
        print(
            f"split_{split}=cases:{metrics['cases']},"
            f"positive:{metrics['positive_cases']},"
            f"negative_seconds:{metrics['negative_audio_seconds']:.3f}"
        )
    evaluation_coverage = report["split_coverage"]["evaluation"]
    for key in ["speakers", "environments", "distances", "devices", "sessions", "categories"]:
        print(f"evaluation_{key}={evaluation_coverage[key]['count']}")
    subgroups = report["subgroups"]
    for key in ["speakers", "environments", "distances", "devices", "sessions", "categories"]:
        print(f"{key}_with_positive_cases={groups_with_positive_cases(subgroups[key])}")
        print(f"{key}_with_negative_audio={groups_with_negative_audio(subgroups[key])}")
    for key, value in report["planning_targets"].items():
        print(f"{key}={value}")
    leakage = report["leakage"]
    print(f"template_case_path_overlaps={len(leakage['template_case_path_overlaps'])}")
    print(f"template_case_audio_overlaps={len(leakage['template_case_audio_overlaps'])}")
    print(
        "template_case_fingerprint_overlaps="
        f"{len(leakage['template_case_fingerprint_overlaps'])}"
    )
    print(f"duplicate_case_paths={len(leakage['duplicate_case_paths'])}")
    print(f"duplicate_case_audio={len(leakage['duplicate_case_audio'])}")
    print(f"duplicate_case_fingerprints={len(leakage['duplicate_case_fingerprints'])}")
    failures = report["failures"]
    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    raise SystemExit(main())
