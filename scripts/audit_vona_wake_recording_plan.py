#!/usr/bin/env python3
"""Audit a vona-wake recording-plan CSV before collecting audio."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from plan_vona_wake_corpus import required_negative_audio_hours, required_wilson_trials


REQUIRED_COLUMNS = {
    "role",
    "id",
    "path",
    "phrase",
    "should_wake",
    "text",
    "expected_phrase",
    "wake_start_ms",
    "speaker_id",
    "environment",
    "distance",
    "device",
    "session_id",
    "category",
    "source_type",
    "split",
    "planned_duration_s",
}

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
    parser.add_argument("csv", type=Path, help="recording-plan CSV")
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
    parser.add_argument("--min-evaluation-speakers", type=int)
    parser.add_argument("--min-evaluation-environments", type=int)
    parser.add_argument("--min-evaluation-distances", type=int)
    parser.add_argument("--min-evaluation-devices", type=int)
    parser.add_argument("--min-evaluation-sessions", type=int)
    parser.add_argument("--min-evaluation-categories", type=int)
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
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero if plan is not ready")
    args = parser.parse_args()

    columns, rows = read_rows(args.csv)
    templates = [row for row in rows if row.get("role", "").lower() == "template"]
    cases = [row for row in rows if row.get("role", "").lower() == "case"]
    positives = [row for row in cases if parse_bool(row.get("should_wake")) is True]
    negatives = [row for row in cases if parse_bool(row.get("should_wake")) is False]
    evaluation = [row for row in cases if row.get("split") == "evaluation"]
    calibration = [row for row in cases if row.get("split") == "calibration"]
    plan_targets = {
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
    metadata = {
        "speakers": values(cases, "speaker_id"),
        "environments": values(cases, "environment"),
        "distances": values(cases, "distance"),
        "devices": values(cases, "device"),
        "sessions": values(cases, "session_id"),
        "categories": values(cases, "category"),
    }
    template_speakers = values(templates, "speaker_id")
    unauthorized_wake_cases = [
        row
        for row in negatives
        if row.get("category") == "unauthorized-wake"
    ]
    unauthorized_wake_speakers = values(unauthorized_wake_cases, "speaker_id")
    evaluation_summary = summarize_cases(evaluation)
    calibration_summary = summarize_cases(calibration)
    split_coverage = {
        "calibration": {
            "speakers": values(calibration, "speaker_id"),
            "environments": values(calibration, "environment"),
            "distances": values(calibration, "distance"),
            "devices": values(calibration, "device"),
            "sessions": values(calibration, "session_id"),
            "categories": values(calibration, "category"),
        },
        "evaluation": {
            "speakers": values(evaluation, "speaker_id"),
            "environments": values(evaluation, "environment"),
            "distances": values(evaluation, "distance"),
            "devices": values(evaluation, "device"),
            "sessions": values(evaluation, "session_id"),
            "categories": values(evaluation, "category"),
        },
    }
    failures = failures_for_plan(
        columns=columns,
        rows=rows,
        templates=templates,
        cases=cases,
        positives=positives,
        negatives=negatives,
        metadata=metadata,
        template_speakers=template_speakers,
        unauthorized_wake_cases=unauthorized_wake_cases,
        unauthorized_wake_speakers=unauthorized_wake_speakers,
        evaluation_summary=evaluation_summary,
        calibration_summary=calibration_summary,
        split_coverage=split_coverage,
        plan_targets=plan_targets,
        args=args,
        required_categories=csv_set(args.required_categories),
        required_calibration_categories=csv_set(args.required_calibration_categories),
        required_evaluation_categories=csv_set(args.required_evaluation_categories),
        required_positive_categories=csv_set(args.required_positive_categories),
        required_negative_categories=csv_set(args.required_negative_categories),
        required_wake_start_buckets=csv_set(args.required_wake_start_buckets),
        required_calibration_wake_start_buckets=csv_set(
            args.required_calibration_wake_start_buckets
        ),
        required_evaluation_wake_start_buckets=csv_set(
            args.required_evaluation_wake_start_buckets
        ),
        required_wake_phrases=csv_set(args.required_wake_phrases),
    )
    report = {
        "csv_path": str(args.csv),
        "ready": not failures,
        "rows": len(rows),
        "templates": len(templates),
        "cases": len(cases),
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
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
            "unauthorized_template_speaker_overlaps": sorted(
                unauthorized_wake_speakers & template_speakers
            ),
        },
        "splits": {
            "calibration": calibration_summary,
            "evaluation": evaluation_summary,
            "missing": summarize_cases([row for row in cases if not row.get("split")]),
        },
        "category_balance": {
            "calibration": category_balance(calibration),
            "evaluation": category_balance(evaluation),
        },
        "speaker_positive_balance": {
            "calibration": speaker_positive_counts(calibration),
            "evaluation": speaker_positive_counts(evaluation),
            "evaluation_heldout_session": heldout_session_positive_counts(
                sessions_by_speaker(templates),
                [row for row in positives if row.get("split") == "evaluation"],
            ),
        },
        "phrase_positive_balance": {
            "calibration": phrase_positive_counts(calibration),
            "evaluation": phrase_positive_counts(evaluation),
        },
        "template_phrase_balance": template_phrase_balance(templates),
        "subgroup_balance": {
            "calibration": {
                key: subgroup_balance(calibration, key)
                for key in ["environment", "distance", "device", "session_id"]
            },
            "evaluation": {
                key: subgroup_balance(evaluation, key)
                for key in ["environment", "distance", "device", "session_id"]
            },
        },
        "wake_start_buckets": {
            "aggregate": sorted(wake_start_buckets(positives)),
            "calibration": sorted(wake_start_buckets([row for row in positives if row.get("split") == "calibration"])),
            "evaluation": sorted(wake_start_buckets([row for row in positives if row.get("split") == "evaluation"])),
        },
        "split_coverage": {
            split: {
                key: {"count": len(found), "values": sorted(found)}
                for key, found in coverage.items()
            }
            for split, coverage in split_coverage.items()
        },
        "source_type_counts": dict(Counter(row.get("source_type") or "missing" for row in rows)),
        "planning_targets": plan_targets,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report)
    return 1 if args.enforce and failures else 0


def read_rows(path: Path) -> tuple[set[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    return columns, rows


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def values(rows: list[dict[str, str]], key: str) -> set[str]:
    return {row.get(key, "").strip() for row in rows if row.get(key, "").strip()}


def csv_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def duration(row: dict[str, str]) -> float:
    try:
        return float(row.get("planned_duration_s") or 0)
    except ValueError:
        return 0.0


def summarize_cases(rows: list[dict[str, str]]) -> dict[str, float | int]:
    positives = [row for row in rows if parse_bool(row.get("should_wake")) is True]
    negatives = [row for row in rows if parse_bool(row.get("should_wake")) is False]
    return {
        "cases": len(rows),
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "positive_audio_seconds": sum(duration(row) for row in positives),
        "negative_audio_seconds": sum(duration(row) for row in negatives),
    }


def category_balance(rows: list[dict[str, str]]) -> dict[str, dict[str, float | int]]:
    balance: dict[str, dict[str, float | int]] = {}
    for row in rows:
        category = row.get("category", "").strip()
        if not category:
            continue
        entry = balance.setdefault(
            category,
            {
                "cases": 0,
                "positive_cases": 0,
                "negative_cases": 0,
                "positive_audio_seconds": 0.0,
                "negative_audio_seconds": 0.0,
            },
        )
        entry["cases"] += 1
        if parse_bool(row.get("should_wake")) is True:
            entry["positive_cases"] += 1
            entry["positive_audio_seconds"] += duration(row)
        elif parse_bool(row.get("should_wake")) is False:
            entry["negative_cases"] += 1
            entry["negative_audio_seconds"] += duration(row)
    return dict(sorted(balance.items()))


def speaker_positive_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if parse_bool(row.get("should_wake")) is True and row.get("speaker_id", "").strip():
            counts[row["speaker_id"].strip()] += 1
    return dict(sorted(counts.items()))


def phrase_positive_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if parse_bool(row.get("should_wake")) is True and row.get("expected_phrase", "").strip():
            counts[row["expected_phrase"].strip()] += 1
    return dict(sorted(counts.items()))


def template_phrase_balance(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    balance: dict[str, Counter[str]] = {}
    for row in rows:
        speaker = row.get("speaker_id", "").strip()
        phrase = row.get("phrase", "").strip()
        if speaker and phrase:
            balance.setdefault(speaker, Counter())[phrase] += 1
    return {speaker: dict(sorted(counts.items())) for speaker, counts in sorted(balance.items())}


def subgroup_balance(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, float | int]]:
    balance: dict[str, dict[str, float | int]] = {}
    for row in rows:
        value = row.get(key, "").strip()
        if not value:
            continue
        entry = balance.setdefault(
            value,
            {
                "cases": 0,
                "positive_cases": 0,
                "negative_cases": 0,
                "positive_audio_seconds": 0.0,
                "negative_audio_seconds": 0.0,
            },
        )
        entry["cases"] += 1
        if parse_bool(row.get("should_wake")) is True:
            entry["positive_cases"] += 1
            entry["positive_audio_seconds"] += duration(row)
        elif parse_bool(row.get("should_wake")) is False:
            entry["negative_cases"] += 1
            entry["negative_audio_seconds"] += duration(row)
    return dict(sorted(balance.items()))


def wake_start_buckets(rows: list[dict[str, str]]) -> set[str]:
    buckets = set()
    for row in rows:
        try:
            wake_start_ms = int(row.get("wake_start_ms", ""))
        except ValueError:
            continue
        buckets.add(wake_start_bucket(wake_start_ms))
    return buckets


def wake_start_bucket(wake_start_ms: int) -> str:
    if wake_start_ms <= 250:
        return "early"
    if wake_start_ms <= 1500:
        return "mid"
    return "late"


def failures_for_plan(
    *,
    columns: set[str],
    rows: list[dict[str, str]],
    templates: list[dict[str, str]],
    cases: list[dict[str, str]],
    positives: list[dict[str, str]],
    negatives: list[dict[str, str]],
    metadata: dict[str, set[str]],
    template_speakers: set[str],
    unauthorized_wake_cases: list[dict[str, str]],
    unauthorized_wake_speakers: set[str],
    evaluation_summary: dict[str, float | int],
    calibration_summary: dict[str, float | int],
    split_coverage: dict[str, dict[str, set[str]]],
    plan_targets: dict[str, int],
    args,
    required_categories: set[str],
    required_calibration_categories: set[str],
    required_evaluation_categories: set[str],
    required_positive_categories: set[str],
    required_negative_categories: set[str],
    required_wake_start_buckets: set[str],
    required_calibration_wake_start_buckets: set[str],
    required_evaluation_wake_start_buckets: set[str],
    required_wake_phrases: set[str],
) -> list[str]:
    failures: list[str] = []
    missing_columns = REQUIRED_COLUMNS - columns
    if missing_columns:
        failures.append(f"missing columns: {', '.join(sorted(missing_columns))}")
    duplicate_ids = duplicate_values([row.get("id", "") for row in rows])
    if duplicate_ids:
        failures.append(f"duplicate ids: {', '.join(duplicate_ids)}")
    duplicate_paths = duplicate_values([row.get("path", "") for row in rows])
    if duplicate_paths:
        failures.append(f"duplicate paths: {', '.join(duplicate_paths[:10])}")
    if not templates:
        failures.append("plan has no template rows")
    if not cases:
        failures.append("plan has no case rows")
    if not positives:
        failures.append("plan has no positive cases")
    if not negatives:
        failures.append("plan has no negative cases")
    for key, minimum in {
        "speakers": args.min_speakers,
        "environments": args.min_environments,
        "distances": args.min_distances,
        "devices": args.min_devices,
        "sessions": args.min_sessions,
        "categories": args.min_categories,
    }.items():
        if len(metadata[key]) < minimum:
            failures.append(f"{key} coverage {len(metadata[key])} below required {minimum}")
    missing_required_categories = sorted(required_categories - metadata["categories"])
    if missing_required_categories:
        failures.append(
            "missing required categories: " + ", ".join(missing_required_categories)
        )
    if len(template_speakers) < args.min_template_speakers:
        failures.append(
            f"template speaker coverage {len(template_speakers)} below required "
            f"{args.min_template_speakers}"
        )
    failures.extend(
        template_phrase_balance_failures(
            template_phrase_balance(templates),
            template_speakers,
            required_wake_phrases,
            args.min_templates_per_speaker_phrase,
        )
    )
    failures.extend(label_semantic_failures(cases, required_wake_phrases))
    if len(unauthorized_wake_cases) < args.min_unauthorized_wake_cases:
        failures.append(
            f"unauthorized wake cases {len(unauthorized_wake_cases)} below required "
            f"{args.min_unauthorized_wake_cases}"
        )
    if len(unauthorized_wake_speakers) < args.min_unauthorized_wake_speakers:
        failures.append(
            f"unauthorized wake speakers {len(unauthorized_wake_speakers)} below required "
            f"{args.min_unauthorized_wake_speakers}"
        )
    speaker_overlap = sorted(unauthorized_wake_speakers & template_speakers)
    if speaker_overlap:
        failures.append(
            "unauthorized wake speakers overlap enrolled template speakers: "
            + ", ".join(speaker_overlap)
        )
    if calibration_summary["positive_cases"] < args.min_calibration_positive_cases:
        failures.append(
            f"calibration positive cases {calibration_summary['positive_cases']} below required "
            f"{args.min_calibration_positive_cases}"
        )
    failures.extend(
        speaker_positive_balance_failures(
            "calibration",
            speaker_positive_counts([row for row in positives if row.get("split") == "calibration"]),
            template_speakers,
            args.min_calibration_speaker_positive_cases,
        )
    )
    failures.extend(
        phrase_positive_balance_failures(
            "calibration",
            phrase_positive_counts([row for row in positives if row.get("split") == "calibration"]),
            required_wake_phrases,
            args.min_calibration_phrase_positive_cases,
        )
    )
    if calibration_summary["negative_audio_seconds"] < args.min_calibration_negative_audio_seconds:
        failures.append(
            f"calibration negative audio seconds "
            f"{calibration_summary['negative_audio_seconds']:.3f} below required "
            f"{args.min_calibration_negative_audio_seconds}"
        )
    evaluation_positive_target = max(
        plan_targets["minimum_precision_trials"], plan_targets["minimum_positive_cases"]
    )
    if evaluation_summary["positive_cases"] < evaluation_positive_target:
        failures.append(
            f"evaluation positive cases {evaluation_summary['positive_cases']} below required "
            f"{evaluation_positive_target}"
        )
    failures.extend(
        speaker_positive_balance_failures(
            "evaluation",
            speaker_positive_counts([row for row in positives if row.get("split") == "evaluation"]),
            template_speakers,
            args.min_evaluation_speaker_positive_cases,
        )
    )
    failures.extend(
        heldout_session_positive_failures(
            templates,
            [row for row in positives if row.get("split") == "evaluation"],
            args.min_evaluation_heldout_session_positive_cases_per_template_speaker,
        )
    )
    failures.extend(
        phrase_positive_balance_failures(
            "evaluation",
            phrase_positive_counts([row for row in positives if row.get("split") == "evaluation"]),
            required_wake_phrases,
            args.min_evaluation_phrase_positive_cases,
        )
    )
    if evaluation_summary["negative_audio_seconds"] < plan_targets["minimum_negative_audio_seconds"]:
        failures.append(
            f"evaluation negative audio seconds "
            f"{evaluation_summary['negative_audio_seconds']:.3f} below required "
            f"{plan_targets['minimum_negative_audio_seconds']}"
        )
    for key, minimum in {
        "speakers": args.min_evaluation_speakers or args.min_speakers,
        "environments": args.min_evaluation_environments or args.min_environments,
        "distances": args.min_evaluation_distances or args.min_distances,
        "devices": args.min_evaluation_devices or args.min_devices,
        "sessions": args.min_evaluation_sessions or args.min_sessions,
        "categories": args.min_evaluation_categories or args.min_categories,
    }.items():
        observed = len(split_coverage["evaluation"][key])
        if observed < minimum:
            failures.append(f"evaluation {key} coverage {observed} below required {minimum}")
    missing_calibration_categories = sorted(
        required_calibration_categories - split_coverage["calibration"]["categories"]
    )
    if missing_calibration_categories:
        failures.append(
            "calibration split missing required categories: "
            + ", ".join(missing_calibration_categories)
        )
    missing_evaluation_categories = sorted(
        required_evaluation_categories - split_coverage["evaluation"]["categories"]
    )
    if missing_evaluation_categories:
        failures.append(
            "evaluation split missing required categories: "
            + ", ".join(missing_evaluation_categories)
        )
    failures.extend(
        category_balance_failures(
            "calibration",
            category_balance([row for row in cases if row.get("split") == "calibration"]),
            required_positive_categories,
            required_negative_categories,
            args.min_calibration_category_positive_cases,
            args.min_calibration_category_negative_audio_seconds,
        )
    )
    failures.extend(
        category_balance_failures(
            "evaluation",
            category_balance([row for row in cases if row.get("split") == "evaluation"]),
            required_positive_categories,
            required_negative_categories,
            args.min_evaluation_category_positive_cases,
            args.min_evaluation_category_negative_audio_seconds,
        )
    )
    aggregate_wake_start_buckets = wake_start_buckets(positives)
    missing_wake_start_buckets = sorted(required_wake_start_buckets - aggregate_wake_start_buckets)
    if missing_wake_start_buckets:
        failures.append(
            "missing required wake_start_ms buckets: " + ", ".join(missing_wake_start_buckets)
        )
    calibration_wake_start_buckets = wake_start_buckets(
        [row for row in positives if row.get("split") == "calibration"]
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
        [row for row in positives if row.get("split") == "evaluation"]
    )
    missing_evaluation_wake_start_buckets = sorted(
        required_evaluation_wake_start_buckets - evaluation_wake_start_buckets
    )
    if missing_evaluation_wake_start_buckets:
        failures.append(
            "evaluation split missing required wake_start_ms buckets: "
            + ", ".join(missing_evaluation_wake_start_buckets)
        )
    for key in ["environment", "distance", "device", "session_id"]:
        coverage_key = subgroup_coverage_key(key)
        failures.extend(
            subgroup_balance_failures(
                "calibration",
                key,
                subgroup_balance([row for row in cases if row.get("split") == "calibration"], key),
                split_coverage["calibration"][coverage_key],
                args.min_calibration_subgroup_positive_cases,
                args.min_calibration_subgroup_negative_audio_seconds,
            )
        )
        failures.extend(
            subgroup_balance_failures(
                "evaluation",
                key,
                subgroup_balance([row for row in cases if row.get("split") == "evaluation"], key),
                split_coverage["evaluation"][coverage_key],
                args.min_evaluation_subgroup_positive_cases,
                args.min_evaluation_subgroup_negative_audio_seconds,
            )
        )
    non_human = [
        row.get("id", f"row-{index}")
        for index, row in enumerate(rows, start=2)
        if row.get("source_type") != "human-recorded"
    ]
    if non_human:
        failures.append(f"{len(non_human)} row(s) missing source_type=human-recorded")
    invalid_splits = [
        row.get("id", f"row-{index}")
        for index, row in enumerate(rows, start=2)
        if row.get("split") not in {"enrollment", "calibration", "evaluation"}
    ]
    if invalid_splits:
        failures.append(f"{len(invalid_splits)} row(s) missing valid split")
    missing_positive_annotations = [
        row.get("id", "")
        for row in positives
        if not row.get("expected_phrase") or row.get("wake_start_ms") == ""
    ]
    if missing_positive_annotations:
        failures.append(
            f"{len(missing_positive_annotations)} positive case(s) missing expected_phrase or wake_start_ms"
        )
    invalid_positive_annotations = [
        row.get("id", "")
        for row in positives
        if positive_annotation_error(row) is not None
    ]
    if invalid_positive_annotations:
        failures.append(
            f"{len(invalid_positive_annotations)} positive case(s) have invalid expected_phrase or wake_start_ms"
        )
    bad_durations = [
        row.get("id", f"row-{index}")
        for index, row in enumerate(rows, start=2)
        if duration(row) <= 0
    ]
    if bad_durations:
        failures.append(f"{len(bad_durations)} row(s) missing positive planned_duration_s")
    return failures


def positive_annotation_error(row: dict[str, str]) -> str | None:
    expected = row.get("expected_phrase", "").strip()
    text = row.get("text", "").strip()
    if expected and text and expected not in text:
        return "expected_phrase is not contained in text"
    try:
        wake_start_ms = int(row.get("wake_start_ms", ""))
    except ValueError:
        return "wake_start_ms is not an integer"
    if wake_start_ms < 0:
        return "wake_start_ms is negative"
    planned_ms = duration(row) * 1000.0
    if planned_ms > 0 and wake_start_ms >= planned_ms:
        return "wake_start_ms is outside planned duration"
    return None


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


def heldout_session_positive_failures(
    templates: list[dict[str, str]],
    evaluation_positives: list[dict[str, str]],
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
    evaluation_positives: list[dict[str, str]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in evaluation_positives:
        speaker = row.get("speaker_id", "").strip()
        session = row.get("session_id", "").strip()
        if not speaker or not session:
            continue
        if session not in template_sessions.get(speaker, set()):
            counts[speaker] += 1
    return dict(counts)


def sessions_by_speaker(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    sessions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        speaker = row.get("speaker_id", "").strip()
        session = row.get("session_id", "").strip()
        if speaker and session:
            sessions[speaker].add(session)
    return dict(sessions)


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
    rows: list[dict[str, str]], required_wake_phrases: set[str]
) -> list[str]:
    failures = []
    required_phrases = {phrase.lower() for phrase in required_wake_phrases}
    positive_bad_expected = []
    unauthorized_missing_phrase = []
    negative_with_expected_phrase = []
    negative_wake_phrase_leaks = []
    for row in rows:
        row_id = row.get("id") or row.get("path") or "row"
        text = normalized_text(row.get("text"))
        expected = normalized_text(row.get("expected_phrase"))
        category = normalized_text(row.get("category"))
        contains_required_phrase = contains_phrase(text, required_phrases)
        should_wake = parse_bool(row.get("should_wake"))
        if should_wake is True:
            if expected and expected not in required_phrases:
                positive_bad_expected.append(row_id)
        elif should_wake is False:
            if expected:
                negative_with_expected_phrase.append(row_id)
            if category == "unauthorized-wake":
                if not contains_required_phrase:
                    unauthorized_missing_phrase.append(row_id)
            elif contains_required_phrase:
                negative_wake_phrase_leaks.append(row_id)
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


def normalized_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


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


def subgroup_coverage_key(key: str) -> str:
    return {
        "environment": "environments",
        "distance": "distances",
        "device": "devices",
        "session_id": "sessions",
    }[key]


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def print_text_report(report: dict[str, object]) -> None:
    print(f"status={'ready' if report['ready'] else 'not_ready'}")
    print(f"rows={report['rows']}")
    print(f"templates={report['templates']}")
    print(f"cases={report['cases']}")
    print(f"positive_cases={report['positive_cases']}")
    print(f"negative_cases={report['negative_cases']}")
    for key in ["speakers", "environments", "distances", "devices", "sessions", "categories"]:
        print(f"{key}={report['metadata'][key]['count']}")
    for split in ["calibration", "evaluation", "missing"]:
        summary = report["splits"][split]
        print(
            f"split_{split}=cases:{summary['cases']},positive:{summary['positive_cases']},"
            f"negative_seconds:{summary['negative_audio_seconds']:.3f}"
        )
    evaluation_coverage = report["split_coverage"]["evaluation"]
    for key in ["speakers", "environments", "distances", "devices", "sessions", "categories"]:
        print(f"evaluation_{key}={evaluation_coverage[key]['count']}")
    print(f"template_speakers={report['speaker_gating']['template_speakers']}")
    print(f"unauthorized_wake_cases={report['speaker_gating']['unauthorized_wake_cases']}")
    print(f"unauthorized_wake_speakers={report['speaker_gating']['unauthorized_wake_speakers']}")
    print(f"source_type_counts={report['source_type_counts']}")
    for key, value in report["planning_targets"].items():
        print(f"{key}={value}")
    failures = report["failures"]
    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    raise SystemExit(main())
