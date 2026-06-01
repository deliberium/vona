#!/usr/bin/env python3
"""Check saved vona-wake audit/evaluation reports for release-grade acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--real-report", type=Path, required=True)
    parser.add_argument("--threshold-report", type=Path)
    parser.add_argument(
        "--require-threshold-selection",
        action="store_true",
        help="Require a passing threshold report selected from calibration and verified on evaluation",
    )
    parser.add_argument("--min-precision", type=float, default=0.98)
    parser.add_argument("--min-recall", type=float, default=0.98)
    parser.add_argument("--min-precision-lower-bound", type=float, default=0.95)
    parser.add_argument("--min-recall-lower-bound", type=float, default=0.95)
    parser.add_argument("--max-false-wakes-per-hour", type=float, default=0.05)
    parser.add_argument("--max-false-wakes-per-hour-upper-bound", type=float, default=0.05)
    parser.add_argument("--max-false-positives", type=int, default=0)
    parser.add_argument("--max-false-negatives", type=int, default=0)
    parser.add_argument("--max-repeated-positive-wake-events", type=int, default=0)
    parser.add_argument("--max-phrase-mismatches", type=int, default=0)
    parser.add_argument("--max-detection-latency-ms", type=int, default=1500)
    parser.add_argument("--min-threshold-sweep-points", type=int, default=5)
    parser.add_argument("--min-calibration-passing-threshold-points", type=int, default=2)
    parser.add_argument("--min-evaluation-passing-threshold-points", type=int, default=1)
    parser.add_argument(
        "--min-evaluation-subgroup-precision",
        type=float,
        help="Minimum precision for each evaluation subgroup with precision evidence",
    )
    parser.add_argument(
        "--min-evaluation-subgroup-recall",
        type=float,
        help="Minimum recall for each evaluation subgroup with positive cases",
    )
    parser.add_argument(
        "--max-evaluation-subgroup-false-wakes-per-hour",
        type=float,
        help="Maximum false wakes/hour for each evaluation subgroup with negative audio",
    )
    parser.add_argument(
        "--max-evaluation-subgroup-repeated-positive-wake-events",
        type=int,
        help="Maximum repeated wake events allowed inside each positive evaluation subgroup",
    )
    parser.add_argument(
        "--max-evaluation-subgroup-detection-latency-ms",
        type=int,
        help="Maximum detection latency for each evaluation subgroup with positive cases",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.min_evaluation_subgroup_precision is None:
        args.min_evaluation_subgroup_precision = args.min_precision
    if args.min_evaluation_subgroup_recall is None:
        args.min_evaluation_subgroup_recall = args.min_recall
    if args.max_evaluation_subgroup_false_wakes_per_hour is None:
        args.max_evaluation_subgroup_false_wakes_per_hour = args.max_false_wakes_per_hour
    if args.max_evaluation_subgroup_repeated_positive_wake_events is None:
        args.max_evaluation_subgroup_repeated_positive_wake_events = (
            args.max_repeated_positive_wake_events
        )
    if args.max_evaluation_subgroup_detection_latency_ms is None:
        args.max_evaluation_subgroup_detection_latency_ms = args.max_detection_latency_ms

    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    real = json.loads(args.real_report.read_text(encoding="utf-8"))
    threshold = (
        json.loads(args.threshold_report.read_text(encoding="utf-8"))
        if args.threshold_report
        else None
    )
    failures = acceptance_failures(audit, real, threshold, args)
    result = {
        "accepted": not failures,
        "audit_report": str(args.audit_report),
        "real_report": str(args.real_report),
        "threshold_report": str(args.threshold_report) if args.threshold_report else None,
        "manifest_sha256": real.get("manifest_sha256"),
        "corpus": real.get("corpus"),
        "precision": real.get("precision"),
        "recall": real.get("recall"),
        "precision_lower_bound": real.get("confidence_intervals", {}).get("precision_lower_bound"),
        "recall_lower_bound": real.get("confidence_intervals", {}).get("recall_lower_bound"),
        "false_wakes_per_hour": real.get("false_wakes_per_hour"),
        "false_wakes_per_hour_upper_bound": real.get("confidence_intervals", {}).get(
            "false_wakes_per_hour_upper_bound"
        ),
        "repeated_positive_wake_events": real.get("repeated_positive_wake_events"),
        "max_detection_latency_ms": real.get("max_detection_latency_ms"),
        "evaluation_subgroups": evaluation_subgroup_metrics(real),
        "selected_threshold": selected_threshold_summary(threshold),
        "failures": failures,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_text_result(result)
    return 0 if not failures else 1


def acceptance_failures(
    audit: dict[str, object],
    real: dict[str, object],
    threshold: dict[str, object] | None,
    args,
) -> list[str]:
    failures: list[str] = []
    if not audit.get("ready", False):
        failures.append("readiness audit did not pass")
        failures.extend(str(failure) for failure in audit.get("failures", []))
    failures.extend(audit_evidence_failures(audit))
    failures.extend(real_provenance_failures(real))
    failures.extend(real_report_consistency_failures(real))
    audit_corpus = corpus_key(audit.get("corpus"))
    real_corpus = corpus_key(real.get("corpus"))
    if audit_corpus != real_corpus:
        failures.append("audit and real reports refer to different corpus metadata")
    audit_manifest_sha = str(audit.get("manifest_sha256") or "").strip()
    real_manifest_sha = str(real.get("manifest_sha256") or "").strip()
    if not audit_manifest_sha:
        failures.append("audit report is missing manifest_sha256")
    if not real_manifest_sha:
        failures.append("real report is missing manifest_sha256")
    if audit_manifest_sha and real_manifest_sha and audit_manifest_sha != real_manifest_sha:
        failures.append("audit and real reports were generated from different manifest contents")
    check_min(failures, "precision", real.get("precision"), args.min_precision)
    check_min(failures, "recall", real.get("recall"), args.min_recall)
    confidence = real.get("confidence_intervals", {})
    check_min(
        failures,
        "precision lower bound",
        confidence.get("precision_lower_bound"),
        args.min_precision_lower_bound,
    )
    check_min(
        failures,
        "recall lower bound",
        confidence.get("recall_lower_bound"),
        args.min_recall_lower_bound,
    )
    check_max(
        failures,
        "false wakes/hour",
        real.get("false_wakes_per_hour"),
        args.max_false_wakes_per_hour,
    )
    check_max(
        failures,
        "false wakes/hour upper bound",
        confidence.get("false_wakes_per_hour_upper_bound"),
        args.max_false_wakes_per_hour_upper_bound,
    )
    check_max(failures, "false positives", real.get("false_positives"), args.max_false_positives)
    check_max(failures, "false negatives", real.get("false_negatives"), args.max_false_negatives)
    check_max(
        failures,
        "repeated positive wake events",
        real.get("repeated_positive_wake_events"),
        args.max_repeated_positive_wake_events,
    )
    check_max(
        failures,
        "phrase mismatches",
        real.get("phrase_mismatches"),
        args.max_phrase_mismatches,
    )
    check_max(
        failures,
        "max detection latency ms",
        real.get("max_detection_latency_ms"),
        args.max_detection_latency_ms,
    )
    leakage = real.get("leakage", {})
    for key in [
        "template_case_path_overlaps",
        "template_case_audio_overlaps",
        "template_case_fingerprint_overlaps",
        "duplicate_case_paths",
        "duplicate_case_audio",
        "duplicate_case_fingerprints",
    ]:
        check_max(failures, key.replace("_", " "), leakage.get(key), 0)
    evaluation = subgroup_by_id(real.get("subgroups", {}).get("splits", []), "evaluation")
    if evaluation is None:
        failures.append("evaluation split metrics are missing")
    else:
        check_min(
            failures,
            "evaluation split precision",
            evaluation.get("precision"),
            args.min_precision,
        )
        check_min(
            failures,
            "evaluation split recall",
            evaluation.get("recall"),
            args.min_recall,
        )
        check_max(
            failures,
            "evaluation split false wakes/hour",
            evaluation.get("false_wakes_per_hour"),
            args.max_false_wakes_per_hour,
        )
        check_max(
            failures,
            "evaluation split repeated positive wake events",
            evaluation.get("repeated_positive_wake_events"),
            args.max_repeated_positive_wake_events,
        )
        check_max(
            failures,
            "evaluation split max detection latency ms",
            evaluation.get("max_detection_latency_ms"),
            args.max_detection_latency_ms,
        )
    failures.extend(evaluation_subgroup_failures(real, args))
    failures.extend(threshold_failures(real, threshold, args))
    return failures


def audit_evidence_failures(audit: dict[str, object]) -> list[str]:
    failures: list[str] = []
    corpus = audit.get("corpus")
    if not isinstance(corpus, dict):
        failures.append("audit report is missing corpus metadata")
    else:
        if not text_value(corpus.get("id")):
            failures.append("audit report corpus.id is missing")
        if not text_value(corpus.get("version")):
            failures.append("audit report corpus.version is missing")
        if text_value(corpus.get("source")) != "human-recorded":
            failures.append("audit report corpus.source is not human-recorded")

    ledger = audit.get("collection_ledger")
    if not isinstance(ledger, dict) or not ledger.get("present"):
        failures.append("audit report collection_ledger is missing")
    else:
        for key in ["consent_protocol", "collection_protocol", "collection_started_at", "collection_completed_at"]:
            if not text_value(ledger.get(key)):
                failures.append(f"audit report collection_ledger.{key} is missing")
        if not ledger.get("collected_by"):
            failures.append("audit report collection_ledger.collected_by is missing")
        for key in ["speaker_ids", "device_ids", "session_ids"]:
            if not ledger.get(key):
                failures.append(f"audit report collection_ledger.{key} is missing")

    source = audit.get("source_provenance")
    if not isinstance(source, dict):
        failures.append("audit report source_provenance is missing")
    else:
        non_human_templates = source.get("non_human_templates")
        non_human_cases = source.get("non_human_cases")
        if non_human_templates:
            failures.append(f"audit report has {len(non_human_templates)} non-human template source(s)")
        if non_human_cases:
            failures.append(f"audit report has {len(non_human_cases)} non-human case source(s)")
        template_counts = source.get("template_source_counts")
        case_counts = source.get("case_source_counts")
        if not isinstance(template_counts, dict) or template_counts.get("human-recorded", 0) <= 0:
            failures.append("audit report template_source_counts lacks human-recorded templates")
        if not isinstance(case_counts, dict) or case_counts.get("human-recorded", 0) <= 0:
            failures.append("audit report case_source_counts lacks human-recorded cases")

    leakage = audit.get("leakage")
    if not isinstance(leakage, dict):
        failures.append("audit report leakage is missing")
    else:
        for key in [
            "template_case_path_overlaps",
            "template_case_audio_overlaps",
            "template_case_fingerprint_overlaps",
            "duplicate_case_paths",
            "duplicate_case_audio",
            "duplicate_case_fingerprints",
        ]:
            value = leakage.get(key)
            if value is None:
                failures.append(f"audit report leakage.{key} is missing")
            elif len(value) != 0:
                failures.append(f"audit report leakage.{key} is not empty")
    return failures


def real_report_consistency_failures(real: dict[str, object]) -> list[str]:
    cases = real.get("cases_detail")
    if not isinstance(cases, list) or not cases or not all(isinstance(case, dict) for case in cases):
        return []
    failures: list[str] = []
    all_metrics = group_metrics("all", cases)
    top_level_fields = [
        "cases",
        "positives",
        "negatives",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "repeated_positive_wake_events",
        "phrase_mismatches",
        "precision",
        "recall",
        "negative_audio_seconds",
        "false_wakes_per_hour",
        "max_detection_latency_ms",
    ]
    for field in top_level_fields:
        compare_metric(failures, f"real report {field}", real.get(field), all_metrics.get(field))
    expected_confidence = confidence_intervals(all_metrics)
    reported_confidence = real.get("confidence_intervals")
    if not isinstance(reported_confidence, dict):
        failures.append("real report confidence_intervals is missing")
    else:
        compare_metric(
            failures,
            "real report confidence level",
            reported_confidence.get("confidence_level"),
            0.95,
        )
        for field in [
            "precision_lower_bound",
            "recall_lower_bound",
            "false_wakes_per_hour_upper_bound",
        ]:
            compare_metric(
                failures,
                f"real report {field}",
                reported_confidence.get(field),
                expected_confidence.get(field),
            )

    evaluation_cases = [
        case for case in cases if str(case.get("split") or "").strip() == "evaluation"
    ]
    reported_evaluation = subgroup_by_id(real.get("subgroups", {}).get("splits", []), "evaluation")
    if evaluation_cases and reported_evaluation is not None:
        evaluation_metrics = group_metrics("evaluation", evaluation_cases)
        split_fields = [
            "cases",
            "positives",
            "negatives",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "repeated_positive_wake_events",
            "phrase_mismatches",
            "precision",
            "recall",
            "negative_audio_seconds",
            "false_wakes_per_hour",
            "max_detection_latency_ms",
        ]
        for field in split_fields:
            compare_metric(
                failures,
                f"evaluation split {field}",
                reported_evaluation.get(field),
                evaluation_metrics.get(field),
            )
    return failures


def real_provenance_failures(real: dict[str, object]) -> list[str]:
    failures: list[str] = []
    corpus = real.get("corpus")
    if not isinstance(corpus, dict):
        failures.append("real report is missing corpus metadata")
    else:
        if not text_value(corpus.get("id")):
            failures.append("real report corpus.id is missing")
        if not text_value(corpus.get("version")):
            failures.append("real report corpus.version is missing")
        if text_value(corpus.get("source")) != "human-recorded":
            failures.append("real report corpus.source is not human-recorded")
    cases = real.get("cases_detail")
    if not isinstance(cases, list) or not cases:
        failures.append("real report cases_detail is missing")
        return failures
    non_human_cases = [
        text_value(case.get("id")) or f"case-{index}"
        for index, case in enumerate(cases, start=1)
        if isinstance(case, dict) and text_value(case.get("source_type")) != "human-recorded"
    ]
    malformed_cases = [
        f"case-{index}"
        for index, case in enumerate(cases, start=1)
        if not isinstance(case, dict)
    ]
    if malformed_cases:
        failures.append(f"{len(malformed_cases)} real report case detail row(s) are malformed")
    if non_human_cases:
        failures.append(
            f"{len(non_human_cases)} real report case(s) missing source_type=human-recorded: "
            + ", ".join(non_human_cases[:10])
        )
    return failures


def threshold_failures(real: dict[str, object], threshold: dict[str, object] | None, args) -> list[str]:
    failures: list[str] = []
    if threshold is None:
        if args.require_threshold_selection:
            failures.append("threshold report is required")
        return failures
    if not threshold.get("accepted", False):
        failures.append("threshold selection did not pass")
        failures.extend(f"threshold: {failure}" for failure in threshold.get("failures", []))
    check_min(
        failures,
        "threshold sweep points",
        threshold.get("threshold_sweep_points"),
        args.min_threshold_sweep_points,
    )
    real_sweep = real.get("threshold_sweep")
    if not isinstance(real_sweep, list) or not real_sweep:
        failures.append("real report threshold_sweep is missing")
    else:
        compare_metric(
            failures,
            "threshold report threshold_sweep_points",
            threshold.get("threshold_sweep_points"),
            len(real_sweep),
        )
    check_min(
        failures,
        "calibration passing threshold points",
        threshold.get("calibration_passing_points"),
        args.min_calibration_passing_threshold_points,
    )
    check_min(
        failures,
        "evaluation passing threshold points",
        threshold.get("evaluation_passing_points"),
        args.min_evaluation_passing_threshold_points,
    )
    if corpus_key(threshold.get("corpus")) != corpus_key(real.get("corpus")):
        failures.append("threshold report and real report refer to different corpus metadata")
    expected_hash = threshold.get("real_report_sha256")
    if not expected_hash:
        failures.append("threshold report is missing real_report_sha256")
    elif getattr(args, "real_report", None) and expected_hash != sha256(args.real_report):
        failures.append("threshold report was generated from a different real report")
    selected = threshold.get("selected") or {}
    if not selected:
        failures.append("threshold report is missing selected threshold")
        return failures
    if selected.get("candidate_threshold") is None:
        failures.append("threshold report selected threshold is missing candidate_threshold")
    if selected.get("accept_threshold") is None:
        failures.append("threshold report selected threshold is missing accept_threshold")
    sweep_point = matching_threshold_sweep_point(real, selected)
    if sweep_point is None:
        failures.append("selected threshold is not present in real report threshold_sweep")
    if not selected.get("calibration"):
        failures.append("selected threshold is missing calibration metrics")
    evaluation = selected.get("evaluation")
    if not evaluation:
        failures.append("selected threshold is missing evaluation metrics")
        return failures
    if sweep_point is not None:
        for split_id in ["calibration", "evaluation"]:
            selected_split = selected.get(split_id)
            sweep_split = subgroup_by_id(sweep_point.get("splits", []), split_id)
            if selected_split is None:
                continue
            if sweep_split is None:
                failures.append(f"real report selected threshold is missing {split_id} split metrics")
                continue
            for field in [
                "cases",
                "positives",
                "negatives",
                "true_positives",
                "false_positives",
                "true_negatives",
                "false_negatives",
                "repeated_positive_wake_events",
                "phrase_mismatches",
                "precision",
                "recall",
                "negative_audio_seconds",
                "false_wakes_per_hour",
                "max_detection_latency_ms",
            ]:
                compare_metric(
                    failures,
                    f"selected {split_id} {field}",
                    selected_split.get(field),
                    sweep_split.get(field),
                )
    check_min(
        failures,
        "selected evaluation precision",
        evaluation.get("precision"),
        args.min_precision,
    )
    check_min(
        failures,
        "selected evaluation recall",
        evaluation.get("recall"),
        args.min_recall,
    )
    check_max(
        failures,
        "selected evaluation repeated positive wake events",
        evaluation.get("repeated_positive_wake_events"),
        args.max_repeated_positive_wake_events,
    )
    check_max(
        failures,
        "selected evaluation false wakes/hour",
        evaluation.get("false_wakes_per_hour"),
        args.max_false_wakes_per_hour,
    )
    check_max(
        failures,
        "selected evaluation max detection latency ms",
        evaluation.get("max_detection_latency_ms"),
        args.max_detection_latency_ms,
    )
    return failures


def subgroup_by_id(groups, group_id: str):
    for group in groups or []:
        if group.get("id") == group_id:
            return group
    return None


def matching_threshold_sweep_point(real: dict[str, object], selected: dict[str, object]):
    selected_candidate = selected.get("candidate_threshold")
    selected_accept = selected.get("accept_threshold")
    if selected_candidate is None or selected_accept is None:
        return None
    for point in real.get("threshold_sweep") or []:
        if not isinstance(point, dict):
            continue
        if numeric_equal(point.get("candidate_threshold"), selected_candidate) and numeric_equal(
            point.get("accept_threshold"),
            selected_accept,
        ):
            return point
    return None


def evaluation_subgroup_failures(real: dict[str, object], args) -> list[str]:
    failures: list[str] = []
    subgroups = evaluation_subgroup_metrics(real)
    if not subgroups:
        failures.append("evaluation subgroup metrics are missing")
        return failures
    for dimension, groups in subgroups.items():
        for group in groups:
            label = f"evaluation {dimension} '{group['id']}'"
            if group["precision_denominator"] > 0:
                check_min(
                    failures,
                    f"{label} precision",
                    group["precision"],
                    args.min_evaluation_subgroup_precision,
                )
            if group["positives"] > 0:
                check_min(
                    failures,
                    f"{label} recall",
                    group["recall"],
                    args.min_evaluation_subgroup_recall,
                )
                check_max(
                    failures,
                    f"{label} repeated positive wake events",
                    group["repeated_positive_wake_events"],
                    args.max_evaluation_subgroup_repeated_positive_wake_events,
                )
                check_max(
                    failures,
                    f"{label} max detection latency ms",
                    group["max_detection_latency_ms"],
                    args.max_evaluation_subgroup_detection_latency_ms,
                )
            if group["negative_audio_seconds"] > 0:
                check_max(
                    failures,
                    f"{label} false wakes/hour",
                    group["false_wakes_per_hour"],
                    args.max_evaluation_subgroup_false_wakes_per_hour,
                )
    return failures


def evaluation_subgroup_metrics(real: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    cases = real.get("cases_detail")
    if not isinstance(cases, list):
        return {}
    evaluation_cases = [
        case
        for case in cases
        if isinstance(case, dict) and str(case.get("split") or "").strip() == "evaluation"
    ]
    if not evaluation_cases:
        return {}
    dimensions = {
        "speaker_id": lambda case: text_value(case.get("speaker_id")),
        "environment": lambda case: text_value(case.get("environment")),
        "distance": lambda case: text_value(case.get("distance")),
        "device": lambda case: text_value(case.get("device")),
        "session_id": lambda case: text_value(case.get("session_id")),
        "category": lambda case: text_value(case.get("category")),
        "expected_phrase": lambda case: text_value(case.get("expected_phrase")),
        "wake_start_bucket": wake_start_bucket,
    }
    metrics: dict[str, list[dict[str, object]]] = {}
    for dimension, key_fn in dimensions.items():
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for case in evaluation_cases:
            key = key_fn(case)
            if key:
                groups[key].append(case)
        if groups:
            metrics[dimension] = [
                group_metrics(group_id, group_cases)
                for group_id, group_cases in sorted(groups.items())
            ]
    return metrics


def group_metrics(group_id: str, cases: list[dict[str, object]]) -> dict[str, object]:
    positives = [case for case in cases if bool(case.get("should_wake"))]
    negatives = [case for case in cases if not bool(case.get("should_wake"))]
    true_positives = sum(1 for case in positives if bool(case.get("woke")))
    false_positives = sum(false_wake_count(case) for case in negatives)
    true_negatives = sum(1 for case in negatives if not bool(case.get("woke")))
    false_negatives = sum(1 for case in positives if not bool(case.get("woke")))
    phrase_mismatches = sum(
        1 for case in positives if bool(case.get("woke")) and not bool(case.get("phrase_matched"))
    )
    repeated_positive_wake_events = sum(
        max(0, false_wake_count(case) - 1)
        for case in positives
    )
    precision_denominator = true_positives + false_positives
    precision = (
        true_positives / precision_denominator
        if precision_denominator > 0
        else None
    )
    recall = true_positives / len(positives) if positives else None
    negative_audio_seconds = sum(float(case.get("duration_ms") or 0) / 1000.0 for case in negatives)
    false_wakes_per_hour = (
        false_positives / (negative_audio_seconds / 3600.0)
        if negative_audio_seconds > 0
        else None
    )
    detection_latencies = [
        int(case["detection_latency_ms"])
        for case in positives
        if case.get("detection_latency_ms") is not None
    ]
    return {
        "id": group_id,
        "cases": len(cases),
        "positives": len(positives),
        "negatives": len(negatives),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "phrase_mismatches": phrase_mismatches,
        "repeated_positive_wake_events": repeated_positive_wake_events,
        "precision_denominator": precision_denominator,
        "precision": precision,
        "recall": recall,
        "negative_audio_seconds": negative_audio_seconds,
        "false_wakes_per_hour": false_wakes_per_hour,
        "max_detection_latency_ms": max(detection_latencies) if detection_latencies else None,
    }


def confidence_intervals(metrics: dict[str, object]) -> dict[str, object]:
    true_positives = int(metrics.get("true_positives") or 0)
    false_positives = int(metrics.get("false_positives") or 0)
    positives = int(metrics.get("positives") or 0)
    negative_audio_seconds = float(metrics.get("negative_audio_seconds") or 0.0)
    return {
        "precision_lower_bound": wilson_lower_bound(
            true_positives,
            true_positives + false_positives,
        ),
        "recall_lower_bound": wilson_lower_bound(true_positives, positives),
        "false_wakes_per_hour_upper_bound": poisson_rate_upper_95(
            false_positives,
            negative_audio_seconds,
        ),
    }


def wilson_lower_bound(successes: int, trials: int) -> float | None:
    if trials == 0:
        return None
    z = 1.959_963_984_540_054
    n = float(trials)
    p = float(successes) / n
    z2 = z * z
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    denominator = 1.0 + z2 / n
    return max(0.0, (center - margin) / denominator)


def poisson_rate_upper_95(events: int, exposure_seconds: float) -> float | None:
    if exposure_seconds <= 0.0:
        return None
    exposure_hours = exposure_seconds / 3600.0
    return poisson_count_upper_95(events) / exposure_hours


def poisson_count_upper_95(events: int) -> float:
    if events == 0:
        return -math.log(0.05)
    z = 1.644_853_626_951_472_2
    degrees_of_freedom = 2.0 * (float(events) + 1.0)
    chi_square_upper = degrees_of_freedom * (
        1.0
        - 2.0 / (9.0 * degrees_of_freedom)
        + z * math.sqrt(2.0 / (9.0 * degrees_of_freedom))
    ) ** 3
    return 0.5 * chi_square_upper


def text_value(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def false_wake_count(case: dict[str, object]) -> int:
    if case.get("wake_event_count") is not None:
        return int(case.get("wake_event_count") or 0)
    return 1 if bool(case.get("woke")) else 0


def wake_start_bucket(case: dict[str, object]) -> str | None:
    if not bool(case.get("should_wake")):
        return None
    try:
        wake_start_ms = int(case.get("wake_start_ms"))
    except (TypeError, ValueError):
        return None
    if wake_start_ms <= 250:
        return "early"
    if wake_start_ms <= 1500:
        return "mid"
    return "late"


def corpus_key(corpus) -> tuple[str, str, str]:
    if not isinstance(corpus, dict):
        return ("", "", "")
    return (
        str(corpus.get("id") or "").strip(),
        str(corpus.get("version") or "").strip(),
        str(corpus.get("source") or "").strip(),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_threshold_summary(threshold: dict[str, object] | None):
    if not threshold:
        return None
    selected = threshold.get("selected") or {}
    return {
        "candidate_threshold": selected.get("candidate_threshold"),
        "accept_threshold": selected.get("accept_threshold"),
        "calibration": selected.get("calibration"),
        "evaluation": selected.get("evaluation"),
    }


def check_min(failures: list[str], name: str, value, minimum: float) -> None:
    if value is None:
        failures.append(f"{name} is missing")
    elif float(value) < minimum:
        failures.append(f"{name} {value} below required {minimum}")


def check_max(failures: list[str], name: str, value, maximum: float) -> None:
    if value is None:
        failures.append(f"{name} is missing")
    elif float(value) > maximum:
        failures.append(f"{name} {value} exceeded allowed {maximum}")


def compare_metric(failures: list[str], name: str, reported, expected) -> None:
    if expected is None:
        if reported is not None:
            failures.append(f"{name} {reported} does not match cases_detail n/a")
        return
    if reported is None:
        failures.append(f"{name} is missing")
        return
    if isinstance(expected, float):
        try:
            observed = float(reported)
        except (TypeError, ValueError):
            failures.append(f"{name} {reported} does not match cases_detail {expected}")
            return
        if not numeric_equal(observed, expected):
            failures.append(f"{name} {reported} does not match cases_detail {expected}")
        return
    try:
        observed_int = int(reported)
    except (TypeError, ValueError):
        failures.append(f"{name} {reported} does not match cases_detail {expected}")
        return
    if observed_int != expected:
        failures.append(f"{name} {reported} does not match cases_detail {expected}")


def numeric_equal(left, right, tolerance: float = 0.001) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def print_text_result(result: dict[str, object]) -> None:
    status = "accepted" if result["accepted"] else "rejected"
    print(f"status={status}")
    for key in [
        "precision",
        "precision_lower_bound",
        "recall",
        "recall_lower_bound",
        "false_wakes_per_hour",
        "false_wakes_per_hour_upper_bound",
        "max_detection_latency_ms",
    ]:
        print(f"{key}={result.get(key)}")
    selected = result.get("selected_threshold") or {}
    if selected:
        print(f"selected_candidate_threshold={selected.get('candidate_threshold')}")
        print(f"selected_accept_threshold={selected.get('accept_threshold')}")
    failures = result["failures"]
    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    raise SystemExit(main())
