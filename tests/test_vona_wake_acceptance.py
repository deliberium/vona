#!/usr/bin/env python3
"""Tests for release-grade vona-wake acceptance evidence checks."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_vona_wake_acceptance.py"
spec = importlib.util.spec_from_file_location("check_vona_wake_acceptance", SCRIPT)
acceptance = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(acceptance)


class VonaWakeAcceptanceTests(unittest.TestCase):
    def test_accepts_clean_evaluation_evidence(self) -> None:
        failures = acceptance.acceptance_failures(
            audit_report(),
            real_report(cases=release_grade_cases()),
            None,
            args(max_false_negatives=0),
        )

        self.assertEqual([], failures)

    def test_rejects_weak_speaker_subgroup_when_aggregate_passes(self) -> None:
        cases = [positive_case("speaker-a", f"a-{index}") for index in range(99)]
        cases.append(positive_case("speaker-b", "b-0", woke=False))
        failures = acceptance.acceptance_failures(
            audit_report(),
            real_report(cases=cases, recall=0.99, false_negatives=1, max_detection_latency_ms=200),
            None,
            args(max_false_negatives=10),
        )

        self.assertIn("evaluation speaker_id 'speaker-b' recall 0.0 below required 0.98", failures)
        self.assertIn("evaluation speaker_id 'speaker-b' max detection latency ms is missing", failures)

    def test_rejects_negative_subgroup_false_wake_rate_when_aggregate_passes(self) -> None:
        failures = acceptance.acceptance_failures(
            audit_report(),
            real_report(
                cases=[
                    positive_case("speaker-a"),
                    negative_case(
                        "speaker-a",
                        "ordinary-speech",
                        duration_ms=3_600_000,
                        woke=True,
                        wake_event_count=1,
                    ),
                ],
                false_positives=0,
                false_wakes_per_hour=0.0,
            ),
            None,
            args(max_false_positives=10),
        )

        self.assertIn(
            "evaluation category 'ordinary-speech' false wakes/hour 1.0 exceeded allowed 0.05",
            failures,
        )

    def test_rejects_normalized_audio_leakage_in_real_report(self) -> None:
        report = real_report(
            cases=[
                positive_case("speaker-a"),
                negative_case("speaker-a", "ordinary-speech", 3_600_000),
            ]
        )
        report["leakage"]["template_case_fingerprint_overlaps"] = 1
        report["leakage"]["duplicate_case_fingerprints"] = 1

        failures = acceptance.acceptance_failures(
            audit_report(),
            report,
            None,
            args(max_false_negatives=0),
        )

        self.assertIn(
            "template case fingerprint overlaps 1 exceeded allowed 0",
            failures,
        )
        self.assertIn(
            "duplicate case fingerprints 1 exceeded allowed 0",
            failures,
        )

    def test_rejects_repeated_positive_wake_events(self) -> None:
        failures = acceptance.acceptance_failures(
            audit_report(),
            real_report(
                cases=[
                    positive_case("speaker-a", wake_event_count=2),
                    negative_case("speaker-a", "ordinary-speech", 3_600_000),
                ],
                repeated_positive_wake_events=1,
            ),
            None,
            args(max_false_negatives=0),
        )

        self.assertIn(
            "repeated positive wake events 1 exceeded allowed 0",
            failures,
        )
        self.assertIn(
            "evaluation speaker_id 'speaker-a' repeated positive wake events 1 exceeded allowed 0",
            failures,
        )

    def test_rejects_non_human_real_report_provenance(self) -> None:
        report = real_report(
            cases=[
                positive_case("speaker-a"),
                negative_case("speaker-a", "ordinary-speech", 3_600_000),
            ],
        )
        report["corpus"]["source"] = "synthetic-generated"
        report["cases_detail"][0]["source_type"] = "deterministic-pseudo-voice"

        failures = acceptance.acceptance_failures(
            audit_report(),
            report,
            None,
            args(max_false_negatives=0),
        )

        self.assertIn("real report corpus.source is not human-recorded", failures)
        self.assertIn(
            "1 real report case(s) missing source_type=human-recorded: positive",
            failures,
        )

    def test_rejects_top_level_metrics_that_do_not_match_case_details(self) -> None:
        report = real_report(
            cases=[
                positive_case("speaker-a"),
                negative_case("speaker-a", "ordinary-speech", 3_600_000),
            ],
        )
        report["true_positives"] = 0

        failures = acceptance.acceptance_failures(
            audit_report(),
            report,
            None,
            args(max_false_negatives=0),
        )

        self.assertIn("real report true_positives 0 does not match cases_detail 1", failures)

    def test_rejects_confidence_bounds_that_do_not_match_case_details(self) -> None:
        report = real_report(cases=release_grade_cases())
        report["confidence_intervals"]["recall_lower_bound"] = 1.0

        failures = acceptance.acceptance_failures(
            audit_report(),
            report,
            None,
            args(max_false_negatives=0),
        )

        self.assertTrue(
            any(
                failure.startswith("real report recall_lower_bound 1.0 does not match cases_detail")
                for failure in failures
            )
        )

    def test_rejects_audit_report_without_direct_evidence_fields(self) -> None:
        audit = audit_report()
        audit.pop("collection_ledger")
        audit.pop("source_provenance")
        audit.pop("leakage")

        failures = acceptance.acceptance_failures(
            audit,
            real_report(
                cases=[
                    positive_case("speaker-a"),
                    negative_case("speaker-a", "ordinary-speech", 3_600_000),
                ]
            ),
            None,
            args(max_false_negatives=0),
        )

        self.assertIn("audit report collection_ledger is missing", failures)
        self.assertIn("audit report source_provenance is missing", failures)
        self.assertIn("audit report leakage is missing", failures)

    def test_rejects_threshold_report_without_real_report_hash(self) -> None:
        threshold = {
            "accepted": True,
            "corpus": corpus(),
            "threshold_sweep_points": 5,
            "calibration_passing_points": 2,
            "evaluation_passing_points": 1,
            "selected": {
                "candidate_threshold": 0.88,
                "accept_threshold": 0.92,
                "calibration": split_metrics(),
                "evaluation": split_metrics(),
            },
            "failures": [],
        }
        failures = acceptance.acceptance_failures(
            audit_report(),
            real_report(
                cases=[
                    positive_case("speaker-a"),
                    negative_case("speaker-a", "ordinary-speech", 3_600_000),
                ]
            ),
            threshold,
            args(max_false_negatives=0, require_threshold_selection=True),
        )

        self.assertIn("threshold report is missing real_report_sha256", failures)

    def test_rejects_threshold_report_for_different_real_report_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real_path = Path(tmp) / "real-report.json"
            real_path.write_text(json.dumps({"fixture": "current"}), encoding="utf-8")
            threshold = {
                "accepted": True,
                "real_report_sha256": "not-the-current-report",
                "corpus": corpus(),
                "threshold_sweep_points": 5,
                "calibration_passing_points": 2,
                "evaluation_passing_points": 1,
                "selected": {
                    "candidate_threshold": 0.88,
                    "accept_threshold": 0.92,
                    "calibration": split_metrics(),
                    "evaluation": split_metrics(),
                },
                "failures": [],
            }
            failures = acceptance.acceptance_failures(
                audit_report(),
                real_report(
                    cases=[
                        positive_case("speaker-a"),
                        negative_case("speaker-a", "ordinary-speech", 3_600_000),
                    ]
                ),
                threshold,
                args(max_false_negatives=0, require_threshold_selection=True, real_report=real_path),
            )

        self.assertIn("threshold report was generated from a different real report", failures)

    def test_rejects_selected_threshold_not_bound_to_real_sweep_point(self) -> None:
        threshold = threshold_report(
            selected={
                "candidate_threshold": 0.11,
                "accept_threshold": 0.22,
                "calibration": split_metrics("calibration"),
                "evaluation": split_metrics("evaluation"),
            }
        )

        failures = acceptance.acceptance_failures(
            audit_report(),
            real_report(cases=release_grade_cases()),
            threshold,
            args(max_false_negatives=0, require_threshold_selection=True),
        )

        self.assertIn("selected threshold is not present in real report threshold_sweep", failures)

    def test_rejects_selected_threshold_metrics_not_matching_real_sweep_point(self) -> None:
        threshold = threshold_report(
            selected={
                "candidate_threshold": 0.88,
                "accept_threshold": 0.92,
                "calibration": split_metrics("calibration"),
                "evaluation": split_metrics("evaluation", recall=0.5),
            }
        )

        failures = acceptance.acceptance_failures(
            audit_report(),
            real_report(cases=release_grade_cases()),
            threshold,
            args(max_false_negatives=0, require_threshold_selection=True),
        )

        self.assertIn("selected evaluation recall 0.5 does not match cases_detail 1.0", failures)


def args(**overrides):
    values = {
        "min_precision": 0.98,
        "min_recall": 0.98,
        "min_precision_lower_bound": 0.95,
        "min_recall_lower_bound": 0.95,
        "max_false_wakes_per_hour": 0.05,
        "max_false_wakes_per_hour_upper_bound": 0.05,
        "max_false_positives": 0,
        "max_false_negatives": 0,
        "max_repeated_positive_wake_events": 0,
        "max_phrase_mismatches": 0,
        "max_detection_latency_ms": 1500,
        "min_threshold_sweep_points": 5,
        "min_calibration_passing_threshold_points": 2,
        "min_evaluation_passing_threshold_points": 1,
        "min_evaluation_subgroup_precision": 0.98,
        "min_evaluation_subgroup_recall": 0.98,
        "max_evaluation_subgroup_false_wakes_per_hour": 0.05,
        "max_evaluation_subgroup_repeated_positive_wake_events": 0,
        "max_evaluation_subgroup_detection_latency_ms": 1500,
        "require_threshold_selection": False,
        "real_report": None,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def audit_report():
    return {
        "ready": True,
        "corpus": corpus(),
        "manifest_sha256": "manifest-hash",
        "collection_ledger": {
            "present": True,
            "consent_protocol": "test-consent-v1",
            "collection_protocol": "test-collection-v1",
            "collected_by": ["operator-a"],
            "collection_started_at": "2026-06-01T09:00:00Z",
            "collection_completed_at": "2026-06-01T10:00:00Z",
            "speaker_ids": ["speaker-a"],
            "device_ids": ["usb-mic"],
            "session_ids": ["session-a"],
        },
        "source_provenance": {
            "template_source_counts": {"human-recorded": 1},
            "case_source_counts": {"human-recorded": 2},
            "non_human_templates": [],
            "non_human_cases": [],
        },
        "leakage": {
            "template_case_path_overlaps": [],
            "template_case_audio_overlaps": [],
            "template_case_fingerprint_overlaps": [],
            "duplicate_case_paths": {},
            "duplicate_case_audio": {},
            "duplicate_case_fingerprints": {},
        },
        "failures": [],
    }


def real_report(
    *,
    cases,
    precision=1.0,
    recall=1.0,
    false_positives=0,
    false_negatives=0,
    repeated_positive_wake_events=0,
    false_wakes_per_hour=0.0,
    max_detection_latency_ms=200,
):
    all_metrics = acceptance.group_metrics("all", cases)
    evaluation_metrics = acceptance.group_metrics(
        "evaluation",
        [case for case in cases if case.get("split") == "evaluation"],
    )
    all_metrics.update(
        {
            "precision": precision,
            "recall": recall,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "repeated_positive_wake_events": repeated_positive_wake_events,
            "false_wakes_per_hour": false_wakes_per_hour,
            "max_detection_latency_ms": max_detection_latency_ms,
        }
    )
    evaluation_metrics.update(
        {
            "precision": precision,
            "recall": recall,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "repeated_positive_wake_events": repeated_positive_wake_events,
            "false_wakes_per_hour": false_wakes_per_hour,
            "max_detection_latency_ms": max_detection_latency_ms,
        }
    )
    return {
        "manifest_sha256": "manifest-hash",
        "corpus": corpus(),
        "cases": all_metrics["cases"],
        "positives": all_metrics["positives"],
        "negatives": all_metrics["negatives"],
        "true_positives": all_metrics["true_positives"],
        "true_negatives": all_metrics["true_negatives"],
        "precision": precision,
        "recall": recall,
        "negative_audio_seconds": all_metrics["negative_audio_seconds"],
        "false_wakes_per_hour": false_wakes_per_hour,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "repeated_positive_wake_events": repeated_positive_wake_events,
        "phrase_mismatches": 0,
        "max_detection_latency_ms": max_detection_latency_ms,
        "confidence_intervals": {
            "confidence_level": 0.95,
            **acceptance.confidence_intervals(all_metrics),
        },
        "leakage": {
            "template_case_path_overlaps": 0,
            "template_case_audio_overlaps": 0,
            "template_case_fingerprint_overlaps": 0,
            "duplicate_case_paths": 0,
            "duplicate_case_audio": 0,
            "duplicate_case_fingerprints": 0,
        },
        "subgroups": {
            "splits": [
                {
                    "id": "evaluation",
                    **evaluation_metrics,
                }
            ]
        },
        "threshold_sweep": [
            threshold_sweep_point(0.88, 0.92),
            threshold_sweep_point(0.84, 0.88),
            threshold_sweep_point(0.80, 0.84),
            threshold_sweep_point(0.76, 0.80),
            threshold_sweep_point(0.72, 0.76),
        ],
        "cases_detail": cases,
    }


def corpus():
    return {
        "id": "fixture",
        "version": "1",
        "source": "human-recorded",
    }


def threshold_report(*, selected=None):
    if selected is None:
        selected = {
            "candidate_threshold": 0.88,
            "accept_threshold": 0.92,
            "calibration": split_metrics("calibration"),
            "evaluation": split_metrics("evaluation"),
        }
    return {
        "accepted": True,
        "real_report_sha256": "not-checked-without-path",
        "corpus": corpus(),
        "threshold_sweep_points": 5,
        "calibration_passing_points": 2,
        "evaluation_passing_points": 1,
        "selected": selected,
        "failures": [],
    }


def threshold_sweep_point(candidate_threshold, accept_threshold):
    return {
        "candidate_threshold": candidate_threshold,
        "accept_threshold": accept_threshold,
        "precision": 1.0,
        "recall": 1.0,
        "repeated_positive_wake_events": 0,
        "false_wakes_per_hour": 0.0,
        "max_detection_latency_ms": 200,
        "splits": [
            split_metrics("calibration"),
            split_metrics("evaluation"),
        ],
    }


def split_metrics(group_id="evaluation", **overrides):
    metrics = {
        "id": group_id,
        "cases": 74,
        "positives": 73,
        "negatives": 1,
        "true_positives": 73,
        "false_positives": 0,
        "true_negatives": 1,
        "false_negatives": 0,
        "phrase_mismatches": 0,
        "negative_audio_seconds": 216_000.0,
        "precision": 1.0,
        "recall": 1.0,
        "repeated_positive_wake_events": 0,
        "false_wakes_per_hour": 0.0,
        "max_detection_latency_ms": 200,
    }
    metrics.update(overrides)
    return metrics


def release_grade_cases():
    cases = [positive_case("speaker-a", f"positive-{index}") for index in range(73)]
    cases.append(negative_case("speaker-a", "ordinary-speech", 216_000_000))
    return cases


def positive_case(speaker_id, case_id="positive", woke=True, wake_event_count=None):
    if wake_event_count is None:
        wake_event_count = 1 if woke else 0
    return {
        "id": case_id,
        "split": "evaluation",
        "speaker_id": speaker_id,
        "environment": "quiet-office",
        "distance": "near",
        "device": "usb-mic",
        "session_id": "session-a",
        "category": "wake-positive",
        "source_type": "human-recorded",
        "expected_phrase": "hey vona",
        "should_wake": True,
        "woke": woke,
        "phrase_matched": True,
        "wake_start_ms": 0,
        "detection_latency_ms": 200 if woke else None,
        "duration_ms": 1000,
        "wake_event_count": wake_event_count,
    }


def negative_case(
    speaker_id,
    category,
    duration_ms,
    *,
    woke=False,
    wake_event_count=0,
):
    return {
        "id": f"negative-{category}",
        "split": "evaluation",
        "speaker_id": speaker_id,
        "environment": "quiet-office",
        "distance": "near",
        "device": "usb-mic",
        "session_id": "session-a",
        "category": category,
        "source_type": "human-recorded",
        "expected_phrase": None,
        "should_wake": False,
        "woke": woke,
        "phrase_matched": True,
        "wake_start_ms": None,
        "detection_latency_ms": None,
        "duration_ms": duration_ms,
        "wake_event_count": wake_event_count,
    }


if __name__ == "__main__":
    unittest.main()
