#!/usr/bin/env python3
"""Tests for vona-wake real evidence status summaries."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_vona_wake_real_evidence.py"
)
spec = importlib.util.spec_from_file_location("summarize_vona_wake_real_evidence", SCRIPT)
status_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(status_mod)

ACCEPTANCE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "check_vona_wake_acceptance.py"
)
acceptance_spec = importlib.util.spec_from_file_location(
    "check_vona_wake_acceptance",
    ACCEPTANCE_SCRIPT,
)
acceptance = importlib.util.module_from_spec(acceptance_spec)
assert acceptance_spec and acceptance_spec.loader
acceptance_spec.loader.exec_module(acceptance)


class VonaWakeEvidenceStatusTests(unittest.TestCase):
    def test_reports_missing_real_evidence_actions(self) -> None:
        status = status_mod.build_status(Path("/tmp/missing-vona-wake-evidence"), {})

        self.assertFalse(status["accepted"])
        self.assertFalse(status["stages"]["real_evaluation"]["passed"])
        self.assertIn("real-report.json is missing", status["failures"])
        self.assertIn(
            "Run VONA_WAKE_REQUIRE_REAL_EVAL=1 scripts/run_vona_wake_eval.sh /path/to/corpus/manifest.json",
            status["next_actions"],
        )

    def test_accepts_complete_saved_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            write_json(report_dir / "generated-report.json", generated_report())
            write_json(report_dir / "generated-manifest.json", {"corpus": {"source": "synthetic-generated"}})
            write_json(report_dir / "audit-report.json", audit_report())
            write_json(report_dir / "real-report.json", real_report())
            write_json(
                report_dir / "threshold-selection-report.json",
                threshold_report(sha256(report_dir / "real-report.json")),
            )
            (report_dir / "summary.md").write_text("# summary\n", encoding="utf-8")

            status = status_mod.build_status(report_dir, status_mod.load_reports(report_dir))

        self.assertTrue(status["accepted"])
        self.assertEqual([], status["failures"])
        self.assertEqual("fixture-corpus", status["corpus"]["id"])
        self.assertEqual(1.0, status["metrics"]["precision"])
        self.assertTrue(status["stages"]["acceptance"]["passed"])


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def generated_report() -> dict[str, object]:
    return {
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
    }


def audit_report() -> dict[str, object]:
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


def real_report() -> dict[str, object]:
    cases = [
        case(f"positive-{index}", True, True, "wake-positive", 1_000, 1)
        for index in range(73)
    ]
    cases.append(case("negative", False, False, "ordinary-speech", 216_000_000, 0))
    all_metrics = acceptance.group_metrics("all", cases)
    evaluation_metrics = acceptance.group_metrics("evaluation", cases)
    return {
        "manifest_sha256": "manifest-hash",
        "corpus": corpus(),
        "cases": all_metrics["cases"],
        "positives": all_metrics["positives"],
        "negatives": all_metrics["negatives"],
        "true_positives": all_metrics["true_positives"],
        "true_negatives": all_metrics["true_negatives"],
        "precision": all_metrics["precision"],
        "recall": all_metrics["recall"],
        "negative_audio_seconds": all_metrics["negative_audio_seconds"],
        "false_wakes_per_hour": all_metrics["false_wakes_per_hour"],
        "false_positives": all_metrics["false_positives"],
        "false_negatives": all_metrics["false_negatives"],
        "repeated_positive_wake_events": all_metrics["repeated_positive_wake_events"],
        "phrase_mismatches": all_metrics["phrase_mismatches"],
        "max_detection_latency_ms": all_metrics["max_detection_latency_ms"],
        "confidence_intervals": {
            "confidence_level": 0.95,
            **acceptance.confidence_intervals(all_metrics),
        },
        "coverage": {
            "speakers": 1,
            "environments": 1,
            "distances": 1,
            "devices": 1,
            "sessions": 1,
            "categories": 2,
            "speaker_ids": ["speaker-a"],
            "environment_ids": ["quiet-office"],
            "distance_ids": ["near"],
            "device_ids": ["usb-mic"],
            "session_ids": ["session-a"],
            "category_ids": ["wake-positive", "ordinary-speech"],
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
        "threshold_sweep": threshold_sweep(),
        "cases_detail": cases,
    }


def threshold_report(real_report_sha256: str) -> dict[str, object]:
    return {
        "accepted": True,
        "real_report_sha256": real_report_sha256,
        "corpus": corpus(),
        "threshold_sweep_points": 5,
        "calibration_passing_points": 2,
        "evaluation_passing_points": 2,
        "selected": {
            "candidate_threshold": 0.88,
            "accept_threshold": 0.92,
            "calibration": split_metrics("calibration"),
            "evaluation": split_metrics("evaluation"),
        },
        "failures": [],
    }


def threshold_sweep() -> list[dict[str, object]]:
    return [
        threshold_sweep_point(0.88, 0.92),
        threshold_sweep_point(0.84, 0.88),
        threshold_sweep_point(0.80, 0.84),
        threshold_sweep_point(0.76, 0.80),
        threshold_sweep_point(0.72, 0.76),
    ]


def threshold_sweep_point(candidate_threshold: float, accept_threshold: float) -> dict[str, object]:
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


def split_metrics(group_id: str) -> dict[str, object]:
    return {
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus() -> dict[str, str]:
    return {
        "id": "fixture-corpus",
        "version": "1",
        "source": "human-recorded",
    }


def case(
    case_id: str,
    should_wake: bool,
    woke: bool,
    category: str,
    duration_ms: int,
    wake_event_count: int,
) -> dict[str, object]:
    return {
        "id": case_id,
        "split": "evaluation",
        "speaker_id": "speaker-a",
        "environment": "quiet-office",
        "distance": "near",
        "device": "usb-mic",
        "session_id": "session-a",
        "category": category,
        "source_type": "human-recorded",
        "expected_phrase": "hey vona" if should_wake else None,
        "should_wake": should_wake,
        "woke": woke,
        "phrase_matched": True,
        "wake_start_ms": 0 if should_wake else None,
        "detection_latency_ms": 200 if should_wake and woke else None,
        "duration_ms": duration_ms,
        "wake_event_count": wake_event_count,
    }


if __name__ == "__main__":
    unittest.main()
