#!/usr/bin/env python3
"""Tests for calibration-selected vona-wake threshold reports."""

from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_vona_wake_threshold.py"
spec = importlib.util.spec_from_file_location("select_vona_wake_threshold", SCRIPT)
thresholds = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(thresholds)


class VonaWakeThresholdSelectionTests(unittest.TestCase):
    def test_requires_more_than_one_calibration_passing_point(self) -> None:
        points = [
            point(0.86, calibration_recall=0.97),
            point(0.89, calibration_recall=0.99),
            point(0.92, calibration_recall=0.97),
        ]
        calibration = thresholds.passing_points(points, "calibration", args())
        evaluation = thresholds.passing_points(points, "evaluation", args())

        self.assertEqual(1, len(calibration))
        self.assertEqual(3, len(evaluation))
        self.assertLess(len(calibration), args().min_calibration_passing_points)

    def test_selects_best_calibration_point_when_region_is_stable(self) -> None:
        points = [
            point(0.86, calibration_recall=0.99, calibration_precision=0.99),
            point(0.89, calibration_recall=1.0, calibration_precision=0.99),
            point(0.92, calibration_recall=1.0, calibration_precision=1.0),
        ]

        selected = thresholds.select_point(thresholds.passing_points(points, "calibration", args()))

        self.assertEqual(0.92, selected["accept_threshold"])

    def test_rejects_threshold_point_with_repeated_positive_wake_events(self) -> None:
        failures = thresholds.metric_failures(
            "evaluation",
            split("evaluation", precision=1.0, recall=1.0, repeated_positive_wake_events=1),
            args(),
        )

        self.assertIn(
            "evaluation repeated positive wake events 1 exceeded allowed 0",
            failures,
        )


def args():
    return types.SimpleNamespace(
        min_precision=0.98,
        min_recall=0.98,
        max_repeated_positive_wake_events=0,
        max_false_wakes_per_hour=0.05,
        max_detection_latency_ms=1500,
        min_calibration_passing_points=2,
        min_evaluation_passing_points=1,
    )


def point(
    accept_threshold: float,
    *,
    calibration_recall: float,
    calibration_precision: float = 1.0,
    evaluation_recall: float = 1.0,
    evaluation_precision: float = 1.0,
) -> dict[str, object]:
    return {
        "candidate_threshold": round(accept_threshold - 0.04, 2),
        "accept_threshold": accept_threshold,
        "splits": [
            split("calibration", calibration_precision, calibration_recall),
            split("evaluation", evaluation_precision, evaluation_recall),
        ],
    }


def split(
    group_id: str,
    precision: float,
    recall: float,
    repeated_positive_wake_events: int = 0,
) -> dict[str, object]:
    return {
        "id": group_id,
        "precision": precision,
        "recall": recall,
        "repeated_positive_wake_events": repeated_positive_wake_events,
        "false_wakes_per_hour": 0.0,
        "max_detection_latency_ms": 200,
    }


if __name__ == "__main__":
    unittest.main()
