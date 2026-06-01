#!/usr/bin/env python3
"""Tests for wake phrase label semantics in real voice plans and manifests."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


corpus_audit = load_script("audit_vona_wake_corpus")
plan_audit = load_script("audit_vona_wake_recording_plan")


class VonaWakeLabelSemanticsTests(unittest.TestCase):
    def test_manifest_audit_rejects_misleading_negative_labels(self) -> None:
        failures = corpus_audit.label_semantic_failures(
            [
                case("unauthorized-a", False, "unauthorized-wake", "hello there"),
                case("ordinary-a", False, "ordinary-speech", "someone said hey vona"),
                case(
                    "negative-expected",
                    False,
                    "near-miss",
                    "hey luna",
                    expected_phrase="hey vona",
                ),
                case("positive-a", True, "wake-positive", "hello assistant", expected_phrase="assistant"),
            ],
            {"hey vona", "vona"},
        )

        self.assertIn(
            "1 unauthorized-wake negative case(s) do not contain a required wake phrase",
            failures,
        )
        self.assertIn(
            "1 non-unauthorized negative case(s) contain a required wake phrase",
            failures,
        )
        self.assertIn("1 negative case(s) should not set expected_phrase", failures)
        self.assertIn(
            "1 positive case(s) use expected_phrase outside required wake phrases",
            failures,
        )

    def test_recording_plan_accepts_valid_wake_phrase_semantics(self) -> None:
        failures = plan_audit.label_semantic_failures(
            [
                plan_row("positive", "true", "wake-positive", "please hey vona now", "hey vona"),
                plan_row("unauthorized", "false", "unauthorized-wake", "hey vona", ""),
                plan_row("near-miss", "false", "near-miss", "hey luna", ""),
                plan_row("token-near-miss", "false", "near-miss", "vonae is close", ""),
                plan_row("ordinary", "false", "ordinary-speech", "open the calendar", ""),
            ],
            {"hey vona", "vona"},
        )

        self.assertEqual([], failures)


def case(
    case_id: str,
    should_wake: bool,
    category: str,
    text: str,
    *,
    expected_phrase: str = "",
) -> dict[str, object]:
    return {
        "id": case_id,
        "should_wake": should_wake,
        "category": category,
        "text": text,
        "expected_phrase": expected_phrase,
    }


def plan_row(
    row_id: str,
    should_wake: str,
    category: str,
    text: str,
    expected_phrase: str,
) -> dict[str, str]:
    return {
        "id": row_id,
        "should_wake": should_wake,
        "category": category,
        "text": text,
        "expected_phrase": expected_phrase,
    }


if __name__ == "__main__":
    unittest.main()
