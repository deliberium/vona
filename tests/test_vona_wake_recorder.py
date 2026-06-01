#!/usr/bin/env python3
"""Tests for the guided vona-wake corpus recorder preflight checks."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_vona_wake_corpus.py"
spec = importlib.util.spec_from_file_location("record_vona_wake_corpus", SCRIPT)
recorder = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(recorder)


class VonaWakeRecorderTests(unittest.TestCase):
    def test_read_rows_rejects_missing_release_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "recordings.csv"
            csv_path.write_text("role,id,path\ncase,case-a,clip.wav\n", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "missing required columns") as raised:
                recorder.read_rows(csv_path)

        self.assertIn("source_type", str(raised.exception))
        self.assertIn("planned_duration_s", str(raised.exception))

    def test_validate_rows_rejects_non_human_source_by_default(self) -> None:
        row = recording_row()
        row["source_type"] = "deterministic-pseudo-voice"

        with self.assertRaisesRegex(
            SystemExit,
            "row 2 source_type must be 'human-recorded' for recording",
        ):
            recorder.validate_rows([row], allow_non_human_source=False)

    def test_validate_rows_allows_non_human_source_when_explicit(self) -> None:
        row = recording_row()
        row["source_type"] = "deterministic-pseudo-voice"

        recorder.validate_rows([row], allow_non_human_source=True)

    def test_validate_rows_rejects_missing_duration_before_prompting(self) -> None:
        row = recording_row()
        row["planned_duration_s"] = ""

        with self.assertRaisesRegex(
            SystemExit,
            "row 2 is missing required value 'planned_duration_s'",
        ):
            recorder.validate_rows([row], allow_non_human_source=False)


def recording_row() -> dict[str, str]:
    return {
        "role": "case",
        "id": "case-a",
        "path": "raw/case-a.wav",
        "phrase": "",
        "should_wake": "true",
        "text": "hey vona",
        "expected_phrase": "hey vona",
        "wake_start_ms": "0",
        "speaker_id": "speaker-a",
        "environment": "quiet-office",
        "distance": "near",
        "device": "built-in-mic",
        "session_id": "session-a",
        "category": "wake-positive",
        "source_type": "human-recorded",
        "split": "evaluation",
        "planned_duration_s": "3",
    }


if __name__ == "__main__":
    unittest.main()
