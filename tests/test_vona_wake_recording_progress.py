#!/usr/bin/env python3
"""Tests for recording progress summaries used during real voice collection."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_vona_wake_recording_progress.py"
spec = importlib.util.spec_from_file_location("check_vona_wake_recording_progress", SCRIPT)
progress = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(progress)


class VonaWakeRecordingProgressTests(unittest.TestCase):
    def test_report_groups_status_by_device_and_session(self) -> None:
        report = progress.build_report(
            "recordings.csv",
            [
                result("ok", "built-in-mic", "session-a"),
                result("missing", "built-in-mic", "session-b"),
                result("quiet", "usb-mic", "session-b"),
            ],
        )

        self.assertFalse(report["complete"])
        self.assertEqual({"missing": 1, "ok": 1}, report["device_counts"]["built-in-mic"])
        self.assertEqual({"quiet": 1}, report["device_counts"]["usb-mic"])
        self.assertEqual({"ok": 1}, report["session_counts"]["session-a"])
        self.assertEqual({"missing": 1, "quiet": 1}, report["session_counts"]["session-b"])

    def test_report_tracks_provenance_failures_without_hiding_audio_status(self) -> None:
        report = progress.build_report(
            "recordings.csv",
            [
                result("ok", "built-in-mic", "session-a", source_type=""),
                result(
                    "missing",
                    "built-in-mic",
                    "session-b",
                    source_type="deterministic-pseudo-voice",
                ),
            ],
        )

        self.assertFalse(report["complete"])
        self.assertEqual(2, report["provenance_failures"])
        self.assertEqual({"missing": 1, "ok": 1}, report["status_counts"])
        self.assertEqual(
            {
                "deterministic-pseudo-voice": 1,
                "missing": 1,
            },
            report["source_type_counts"],
        )
        self.assertEqual(2, len(report["failures"]))

    def test_source_type_error_is_disabled_for_non_release_experiments(self) -> None:
        row = {"source_type": "deterministic-pseudo-voice"}

        self.assertIsNone(progress.source_type_error(row, allow_non_human_source=True))
        self.assertEqual(
            "source_type 'deterministic-pseudo-voice' is not human-recorded",
            progress.source_type_error(row, allow_non_human_source=False),
        )

    def test_read_rows_rejects_missing_release_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "recordings.csv"
            csv_path.write_text("role,id,path\ncase,case-a,clip.wav\n", encoding="utf-8")

            with self.assertRaisesRegex(
                SystemExit,
                "missing required columns",
            ) as raised:
                progress.read_rows(csv_path)

        self.assertIn("source_type", str(raised.exception))
        self.assertIn("planned_duration_s", str(raised.exception))


def result(
    status: str,
    device: str,
    session_id: str,
    *,
    source_type: str = "human-recorded",
) -> dict[str, object]:
    source_error = None
    if source_type == "":
        source_error = "source_type is missing; expected human-recorded"
    elif source_type != "human-recorded":
        source_error = f"source_type {source_type!r} is not human-recorded"
    return {
        "row": 2,
        "id": f"{status}-{device}-{session_id}",
        "role": "case",
        "path": "clip.wav",
        "category": "wake-positive",
        "device": device,
        "session_id": session_id,
        "source_type": source_type,
        "provenance_error": source_error,
        "status": status,
        "error": "fixture",
    }


if __name__ == "__main__":
    unittest.main()
