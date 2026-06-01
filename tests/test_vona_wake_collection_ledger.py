#!/usr/bin/env python3
"""Tests for vona-wake real corpus collection ledger metadata."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import wave
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


builder = load_script("build_vona_wake_corpus")
auditor = load_script("audit_vona_wake_corpus")
plan_auditor = load_script("audit_vona_wake_recording_plan")


class VonaWakeCollectionLedgerTests(unittest.TestCase):
    def test_builder_embeds_collection_ledger_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            wav_path = base / "clip.wav"
            write_wav(wav_path)
            ledger_path = base / "ledger.json"
            ledger = collection_ledger("speaker-a")
            ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

            manifest = builder.build_manifest(
                [
                    template_row(wav_path),
                    positive_row(wav_path),
                    negative_row(wav_path),
                ],
                base,
                base / "corpus",
                copy=True,
                convert=False,
                corpus_id="ledger-test",
                corpus_version="1",
                corpus_created_by="tests",
                corpus_notes="",
                collection_ledger_path=ledger_path,
            )

            self.assertEqual(ledger, manifest["collection_ledger"])
            self.assertEqual(builder.sha256(ledger_path), manifest["corpus"]["collection_ledger_sha256"])
            self.assertEqual("session-a", manifest["templates"][0]["session_id"])
            self.assertEqual("session-a", manifest["cases"][0]["session_id"])

    def test_builder_preserves_distinct_template_and_case_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            wav_path = base / "clip.wav"
            write_wav(wav_path)
            template = template_row(wav_path)
            template["session_id"] = "template-session"
            case = positive_row(wav_path)
            case["session_id"] = "case-session"

            manifest = builder.build_manifest(
                [template, case],
                base,
                base / "corpus",
                copy=True,
                convert=False,
                corpus_id="session-test",
                corpus_version="1",
                corpus_created_by="tests",
                corpus_notes="",
                collection_ledger_path=None,
            )

        self.assertEqual("template-session", manifest["templates"][0]["session_id"])
        self.assertEqual("case-session", manifest["cases"][0]["session_id"])

    def test_builder_rejects_missing_source_type_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            wav_path = base / "clip.wav"
            write_wav(wav_path)
            row = template_row(wav_path)
            row.pop("source_type")

            with self.assertRaisesRegex(
                SystemExit,
                "row 2 is missing source_type='human-recorded' for a real corpus",
            ):
                builder.build_manifest(
                    [row, positive_row(wav_path)],
                    base,
                    base / "corpus",
                    copy=True,
                    convert=False,
                    corpus_id="source-test",
                    corpus_version="1",
                    corpus_created_by="tests",
                    corpus_notes="",
                    collection_ledger_path=None,
                )

    def test_builder_rejects_non_human_source_type_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            wav_path = base / "clip.wav"
            write_wav(wav_path)
            row = positive_row(wav_path)
            row["source_type"] = "deterministic-pseudo-voice"

            with self.assertRaisesRegex(
                SystemExit,
                "row 3 source_type must be 'human-recorded' for a real corpus",
            ):
                builder.build_manifest(
                    [template_row(wav_path), row],
                    base,
                    base / "corpus",
                    copy=True,
                    convert=False,
                    corpus_id="source-test",
                    corpus_version="1",
                    corpus_created_by="tests",
                    corpus_notes="",
                    collection_ledger_path=None,
                )

    def test_builder_allows_non_human_source_type_when_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            wav_path = base / "clip.wav"
            write_wav(wav_path)
            row = positive_row(wav_path)
            row["source_type"] = "deterministic-pseudo-voice"

            manifest = builder.build_manifest(
                [template_row(wav_path), row],
                base,
                base / "corpus",
                copy=True,
                convert=False,
                corpus_id="source-test",
                corpus_version="1",
                corpus_created_by="tests",
                corpus_notes="",
                collection_ledger_path=None,
                allow_non_human_source=True,
            )

        self.assertEqual("deterministic-pseudo-voice", manifest["cases"][0]["source_type"])

    def test_auditor_rejects_missing_collection_ledger_speaker(self) -> None:
        failures = auditor.collection_ledger_failures(
            auditor.collection_ledger_metadata(
                {"collection_ledger": collection_ledger("speaker-a")}
            ),
            [template_row(Path("template.wav"))],
            [negative_row(Path("guest.wav"), speaker_id="guest-a")],
        )

        self.assertIn(
            "collection_ledger missing speaker consent/provenance entries for: guest-a",
            failures,
        )

    def test_auditor_rejects_missing_collection_ledger_device(self) -> None:
        failures = auditor.collection_ledger_failures(
            auditor.collection_ledger_metadata(
                {"collection_ledger": collection_ledger("speaker-a", devices=["built-in-mic"])}
            ),
            [template_row(Path("template.wav"))],
            [negative_row(Path("guest.wav"), device="usb-mic")],
        )

        self.assertIn(
            "collection_ledger missing device provenance entries for: usb-mic",
            failures,
        )

    def test_auditor_rejects_missing_collection_ledger_session(self) -> None:
        failures = auditor.collection_ledger_failures(
            auditor.collection_ledger_metadata(
                {"collection_ledger": collection_ledger("speaker-a", sessions=["session-a"])}
            ),
            [template_row(Path("template.wav"))],
            [negative_row(Path("guest.wav"), session_id="session-b")],
        )

        self.assertIn(
            "collection_ledger missing session provenance entries for: session-b",
            failures,
        )

    def test_auditor_rejects_missing_template_session_provenance(self) -> None:
        template = template_row(Path("template.wav"))
        template["session_id"] = "template-session"
        failures = auditor.collection_ledger_failures(
            auditor.collection_ledger_metadata(
                {"collection_ledger": collection_ledger("speaker-a", sessions=["session-a"])}
            ),
            [template],
            [negative_row(Path("guest.wav"), session_id="session-a")],
        )

        self.assertIn(
            "collection_ledger missing session provenance entries for: template-session",
            failures,
        )

    def test_auditor_detects_gain_changed_audio_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            template_path = base / "template.wav"
            leaked_case_path = base / "leaked-case.wav"
            duplicate_case_path = base / "duplicate-case.wav"
            write_pattern_wav(template_path, gain=900)
            write_pattern_wav(leaked_case_path, gain=450)
            write_pattern_wav(duplicate_case_path, gain=225)

            leakage = auditor.leakage_report(
                [template_path],
                [leaked_case_path, duplicate_case_path],
                [
                    positive_row(leaked_case_path, case_id="leaked"),
                    positive_row(duplicate_case_path, case_id="duplicate"),
                ],
            )

        self.assertEqual([], leakage["template_case_audio_overlaps"])
        self.assertEqual({}, leakage["duplicate_case_audio"])
        self.assertEqual(1, len(leakage["template_case_fingerprint_overlaps"]))
        self.assertEqual(
            [["duplicate", "leaked"]],
            list(leakage["duplicate_case_fingerprints"].values()),
        )

    def test_auditors_require_evaluation_positives_from_heldout_sessions(self) -> None:
        template = template_row(Path("template.wav"))
        same_session_case = positive_row(Path("same-session.wav"))
        heldout_case = positive_row(Path("heldout-session.wav"), case_id="heldout")
        heldout_case["session_id"] = "session-b"

        failures = auditor.heldout_session_positive_failures(
            [template],
            [same_session_case],
            minimum=1,
        )
        self.assertIn(
            "evaluation speaker speaker-a held-out-session positive cases 0 below required 1 "
            "(template sessions: session-a)",
            failures,
        )
        self.assertEqual(
            [],
            auditor.heldout_session_positive_failures(
                [template],
                [same_session_case, heldout_case],
                minimum=1,
            ),
        )
        self.assertEqual(
            [],
            plan_auditor.heldout_session_positive_failures(
                [template],
                [same_session_case, heldout_case],
                minimum=1,
            ),
        )


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes((1000).to_bytes(2, "little", signed=True) * 320)


def write_pattern_wav(path: Path, gain: int) -> None:
    pattern = [0, gain, 0, -gain, gain // 2, 0, -(gain // 2), 0]
    samples = pattern * 400
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))


def collection_ledger(
    speaker_id: str,
    devices: list[str] | None = None,
    sessions: list[str] | None = None,
) -> dict[str, object]:
    devices = devices or ["usb"]
    sessions = sessions or ["session-a"]
    return {
        "consent_protocol": "test-consent-v1",
        "collection_protocol": "test-recording-v1",
        "collected_by": ["operator-a"],
        "collection_started_at": "2026-06-01T09:00:00Z",
        "collection_completed_at": "2026-06-01T09:10:00Z",
        "speakers": [
            {
                "speaker_id": speaker_id,
                "consent_record": f"consent/{speaker_id}.md",
                "consent_obtained_at": "2026-06-01T09:00:00Z",
            }
        ],
        "devices": [
            {
                "device_id": device,
                "recorder": "test-recorder",
                "sample_rate_hz": 16000,
            }
            for device in devices
        ],
        "sessions": [
            {
                "session_id": session,
                "collected_at": "2026-06-01T09:00:00Z",
                "operator": "operator-a",
            }
            for session in sessions
        ],
    }


def template_row(path: Path) -> dict[str, str]:
    return {
        "role": "template",
        "id": "template-a",
        "path": str(path),
        "phrase": "hey vona",
        "speaker_id": "speaker-a",
        "source_type": "human-recorded",
        "session_id": "session-a",
    }


def positive_row(path: Path, case_id: str = "positive-a") -> dict[str, str]:
    return {
        "role": "case",
        "id": case_id,
        "path": str(path),
        "should_wake": "true",
        "text": "hey vona",
        "expected_phrase": "hey vona",
        "wake_start_ms": "0",
        "speaker_id": "speaker-a",
        "environment": "quiet",
        "distance": "near",
        "device": "usb",
        "session_id": "session-a",
        "category": "wake-positive",
        "source_type": "human-recorded",
        "split": "evaluation",
    }


def negative_row(
    path: Path,
    speaker_id: str = "speaker-a",
    device: str = "usb",
    session_id: str = "session-a",
) -> dict[str, str]:
    return {
        "role": "case",
        "id": "negative-a",
        "path": str(path),
        "should_wake": "false",
        "text": "hey luna",
        "speaker_id": speaker_id,
        "environment": "quiet",
        "distance": "near",
        "device": device,
        "session_id": session_id,
        "category": "near-miss",
        "source_type": "human-recorded",
        "split": "evaluation",
    }


if __name__ == "__main__":
    unittest.main()
