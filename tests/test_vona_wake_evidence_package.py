#!/usr/bin/env python3
"""Tests for vona-wake evidence bundle summaries."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_vona_wake_evidence.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("package_vona_wake_evidence", SCRIPT)
packager = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(packager)


class VonaWakeEvidencePackageTests(unittest.TestCase):
    def test_enforced_cli_rejects_incomplete_real_evidence_but_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            report_dir = base / "reports"
            output_dir = base / "evidence"
            report_dir.mkdir()
            (report_dir / "generated-report.json").write_text(
                json.dumps(
                    {
                        "false_positives": 0,
                        "false_negatives": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            (report_dir / "generated-manifest.json").write_text(
                json.dumps({"corpus": {"source": "synthetic-generated"}}),
                encoding="utf-8",
            )
            (report_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    str(SCRIPT),
                    "--report-dir",
                    str(report_dir),
                    "--output-dir",
                    str(output_dir),
                    "--enforce",
                ],
                cwd=SCRIPT.parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(0, completed.returncode)
            manifest = json.loads((output_dir / "evidence-manifest.json").read_text())
            summary = (output_dir / "evidence-summary.md").read_text(encoding="utf-8")
            self.assertFalse(manifest["accepted"])
            self.assertFalse(manifest["real_evidence_status"]["accepted"])
            self.assertIn("| `generated_regression` | pass |", summary)
            self.assertIn("| `readiness_audit` | fail |", summary)
            self.assertIn("| `real_evaluation` | fail |", summary)
            self.assertIn("| `acceptance` | fail |", summary)
            self.assertIn("- audit-report.json is missing", summary)
            self.assertIn("- real-report.json is missing", summary)
            self.assertIn("- threshold-selection-report.json is missing", summary)
            self.assertIn("## Next Actions", summary)
            self.assertIn(
                "- Run VONA_WAKE_REQUIRE_REAL_EVAL=1 scripts/run_vona_wake_eval.sh /path/to/corpus/manifest.json",
                summary,
            )

    def test_markdown_summary_includes_coverage_and_provenance(self) -> None:
        markdown = packager.markdown_summary(
            {
                "accepted": True,
                "corpus": {"id": "fixture", "version": "1"},
                "manifest_sha256": "manifest-hash",
                "git": {"branch": "test", "commit": "abc123"},
                "acceptance": {
                    "precision": 1.0,
                    "precision_lower_bound": 0.99,
                    "recall": 1.0,
                    "recall_lower_bound": 0.99,
                    "repeated_positive_wake_events": 0,
                    "false_wakes_per_hour": 0.0,
                    "false_wakes_per_hour_upper_bound": 0.01,
                    "max_detection_latency_ms": 200,
                    "threshold_selection": {
                        "accepted": True,
                        "threshold_sweep_points": 5,
                        "calibration_passing_points": 3,
                        "evaluation_passing_points": 2,
                        "selected": {
                            "candidate_threshold": 0.88,
                            "accept_threshold": 0.92,
                        },
                    },
                    "failures": [],
                },
                "real_evidence_status": {
                    "accepted": True,
                    "corpus": {"id": "fixture", "version": "1"},
                    "stages": {
                        "generated_regression": {"passed": True},
                        "readiness_audit": {"passed": True},
                        "real_evaluation": {"passed": True},
                        "threshold_selection": {"passed": True},
                        "acceptance": {"passed": True},
                    },
                },
                "coverage": {
                    "speakers": 2,
                    "speaker_ids": ["speaker-a", "speaker-b"],
                    "sessions": 2,
                    "session_ids": ["session-a", "session-b"],
                    "devices": 1,
                    "device_ids": ["usb-mic"],
                },
                "provenance": {
                    "collection_ledger_present": True,
                    "collected_by": ["operator-a"],
                    "speakers": ["speaker-a", "speaker-b"],
                    "devices": ["usb-mic"],
                    "sessions": ["session-a", "session-b"],
                },
                "artifacts": [
                    {"name": "real-report.json", "bytes": 123, "sha256": "hash"},
                ],
            }
        )

        self.assertIn("| `sessions` | 2 | `session-a`, `session-b` |", markdown)
        self.assertIn("- Device records: `1`", markdown)
        self.assertIn("- Session records: `2`", markdown)
        self.assertIn("- Calibration passing points: `3`", markdown)
        self.assertIn("- Repeated positive wake events: `0`", markdown)
        self.assertIn("## Real Evidence Status", markdown)
        self.assertIn("| `generated_regression` | pass |", markdown)
        self.assertIn("| `acceptance` | pass |", markdown)

    def test_markdown_summary_surfaces_failed_real_evidence_stages(self) -> None:
        markdown = packager.markdown_summary(
            {
                "accepted": False,
                "corpus": {},
                "manifest_sha256": "",
                "git": {"branch": "test", "commit": "abc123"},
                "acceptance": {
                    "failures": ["audit-report.json and real-report.json are required for acceptance"],
                },
                "real_evidence_status": {
                    "accepted": False,
                    "corpus": {},
                    "failures": [
                        "audit-report.json is missing",
                        "real-report.json is missing",
                    ],
                    "next_actions": [
                        "Run scripts/audit_vona_wake_corpus.py --enforce --json /path/to/corpus/manifest.json",
                    ],
                    "stages": {
                        "generated_regression": {"passed": True},
                        "readiness_audit": {"passed": False},
                        "real_evaluation": {"passed": False},
                        "threshold_selection": {"passed": False},
                        "acceptance": {"passed": False},
                    },
                },
                "coverage": {},
                "provenance": {},
                "artifacts": [
                    {"name": "generated-report.json", "bytes": 123, "sha256": "hash"},
                ],
            }
        )

        self.assertIn("- Accepted: `false`", markdown)
        self.assertIn("| `generated_regression` | pass |", markdown)
        self.assertIn("| `readiness_audit` | fail |", markdown)
        self.assertIn("| `real_evaluation` | fail |", markdown)
        self.assertIn("| `threshold_selection` | fail |", markdown)
        self.assertIn("- audit-report.json is missing", markdown)
        self.assertIn("- real-report.json is missing", markdown)
        self.assertIn("## Next Actions", markdown)


if __name__ == "__main__":
    unittest.main()
