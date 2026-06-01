#!/usr/bin/env python3
"""Tests for the integrated vona-wake evaluation runner."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_vona_wake_eval.sh"


class VonaWakeEvalRunnerTests(unittest.TestCase):
    def test_required_real_eval_without_manifest_fails_with_status_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "vona-wake-eval"
            env = {
                **os.environ,
                "VONA_WAKE_REQUIRE_REAL_EVAL": "1",
                "VONA_WAKE_EVAL_REPORT_DIR": str(report_dir),
                "PYTHONPYCACHEPREFIX": "/private/tmp/vona-wake-pycache",
            }

            completed = subprocess.run(
                [str(RUNNER)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(0, completed.returncode)
            self.assertIn(
                "VONA_WAKE_REQUIRE_REAL_EVAL=1 but no real manifest was provided",
                completed.stdout,
            )
            status = report_dir / "real-evidence-status.md"
            self.assertTrue(status.exists())
            status_text = status.read_text(encoding="utf-8")
            self.assertIn("- Accepted: `false`", status_text)
            self.assertIn("- real-report.json is missing", status_text)
            self.assertIn("`acceptance` | fail", status_text)
            self.assertTrue((report_dir / "generated-report.json").exists())
            self.assertTrue((report_dir / "generated-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
