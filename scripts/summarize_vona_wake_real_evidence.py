#!/usr/bin/env python3
"""Summarize release-grade vona-wake real-voice evidence status."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ARTIFACTS = {
    "generated_report": "generated-report.json",
    "generated_manifest": "generated-manifest.json",
    "audit_report": "audit-report.json",
    "real_report": "real-report.json",
    "threshold_report": "threshold-selection-report.json",
    "summary": "summary.md",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("target/vona-wake-eval"))
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero unless real evidence is accepted")
    args = parser.parse_args()

    reports = load_reports(args.report_dir)
    status = build_status(args.report_dir, reports)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown_status(status), encoding="utf-8")
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print_text_status(status)
    return 1 if args.enforce and not status["accepted"] else 0


def load_reports(report_dir: Path) -> dict[str, object]:
    loaded: dict[str, object] = {}
    for key, filename in ARTIFACTS.items():
        path = report_dir / filename
        if not path.exists():
            loaded[key] = None
            continue
        if filename.endswith(".json"):
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            loaded[key] = {"path": str(path)}
    return loaded


def build_status(report_dir: Path, reports: dict[str, object]) -> dict[str, object]:
    generated = reports.get("generated_report")
    audit = reports.get("audit_report")
    real = reports.get("real_report")
    threshold = reports.get("threshold_report")
    acceptance = acceptance_status(report_dir, audit, real, threshold)
    failures: list[str] = []
    next_actions: list[str] = []
    stages = {
        "generated_regression": generated_stage(generated, reports.get("generated_manifest")),
        "readiness_audit": audit_stage(audit),
        "real_evaluation": real_stage(real),
        "threshold_selection": threshold_stage(threshold),
        "acceptance": acceptance,
    }
    for stage in stages.values():
        failures.extend(stage.get("failures", []))
        next_actions.extend(stage.get("next_actions", []))
    corpus = corpus_info(real, audit)
    manifest_hashes = manifest_hashes_for(audit, real)
    return {
        "accepted": bool(acceptance.get("passed")),
        "report_dir": str(report_dir),
        "corpus": corpus,
        "manifest_sha256": manifest_hashes,
        "stages": stages,
        "metrics": real_metrics(real),
        "coverage": coverage(real),
        "failures": dedupe(failures),
        "next_actions": dedupe(next_actions),
    }


def generated_stage(generated, generated_manifest) -> dict[str, object]:
    failures = []
    next_actions = []
    if not generated:
        failures.append("generated-report.json is missing")
        next_actions.append("Run scripts/run_vona_wake_eval.sh to create generated regression artifacts")
        return stage(False, failures, next_actions)
    if not generated_manifest:
        failures.append("generated-manifest.json is missing")
        next_actions.append("Re-run scripts/run_vona_wake_eval.sh to emit generated-manifest.json")
    if generated.get("false_positives") != 0 or generated.get("false_negatives") != 0:
        failures.append("generated regression has wake classification failures")
        next_actions.append("Fix generated regression failures before collecting release evidence")
    return stage(not failures, failures, next_actions)


def audit_stage(audit) -> dict[str, object]:
    if not audit:
        return stage(
            False,
            ["audit-report.json is missing"],
            ["Run scripts/audit_vona_wake_corpus.py --enforce --json /path/to/corpus/manifest.json"],
        )
    failures = [str(failure) for failure in audit.get("failures", [])]
    if not audit.get("ready", False) and not failures:
        failures.append("readiness audit did not pass")
    next_actions = ["Fix readiness audit failures and rebuild the corpus manifest"] if failures else []
    return stage(not failures, failures, next_actions)


def real_stage(real) -> dict[str, object]:
    if not real:
        return stage(
            False,
            ["real-report.json is missing"],
            ["Run VONA_WAKE_REQUIRE_REAL_EVAL=1 scripts/run_vona_wake_eval.sh /path/to/corpus/manifest.json"],
        )
    failures = []
    corpus = real.get("corpus") or {}
    if corpus.get("source") != "human-recorded":
        failures.append("real report corpus.source is not human-recorded")
    for key in [
        "precision",
        "recall",
        "repeated_positive_wake_events",
        "false_wakes_per_hour",
        "max_detection_latency_ms",
        "manifest_sha256",
    ]:
        if real.get(key) is None:
            failures.append(f"real report missing {key}")
    next_actions = ["Re-run real voice evaluation with the release corpus"] if failures else []
    return stage(not failures, failures, next_actions)


def threshold_stage(threshold) -> dict[str, object]:
    if not threshold:
        return stage(
            False,
            ["threshold-selection-report.json is missing"],
            ["Run scripts/select_vona_wake_threshold.py --real-report target/vona-wake-eval/real-report.json --json"],
        )
    failures = [f"threshold: {failure}" for failure in threshold.get("failures", [])]
    if not threshold.get("accepted", False) and not failures:
        failures.append("threshold selection did not pass")
    next_actions = ["Fix threshold selection failures or collect stronger calibration/evaluation data"] if failures else []
    return stage(not failures, failures, next_actions)


def acceptance_status(report_dir: Path, audit, real, threshold) -> dict[str, object]:
    if not audit or not real:
        return stage(
            False,
            ["audit and real reports are required for acceptance"],
            ["Produce audit-report.json and real-report.json before running acceptance"],
        )
    command = [
        "scripts/check_vona_wake_acceptance.py",
        "--json",
        "--audit-report",
        str(report_dir / ARTIFACTS["audit_report"]),
        "--real-report",
        str(report_dir / ARTIFACTS["real_report"]),
        "--require-threshold-selection",
    ]
    if threshold:
        command.extend(["--threshold-report", str(report_dir / ARTIFACTS["threshold_report"])])
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return stage(
            False,
            [completed.stderr.strip() or "acceptance checker did not emit JSON"],
            ["Re-run scripts/check_vona_wake_acceptance.py manually to inspect the error"],
        )
    failures = [str(failure) for failure in result.get("failures", [])]
    next_actions = ["Fix acceptance failures before making a real-world reliability claim"] if failures else []
    output = stage(not failures, failures, next_actions)
    output["details"] = result
    return output


def stage(passed: bool, failures: list[str], next_actions: list[str]) -> dict[str, object]:
    return {
        "passed": passed,
        "failures": failures,
        "next_actions": next_actions,
    }


def corpus_info(real, audit) -> dict[str, object]:
    for report in (real, audit):
        if isinstance(report, dict) and isinstance(report.get("corpus"), dict):
            return report["corpus"]
    return {}


def manifest_hashes_for(audit, real) -> dict[str, str]:
    return {
        "audit": str(audit.get("manifest_sha256") or "") if isinstance(audit, dict) else "",
        "real": str(real.get("manifest_sha256") or "") if isinstance(real, dict) else "",
    }


def real_metrics(real) -> dict[str, object]:
    if not isinstance(real, dict):
        return {}
    confidence = real.get("confidence_intervals") or {}
    return {
        "precision": real.get("precision"),
        "precision_lower_bound": confidence.get("precision_lower_bound"),
        "recall": real.get("recall"),
        "recall_lower_bound": confidence.get("recall_lower_bound"),
        "repeated_positive_wake_events": real.get("repeated_positive_wake_events"),
        "false_wakes_per_hour": real.get("false_wakes_per_hour"),
        "false_wakes_per_hour_upper_bound": confidence.get("false_wakes_per_hour_upper_bound"),
        "max_detection_latency_ms": real.get("max_detection_latency_ms"),
    }


def coverage(real) -> dict[str, object]:
    if not isinstance(real, dict):
        return {}
    return real.get("coverage") or {}


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def print_text_status(status: dict[str, object]) -> None:
    print(f"status={'accepted' if status['accepted'] else 'not_accepted'}")
    corpus = status.get("corpus") or {}
    if corpus:
        print(f"corpus={corpus.get('id', '')}@{corpus.get('version', '')}")
    for name, stage_info in status["stages"].items():
        print(f"{name}={'pass' if stage_info['passed'] else 'fail'}")
    failures = status["failures"]
    if failures:
        print("failures:")
        for failure in failures[:50]:
            print(f"- {failure}")
        if len(failures) > 50:
            print(f"- ... {len(failures) - 50} more")
    next_actions = status["next_actions"]
    if next_actions:
        print("next_actions:")
        for action in next_actions:
            print(f"- {action}")


def markdown_status(status: dict[str, object]) -> str:
    lines = [
        "# Vona Wake Real Evidence Status",
        "",
        f"- Accepted: `{str(status['accepted']).lower()}`",
        f"- Corpus: `{corpus_label(status.get('corpus', {}))}`",
        f"- Audit manifest SHA-256: `{status['manifest_sha256'].get('audit') or 'n/a'}`",
        f"- Real manifest SHA-256: `{status['manifest_sha256'].get('real') or 'n/a'}`",
        "",
        "## Stages",
        "",
        "| Stage | Status |",
        "|---|---|",
    ]
    for name, stage_info in status["stages"].items():
        lines.append(f"| `{name}` | {'pass' if stage_info['passed'] else 'fail'} |")
    lines.extend(["", "## Metrics", "", "| Metric | Value |", "|---|---:|"])
    for key, value in status["metrics"].items():
        lines.append(f"| `{key}` | `{value if value is not None else 'n/a'}` |")
    if status["coverage"]:
        coverage_data = status["coverage"]
        lines.extend(["", "## Coverage", "", "| Dimension | Count | IDs |", "|---|---:|---|"])
        for dimension in ["speakers", "environments", "distances", "devices", "sessions", "categories"]:
            ids = coverage_data.get(f"{dimension[:-1]}_ids") or coverage_data.get(f"{dimension}_ids") or []
            lines.append(f"| `{dimension}` | `{coverage_data.get(dimension, 'n/a')}` | {csv_ids(ids)} |")
    if status["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in status["failures"])
    if status["next_actions"]:
        lines.extend(["", "## Next Actions", ""])
        lines.extend(f"- {action}" for action in status["next_actions"])
    lines.append("")
    return "\n".join(lines)


def corpus_label(corpus: object) -> str:
    if not isinstance(corpus, dict) or not corpus.get("id"):
        return "n/a"
    return f"{corpus.get('id')}@{corpus.get('version') or 'unversioned'}"


def csv_ids(values: object) -> str:
    if not values:
        return "n/a"
    return ", ".join(f"`{value}`" for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
