#!/usr/bin/env python3
"""Package vona-wake evaluation artifacts into a reproducible evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from summarize_vona_wake_real_evidence import build_status, load_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("target/vona-wake-eval"),
        help="Directory containing wake evaluation reports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("target/vona-wake-evidence"),
        help="Directory to write the evidence bundle",
    )
    parser.add_argument("--zip", action="store_true", help="Also write a ZIP archive")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero if acceptance failed")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = collect_artifacts(args.report_dir)
    copied = copy_artifacts(artifact_paths, args.output_dir)
    acceptance = acceptance_result(copied)
    real_evidence_status = evidence_status(args.output_dir)
    accepted = bool(acceptance.get("accepted")) and bool(real_evidence_status.get("accepted"))
    corpus = corpus_info(copied)
    coverage = coverage_info(copied)
    provenance = provenance_info(copied)
    manifest = {
        "git": git_info(),
        "corpus": corpus,
        "coverage": coverage,
        "provenance": provenance,
        "manifest_sha256": manifest_sha256(copied),
        "report_dir": str(args.report_dir),
        "output_dir": str(args.output_dir),
        "accepted": accepted,
        "acceptance": acceptance,
        "real_evidence_status": real_evidence_status,
        "artifacts": [
            {
                "name": name,
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sorted(copied.items())
        ],
    }
    manifest_path = args.output_dir / "evidence-manifest.json"
    summary_path = args.output_dir / "evidence-summary.md"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(markdown_summary(manifest), encoding="utf-8")
    if args.zip:
        write_zip(args.output_dir)
    print(summary_path)
    return 1 if args.enforce and not accepted else 0


def collect_artifacts(report_dir: Path) -> dict[str, Path]:
    required = {
        "generated-report.json": report_dir / "generated-report.json",
        "summary.md": report_dir / "summary.md",
    }
    optional = {
        "generated-manifest.json": report_dir / "generated-manifest.json",
        "audit-report.json": report_dir / "audit-report.json",
        "real-report.json": report_dir / "real-report.json",
        "threshold-selection-report.json": report_dir / "threshold-selection-report.json",
        "real-evidence-status.md": report_dir / "real-evidence-status.md",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing required evidence artifact(s): {', '.join(missing)}")
    artifacts = dict(required)
    artifacts.update({name: path for name, path in optional.items() if path.exists()})
    return artifacts


def copy_artifacts(artifacts: dict[str, Path], output_dir: Path) -> dict[str, Path]:
    copied = {}
    for name, source in artifacts.items():
        target = output_dir / name
        shutil.copy2(source, target)
        copied[name] = target
    return copied


def acceptance_result(artifacts: dict[str, Path]) -> dict[str, object]:
    audit = artifacts.get("audit-report.json")
    real = artifacts.get("real-report.json")
    threshold = artifacts.get("threshold-selection-report.json")
    if not audit or not real:
        return {
            "accepted": False,
            "failures": ["audit-report.json and real-report.json are required for acceptance"],
        }
    command = [
        "scripts/check_vona_wake_acceptance.py",
        "--json",
        "--audit-report",
        str(audit),
        "--real-report",
        str(real),
        "--require-threshold-selection",
    ]
    if threshold:
        command.extend(["--threshold-report", str(threshold)])
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.stdout.strip():
        result = json.loads(completed.stdout)
    else:
        return {
            "accepted": False,
            "failures": [completed.stderr.strip() or "acceptance checker produced no JSON"],
        }
    if threshold:
        result["threshold_selection"] = json.loads(threshold.read_text(encoding="utf-8"))
    return result


def evidence_status(report_dir: Path) -> dict[str, object]:
    return build_status(report_dir, load_reports(report_dir))


def corpus_info(artifacts: dict[str, Path]) -> dict[str, object]:
    for artifact_name in ("real-report.json", "audit-report.json"):
        artifact = artifacts.get(artifact_name)
        if not artifact:
            continue
        report = json.loads(artifact.read_text(encoding="utf-8"))
        corpus = report.get("corpus")
        if isinstance(corpus, dict):
            return {
                "id": corpus.get("id") or "",
                "version": corpus.get("version") or "",
                "source": corpus.get("source") or "",
                "created_by": corpus.get("created_by") or "",
                "notes": corpus.get("notes") or "",
                "collection_ledger_sha256": corpus.get("collection_ledger_sha256") or "",
            }
    return {}


def coverage_info(artifacts: dict[str, Path]) -> dict[str, object]:
    real = artifacts.get("real-report.json")
    if not real:
        return {}
    report = json.loads(real.read_text(encoding="utf-8"))
    coverage = report.get("coverage")
    return coverage if isinstance(coverage, dict) else {}


def provenance_info(artifacts: dict[str, Path]) -> dict[str, object]:
    audit = artifacts.get("audit-report.json")
    if not audit:
        return {}
    report = json.loads(audit.read_text(encoding="utf-8"))
    ledger = report.get("collection_ledger")
    if not isinstance(ledger, dict):
        return {}
    return {
        "collection_ledger_present": bool(ledger.get("present")),
        "collected_by": ledger.get("collected_by") or [],
        "speakers": ledger.get("speaker_ids") or [],
        "devices": ledger.get("device_ids") or [],
        "sessions": ledger.get("session_ids") or [],
    }


def manifest_sha256(artifacts: dict[str, Path]) -> str:
    for artifact_name in ("real-report.json", "audit-report.json"):
        artifact = artifacts.get(artifact_name)
        if not artifact:
            continue
        report = json.loads(artifact.read_text(encoding="utf-8"))
        value = str(report.get("manifest_sha256") or "").strip()
        if value:
            return value
    return ""


def git_info() -> dict[str, str]:
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "status": run_git("status", "--short"),
    }


def run_git(*args: str) -> str:
    completed = subprocess.run(["git", *args], check=False, text=True, capture_output=True)
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_summary(manifest: dict[str, object]) -> str:
    acceptance = manifest["acceptance"]
    lines = [
        "# Vona Wake Evidence Bundle",
        "",
        f"- Accepted: `{str(manifest['accepted']).lower()}`",
        f"- Corpus: `{corpus_label(manifest.get('corpus', {}))}`",
        f"- Manifest SHA-256: `{manifest.get('manifest_sha256') or 'n/a'}`",
        f"- Git branch: `{manifest['git']['branch']}`",
        f"- Git commit: `{manifest['git']['commit']}`",
        "",
        "## Acceptance",
        "",
        f"- Precision: `{acceptance.get('precision', 'n/a')}`",
        f"- Precision lower bound: `{acceptance.get('precision_lower_bound', 'n/a')}`",
        f"- Recall: `{acceptance.get('recall', 'n/a')}`",
        f"- Recall lower bound: `{acceptance.get('recall_lower_bound', 'n/a')}`",
        f"- Repeated positive wake events: `{acceptance.get('repeated_positive_wake_events', 'n/a')}`",
        f"- False wakes/hour: `{acceptance.get('false_wakes_per_hour', 'n/a')}`",
        f"- False wakes/hour upper bound: `{acceptance.get('false_wakes_per_hour_upper_bound', 'n/a')}`",
        f"- Max detection latency ms: `{acceptance.get('max_detection_latency_ms', 'n/a')}`",
        "",
    ]
    status = manifest.get("real_evidence_status")
    if isinstance(status, dict):
        lines.extend(
            [
                "## Real Evidence Status",
                "",
                f"- Accepted: `{str(status.get('accepted', False)).lower()}`",
                f"- Corpus: `{corpus_label(status.get('corpus', {}))}`",
                "",
                "| Stage | Status |",
                "|---|---|",
            ]
        )
        for name, stage_info in (status.get("stages") or {}).items():
            passed = bool(stage_info.get("passed")) if isinstance(stage_info, dict) else False
            lines.append(f"| `{name}` | {'pass' if passed else 'fail'} |")
        lines.append("")
    threshold = acceptance.get("threshold_selection")
    if threshold:
        selected = threshold.get("selected") or {}
        lines.extend(
            [
                "## Threshold Selection",
                "",
                f"- Accepted: `{str(threshold.get('accepted', False)).lower()}`",
                f"- Sweep points: `{threshold.get('threshold_sweep_points', 'n/a')}`",
                f"- Calibration passing points: `{threshold.get('calibration_passing_points', 'n/a')}`",
                f"- Evaluation passing points: `{threshold.get('evaluation_passing_points', 'n/a')}`",
                f"- Candidate threshold: `{selected.get('candidate_threshold', 'n/a')}`",
                f"- Accept threshold: `{selected.get('accept_threshold', 'n/a')}`",
                "",
            ]
        )
    coverage = manifest.get("coverage")
    if isinstance(coverage, dict) and coverage:
        lines.extend(
            [
                "## Coverage",
                "",
                "| Dimension | Count | Values |",
                "|---|---:|---|",
            ]
        )
        for dimension, ids_key in [
            ("speakers", "speaker_ids"),
            ("environments", "environment_ids"),
            ("distances", "distance_ids"),
            ("devices", "device_ids"),
            ("sessions", "session_ids"),
            ("categories", "category_ids"),
        ]:
            lines.append(
                f"| `{dimension}` | {coverage.get(dimension, 'n/a')} | {csv_values(coverage.get(ids_key, []))} |"
            )
        lines.append("")
    provenance = manifest.get("provenance")
    if isinstance(provenance, dict) and provenance:
        lines.extend(
            [
                "## Provenance",
                "",
                f"- Collection ledger present: `{str(provenance.get('collection_ledger_present', False)).lower()}`",
                f"- Operators: {csv_values(provenance.get('collected_by', []))}",
                f"- Speaker records: `{len(provenance.get('speakers', []))}`",
                f"- Device records: `{len(provenance.get('devices', []))}`",
                f"- Session records: `{len(provenance.get('sessions', []))}`",
                "",
            ]
        )
    failures = combined_failures(acceptance, status)
    if failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
        lines.append("")
    next_actions = combined_next_actions(status)
    if next_actions:
        lines.extend(["## Next Actions", ""])
        lines.extend(f"- {action}" for action in next_actions)
        lines.append("")
    lines.extend(
        [
            "## Artifacts",
            "",
            "| Name | Bytes | SHA-256 |",
            "|---|---:|---|",
        ]
    )
    for artifact in manifest["artifacts"]:
        lines.append(f"| `{artifact['name']}` | {artifact['bytes']} | `{artifact['sha256']}` |")
    lines.append("")
    return "\n".join(lines)


def combined_failures(acceptance: dict[str, object], status: object) -> list[str]:
    failures: list[str] = []
    failures.extend(str(failure) for failure in acceptance.get("failures", []))
    if isinstance(status, dict):
        failures.extend(str(failure) for failure in status.get("failures", []))
    return dedupe(failures)


def combined_next_actions(status: object) -> list[str]:
    if not isinstance(status, dict):
        return []
    return dedupe(str(action) for action in status.get("next_actions", []))


def dedupe(values) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def corpus_label(corpus: object) -> str:
    if not isinstance(corpus, dict) or not corpus.get("id"):
        return "n/a"
    version = corpus.get("version") or "unversioned"
    return f"{corpus['id']}@{version}"


def csv_values(values: object) -> str:
    if not isinstance(values, list) or not values:
        return "`n/a`"
    return ", ".join(f"`{value}`" for value in values)


def write_zip(output_dir: Path) -> None:
    archive = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                handle.write(path, path.name)


if __name__ == "__main__":
    raise SystemExit(main())
