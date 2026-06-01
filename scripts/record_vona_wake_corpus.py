#!/usr/bin/env python3
"""Record WAV files from a vona-wake recording-plan CSV."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import shlex
import subprocess
import sys
from pathlib import Path


REQUIRED_COLUMNS = {
    "role",
    "id",
    "path",
    "phrase",
    "should_wake",
    "text",
    "expected_phrase",
    "wake_start_ms",
    "speaker_id",
    "environment",
    "distance",
    "device",
    "session_id",
    "category",
    "source_type",
    "split",
    "planned_duration_s",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="recording-plan CSV")
    parser.add_argument(
        "--recorder",
        choices=["auto", "afrecord", "sox", "rec", "arecord"],
        default="auto",
        help="Recorder backend to use; default auto-detects a local recorder",
    )
    parser.add_argument(
        "--record-command",
        help=(
            "Custom command template. Supports {path}, {duration}, {id}, "
            "{text}, {speaker_id}, {category}. Example: rec -r 16000 -c 1 -b 16 {path} trim 0 {duration}"
        ),
    )
    parser.add_argument("--limit", type=int, help="Maximum rows to record")
    parser.add_argument("--start-id", help="Skip rows until this id is reached")
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Record rows even when the target WAV already exists",
    )
    parser.add_argument(
        "--roles",
        default="template,case",
        help="Comma-separated role filter, default template,case",
    )
    parser.add_argument(
        "--splits",
        help="Optional comma-separated split filter such as enrollment,calibration,evaluation",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Record without per-row confirmation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the recording worklist and commands without recording",
    )
    parser.add_argument(
        "--allow-non-human-source",
        action="store_true",
        help=(
            "Allow rows whose source_type is missing or not human-recorded. "
            "This is for non-release experiments only."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    validate_rows(rows, args.allow_non_human_source)
    rows = select_rows(rows, args)
    recorder = resolve_recorder(args)
    results = []
    for row in rows:
        path = resolve_path(args.csv.parent, Path(row.get("path", "")))
        command = command_for(row, path, recorder, args.record_command)
        result = {
            "id": row.get("id", ""),
            "path": str(path),
            "role": row.get("role", ""),
            "split": row.get("split", ""),
            "category": row.get("category", ""),
            "duration_s": duration(row),
            "recorded": False,
            "skipped": False,
            "command": command,
            "error": None,
        }
        if path.exists() and not args.include_existing:
            result["skipped"] = True
            results.append(result)
            continue
        if args.dry_run:
            results.append(result)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print_prompt(row, path, command)
        if not args.yes and not confirm():
            result["skipped"] = True
            results.append(result)
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            result["recorded"] = True
        else:
            result["error"] = f"recorder exited with {completed.returncode}"
            results.append(result)
            return emit(results, args, exit_code=1)
        results.append(result)
    return emit(results, args)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def validate_rows(rows: list[dict[str, str]], allow_non_human_source: bool) -> None:
    for row_number, row in enumerate(rows, start=2):
        for key in ["role", "id", "path", "planned_duration_s"]:
            if not row.get(key, "").strip():
                raise SystemExit(f"row {row_number} is missing required value {key!r}")
        role = row.get("role", "").lower()
        if role not in {"template", "case"}:
            raise SystemExit(f"row {row_number} role must be template or case, got {role!r}")
        if not allow_non_human_source:
            source_type = row.get("source_type", "").strip()
            if not source_type:
                raise SystemExit(
                    f"row {row_number} is missing source_type='human-recorded' for recording"
                )
            if source_type != "human-recorded":
                raise SystemExit(
                    f"row {row_number} source_type must be 'human-recorded' for recording, "
                    f"got {source_type!r}"
                )
        duration(row)


def select_rows(rows: list[dict[str, str]], args) -> list[dict[str, str]]:
    roles = split_filter(args.roles)
    splits = split_filter(args.splits) if args.splits else None
    selected = []
    started = args.start_id is None
    for row in rows:
        if not started:
            started = row.get("id") == args.start_id
            if not started:
                continue
        if roles and row.get("role", "").lower() not in roles:
            continue
        if splits and row.get("split", "").lower() not in splits:
            continue
        selected.append(row)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def split_filter(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def resolve_recorder(args) -> str:
    if args.record_command:
        return "custom"
    if args.recorder != "auto":
        if not args.dry_run:
            require_command(args.recorder)
        return args.recorder
    for candidate in ["afrecord", "sox", "rec", "arecord"]:
        if shutil.which(candidate):
            return candidate
    if args.dry_run:
        return "afrecord"
    raise SystemExit(
        "no supported recorder found; install afrecord/sox/rec/arecord or pass --record-command"
    )


def require_command(command: str) -> None:
    if not shutil.which(command):
        raise SystemExit(f"recorder not found on PATH: {command}")


def command_for(
    row: dict[str, str],
    path: Path,
    recorder: str,
    custom_command: str | None,
) -> list[str]:
    seconds = duration(row)
    values = {
        "path": str(path),
        "duration": format_duration(seconds),
        "id": row.get("id", ""),
        "text": row.get("text", ""),
        "speaker_id": row.get("speaker_id", ""),
        "category": row.get("category", ""),
    }
    if custom_command:
        return shlex.split(custom_command.format(**values))
    if recorder == "afrecord":
        return [
            "afrecord",
            "-f",
            "WAVE",
            "-d",
            "LEI16@16000",
            "-c",
            "1",
            "-r",
            "16000",
            "-t",
            format_duration(seconds),
            str(path),
        ]
    if recorder == "sox":
        return [
            "sox",
            "-d",
            "-r",
            "16000",
            "-c",
            "1",
            "-b",
            "16",
            str(path),
            "trim",
            "0",
            format_duration(seconds),
        ]
    if recorder == "rec":
        return [
            "rec",
            "-r",
            "16000",
            "-c",
            "1",
            "-b",
            "16",
            str(path),
            "trim",
            "0",
            format_duration(seconds),
        ]
    if recorder == "arecord":
        return [
            "arecord",
            "-q",
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            "-d",
            str(int(round(seconds))),
            str(path),
        ]
    raise AssertionError(f"unknown recorder: {recorder}")


def duration(row: dict[str, str]) -> float:
    raw = row.get("planned_duration_s", "")
    if not raw:
        return 2.0
    seconds = float(raw)
    if seconds <= 0:
        raise SystemExit(f"row {row.get('id', '')} has non-positive planned_duration_s")
    return seconds


def format_duration(seconds: float) -> str:
    return str(int(seconds)) if seconds.is_integer() else f"{seconds:.3f}"


def resolve_path(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else base / path


def print_prompt(row: dict[str, str], path: Path, command: list[str]) -> None:
    print()
    print(f"id: {row.get('id', '')}")
    print(f"path: {path}")
    print(f"role: {row.get('role', '')}  split: {row.get('split', '')}")
    print(f"speaker: {row.get('speaker_id', '')}")
    print(f"environment: {row.get('environment', '')}")
    print(
        f"distance: {row.get('distance', '')}  "
        f"device: {row.get('device', '')}  "
        f"session: {row.get('session_id', '')}"
    )
    print(f"category: {row.get('category', '')}")
    if row.get("phrase"):
        print(f"enrollment phrase: {row['phrase']}")
    if row.get("text"):
        print(f"text: {row['text']}")
    if row.get("wake_start_ms"):
        print(f"wake_start_ms: {row['wake_start_ms']}")
    print(f"duration_s: {format_duration(duration(row))}")
    print(f"command: {shlex.join(command)}")


def confirm() -> bool:
    answer = input("Press Enter to record, 's' to skip, or 'q' to quit: ").strip().lower()
    if answer == "q":
        raise SystemExit(130)
    return answer != "s"


def emit(results: list[dict[str, object]], args, exit_code: int = 0) -> int:
    recorded = sum(1 for result in results if result["recorded"])
    skipped = sum(1 for result in results if result["skipped"])
    failed = sum(1 for result in results if result["error"])
    summary = {
        "rows": len(results),
        "recorded": recorded,
        "skipped": skipped,
        "failed": failed,
        "dry_run": args.dry_run,
        "results": results,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"rows={len(results)}")
        print(f"recorded={recorded}")
        print(f"skipped={skipped}")
        print(f"failed={failed}")
        if args.dry_run:
            for result in results[:50]:
                print(f"- {result['id']}: {shlex.join(result['command'])}")
            if len(results) > 50:
                print(f"- ... {len(results) - 50} more")
    return exit_code if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
