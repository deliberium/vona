#!/usr/bin/env python3
"""Check recording progress against a vona-wake corpus CSV plan."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from array import array
from collections import Counter, defaultdict
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
        "--min-duration-ratio",
        type=float,
        default=0.9,
        help="Minimum actual/planned duration ratio for recorded files (default: 0.9)",
    )
    parser.add_argument("--min-template-rms", type=float, default=0.001)
    parser.add_argument("--min-positive-rms", type=float, default=0.001)
    parser.add_argument("--min-negative-rms", type=float, default=0.0001)
    parser.add_argument("--max-clipped-ratio", type=float, default=0.01)
    parser.add_argument(
        "--allow-non-human-source",
        action="store_true",
        help=(
            "Allow rows whose source_type is missing or not human-recorded. "
            "This is for non-release experiments only."
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero unless all rows pass")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    results = [
        check_row(args.csv.parent, index, row, args)
        for index, row in enumerate(rows, start=2)
    ]
    report = build_report(str(args.csv), results)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report)
    return 1 if args.enforce and not report["complete"] else 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def check_row(base: Path, row_number: int, row: dict[str, str], args) -> dict[str, object]:
    path = resolve_path(base, Path(row.get("path", "")))
    planned_duration = optional_float(row.get("planned_duration_s"))
    result: dict[str, object] = {
        "row": row_number,
        "id": row.get("id", ""),
        "role": row.get("role", ""),
        "path": str(path),
        "category": row.get("category", ""),
        "device": row.get("device", ""),
        "session_id": row.get("session_id", ""),
        "source_type": row.get("source_type", ""),
        "provenance_error": source_type_error(row, args.allow_non_human_source),
        "planned_duration_s": planned_duration,
        "status": "ok",
        "duration_s": None,
        "rms": None,
        "peak": None,
        "clipped_ratio": None,
        "error": None,
    }
    if not path.exists():
        result["status"] = "missing"
        result["error"] = "file missing"
        return result
    try:
        stats = wav_stats(path)
    except (OSError, wave.Error, AssertionError) as exc:
        result["status"] = "invalid"
        result["error"] = str(exc)
        return result
    result.update(stats)
    if planned_duration is not None and stats["duration_s"] < planned_duration * args.min_duration_ratio:
        result["status"] = "short"
        result["error"] = (
            f"duration {stats['duration_s']:.3f}s below {args.min_duration_ratio:.2f} of planned "
            f"{planned_duration:.3f}s"
        )
        return result
    min_rms = min_rms_for(row, args)
    if stats["rms"] < min_rms:
        result["status"] = "quiet"
        result["error"] = f"rms {stats['rms']:.6f} below required {min_rms:.6f}"
        return result
    if stats["clipped_ratio"] > args.max_clipped_ratio:
        result["status"] = "clipped"
        result["error"] = (
            f"clipped ratio {stats['clipped_ratio']:.6f} exceeded allowed "
            f"{args.max_clipped_ratio:.6f}"
        )
    return result


def resolve_path(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else base / path


def optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def min_rms_for(row: dict[str, str], args) -> float:
    if row.get("role", "").lower() == "template":
        return args.min_template_rms
    if row.get("should_wake", "").lower() == "true":
        return args.min_positive_rms
    return args.min_negative_rms


def source_type_error(row: dict[str, str], allow_non_human_source: bool) -> str | None:
    if allow_non_human_source:
        return None
    source_type = row.get("source_type", "").strip()
    if not source_type:
        return "source_type is missing; expected human-recorded"
    if source_type != "human-recorded":
        return f"source_type {source_type!r} is not human-recorded"
    return None


def wav_stats(path: Path) -> dict[str, float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        assert channels == 1, f"{path} has {channels} channel(s), expected mono"
        assert sample_rate == 16_000, f"{path} has {sample_rate} Hz, expected 16000 Hz"
        assert sample_width == 2, f"{path} has {sample_width * 8}-bit samples, expected 16-bit"
        assert frames > 0, f"{path} has no audio frames"
        total = 0
        square_sum = 0
        peak = 0
        clipped = 0
        while data := wav.readframes(65_536):
            samples = array("h")
            samples.frombytes(data)
            if sys.byteorder != "little":
                samples.byteswap()
            for sample in samples:
                absolute = abs(sample)
                total += 1
                square_sum += sample * sample
                peak = max(peak, absolute)
                if absolute >= 32_760:
                    clipped += 1
    rms = (square_sum / max(1, total)) ** 0.5 / 32_768.0
    return {
        "duration_s": frames / sample_rate,
        "rms": rms,
        "peak": peak / 32_768.0,
        "clipped_ratio": clipped / max(1, total),
    }


def build_report(csv_path: str, results: list[dict[str, object]]) -> dict[str, object]:
    status_counts = Counter(str(result["status"]) for result in results)
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    device_counts: dict[str, Counter[str]] = defaultdict(Counter)
    session_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_type_counts: Counter[str] = Counter()
    provenance_failures = 0
    for result in results:
        role_counts[str(result["role"])][str(result["status"])] += 1
        category_counts[str(result["category"])][str(result["status"])] += 1
        device_counts[str(result["device"]) or "missing"][str(result["status"])] += 1
        session_counts[str(result["session_id"]) or "missing"][str(result["status"])] += 1
        source_type_counts[str(result.get("source_type") or "missing")] += 1
        if result.get("provenance_error"):
            provenance_failures += 1
    return {
        "csv_path": csv_path,
        "complete": status_counts.get("ok", 0) == len(results) and provenance_failures == 0,
        "rows": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "provenance_failures": provenance_failures,
        "role_counts": {key: dict(sorted(value.items())) for key, value in sorted(role_counts.items())},
        "category_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(category_counts.items())
        },
        "device_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(device_counts.items())
        },
        "session_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(session_counts.items())
        },
        "failures": [
            result for result in results if result["status"] != "ok" or result.get("provenance_error")
        ],
    }


def print_text_report(report: dict[str, object]) -> None:
    status = "complete" if report["complete"] else "incomplete"
    print(f"status={status}")
    print(f"rows={report['rows']}")
    for key, count in report["status_counts"].items():
        print(f"{key}={count}")
    for key, counts in report["device_counts"].items():
        print(f"device_{key}={counts}")
    for key, counts in report["session_counts"].items():
        print(f"session_{key}={counts}")
    for key, count in report["source_type_counts"].items():
        print(f"source_type_{key}={count}")
    print(f"provenance_failures={report['provenance_failures']}")
    failures = report["failures"]
    if failures:
        print("failures:")
        for failure in failures[:50]:
            details = [str(failure["status"])]
            if failure.get("error"):
                details.append(str(failure["error"]))
            if failure.get("provenance_error"):
                details.append(str(failure["provenance_error"]))
            print(f"- row {failure['row']} {failure['id']}: {'; '.join(details)}")
        if len(failures) > 50:
            print(f"- ... {len(failures) - 50} more")


if __name__ == "__main__":
    raise SystemExit(main())
