#!/usr/bin/env python3
"""Build a vona-wake real-voice evaluation corpus manifest.

Input is a CSV with one row per recording. The script validates that every
recording is 16 kHz mono 16-bit PCM WAV, copies it into a stable corpus layout,
and writes a manifest accepted by:

    cargo run -p vona-wake --example real_voice_eval -- /path/to/manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path


REQUIRED_COLUMNS = {"role", "id", "path"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV describing recordings")
    parser.add_argument("output_dir", type=Path, help="Directory for corpus files")
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest filename inside output_dir (default: manifest.json)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Reference source WAV paths instead of copying them into output_dir",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert source audio to 16 kHz mono 16-bit WAV inside output_dir",
    )
    parser.add_argument(
        "--corpus-id",
        required=True,
        help="Stable identifier for this recording corpus, such as vona-wake-office-v1",
    )
    parser.add_argument(
        "--corpus-version",
        required=True,
        help="Immutable corpus version, such as 2026-06-01 or v1.0.0",
    )
    parser.add_argument(
        "--corpus-created-by",
        default="",
        help="Optional operator/team that assembled the corpus",
    )
    parser.add_argument(
        "--corpus-notes",
        default="",
        help="Optional short notes about collection scope or consent record location",
    )
    parser.add_argument(
        "--collection-ledger",
        type=Path,
        help=(
            "Optional JSON collection ledger with consent/provenance metadata. "
            "The ledger is embedded into the manifest with its SHA-256."
        ),
    )
    parser.add_argument(
        "--allow-non-human-source",
        action="store_true",
        help=(
            "Allow rows whose source_type is missing or not human-recorded. "
            "This is for experiments only; release-grade real corpora should omit it."
        ),
    )
    args = parser.parse_args()
    if args.no_copy and args.convert:
        raise SystemExit("--convert cannot be combined with --no-copy")

    rows = read_rows(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        rows,
        args.csv.parent,
        args.output_dir,
        copy=not args.no_copy,
        convert=args.convert,
        corpus_id=args.corpus_id,
        corpus_version=args.corpus_version,
        corpus_created_by=args.corpus_created_by,
        corpus_notes=args.corpus_notes,
        collection_ledger_path=args.collection_ledger,
        allow_non_human_source=args.allow_non_human_source,
    )
    manifest_path = args.output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def build_manifest(
    rows: list[dict[str, str]],
    input_dir: Path,
    output_dir: Path,
    *,
    copy: bool,
    convert: bool,
    corpus_id: str,
    corpus_version: str,
    corpus_created_by: str,
    corpus_notes: str,
    collection_ledger_path: Path | None,
    allow_non_human_source: bool = False,
) -> dict[str, object]:
    collection_ledger = load_collection_ledger(collection_ledger_path)
    manifest: dict[str, object] = {
        "corpus": compact(
            {
                "id": corpus_id.strip(),
                "version": corpus_version.strip(),
                "source": "human-recorded",
                "created_by": corpus_created_by.strip(),
                "notes": corpus_notes.strip(),
                "collection_ledger_sha256": sha256(collection_ledger_path)
                if collection_ledger_path
                else "",
            }
        ),
        "templates": [],
        "policy": {
            "candidate_threshold": 0.88,
            "accept_threshold": 0.92,
            "speaker_threshold": 0.78,
            "min_energy": 0.0005,
            "preroll_ms": 1200,
            "rearm_ms": 1200,
            "require_speaker_verification": True,
        },
        "cases": [],
    }
    if collection_ledger is not None:
        manifest["collection_ledger"] = collection_ledger

    templates: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        recording_id = require(row, "id", row_number)
        if recording_id in seen_ids:
            raise SystemExit(f"duplicate id at row {row_number}: {recording_id}")
        seen_ids.add(recording_id)

        role = require(row, "role", row_number).lower()
        source = resolve(input_dir, require(row, "path", row_number))

        if role == "template":
            phrase = require(row, "phrase", row_number)
            source_type = source_type_for(row, row_number, allow_non_human_source)
            target = materialize(
                source,
                output_dir,
                Path("enrollment") / f"{slug(recording_id)}.wav",
                copy=copy,
                convert=convert,
                row_number=row_number,
            )
            templates.append(compact({
                "phrase": phrase,
                "path": target,
                "speaker_id": row.get("speaker_id"),
                "source_type": source_type,
                "session_id": row.get("session_id"),
            }))
        elif role == "case":
            should_wake = parse_bool(require(row, "should_wake", row_number), row_number)
            source_type = source_type_for(row, row_number, allow_non_human_source)
            folder = "positives" if should_wake else "negatives"
            target = materialize(
                source,
                output_dir,
                Path(folder) / f"{slug(recording_id)}.wav",
                copy=copy,
                convert=convert,
                row_number=row_number,
            )
            case = compact(
                {
                    "id": recording_id,
                    "path": target,
                    "text": row.get("text"),
                    "wake_start_ms": optional_int(row.get("wake_start_ms"), row_number, "wake_start_ms"),
                    "speaker_id": row.get("speaker_id"),
                    "environment": row.get("environment"),
                    "distance": row.get("distance"),
                    "device": row.get("device"),
                    "session_id": row.get("session_id"),
                    "category": row.get("category"),
                    "source_type": source_type,
                    "split": row.get("split"),
                    "should_wake": should_wake,
                    "expected_phrase": row.get("expected_phrase"),
                }
            )
            cases.append(case)
        else:
            raise SystemExit(f"row {row_number} role must be template or case, got {role!r}")

    if not templates:
        raise SystemExit("at least one template row is required")
    if not cases:
        raise SystemExit("at least one case row is required")
    manifest["templates"] = templates
    manifest["cases"] = cases
    return manifest


def load_collection_ledger(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"collection ledger is not valid JSON: {path}") from exc
    if not isinstance(ledger, dict):
        raise SystemExit("collection ledger must be a JSON object")
    return ledger


def sha256(path: Path | None) -> str:
    if path is None:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(row: dict[str, str], key: str, row_number: int) -> str:
    value = row.get(key, "").strip()
    if not value:
        raise SystemExit(f"row {row_number} is missing required value {key!r}")
    return value


def source_type_for(row: dict[str, str], row_number: int, allow_non_human_source: bool) -> str:
    source_type = row.get("source_type", "").strip()
    if allow_non_human_source:
        return source_type
    if not source_type:
        raise SystemExit(
            f"row {row_number} is missing source_type='human-recorded' for a real corpus"
        )
    if source_type != "human-recorded":
        raise SystemExit(
            f"row {row_number} source_type must be 'human-recorded' for a real corpus, "
            f"got {source_type!r}"
        )
    return source_type


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def validate_wav(path: Path, row_number: int) -> bool:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
    except FileNotFoundError as exc:
        raise SystemExit(f"row {row_number} audio file not found: {path}") from exc
    except wave.Error as exc:
        raise SystemExit(f"row {row_number} is not a valid PCM WAV: {path}") from exc

    if channels != 1 or sample_rate != 16_000 or sample_width != 2:
        raise SystemExit(
            f"row {row_number} WAV must be 16 kHz mono 16-bit PCM: "
            f"{path} has {channels} channel(s), {sample_rate} Hz, {sample_width * 8}-bit"
        )
    if frames == 0:
        raise SystemExit(f"row {row_number} WAV has no audio frames: {path}")
    return True


def materialize(
    source: Path,
    output_dir: Path,
    relative_target: Path,
    *,
    copy: bool,
    convert: bool,
    row_number: int,
) -> str:
    if not copy:
        validate_wav(source, row_number)
        return str(source)
    target = output_dir / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    if convert:
        convert_audio(source, target, row_number)
    else:
        validate_wav(source, row_number)
        shutil.copy2(source, target)
    validate_wav(target, row_number)
    return target.relative_to(output_dir).as_posix()


def convert_audio(source: Path, target: Path, row_number: int) -> None:
    if shutil.which("afconvert"):
        command = [
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            "LEI16@16000",
            "-c",
            "1",
            str(source),
            str(target),
        ]
    elif shutil.which("ffmpeg"):
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(target),
        ]
    else:
        raise SystemExit("audio conversion requires afconvert or ffmpeg")

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"row {row_number} audio conversion failed for {source}: {stderr}") from exc


def parse_bool(value: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise SystemExit(f"row {row_number} should_wake must be true or false, got {value!r}")


def optional_int(value: str | None, row_number: int, key: str) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"row {row_number} {key} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise SystemExit(f"row {row_number} {key} must be non-negative, got {parsed}")
    return parsed


def compact(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if item not in ("", None)}


def slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    slugged = "".join(chars).strip("-")
    while "--" in slugged:
        slugged = slugged.replace("--", "-")
    return slugged or "recording"


if __name__ == "__main__":
    sys.exit(main())
