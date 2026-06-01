#!/usr/bin/env python3
"""Generate a balanced recording CSV plan for a vona-wake real corpus."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from plan_vona_wake_corpus import required_negative_audio_hours, required_wilson_trials


COLUMNS = [
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
]

WAKE_PHRASES = ["hey vona", "vona"]
POSITIVE_SCENARIOS = [
    ("early", "hey vona", "hey vona", 0, 3),
    ("early", "vona", "vona", 0, 3),
    ("mid", "hey vona", "okay hey vona can you help", 1000, 4),
    ("mid", "vona", "please wait vona start listening", 1000, 4),
    ("late", "hey vona", "when you are ready hey vona can you help", 2000, 5),
    ("late", "vona", "after this sentence vona start listening", 2000, 5),
]
NEGATIVE_SCENARIOS = [
    ("unauthorized-wake", "hey vona"),
    ("unauthorized-wake", "vona"),
    ("near-miss", "hey luna"),
    ("near-miss", "hey mona"),
    ("near-miss", "vonae is not the wake word"),
    ("ordinary-speech", "the meeting starts in ten minutes"),
    ("ordinary-command", "can you turn the volume down"),
    ("background-speech", "background office conversation with no wake phrase"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="CSV path; defaults to stdout")
    parser.add_argument("--instructions-output", help="Markdown recording guide path")
    parser.add_argument("--speakers", default="speaker-a,speaker-b,speaker-c,speaker-d,speaker-e")
    parser.add_argument("--unauthorized-speakers", default="guest-a,guest-b")
    parser.add_argument("--environments", default="quiet-office,meeting-room,open-office")
    parser.add_argument("--distances", default="near,mid,far")
    parser.add_argument("--devices", default="built-in-mic,usb-mic")
    parser.add_argument("--sessions", default="session-a,session-b,session-c")
    parser.add_argument("--observed-precision", type=float, default=0.98)
    parser.add_argument("--precision-lower-bound", type=float, default=0.95)
    parser.add_argument("--observed-recall", type=float, default=0.98)
    parser.add_argument("--recall-lower-bound", type=float, default=0.95)
    parser.add_argument("--false-wake-events", type=int, default=0)
    parser.add_argument("--false-wakes-per-hour-upper-bound", type=float, default=0.05)
    parser.add_argument("--positive-cases", type=int)
    parser.add_argument("--negative-audio-seconds", type=int)
    parser.add_argument("--negative-clip-seconds", type=int, default=600)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 <= args.calibration_fraction < 1.0:
        raise SystemExit("--calibration-fraction must be >= 0 and < 1")

    speakers = split_values(args.speakers, "speakers")
    unauthorized_speakers = split_values(args.unauthorized_speakers, "unauthorized speakers")
    environments = split_values(args.environments, "environments")
    distances = split_values(args.distances, "distances")
    devices = split_values(args.devices, "devices")
    sessions = split_values(args.sessions, "sessions")
    evaluation_positive_cases = args.positive_cases or max(
        required_wilson_trials(args.observed_precision, args.precision_lower_bound),
        required_wilson_trials(args.observed_recall, args.recall_lower_bound),
    )
    evaluation_negative_audio_seconds = args.negative_audio_seconds or math.ceil(
        required_negative_audio_hours(
            args.false_wake_events, args.false_wakes_per_hour_upper_bound
        )
        * 3600
    )
    calibration_positive_cases = calibration_overhead(
        evaluation_positive_cases, args.calibration_fraction
    )
    calibration_negative_audio_seconds = round_up_to_clip(
        calibration_overhead(
            evaluation_negative_audio_seconds, args.calibration_fraction
        ),
        args.negative_clip_seconds,
    )
    rows = build_rows(
        speakers=speakers,
        unauthorized_speakers=unauthorized_speakers,
        environments=environments,
        distances=distances,
        devices=devices,
        sessions=sessions,
        calibration_positive_cases=calibration_positive_cases,
        evaluation_positive_cases=evaluation_positive_cases,
        calibration_negative_audio_seconds=calibration_negative_audio_seconds,
        evaluation_negative_audio_seconds=evaluation_negative_audio_seconds,
        negative_clip_seconds=args.negative_clip_seconds,
    )
    write_rows(rows, args.output)
    if args.instructions_output:
        Path(args.instructions_output).write_text(
            recording_instructions(rows, args.output),
            encoding="utf-8",
        )
    return 0


def split_values(value: str, name: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"{name} must contain at least one value")
    return values


def calibration_overhead(evaluation_target: int, fraction: float) -> int:
    if fraction <= 0.0:
        return 0
    return math.ceil(evaluation_target * fraction / (1.0 - fraction))


def round_up_to_clip(seconds: int, clip_seconds: int) -> int:
    if seconds == 0:
        return 0
    return math.ceil(seconds / clip_seconds) * clip_seconds


def build_rows(
    *,
    speakers: list[str],
    unauthorized_speakers: list[str],
    environments: list[str],
    distances: list[str],
    devices: list[str],
    sessions: list[str],
    calibration_positive_cases: int,
    evaluation_positive_cases: int,
    calibration_negative_audio_seconds: int,
    evaluation_negative_audio_seconds: int,
    negative_clip_seconds: int,
) -> list[dict[str, object]]:
    if negative_clip_seconds <= 0:
        raise SystemExit("negative clip seconds must be greater than zero")
    rows: list[dict[str, object]] = []
    for speaker in speakers:
        for phrase in WAKE_PHRASES:
            recording_id = f"template-{slug(speaker)}-{slug(phrase)}"
            rows.append(
                row(
                    role="template",
                    recording_id=recording_id,
                    path=f"raw/enrollment/{recording_id}.wav",
                    phrase=phrase,
                    speaker_id=speaker,
                    environment=environments[0],
                    distance=distances[0],
                    device=devices[0],
                    session_id=sessions[0],
                    category="enrollment",
                    source_type="human-recorded",
                    split="enrollment",
                    planned_duration_s=2,
                )
            )

    total_positive_cases = calibration_positive_cases + evaluation_positive_cases
    for index in range(total_positive_cases):
        onset_bucket, phrase, text, wake_start_ms, planned_duration_s = POSITIVE_SCENARIOS[
            index % len(POSITIVE_SCENARIOS)
        ]
        speaker = speakers[index % len(speakers)]
        environment = environments[(index // len(speakers)) % len(environments)]
        distance = distances[(index // (len(speakers) * len(environments))) % len(distances)]
        device = devices[(index // (len(speakers) * len(environments) * len(distances))) % len(devices)]
        session_id = sessions[index % len(sessions)]
        recording_id = f"positive-{index + 1:04d}-{onset_bucket}-{slug(speaker)}-{slug(phrase)}"
        rows.append(
            row(
                role="case",
                recording_id=recording_id,
                path=f"raw/positives/{recording_id}.wav",
                should_wake="true",
                text=text,
                expected_phrase=phrase,
                wake_start_ms=wake_start_ms,
                speaker_id=speaker,
                environment=environment,
                distance=distance,
                device=device,
                session_id=session_id,
                category="wake-positive",
                source_type="human-recorded",
                split="calibration" if index < calibration_positive_cases else "evaluation",
                planned_duration_s=planned_duration_s,
            )
        )

    total_negative_audio_seconds = calibration_negative_audio_seconds + evaluation_negative_audio_seconds
    negative_clips = math.ceil(total_negative_audio_seconds / negative_clip_seconds)
    for index in range(negative_clips):
        category, text = NEGATIVE_SCENARIOS[index % len(NEGATIVE_SCENARIOS)]
        speaker_pool = unauthorized_speakers if category == "unauthorized-wake" else speakers
        speaker = speaker_pool[index % len(speaker_pool)]
        environment = environments[(index // len(speakers)) % len(environments)]
        distance = distances[(index // (len(speakers) * len(environments))) % len(distances)]
        device = devices[(index // (len(speakers) * len(environments) * len(distances))) % len(devices)]
        session_id = sessions[index % len(sessions)]
        planned_duration = min(
            negative_clip_seconds,
            max(1, total_negative_audio_seconds - index * negative_clip_seconds),
        )
        elapsed_before = index * negative_clip_seconds
        split = (
            "calibration"
            if elapsed_before < calibration_negative_audio_seconds
            else "evaluation"
        )
        recording_id = f"negative-{index + 1:04d}-{slug(speaker)}"
        rows.append(
            row(
                role="case",
                recording_id=recording_id,
                path=f"raw/negatives/{recording_id}.wav",
                should_wake="false",
                text=text,
                speaker_id=speaker,
                environment=environment,
                distance=distance,
                device=device,
                session_id=session_id,
                category=category,
                source_type="human-recorded",
                split=split,
                planned_duration_s=planned_duration,
            )
        )
    return rows


def row(**values: object) -> dict[str, object]:
    output = {column: "" for column in COLUMNS}
    if "recording_id" in values:
        output["id"] = values.pop("recording_id")
    output.update(values)
    return output


def write_rows(rows: list[dict[str, object]], output: str | None) -> None:
    handle = open(output, "w", newline="", encoding="utf-8") if output else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output:
            handle.close()


def recording_instructions(rows: list[dict[str, object]], csv_path: str | None) -> str:
    templates = [row for row in rows if row["role"] == "template"]
    positives = [row for row in rows if row["role"] == "case" and row["should_wake"] == "true"]
    negatives = [row for row in rows if row["role"] == "case" and row["should_wake"] == "false"]
    calibration_cases = [row for row in rows if row.get("split") == "calibration"]
    evaluation_cases = [row for row in rows if row.get("split") == "evaluation"]
    negative_seconds = sum(int(row["planned_duration_s"]) for row in negatives)
    lines = [
        "# Vona Wake Recording Plan",
        "",
        "## Summary",
        "",
        f"- CSV: `{csv_path or 'stdout'}`",
        f"- Enrollment clips: {len(templates)}",
        f"- Positive wake clips: {len(positives)}",
        f"- Negative/background clips: {len(negatives)}",
        f"- Calibration cases: {len(calibration_cases)}",
        f"- Evaluation cases: {len(evaluation_cases)}",
        f"- Planned negative audio: {negative_seconds} seconds ({negative_seconds / 3600:.3f} hours)",
        "",
        "## Recording Rules",
        "",
        "- Record every file as 16 kHz mono 16-bit PCM WAV, or record in a convenient format and use `build_vona_wake_corpus.py --convert`.",
        "- Save each recording at exactly the path listed in the CSV `path` column.",
        "- Do not reuse enrollment audio as a positive or negative case.",
        "- For positive clips, speak the `text` phrase naturally and keep `wake_start_ms` accurate; update the CSV if the phrase does not start at 0 ms.",
        "- For negative clips, avoid saying the actual wake phrase unless the row text intentionally contains a near miss.",
        "- Rows with category `unauthorized-wake` should be recorded by speakers who do not appear in enrollment rows; these are negative cases even though they say the wake phrase.",
        "- Keep speaker, environment, distance, device, session_id, and category labels exactly as written unless deliberately changing the plan.",
        "- Keep `source_type` as `human-recorded` for real reliability evidence; generated or synthetic clips belong only in regression tests.",
        "- Use `split=calibration` clips for threshold exploration and `split=evaluation` clips for the final reliability claim.",
        "",
        "## Enrollment Clips",
        "",
        "| ID | Speaker | Phrase | Split | Path | Duration s |",
        "|---|---|---|---|---|---:|",
    ]
    for row in templates:
        lines.append(
            f"| `{row['id']}` | `{row['speaker_id']}` | {row['phrase']} | `{row['split']}` | `{row['path']}` | {row['planned_duration_s']} |"
        )
    lines.extend(
        [
            "",
            "## Positive Wake Clips",
            "",
            "| ID | Speaker | Environment | Distance | Device | Session | Split | Prompt | Path | Duration s |",
            "|---|---|---|---|---|---|---|---|---|---:|",
        ]
    )
    for row in positives:
        lines.append(
            f"| `{row['id']}` | `{row['speaker_id']}` | `{row['environment']}` | `{row['distance']}` | `{row['device']}` | `{row['session_id']}` | `{row['split']}` | {row['text']} | `{row['path']}` | {row['planned_duration_s']} |"
        )
    lines.extend(
        [
            "",
            "## Negative/Background Clips",
            "",
            "| ID | Category | Speaker | Environment | Distance | Device | Session | Split | Prompt/Scenario | Path | Duration s |",
            "|---|---|---|---|---|---|---|---|---|---|---:|",
        ]
    )
    for row in negatives:
        lines.append(
            f"| `{row['id']}` | `{row['category']}` | `{row['speaker_id']}` | `{row['environment']}` | `{row['distance']}` | `{row['device']}` | `{row['session_id']}` | `{row['split']}` | {row['text']} | `{row['path']}` | {row['planned_duration_s']} |"
        )
    lines.extend(
        [
            "",
            "## After Recording",
            "",
        "```bash",
        f"scripts/record_vona_wake_corpus.py {csv_path or '/path/to/recordings.csv'}",
        f"scripts/check_vona_wake_recording_progress.py --enforce {csv_path or '/path/to/recordings.csv'}",
        (
            "python3 scripts/build_vona_wake_corpus.py --convert "
            "--corpus-id vona-wake-office-v1 --corpus-version 2026-06-01 "
                f"{csv_path or '/path/to/recordings.csv'} /path/to/corpus"
            ),
            "scripts/audit_vona_wake_corpus.py --enforce /path/to/corpus/manifest.json",
            "VONA_WAKE_REQUIRE_REAL_EVAL=1 scripts/run_vona_wake_eval.sh /path/to/corpus/manifest.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")


if __name__ == "__main__":
    raise SystemExit(main())
