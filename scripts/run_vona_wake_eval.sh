#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() {
  printf "\n[%s] %s\n" "vona-wake-eval" "$1"
}

REAL_MANIFEST="${1:-${VONA_WAKE_REAL_EVAL_MANIFEST:-}}"
CARGO_FLAGS="${VONA_WAKE_EVAL_CARGO_FLAGS:---offline}"
REPORT_DIR="${VONA_WAKE_EVAL_REPORT_DIR:-${ROOT_DIR}/target/vona-wake-eval}"
GENERATED_REPORT="${VONA_WAKE_EVAL_REPORT:-${REPORT_DIR}/generated-report.json}"
GENERATED_MANIFEST="${VONA_WAKE_EVAL_MANIFEST:-${REPORT_DIR}/generated-manifest.json}"
REAL_REPORT="${VONA_WAKE_REAL_EVAL_REPORT:-${REPORT_DIR}/real-report.json}"
AUDIT_REPORT="${VONA_WAKE_REAL_EVAL_AUDIT_REPORT:-${REPORT_DIR}/audit-report.json}"
THRESHOLD_REPORT="${VONA_WAKE_REAL_EVAL_THRESHOLD_REPORT:-${REPORT_DIR}/threshold-selection-report.json}"
SUMMARY_REPORT="${VONA_WAKE_EVAL_SUMMARY:-${REPORT_DIR}/summary.md}"
STATUS_REPORT="${VONA_WAKE_REAL_EVIDENCE_STATUS:-${REPORT_DIR}/real-evidence-status.md}"
AUDIT_PERFORMED=0
THRESHOLD_SELECTION_PERFORMED=0
EXIT_CODE=0

mkdir -p "$REPORT_DIR"

log "Checking vona-wake crate"
# shellcheck disable=SC2086
cargo test -p vona-wake $CARGO_FLAGS

log "Running generated wake regression evaluation"
VONA_WAKE_EVAL_ENFORCE=1 \
VONA_WAKE_EVAL_REPORT="$GENERATED_REPORT" \
VONA_WAKE_EVAL_MANIFEST="$GENERATED_MANIFEST" \
  cargo run -p vona-wake --example generated_voice_eval $CARGO_FLAGS >/dev/null
log "Generated wake report: $GENERATED_REPORT"
log "Generated wake manifest: $GENERATED_MANIFEST"

if [[ -n "$REAL_MANIFEST" ]]; then
  log "Running real voice corpus evaluation: $REAL_MANIFEST"
  : "${VONA_WAKE_REAL_EVAL_MIN_PRECISION:=0.98}"
  : "${VONA_WAKE_REAL_EVAL_MIN_RECALL:=0.98}"
  : "${VONA_WAKE_REAL_EVAL_MAX_FALSE_POSITIVES:=0}"
  : "${VONA_WAKE_REAL_EVAL_MAX_FALSE_NEGATIVES:=0}"
  : "${VONA_WAKE_REAL_EVAL_MAX_PHRASE_MISMATCHES:=0}"
  : "${VONA_WAKE_REAL_EVAL_MAX_FALSE_WAKES_PER_HOUR:=0.05}"
  : "${VONA_WAKE_REAL_EVAL_MAX_FIRST_WAKE_MS:=1500}"
  : "${VONA_WAKE_REAL_EVAL_MAX_DETECTION_LATENCY_MS:=$VONA_WAKE_REAL_EVAL_MAX_FIRST_WAKE_MS}"
  : "${VONA_WAKE_REAL_EVAL_MIN_SUBGROUP_PRECISION:=$VONA_WAKE_REAL_EVAL_MIN_PRECISION}"
  : "${VONA_WAKE_REAL_EVAL_MIN_SUBGROUP_RECALL:=$VONA_WAKE_REAL_EVAL_MIN_RECALL}"
  : "${VONA_WAKE_REAL_EVAL_MAX_SUBGROUP_FALSE_WAKES_PER_HOUR:=$VONA_WAKE_REAL_EVAL_MAX_FALSE_WAKES_PER_HOUR}"
  : "${VONA_WAKE_REAL_EVAL_MAX_SUBGROUP_DETECTION_LATENCY_MS:=$VONA_WAKE_REAL_EVAL_MAX_DETECTION_LATENCY_MS}"
  : "${VONA_WAKE_REAL_EVAL_MAX_TEMPLATE_CASE_PATH_OVERLAPS:=0}"
  : "${VONA_WAKE_REAL_EVAL_MAX_TEMPLATE_CASE_AUDIO_OVERLAPS:=0}"
  : "${VONA_WAKE_REAL_EVAL_MAX_TEMPLATE_CASE_FINGERPRINT_OVERLAPS:=0}"
  : "${VONA_WAKE_REAL_EVAL_MAX_DUPLICATE_CASE_PATHS:=0}"
  : "${VONA_WAKE_REAL_EVAL_MAX_DUPLICATE_CASE_AUDIO:=0}"
  : "${VONA_WAKE_REAL_EVAL_MAX_DUPLICATE_CASE_FINGERPRINTS:=0}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_SPEAKERS:=5}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_ENVIRONMENTS:=3}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_DISTANCES:=3}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_DEVICES:=2}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_SESSIONS:=2}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_CATEGORIES:=4}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_SPEAKERS:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_SPEAKERS}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_ENVIRONMENTS:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_ENVIRONMENTS}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_DISTANCES:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_DISTANCES}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_DEVICES:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_DEVICES}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_SESSIONS:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_SESSIONS}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_CATEGORIES:=1}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_SPEAKERS:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_SPEAKERS}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_ENVIRONMENTS:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_ENVIRONMENTS}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_DISTANCES:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_DISTANCES}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_DEVICES:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_DEVICES}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_SESSIONS:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_SESSIONS}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_CATEGORIES:=$VONA_WAKE_REAL_EVAL_AUDIT_MIN_CATEGORIES}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_OBSERVED_PRECISION:=$VONA_WAKE_REAL_EVAL_MIN_PRECISION}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_PRECISION_LOWER_BOUND:=0.95}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_OBSERVED_RECALL:=$VONA_WAKE_REAL_EVAL_MIN_RECALL}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_RECALL_LOWER_BOUND:=0.95}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_FALSE_WAKE_EVENTS:=0}"
  : "${VONA_WAKE_REAL_EVAL_AUDIT_FALSE_WAKES_PER_HOUR_UPPER_BOUND:=$VONA_WAKE_REAL_EVAL_MAX_FALSE_WAKES_PER_HOUR}"
  export VONA_WAKE_REAL_EVAL_MIN_PRECISION
  export VONA_WAKE_REAL_EVAL_MIN_RECALL
  export VONA_WAKE_REAL_EVAL_MAX_FALSE_POSITIVES
  export VONA_WAKE_REAL_EVAL_MAX_FALSE_NEGATIVES
  export VONA_WAKE_REAL_EVAL_MAX_PHRASE_MISMATCHES
  export VONA_WAKE_REAL_EVAL_MAX_FALSE_WAKES_PER_HOUR
  export VONA_WAKE_REAL_EVAL_MAX_FIRST_WAKE_MS
  export VONA_WAKE_REAL_EVAL_MAX_DETECTION_LATENCY_MS
  export VONA_WAKE_REAL_EVAL_MIN_SUBGROUP_PRECISION
  export VONA_WAKE_REAL_EVAL_MIN_SUBGROUP_RECALL
  export VONA_WAKE_REAL_EVAL_MAX_SUBGROUP_FALSE_WAKES_PER_HOUR
  export VONA_WAKE_REAL_EVAL_MAX_SUBGROUP_DETECTION_LATENCY_MS
  export VONA_WAKE_REAL_EVAL_MAX_TEMPLATE_CASE_PATH_OVERLAPS
  export VONA_WAKE_REAL_EVAL_MAX_TEMPLATE_CASE_AUDIO_OVERLAPS
  export VONA_WAKE_REAL_EVAL_MAX_TEMPLATE_CASE_FINGERPRINT_OVERLAPS
  export VONA_WAKE_REAL_EVAL_MAX_DUPLICATE_CASE_PATHS
  export VONA_WAKE_REAL_EVAL_MAX_DUPLICATE_CASE_AUDIO
  export VONA_WAKE_REAL_EVAL_MAX_DUPLICATE_CASE_FINGERPRINTS

  if [[ "${VONA_WAKE_REAL_EVAL_AUDIT:-${VONA_WAKE_REQUIRE_REAL_EVAL:-0}}" == "1" ]]; then
    AUDIT_PERFORMED=1
    log "Auditing real voice corpus readiness: $REAL_MANIFEST"
    scripts/audit_vona_wake_corpus.py \
      --min-speakers "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_SPEAKERS" \
      --min-environments "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_ENVIRONMENTS" \
      --min-distances "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_DISTANCES" \
      --min-devices "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_DEVICES" \
      --min-sessions "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_SESSIONS" \
      --min-categories "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_CATEGORIES" \
      --min-positive-speakers "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_SPEAKERS" \
      --min-positive-environments "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_ENVIRONMENTS" \
      --min-positive-distances "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_DISTANCES" \
      --min-positive-devices "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_DEVICES" \
      --min-positive-sessions "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_SESSIONS" \
      --min-positive-categories "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_CATEGORIES" \
      --min-negative-speakers "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_SPEAKERS" \
      --min-negative-environments "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_ENVIRONMENTS" \
      --min-negative-distances "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_DISTANCES" \
      --min-negative-devices "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_DEVICES" \
      --min-negative-sessions "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_SESSIONS" \
      --min-negative-categories "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_CATEGORIES" \
      --observed-precision "$VONA_WAKE_REAL_EVAL_AUDIT_OBSERVED_PRECISION" \
      --precision-lower-bound "$VONA_WAKE_REAL_EVAL_AUDIT_PRECISION_LOWER_BOUND" \
      --observed-recall "$VONA_WAKE_REAL_EVAL_AUDIT_OBSERVED_RECALL" \
      --recall-lower-bound "$VONA_WAKE_REAL_EVAL_AUDIT_RECALL_LOWER_BOUND" \
      --false-wake-events "$VONA_WAKE_REAL_EVAL_AUDIT_FALSE_WAKE_EVENTS" \
      --false-wakes-per-hour-upper-bound "$VONA_WAKE_REAL_EVAL_AUDIT_FALSE_WAKES_PER_HOUR_UPPER_BOUND" \
      --json \
      "$REAL_MANIFEST" > "$AUDIT_REPORT"
    scripts/audit_vona_wake_corpus.py \
      --min-speakers "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_SPEAKERS" \
      --min-environments "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_ENVIRONMENTS" \
      --min-distances "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_DISTANCES" \
      --min-devices "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_DEVICES" \
      --min-sessions "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_SESSIONS" \
      --min-categories "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_CATEGORIES" \
      --min-positive-speakers "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_SPEAKERS" \
      --min-positive-environments "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_ENVIRONMENTS" \
      --min-positive-distances "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_DISTANCES" \
      --min-positive-devices "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_DEVICES" \
      --min-positive-sessions "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_SESSIONS" \
      --min-positive-categories "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_POSITIVE_CATEGORIES" \
      --min-negative-speakers "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_SPEAKERS" \
      --min-negative-environments "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_ENVIRONMENTS" \
      --min-negative-distances "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_DISTANCES" \
      --min-negative-devices "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_DEVICES" \
      --min-negative-sessions "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_SESSIONS" \
      --min-negative-categories "$VONA_WAKE_REAL_EVAL_AUDIT_MIN_NEGATIVE_CATEGORIES" \
      --observed-precision "$VONA_WAKE_REAL_EVAL_AUDIT_OBSERVED_PRECISION" \
      --precision-lower-bound "$VONA_WAKE_REAL_EVAL_AUDIT_PRECISION_LOWER_BOUND" \
      --observed-recall "$VONA_WAKE_REAL_EVAL_AUDIT_OBSERVED_RECALL" \
      --recall-lower-bound "$VONA_WAKE_REAL_EVAL_AUDIT_RECALL_LOWER_BOUND" \
      --false-wake-events "$VONA_WAKE_REAL_EVAL_AUDIT_FALSE_WAKE_EVENTS" \
      --false-wakes-per-hour-upper-bound "$VONA_WAKE_REAL_EVAL_AUDIT_FALSE_WAKES_PER_HOUR_UPPER_BOUND" \
      --enforce \
      "$REAL_MANIFEST"
    log "Real voice audit report: $AUDIT_REPORT"
  fi

  VONA_WAKE_REAL_EVAL_ENFORCE=1 \
  VONA_WAKE_REAL_EVAL_REPORT="$REAL_REPORT" \
    cargo run -p vona-wake --example real_voice_eval $CARGO_FLAGS -- "$REAL_MANIFEST" >/dev/null
  log "Real voice report: $REAL_REPORT"

  if [[ "${VONA_WAKE_REAL_EVAL_THRESHOLD_SELECTION:-${VONA_WAKE_REQUIRE_REAL_EVAL:-0}}" == "1" ]]; then
    THRESHOLD_SELECTION_PERFORMED=1
    log "Selecting threshold from calibration split"
    scripts/select_vona_wake_threshold.py \
      --json \
      --real-report "$REAL_REPORT" > "$THRESHOLD_REPORT"
    scripts/select_vona_wake_threshold.py \
      --enforce \
      --real-report "$REAL_REPORT"
    log "Threshold selection report: $THRESHOLD_REPORT"
  fi

  if [[ "${VONA_WAKE_REAL_EVAL_ACCEPTANCE:-${VONA_WAKE_REQUIRE_REAL_EVAL:-0}}" == "1" ]]; then
    log "Checking real voice acceptance evidence"
    ACCEPTANCE_ARGS=(
      --audit-report "$AUDIT_REPORT"
      --real-report "$REAL_REPORT"
    )
    if [[ "$THRESHOLD_SELECTION_PERFORMED" == "1" && -f "$THRESHOLD_REPORT" ]]; then
      ACCEPTANCE_ARGS+=(--threshold-report "$THRESHOLD_REPORT")
    fi
    if [[ "${VONA_WAKE_REQUIRE_REAL_EVAL:-0}" == "1" ]]; then
      ACCEPTANCE_ARGS+=(--require-threshold-selection)
    fi
    scripts/check_vona_wake_acceptance.py "${ACCEPTANCE_ARGS[@]}"
  fi
elif [[ "${VONA_WAKE_REQUIRE_REAL_EVAL:-0}" == "1" ]]; then
  rm -f "$REAL_REPORT" "$AUDIT_REPORT" "$THRESHOLD_REPORT"
  log "VONA_WAKE_REQUIRE_REAL_EVAL=1 but no real manifest was provided"
  EXIT_CODE=1
else
  rm -f "$REAL_REPORT" "$AUDIT_REPORT" "$THRESHOLD_REPORT"
  log "No real voice manifest provided; set VONA_WAKE_REAL_EVAL_MANIFEST or pass one as argv[1]"
fi

log "Writing Markdown summary: $SUMMARY_REPORT"
REAL_SUMMARY_ARG="${REAL_MANIFEST:+$REAL_REPORT}"
AUDIT_SUMMARY_ARG=""
if [[ "$AUDIT_PERFORMED" == "1" && -f "$AUDIT_REPORT" ]]; then
  AUDIT_SUMMARY_ARG="$AUDIT_REPORT"
fi
THRESHOLD_SUMMARY_ARG=""
if [[ "$THRESHOLD_SELECTION_PERFORMED" == "1" && -f "$THRESHOLD_REPORT" ]]; then
  THRESHOLD_SUMMARY_ARG="$THRESHOLD_REPORT"
fi
python3 - "$GENERATED_REPORT" "$REAL_SUMMARY_ARG" "$AUDIT_SUMMARY_ARG" "$THRESHOLD_SUMMARY_ARG" "$SUMMARY_REPORT" <<'PY'
import json
import sys
from pathlib import Path

generated_path = Path(sys.argv[1])
real_path = Path(sys.argv[2]) if sys.argv[2] else None
audit_path = Path(sys.argv[3]) if sys.argv[3] else None
threshold_path = Path(sys.argv[4]) if sys.argv[4] else None
summary_path = Path(sys.argv[5])

generated = json.loads(generated_path.read_text())
real = json.loads(real_path.read_text()) if real_path and real_path.exists() else None
audit = json.loads(audit_path.read_text()) if audit_path and audit_path.exists() else None
threshold = json.loads(threshold_path.read_text()) if threshold_path and threshold_path.exists() else None

def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def csv(values):
    if not values:
        return "n/a"
    return ", ".join(f"`{value}`" for value in values)


def weakest_groups(subgroups):
    rows = []
    for dimension, groups in subgroups.items():
        for group in groups:
            rows.append(
                {
                    "dimension": dimension,
                    "id": group["id"],
                    "cases": group["cases"],
                    "positives": group["positives"],
                    "negatives": group["negatives"],
                    "precision": group["precision"],
                    "recall": group["recall"],
                    "false_wakes_per_hour": group["false_wakes_per_hour"],
                    "repeated_positive_wake_events": group.get(
                        "repeated_positive_wake_events",
                        0,
                    ),
                    "max_detection_latency_ms": group.get("max_detection_latency_ms"),
                }
            )
    rows.sort(
        key=lambda row: (
            row["recall"],
            row["precision"],
            -row["false_wakes_per_hour"],
            row["dimension"],
            row["id"],
        )
    )
    return rows[:12]


lines = [
    "# Vona Wake Evaluation Summary",
    "",
    "## Generated Regression",
    "",
    "| Metric | Value |",
    "|---|---:|",
    f"| Positives | {generated['positives']} |",
    f"| Negatives | {generated['negatives']} |",
    f"| True positives | {generated['true_positives']} |",
    f"| False negatives | {generated['false_negatives']} |",
    f"| True negatives | {generated['true_negatives']} |",
    f"| False positives | {generated['false_positives']} |",
    f"| Precision | {generated['precision']:.4f} |",
    f"| Recall | {generated['recall']:.4f} |",
    f"| Unauthorized speaker rejected | {str(generated['unauthorized_rejected']).lower()} |",
    f"| Privacy suppressed | {str(generated['privacy_suppressed']).lower()} |",
    "",
]

if real:
    coverage = real.get("coverage", {})
    leakage = real.get("leakage", {})
    confidence = real.get("confidence_intervals", {})
    subgroup_rows = weakest_groups(real.get("subgroups", {}))
    lines.extend(
        [
            "## Real Voice Corpus",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Manifest | `{real['manifest_path']}` |",
            f"| Templates | {real['templates']} |",
            f"| Cases | {real['cases']} |",
            f"| Positives | {real['positives']} |",
            f"| Negatives | {real['negatives']} |",
            f"| True positives | {real['true_positives']} |",
            f"| False negatives | {real['false_negatives']} |",
            f"| True negatives | {real['true_negatives']} |",
            f"| False wake events | {real['false_positives']} |",
            f"| Repeated positive wake events | {real.get('repeated_positive_wake_events', 0)} |",
            f"| Phrase mismatches | {real['phrase_mismatches']} |",
            f"| Precision | {real['precision']:.4f} |",
            f"| Precision lower bound ({confidence.get('confidence_level', 0.95):.0%}) | {fmt(confidence.get('precision_lower_bound'))} |",
            f"| Recall | {real['recall']:.4f} |",
            f"| Recall lower bound ({confidence.get('confidence_level', 0.95):.0%}) | {fmt(confidence.get('recall_lower_bound'))} |",
            f"| Positive audio seconds | {real['positive_audio_seconds']:.3f} |",
            f"| Negative audio seconds | {real['negative_audio_seconds']:.3f} |",
            f"| False wakes per hour | {real['false_wakes_per_hour']:.4f} |",
            f"| False wakes/hour upper bound ({confidence.get('confidence_level', 0.95):.0%}) | {fmt(confidence.get('false_wakes_per_hour_upper_bound'))} |",
            f"| Mean detection latency ms | {fmt(real.get('mean_detection_latency_ms'))} |",
            f"| Max detection latency ms | {fmt(real.get('max_detection_latency_ms'))} |",
            f"| Template/case path overlaps | {leakage.get('template_case_path_overlaps', 0)} |",
            f"| Template/case audio overlaps | {leakage.get('template_case_audio_overlaps', 0)} |",
            f"| Duplicate case paths | {leakage.get('duplicate_case_paths', 0)} |",
            f"| Duplicate case audio groups | {leakage.get('duplicate_case_audio', 0)} |",
            "",
            "## Corpus Coverage",
            "",
            "| Dimension | Count | Values |",
            "|---|---:|---|",
            f"| Speakers | {coverage.get('speakers', 0)} | {csv(coverage.get('speaker_ids', []))} |",
            f"| Environments | {coverage.get('environments', 0)} | {csv(coverage.get('environment_ids', []))} |",
            f"| Distances | {coverage.get('distances', 0)} | {csv(coverage.get('distance_ids', []))} |",
            f"| Devices | {coverage.get('devices', 0)} | {csv(coverage.get('device_ids', []))} |",
            f"| Sessions | {coverage.get('sessions', 0)} | {csv(coverage.get('session_ids', []))} |",
            f"| Categories | {coverage.get('categories', 0)} | {csv(coverage.get('category_ids', []))} |",
            "",
            "## Weakest Subgroups",
            "",
            "| Dimension | ID | Cases | Pos | Neg | Precision | Recall | Repeated positive wakes | False wakes/hour | Max latency ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if subgroup_rows:
        for row in subgroup_rows:
            lines.append(
                f"| {row['dimension']} | `{row['id']}` | {row['cases']} | "
                f"{row['positives']} | {row['negatives']} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['repeated_positive_wake_events']} | "
                f"{row['false_wakes_per_hour']:.4f} | "
                f"{fmt(row['max_detection_latency_ms'])} |"
            )
        lines.append("")
    else:
        lines.extend(["| n/a | n/a | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |", ""])

    if audit:
        lines.extend(
            [
                "## Corpus Readiness Audit",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Ready | {str(audit.get('ready', False)).lower()} |",
                f"| Positive cases | {audit.get('positive_cases', 0)} |",
                f"| Required positive cases | {audit.get('planning_targets', {}).get('minimum_positive_cases', 'n/a')} |",
                f"| Negative audio seconds | {fmt(audit.get('negative_audio_seconds'))} |",
                f"| Required negative audio seconds | {audit.get('planning_targets', {}).get('minimum_negative_audio_seconds', 'n/a')} |",
                f"| Speakers | {audit.get('metadata', {}).get('speakers', {}).get('count', 0)} |",
                f"| Speakers with positives | {sum(1 for group in audit.get('subgroups', {}).get('speakers', {}).values() if group.get('positive_cases', 0) > 0)} |",
                f"| Speakers with negative audio | {sum(1 for group in audit.get('subgroups', {}).get('speakers', {}).values() if group.get('negative_audio_seconds', 0) > 0)} |",
                f"| Environments | {audit.get('metadata', {}).get('environments', {}).get('count', 0)} |",
                f"| Environments with positives | {sum(1 for group in audit.get('subgroups', {}).get('environments', {}).values() if group.get('positive_cases', 0) > 0)} |",
                f"| Environments with negative audio | {sum(1 for group in audit.get('subgroups', {}).get('environments', {}).values() if group.get('negative_audio_seconds', 0) > 0)} |",
                f"| Distances | {audit.get('metadata', {}).get('distances', {}).get('count', 0)} |",
                f"| Distances with positives | {sum(1 for group in audit.get('subgroups', {}).get('distances', {}).values() if group.get('positive_cases', 0) > 0)} |",
                f"| Distances with negative audio | {sum(1 for group in audit.get('subgroups', {}).get('distances', {}).values() if group.get('negative_audio_seconds', 0) > 0)} |",
                f"| Devices | {audit.get('metadata', {}).get('devices', {}).get('count', 0)} |",
                f"| Devices with positives | {sum(1 for group in audit.get('subgroups', {}).get('devices', {}).values() if group.get('positive_cases', 0) > 0)} |",
                f"| Devices with negative audio | {sum(1 for group in audit.get('subgroups', {}).get('devices', {}).values() if group.get('negative_audio_seconds', 0) > 0)} |",
                f"| Sessions | {audit.get('metadata', {}).get('sessions', {}).get('count', 0)} |",
                f"| Sessions with positives | {sum(1 for group in audit.get('subgroups', {}).get('sessions', {}).values() if group.get('positive_cases', 0) > 0)} |",
                f"| Sessions with negative audio | {sum(1 for group in audit.get('subgroups', {}).get('sessions', {}).values() if group.get('negative_audio_seconds', 0) > 0)} |",
                f"| Categories | {audit.get('metadata', {}).get('categories', {}).get('count', 0)} |",
                f"| Categories with positives | {sum(1 for group in audit.get('subgroups', {}).get('categories', {}).values() if group.get('positive_cases', 0) > 0)} |",
                f"| Categories with negative audio | {sum(1 for group in audit.get('subgroups', {}).get('categories', {}).values() if group.get('negative_audio_seconds', 0) > 0)} |",
                f"| Template speakers | {audit.get('speaker_gating', {}).get('template_speakers', 0)} |",
                f"| Unauthorized wake cases | {audit.get('speaker_gating', {}).get('unauthorized_wake_cases', 0)} |",
                f"| Unauthorized wake speakers | {audit.get('speaker_gating', {}).get('unauthorized_wake_speakers', 0)} |",
                f"| Unauthorized/enrolled speaker overlaps | {len(audit.get('speaker_gating', {}).get('unauthorized_template_speaker_overlaps', []))} |",
                f"| Template source counts | {audit.get('source_provenance', {}).get('template_source_counts', {})} |",
                f"| Case source counts | {audit.get('source_provenance', {}).get('case_source_counts', {})} |",
                f"| Calibration positive cases | {audit.get('splits', {}).get('calibration', {}).get('positive_cases', 0)} |",
                f"| Calibration negative audio seconds | {fmt(audit.get('splits', {}).get('calibration', {}).get('negative_audio_seconds', 0))} |",
                f"| Evaluation positive cases | {audit.get('splits', {}).get('evaluation', {}).get('positive_cases', 0)} |",
                f"| Evaluation negative audio seconds | {fmt(audit.get('splits', {}).get('evaluation', {}).get('negative_audio_seconds', 0))} |",
                f"| Failures | {len(audit.get('failures', []))} |",
                "",
            ]
        )

    if threshold:
        selected = threshold.get("selected") or {}
        calibration = selected.get("calibration") or {}
        evaluation = selected.get("evaluation") or {}
        lines.extend(
            [
                "## Threshold Selection",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Accepted | {str(threshold.get('accepted', False)).lower()} |",
                f"| Candidate threshold | {fmt(selected.get('candidate_threshold'))} |",
                f"| Accept threshold | {fmt(selected.get('accept_threshold'))} |",
                f"| Calibration precision | {fmt(calibration.get('precision'))} |",
                f"| Calibration recall | {fmt(calibration.get('recall'))} |",
                f"| Calibration repeated positive wake events | {calibration.get('repeated_positive_wake_events', 'n/a')} |",
                f"| Calibration false wakes/hour | {fmt(calibration.get('false_wakes_per_hour'))} |",
                f"| Evaluation precision | {fmt(evaluation.get('precision'))} |",
                f"| Evaluation recall | {fmt(evaluation.get('recall'))} |",
                f"| Evaluation repeated positive wake events | {evaluation.get('repeated_positive_wake_events', 'n/a')} |",
                f"| Evaluation false wakes/hour | {fmt(evaluation.get('false_wakes_per_hour'))} |",
                f"| Failures | {len(threshold.get('failures', []))} |",
                "",
            ]
        )

    sweep_rows = real.get("threshold_sweep", [])
    lines.extend(
        [
            "## Threshold Sweep",
            "",
            "| Candidate | Accept | TP | FP Events | TN | FN | Repeated positive wakes | Precision | Recall | False wakes/hour | Max latency ms |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if sweep_rows:
        for row in sweep_rows:
            lines.append(
                f"| {row['candidate_threshold']:.2f} | {row['accept_threshold']:.2f} | "
                f"{row['true_positives']} | {row['false_positives']} | "
                f"{row['true_negatives']} | {row['false_negatives']} | "
                f"{row.get('repeated_positive_wake_events', 0)} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['false_wakes_per_hour']:.4f} | {fmt(row.get('max_detection_latency_ms'))} |"
            )
        lines.append("")
    else:
        lines.extend(["| n/a | n/a | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a |", ""])
else:
    lines.extend(
        [
            "## Real Voice Corpus",
            "",
            "No real voice corpus manifest was supplied for this run.",
            "",
        ]
    )

summary_path.write_text("\n".join(lines), encoding="utf-8")
PY
log "Summary report: $SUMMARY_REPORT"

log "Writing real evidence status: $STATUS_REPORT"
scripts/summarize_vona_wake_real_evidence.py \
  --report-dir "$REPORT_DIR" \
  --output "$STATUS_REPORT"
log "Real evidence status report: $STATUS_REPORT"

log "Wake evaluation complete"
exit "$EXIT_CODE"
