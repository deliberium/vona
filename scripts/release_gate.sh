#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() {
  printf "\n[%s] %s\n" "release-gate" "$1"
}

resolve_libopus_dir() {
  if [[ -n "${LIBOPUS_LIB_DIR:-}" ]]; then
    local configured
    configured="$LIBOPUS_LIB_DIR"

    # `audiopus_sys` appends `/lib` to this value, so normalize any
    # accidentally provided libdir (e.g. /opt/homebrew/lib) to its prefix.
    if [[ "$configured" == */lib ]]; then
      configured="$(dirname "$configured")"
    fi

    echo "$configured"
    return 0
  fi

  if [[ -d "/opt/homebrew/lib" ]] && [[ -f "/opt/homebrew/lib/libopus.a" ]]; then
    echo "/opt/homebrew"
    return 0
  fi

  if [[ -d "/usr/local/lib" ]] && [[ -f "/usr/local/lib/libopus.a" ]]; then
    echo "/usr/local"
    return 0
  fi

  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists opus; then
    local opus_libdir
    opus_libdir="$(pkg-config --variable=libdir opus)"
    if [[ -n "$opus_libdir" ]]; then
      echo "$(dirname "$opus_libdir")"
      return 0
    fi
  fi

  return 1
}

if LIBOPUS_RESOLVED="$(resolve_libopus_dir)"; then
  export LIBOPUS_LIB_DIR="$LIBOPUS_RESOLVED"
  log "Using LIBOPUS_LIB_DIR=$LIBOPUS_LIB_DIR"
else
  log "Failed to locate libopus. Set LIBOPUS_LIB_DIR to your prefix (e.g. /opt/homebrew) and rerun."
  exit 1
fi

export CARGO_TERM_COLOR=always

# Ensure we do not accidentally disable pkg-config probing through inherited env.
unset LIBOPUS_NO_PKG || true
unset OPUS_NO_PKG || true

log "Step 1/6: cargo check --workspace --locked"
cargo check --workspace --locked

log "Step 2/6: deterministic crate test matrix"
cargo test -p vona --locked
cargo test -p vona-core --locked
cargo test -p vona-test-harness --locked
cargo test -p vona-ollama --locked
cargo test -p vona-mlx --locked
cargo test -p vona-mlx-speech --locked
cargo test -p vona-mlx-whisper --locked
cargo test -p vona-mlx-qwen3-tts --locked
cargo test -p vona-seamless --locked
cargo test -p vona-transport-local --locked
cargo test -p vona-sidecar --locked
cargo test -p vona-moshi --locked
cargo test -p vona-openai-realtime --locked
cargo test -p vona-gemini-live --locked
cargo test -p vona-azure-speech --locked
cargo test -p vona-elevenlabs --locked
cargo test -p vona-deepgram --locked
cargo test -p vona-qwen --locked
cargo test -p vona-model-provisioning --locked

log "Step 2b/6: optional adapter feature compile matrix"
cargo check -p vona --features "ollama mlx mlx-whisper mlx-qwen3-tts model-provisioning" --locked

if [[ "$(uname -s)" == "Darwin" ]] && command -v xcrun >/dev/null 2>&1 && xcrun -f metal >/dev/null 2>&1; then
  log "Step 2c/6: native MLX compile matrix"
  cargo check -p vona-mlx-speech --features native-mlx --locked
  cargo check -p vona-mlx --features "native-mlx mlx-models-loader" --locked
  cargo check -p vona-mlx-whisper --features native-mlx --locked
  cargo check -p vona-mlx-qwen3-tts --features native-mlx --locked
  cargo check -p vona --features "ollama mlx-whisper-native mlx-qwen3-tts-native model-provisioning" --locked
else
  log "Step 2c/6: skipping native MLX compile matrix; macOS Metal compiler not available"
fi

log "Step 3/6: cargo check --workspace --all-targets --locked"
cargo check --workspace --all-targets --locked

log "Step 4/6: cargo clippy --workspace --all-targets --locked -- -D warnings"
cargo clippy --workspace --all-targets --locked -- -D warnings

log "Step 5/6: deterministic transport benchmark smoke run"
BENCH_OUTPUT_FILE="${ROOT_DIR}/target/release-gate-transport-bench.log"
cargo run --release -p vona-transport-local --example seamless_m4t_transport_bench -- \
  --iterations=8 \
  --sample-count=320 \
  --live \
  --mock-live \
  > "$BENCH_OUTPUT_FILE"

if ! grep -q "http_round_trip_avg_ms=" "$BENCH_OUTPUT_FILE"; then
  log "Benchmark output missing http_round_trip_avg_ms"
  exit 1
fi
if ! grep -q "ipc_round_trip_avg_ms=" "$BENCH_OUTPUT_FILE"; then
  log "Benchmark output missing ipc_round_trip_avg_ms"
  exit 1
fi
if ! grep -q "live_latency_ratio_http_over_ipc=" "$BENCH_OUTPUT_FILE"; then
  log "Benchmark output missing live_latency_ratio_http_over_ipc"
  exit 1
fi

log "Step 6/6: writing docs/benchmark-results.md"

RESAMPLE_LOG="${ROOT_DIR}/target/release-gate-resample-bench.log"
SLO_LOG="${ROOT_DIR}/target/release-gate-slo-bench.log"
REALTIME_LOG="${ROOT_DIR}/target/release-gate-realtime-bench.log"
RESULTS_MD="${ROOT_DIR}/docs/benchmark-results.md"
RUN_DATE="$(date -u '+%Y-%m-%d %H:%M UTC')"

# Run Criterion benchmarks and capture output.
# --output-format bencher emits "test <name> ... bench: <median> ns/iter (+/- <mad>)" lines.
cargo bench -p vona-seamless --bench resample -- \
  --warm-up-time 1 --measurement-time 3 --output-format bencher \
  2>/dev/null > "$RESAMPLE_LOG"

cargo bench -p vona-test-harness --bench slo -- \
  --warm-up-time 1 --measurement-time 3 --output-format bencher \
  2>/dev/null > "$SLO_LOG"

cargo bench -p vona-test-harness --bench realtime -- \
  --warm-up-time 1 --measurement-time 3 --output-format bencher \
  2>/dev/null > "$REALTIME_LOG"

# ── Parse one bench line into a table row ───────────────────────────────────
# Input: "test resample_mono/8k_to_16k ... bench:      15,993 ns/iter (+/-  123)"
# Output: "| resample_mono/8k_to_16k | 15993 ns/iter | ± 123 |"
parse_bench_lines() {
  local logfile="$1"
  grep "^test " "$logfile" 2>/dev/null | while IFS= read -r line; do
    # Extract name (field 2), median (field 5 with commas stripped), mad (field 7)
    name="$(echo "$line" | awk '{print $2}')"
    median="$(echo "$line" | awk '{print $5}' | tr -d ',')"
    unit="$(echo "$line" | awk '{print $6}')"
    mad="$(echo "$line" | awk '{print $8}' | tr -d ')')"
    printf "| \`%s\` | %s %s | ± %s |\n" "$name" "$median" "$unit" "$mad"
  done
}

require_bench_rows() {
  local logfile="$1"
  local expected="$2"
  local label="$3"
  local actual

  actual="$(grep -c "^test " "$logfile" 2>/dev/null || true)"
  if [[ "$actual" -lt "$expected" ]]; then
    log "Benchmark output for ${label} has ${actual} rows; expected at least ${expected}"
    exit 1
  fi
}

require_bench_rows "$RESAMPLE_LOG" 8 "vona-seamless resample"
require_bench_rows "$SLO_LOG" 9 "vona-test-harness SLO"
require_bench_rows "$REALTIME_LOG" 8 "vona-test-harness realtime/provider/provisioning"

# ── Extract transport gate metrics ──────────────────────────────────────────
extract_transport_metric() {
  local key="$1"
  grep "${key}=" "$BENCH_OUTPUT_FILE" 2>/dev/null | head -1 | sed "s/.*${key}=//"
}

HTTP_MS="$(extract_transport_metric http_round_trip_avg_ms)"
IPC_MS="$(extract_transport_metric ipc_round_trip_avg_ms)"
RATIO="$(extract_transport_metric live_latency_ratio_http_over_ipc)"

# ── Write markdown ───────────────────────────────────────────────────────────
{
  printf "# Benchmark Results\n\n"
  printf "> **Generated by \`scripts/release_gate.sh\` on %s**\n" "$RUN_DATE"
  printf "> Do not edit manually — re-run the gate to refresh.\n\n"

  printf "## Transport Smoke Benchmarks\n\n"
  printf "These run deterministically in every gate pass using \`--mock-live --iterations=8 --sample-count=320\`.\n\n"
  printf "| Metric | Value |\n"
  printf "|--------|-------|\n"
  printf "| HTTP round-trip avg | %s ms |\n" "$HTTP_MS"
  printf "| IPC round-trip avg  | %s ms |\n" "$IPC_MS"
  printf "| HTTP / IPC latency ratio | %s |\n" "$RATIO"
  printf "\n"

  printf "## Resample Throughput (vona-seamless)\n\n"
  printf "Criterion micro-benchmarks. Each row is the median of 100 samples over a 3-second measurement window.\n\n"
  printf "| Benchmark | Median | MAD |\n"
  printf "|-----------|--------|-----|\n"
  parse_bench_lines "$RESAMPLE_LOG"
  printf "\n"

  printf "## Session / Transport SLO (vona-test-harness)\n\n"
  printf "Criterion micro-benchmarks using the deterministic \`MockBackend\` and \`ScriptedTransport\` — no external services required.\n\n"
  printf "| Benchmark | Median | MAD |\n"
  printf "|-----------|--------|-----|\n"
  parse_bench_lines "$SLO_LOG"
  printf "\n"

  printf "## Realtime / Provider / Provisioning SLO (vona-test-harness)\n\n"
  printf "Criterion micro-benchmarks for the provider-neutral realtime contract, hosted-provider protocol mapping, and local model provisioning validation. These are deterministic and do not call external services.\n\n"
  printf "| Benchmark | Median | MAD |\n"
  printf "|-----------|--------|-----|\n"
  parse_bench_lines "$REALTIME_LOG"
  printf "\n"

  printf "## SLO Targets\n\n"
  printf "| Benchmark group | Target |\n"
  printf "|-----------------|--------|\n"
  printf "| \`backend_step\` (all frame sizes) | P99 < 500 µs |\n"
  printf "| \`session_lifecycle\` | P99 < 1 ms |\n"
  printf "| \`transport_loopback\` (all frame sizes) | P99 < 100 µs |\n"
  printf "| \`inject_and_drain_10_events\` | P99 < 200 µs |\n"
  printf "| \`scripted_realtime_event_flow\` | P99 < 500 µs |\n"
  printf "| \`provider_mapping\` audio/tool/config events | P99 < 500 µs |\n"
  printf "| \`model_provisioning\` manifest/checksum validation | P99 < 5 ms |\n"
  printf "| \`resample_mono\` 8k→16k 1 s | > 50× real-time |\n"
  printf "| \`resample_mono\` 48k→16k 1 s | > 30× real-time |\n"
  printf "| \`resample_mono\` identity 1 s | > 500× real-time |\n"
} > "$RESULTS_MD"

log "Release gate PASSED"
log "Benchmark log saved at target/release-gate-transport-bench.log"
log "Benchmark results written to docs/benchmark-results.md"
