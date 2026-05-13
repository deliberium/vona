# Vona-rs Release Readiness Checklist

This checklist is the single-page release gate for Vona-rs.

## How To Run

From the workspace root:

```bash
bash scripts/release_gate.sh
```

The script is deterministic by design:

- Resolves `LIBOPUS_LIB_DIR` using standard prefixes or `pkg-config`
- Normalizes `LIBOPUS_LIB_DIR` to a prefix path (`/opt/homebrew`), not a raw libdir (`/opt/homebrew/lib`)
- Uses a fixed benchmark input (`--iterations=8 --sample-count=320`)
- Uses `--live --mock-live` to avoid external sidecar/network dependencies

## Pass/Fail Criteria

Release readiness is **PASS** only when all items below succeed in a single run.

1. `cargo check --workspace --locked` exits with code `0`
2. Deterministic per-crate test matrix exits with code `0`:
   - `cargo test -p vona --locked`
   - `cargo test -p vona-test-harness --locked`
   - `cargo test -p vona-seamless --locked`
   - `cargo test -p vona-transport-local --locked`
   - `cargo test -p vona-sidecar --locked`
   - `cargo test -p vona-moshi --locked`
3. `cargo check --workspace --all-targets --locked` exits with code `0`
4. `cargo clippy --workspace --all-targets --locked -- -D warnings` exits with code `0`
5. Transport benchmark smoke run exits with code `0`
6. Benchmark log contains all required metrics keys:
   - `http_round_trip_avg_ms=`
   - `ipc_round_trip_avg_ms=`
   - `live_latency_ratio_http_over_ipc=`

Any failure above is a **FAIL** and blocks release.

## Evidence Artifacts

The gate produces one benchmark artifact:

- `target/release-gate-transport-bench.log`

Keep this file in CI artifacts for each release candidate.

## Notes

- If `libopus` is not installed, the gate fails early with a clear setup message.
- This gate validates build/test/benchmark determinism and baseline transport performance smoke behavior.
- Production latency SLO validation should be layered on top of this gate in environment-specific performance jobs.
