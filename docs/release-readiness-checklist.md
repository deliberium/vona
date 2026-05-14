# Vona-rs Release Readiness Checklist

This checklist is the single-page release gate for Vona-rs.

## How To Run

From the workspace root:

```bash
bash scripts/release_gate.sh
```

For release packaging and crates.io publishing:

```bash
scripts/release_crates.sh --release patch
scripts/release_crates.sh --release patch --publish
scripts/release_crates.sh --release current --bootstrap
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
   - `cargo test -p vona-core --locked`
   - `cargo test -p vona-test-harness --locked`
   - `cargo test -p vona-seamless --locked`
   - `cargo test -p vona-transport-local --locked`
   - `cargo test -p vona-sidecar --locked`
   - `cargo test -p vona-moshi --locked`
   - `cargo test -p vona-openai-realtime --locked`
   - `cargo test -p vona-gemini-live --locked`
   - `cargo test -p vona-azure-speech --locked`
   - `cargo test -p vona-elevenlabs --locked`
   - `cargo test -p vona-deepgram --locked`
   - `cargo test -p vona-model-provisioning --locked`
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

## Crates.io Publish Order

Publish workspace crates in dependency order:

1. `vona-core`
2. `vona-model-provisioning`
3. Provider and adapter crates: `vona-openai-realtime`, `vona-gemini-live`, `vona-azure-speech`, `vona-elevenlabs`, `vona-deepgram`, `vona-seamless`, `vona-moshi`, `vona-test-harness`, `vona-transport-local`
4. `vona`

Until `vona-core` is live on crates.io, provider crates that depend on it cannot be fully verified by `cargo package`.

The `release.yml` GitHub workflow wraps `scripts/release_crates.sh` with a manual `current`, `patch`, `minor`, or `major` input. It updates Cargo versions, refreshes `CHANGELOG.md`, runs the release gate, creates a local release commit and tag, publishes in order when requested, then pushes the release commit/tag and creates the GitHub release. Publish mode waits for each crate to become visible in the crates.io index before publishing dependents.

## Notes

- If `libopus` is not installed, the gate fails early with a clear setup message.
- This gate validates build/test/benchmark determinism and baseline transport performance smoke behavior.
- Production latency SLO validation should be layered on top of this gate in environment-specific performance jobs.
