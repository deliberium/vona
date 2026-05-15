# Vona Release Readiness Checklist

This checklist is the single-page release gate for Vona.

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
scripts/release_crates.sh --release current --bootstrap --skip-package-dry-run
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
7. Criterion benchmark collection exits with code `0` and emits the expected row counts:
   - at least 8 `vona-seamless` resample rows
   - at least 9 `vona-test-harness` session/transport SLO rows
   - at least 8 `vona-test-harness` realtime/provider/provisioning SLO rows

Any failure above is a **FAIL** and blocks release.

## Evidence Artifacts

The gate produces benchmark artifacts:

- `target/release-gate-transport-bench.log`
- `target/release-gate-resample-bench.log`
- `target/release-gate-slo-bench.log`
- `target/release-gate-realtime-bench.log`
- `docs/benchmark-results.md`

Keep these files in CI artifacts for each release candidate.

## Crates.io Publish Order

Publish workspace crates in dependency order:

1. `vona-core`
2. `vona-model-provisioning`
3. Provider and adapter crates: `vona-openai-realtime`, `vona-gemini-live`, `vona-azure-speech`, `vona-elevenlabs`, `vona-deepgram`, `vona-seamless`, `vona-moshi`, `vona-test-harness`, `vona-transport-local`
4. `vona`

Until `vona-core` is live on crates.io, provider crates that depend on it cannot be fully verified by `cargo package`.

The `release.yml` GitHub workflow wraps `scripts/release_crates.sh` with a manual `current`, `patch`, `minor`, or `major` input. It updates Cargo versions, refreshes `CHANGELOG.md`, runs the release gate, creates a local release commit and tag, publishes in order when requested, then pushes the release commit/tag and creates the GitHub release. Publish mode skips the offline package dry-run during preparation because `cargo publish` performs packaging against the live registry. Publish mode can skip already-published versions when resuming a partial publish, retries dependent publishes while the crates.io index catches up, and waits/retries when crates.io returns a new-crate rate-limit response.

## Notes

- If `libopus` is not installed, the gate fails early with a clear setup message.
- This gate validates build/test/benchmark determinism, baseline transport performance smoke behavior, realtime event flow overhead, hosted-provider protocol mapping overhead, and local model provisioning validation overhead.
- Production latency SLO validation should be layered on top of this gate in environment-specific performance jobs.
