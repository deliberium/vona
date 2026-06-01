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
- Checks optional facade feature combinations for hosted providers, Ollama, MLX, and model provisioning
- Runs native MLX compile checks on macOS when `xcrun metal` is available

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
   - `cargo test -p vona-qwen --locked`
   - `cargo test -p vona-ollama --locked`
   - `cargo test -p vona-model-provisioning --locked`
   - `cargo test -p vona-mlx --locked`
   - `cargo test -p vona-mlx-speech --locked`
   - `cargo test -p vona-mlx-whisper --locked`
   - `cargo test -p vona-mlx-qwen3-tts --locked`
3. `cargo check --workspace --all-targets --locked` exits with code `0`
4. `cargo clippy --workspace --all-targets --locked -- -D warnings` exits with code `0`
5. Facade feature compile matrix exits with code `0`:
   - hosted provider cloud features
   - local Ollama feature
   - MLX adapter features without native Metal
   - combined Ollama + MLX + model-provisioning features
6. Native MLX compile matrix exits with code `0` on macOS hosts with `xcrun metal` available:
   - `vona-mlx-speech --features native-mlx`
   - `vona-mlx --features "native-mlx mlx-models-loader"`
   - `vona-mlx-whisper --features native-mlx`
   - `vona-mlx-qwen3-tts --features native-mlx`
   - `vona --features "ollama mlx-whisper-native mlx-qwen3-tts-native model-provisioning"`
7. Transport benchmark smoke run exits with code `0`
8. Benchmark log contains all required metrics keys:
   - `http_round_trip_avg_ms=`
   - `ipc_round_trip_avg_ms=`
   - `live_latency_ratio_http_over_ipc=`
9. `scripts/run_vona_wake_eval.sh` exits with code `0`
10. `python3 -m unittest tests/test_vona_wake_acceptance.py tests/test_vona_wake_collection_ledger.py tests/test_vona_wake_eval_runner.py tests/test_vona_wake_evidence_package.py tests/test_vona_wake_evidence_status.py tests/test_vona_wake_label_semantics.py tests/test_vona_wake_recorder.py tests/test_vona_wake_recording_progress.py tests/test_vona_wake_threshold_selection.py` exits with code `0`
11. Wake evaluation output includes:
   - enforced generated wake regression report
   - real voice corpus report when `VONA_WAKE_REAL_EVAL_MANIFEST` is set
   - real voice readiness audit report when `VONA_WAKE_REAL_EVAL_AUDIT=1` or `VONA_WAKE_REQUIRE_REAL_EVAL=1`
   - real voice acceptance check when `VONA_WAKE_REAL_EVAL_ACCEPTANCE=1` or `VONA_WAKE_REQUIRE_REAL_EVAL=1`
   - real voice statistical confidence bounds when a real corpus report is present
   - real voice threshold sweep when a real corpus report is present
   - real voice readiness audit coverage for enrolled template speakers and `unauthorized-wake` negative cases
   - real voice readiness audit session coverage and session-provenance checks
   - real voice readiness audit leakage checks for exact audio duplicates and gain-normalized audio fingerprint duplicates
   - real voice readiness audit label-semantic checks for wake-positive, unauthorized-wake, and non-wake negative categories
   - real voice source provenance showing `source_type=human-recorded` for templates and cases
   - real voice corpus identity showing top-level `corpus.id`, `corpus.version`, and `corpus.source=human-recorded`
   - matching `manifest_sha256` values between readiness and real voice reports
   - real voice calibration/evaluation split reporting, with evaluation clips large enough for the confidence targets
   - release failure when `VONA_WAKE_REQUIRE_REAL_EVAL=1` and no real manifest is supplied
   - status artifacts when `VONA_WAKE_REQUIRE_REAL_EVAL=1` fails because no real manifest is supplied
12. `scripts/package_vona_wake_evidence.py --report-dir target/vona-wake-eval --output-dir target/vona-wake-evidence` exits with code `0` and writes:
   - `target/vona-wake-evidence/evidence-manifest.json`
   - `target/vona-wake-evidence/evidence-summary.md`
   - real-evidence stage status in both packaged artifacts
13. Criterion benchmark collection exits with code `0` and emits the expected row counts:
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
- `target/vona-wake-eval/generated-report.json`
- `target/vona-wake-eval/generated-manifest.json`
- `target/vona-wake-eval/audit-report.json` when a real voice readiness audit is enabled
- `target/vona-wake-eval/real-report.json` when a real voice manifest is supplied
- `target/vona-wake-eval/threshold-selection-report.json` when threshold selection is enabled
- `target/vona-wake-eval/summary.md`
- `target/vona-wake-eval/real-evidence-status.md`
- `target/vona-wake-evidence/evidence-manifest.json` when wake evidence is packaged
- `target/vona-wake-evidence/evidence-summary.md` when wake evidence is packaged
- `docs/benchmark-results.md`

Keep these files in CI artifacts for each release candidate.

For wakeword release candidates that claim real-world reliability, run
`scripts/plan_vona_wake_corpus.py` first, make the real corpus at least as large
as the planner recommends for the chosen confidence targets, and require
`scripts/audit_vona_wake_recording_plan.py --enforce /path/to/recordings.csv` to
pass before collection begins. After recording, require
`scripts/audit_vona_wake_corpus.py --enforce /path/to/corpus/manifest.json` to
pass before running the real scorer. The real corpus must include enrolled
template `speaker_id` values plus `unauthorized-wake` negative clips from
non-enrolled speakers, with zero unauthorized/enrolled speaker overlaps, if the
release claim includes speaker-gated wake access. Every template and case used
for the real-world reliability claim must be marked `source_type=human-recorded`,
and `scripts/build_vona_wake_corpus.py` must be run without
`--allow-non-human-source`.
`scripts/record_vona_wake_corpus.py` must also be run without
`--allow-non-human-source` for release recordings, and its preflight must accept
the CSV before operators begin recording.
The pre-recording plan and built corpus must include the required release-grade
categories in aggregate, calibration, and evaluation coverage: `wake-positive`,
`unauthorized-wake`, `near-miss`, `ordinary-speech`, `ordinary-command`, and
`background-speech`. Each required negative category must have at least one
second of calibration negative audio and at least 600 seconds of evaluation
negative audio by default, and `wake-positive` must have calibration and
evaluation positive cases. Positive cases must also cover early (`<=250 ms`),
mid (`251-1500 ms`), and late (`>1500 ms`) `wake_start_ms` buckets in both
calibration and evaluation splits. Every enrolled template speaker must have
templates for each required wake phrase, and the default required phrases are
`hey vona` and `vona`. Each required phrase must have at least one calibration
positive case and ten evaluation positive cases. Every enrolled template
speaker must have at least two calibration positive cases and ten evaluation
positive cases by default. Each environment, distance, device, and session value must
also have at least one positive case and one second of negative audio in
calibration, plus at least one positive case and 600 seconds of negative audio
in evaluation.
Each enrolled speaker must also have at least one evaluation positive recorded
in a session that was not used for that speaker's enrollment templates. This
keeps speaker-gated wake claims from relying only on same-session enrollment
acoustics.
The manifest must also carry top-level `corpus.id`, `corpus.version`, and
`corpus.source=human-recorded` metadata so the accepted evidence bundle is tied
to a named, versioned recording set. The readiness and real-eval reports must
also contain the same `manifest_sha256`, proving they were generated from the
same manifest contents.
Release-grade real corpora must also include a top-level `collection_ledger`
with consent protocol, collection protocol, collection operators, start/end
timestamps, and consent/provenance entries for every `speaker_id` referenced by
templates or cases. The ledger must also include device provenance entries for
every manifest `device`, including recorder information and sample rate, plus
session provenance entries for every manifest `session_id`, including collection
time and operator, so device/session-slice metrics are tied to known capture
setups. The corpus metadata must record the ledger SHA-256 so the
accepted evidence can be tied back to the collection records without embedding
private consent details in every report.
The readiness audit must report zero template/case path overlaps, zero exact
template/case audio hash overlaps, zero gain-normalized template/case audio
fingerprint overlaps, and zero duplicate case groups by path, exact audio hash,
or gain-normalized audio fingerprint. This prevents enrollment utterances or
copied evaluation clips from inflating the measured accuracy.
The recording-plan audit and manifest readiness audit must also validate label
semantics: positive cases use a configured wake phrase as `expected_phrase`,
`unauthorized-wake` negatives actually contain a configured wake phrase, other
negative categories do not contain configured wake phrases, and negative cases
do not set `expected_phrase`. This prevents mislabeled text from inflating
recall or hiding wake-phrase false positives.
Calibration clips may be used for threshold exploration, but the final
reliability claim must pass on `split=evaluation` clips that independently
satisfy the corpus-size, negative-exposure, and metadata-coverage targets.
After scoring, require
`scripts/select_vona_wake_threshold.py --real-report target/vona-wake-eval/real-report.json --enforce`
to choose an operating point from calibration audio and verify that point on
evaluation audio. By default the selector also requires at least two
calibration-passing threshold sweep points and at least one evaluation-passing
point, so a single knife-edge threshold is not enough for a reliability claim.
Save the JSON threshold-selection report beside the real report so its
real-report SHA-256 can be checked. Acceptance must reject threshold reports
that are missing `real_report_sha256`, refer to a different real report, or omit
selected calibration/evaluation metrics. Then require
`scripts/check_vona_wake_acceptance.py --audit-report target/vona-wake-eval/audit-report.json --real-report target/vona-wake-eval/real-report.json --threshold-report target/vona-wake-eval/threshold-selection-report.json --require-threshold-selection`
to pass before making a real-world wakeword reliability claim. The acceptance
checker must independently reject non-human real-report provenance, missing
audit collection-ledger/source-provenance evidence, non-empty audit leakage
groups, and pass evaluation subgroup gates recomputed from per-case
details for speaker, environment, distance, device, session, category, expected phrase,
and early/mid/late wake onset buckets, including repeated-positive-wake checks,
so a weak subgroup cannot be hidden by aggregate scores. It must also verify
that saved top-level/evaluation-split metrics and confidence bounds match
independently recomputed `cases_detail` metrics, and that the selected threshold
is present in the real report's threshold sweep with matching split metrics.
Package the accepted artifacts with
`scripts/package_vona_wake_evidence.py --report-dir target/vona-wake-eval --output-dir target/vona-wake-evidence --zip --enforce`.

## Crates.io Publish Order

Publish workspace crates in dependency order:

1. `vona-core`
2. `vona-model-provisioning`
3. `vona-ollama`
4. `vona-mlx-speech`
5. `vona-mlx`
6. `vona-mlx-whisper`
7. `vona-mlx-qwen3-tts`
8. Provider and adapter crates: `vona-openai-realtime`, `vona-gemini-live`, `vona-azure-speech`, `vona-elevenlabs`, `vona-deepgram`, `vona-qwen`, `vona-seamless`, `vona-moshi`, `vona-test-harness`, `vona-transport-local`
9. `vona-sidecar`
10. `vona`

Until `vona-core` is live on crates.io, provider crates that depend on it cannot be fully verified by `cargo package`.

The `release.yml` GitHub workflow wraps `scripts/release_crates.sh` with a manual `current`, `patch`, `minor`, or `major` input. It updates Cargo versions, refreshes `CHANGELOG.md`, runs the release gate, creates a local release commit and tag, publishes in order when requested, then pushes the release commit/tag and creates the GitHub release. Publish mode skips the offline package dry-run during preparation because `cargo publish` performs packaging against the live registry. Publish mode can skip already-published versions when resuming a partial publish, retries dependent publishes while the crates.io index catches up, and waits/retries when crates.io returns a new-crate rate-limit response.

## Notes

- If `libopus` is not installed, the gate fails early with a clear setup message.
- If Metal tooling is unavailable, native MLX checks are skipped with a clear message; non-native MLX facade checks still run.
- This gate validates build/test/benchmark determinism, baseline transport performance smoke behavior, realtime event flow overhead, hosted-provider protocol mapping overhead, and local model provisioning validation overhead.
- Production latency SLO validation should be layered on top of this gate in environment-specific performance jobs.
