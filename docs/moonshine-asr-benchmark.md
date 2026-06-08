# Vona Moonshine ASR Benchmark

This benchmark measures Vona's native Moonshine ASR path against a labelled
2,000-case generated-voice corpus. It is publishable regression evidence for
runtime speed and category behavior, but it is not human-recorded evidence.

No downstream host application is named or required for the report. The measured
component is the Vona ASR path directly: `vona-moonshine` with the native
`libmoonshine` binding and the `MEDIUM_STREAMING` English model, using the same
configuration shape a consumer host can wire into its voice runtime.

## Corpus

- Cases: 2,000
- Categories: 10, with 200 cases each
- Categories: complex reasoning, conversational fragments, long form, numbers
  and dates, proper nouns, punctuation-heavy prompts, real-world tasks, short
  commands, simple questions, technical debugging
- Audio source: generated macOS voice (`generated-macos-say`)
- Audio format: raw 16 kHz mono PCM16LE
- Manifest: `target/asr-corpus/generated-2000/manifest.jsonl`
- Manifest SHA-256:
  `14d9e5c0d7754ce566b8150e0f009d418fcd15d99fe583d11ed15afc4161937e`
- Report: `target/asr-corpus/generated-2000/moonshine-native-report.json`
- Report SHA-256:
  `00b8c63ac02f5de8ba0e10b561af77e139cfb1e90b597ba2872f3b1e9c23af5c`

## Result

| Metric | Result |
| --- | ---: |
| Cases scored | 2,000 |
| Model arch | `MEDIUM_STREAMING` |
| Load time | 283.8 ms |
| Average STT latency | 601.0 ms |
| p50 STT latency | 663.4 ms |
| p95 STT latency | 904.0 ms |
| p99 STT latency | 977.8 ms |
| Average real-time factor | 0.0887 |
| Average WER | 0.301 |
| p95 WER | 1.0 |
| Near-exact cases, WER <= 0.10 | 623 / 2,000 |
| Near-exact rate | 31.15% |

The speed result is strong: Moonshine ran at roughly 8.9% of realtime on this
machine and stayed under one second at p99 for the generated corpus. The accuracy
result is not release-grade as a standalone ASR policy. Average WER was 30.1%,
and only 31.15% of cases were near-exact.

## Category Rollup

| Category | Cases | Avg WER | Near-exact rate | Avg STT ms |
| --- | ---: | ---: | ---: | ---: |
| complex_reasoning | 200 | 0.152 | 71.5% | 621.6 |
| simple_question | 200 | 0.160 | 55.0% | 269.4 |
| conversational_fragment | 200 | 0.185 | 55.5% | 305.8 |
| long_form | 200 | 0.189 | 43.0% | 756.2 |
| technical_debug | 200 | 0.244 | 20.5% | 711.1 |
| punctuation_heavy | 200 | 0.253 | 24.0% | 796.7 |
| proper_nouns | 200 | 0.267 | 15.5% | 718.1 |
| real_world_task | 200 | 0.332 | 19.0% | 675.2 |
| short_command | 200 | 0.572 | 7.5% | 354.8 |
| numbers_dates | 200 | 0.651 | 0.0% | 800.9 |

The main failures were numbers/dates, short commands with synthetic identifiers,
and proper-noun-like vocabulary. Some high-WER examples also showed stale or
repeated transcript content rather than simple acoustic confusion. That makes the
safe product conclusion clear: the current local Moonshine path is fast enough
to be useful, but Vona should keep ASR acceptance gates and hybrid fallback policy
available for release-grade applications.

## Reproduction

Generate the corpus:

```bash
python3 bench/generate_asr_corpus.py \
  --count 2000 \
  --out-dir target/asr-corpus/generated-2000 \
  --jobs 4 \
  --source-type generated-macos-say
```

Run the native Moonshine scorer:

```bash
VONA_MOONSHINE_LIBRARY_PATH=/path/to/libmoonshine.dylib \
VONA_MOONSHINE_MODEL_PATH=/path/to/medium-streaming-en/quantized \
VONA_MOONSHINE_ARCH=MEDIUM_STREAMING \
VONA_MOONSHINE_BENCH_MANIFEST=target/asr-corpus/generated-2000/manifest.jsonl \
VONA_MOONSHINE_BENCH_OUTPUT=target/asr-corpus/generated-2000/moonshine-native-report.json \
cargo run --release -p vona-moonshine --example native_corpus_benchmark
```

## Evidence Status

This benchmark is useful for regression and architecture decisions, but it does
not replace human-recorded corpus evidence. A release claim should require a
separate human-recorded corpus with microphone/environment metadata, speaker
coverage, audio quality measurements, and pass/fail thresholds that reflect the
target application.
