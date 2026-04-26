# Local Mock Example

This example slot is reserved for the first end-to-end in-process mock transport demo.

This example directory now pairs with committed waveform fixtures under `vona-rs/tests/fixtures/`.

Use the deterministic loopback harness in `vona-test-harness` to validate fixture replay and in-process latency:

```bash
cargo test -p vona-test-harness waveform_fixture_round_trips_through_scripted_transport -- --nocapture
cargo test -p vona-test-harness scripted_transport_loopback_latency_stays_low_for_fixture_audio -- --nocapture
```

The current fixtures are intentionally tiny and deterministic:

- `sine-16khz-mono.json` validates sample-shape preservation.
- `impulse-16khz-mono.json` validates low-latency loopback behavior.
  Initial implementation work is focused on the core crates and Deliberium runtime seam.
