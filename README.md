# Vona-rs

Vona-rs is a Rust workspace for real-time speech-to-speech runtimes.

Current scope:

- `vona`: core traits, event types, skill registry, and runtime policy surfaces.
- `vona-test-harness`: deterministic mock backend and transport coverage.

This workspace is intentionally backend-agnostic. Provider-specific STS integrations belong in adapter crates rather than in the core API.

## Release Gate

Run the deterministic release gate from the workspace root:

```bash
bash scripts/release_gate.sh
```

Readiness criteria and artifact expectations are documented in `docs/release-readiness-checklist.md`.
