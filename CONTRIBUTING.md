# Contributing to vona-rs

Thank you for your interest in contributing!

## Getting Started

1. Fork and clone the repository.
2. Install system dependencies:
   - **macOS**: `brew install opus`
   - **Linux**: `sudo apt-get install libopus-dev pkg-config`
3. Verify your setup with the release gate:
   ```bash
   bash scripts/release_gate.sh
   ```

## Development Workflow

- All changes must keep `bash scripts/release_gate.sh` green.
- New adapter crates go under `crates/` and must not add internal dependencies to `deliberium`.
- Keep crates focused: core traits in `vona`, provider adapters in separate crates.

## Pull Requests

- Target the `main` branch.
- Include a clear description of what changes and why.
- Keep PRs small and focused — one feature or fix per PR.
- All CI checks must pass before merge.

## Tests

Run the deterministic test matrix:

```bash
cargo test -p vona
cargo test -p vona-test-harness
```

Run all compile-checked targets:

```bash
cargo check --workspace --all-targets --locked
```

## Bug Reports & Feature Requests

Open a GitHub Issue with:

- A minimal reproduction case (for bugs)
- The expected vs. actual behavior

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and constructive.
