# Model Provisioning

Vona should own local model cache layout for the model backends it supports. That keeps host applications from hard-coding provider-specific paths and makes release checks more repeatable.

The initial `vona-model-provisioning` crate provides:

- `ModelManifest` for local model identity, provider family, and artifacts
- `ModelArtifact` for expected relative paths, optional source URLs, size hints, and future checksums
- `ModelCache` for deriving cache paths and inspecting whether artifacts are present
- manifest validation that rejects empty file-backed manifests and absolute artifact paths
- `HttpModelProvisioner` for explicit source-URL downloads into the Vona cache

The default cache environment variable is:

```text
VONA_MODEL_CACHE_DIR
```

The crate can download missing artifacts when the host explicitly calls `HttpModelProvisioner::provision_missing`. It does not download automatically from backend constructors. That keeps model licenses, user prompts, credentials, and spend/network policy under application control while still letting Vona own cache layout and artifact placement.

Recommended next step:

1. Add checksum enforcement.
2. Add provider-aware helpers for Hugging Face and Ollama-style local model pulls.
3. Wire `SeamlessM4tLocalConfig::from_env` through a provisioning plan so missing ONNX artifacts produce actionable setup errors.
