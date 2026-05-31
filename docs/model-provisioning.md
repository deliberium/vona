# Model Provisioning

Vona should own local model cache layout for the model backends it supports. That keeps host applications from hard-coding provider-specific paths and makes release checks more repeatable.

The initial `vona-model-provisioning` crate provides:

- `ModelManifest` for local model identity, provider family, and artifacts
- `ModelArtifact` for expected relative paths, optional source URLs, size hints, and future checksums
- `ModelCache` for deriving cache paths and inspecting whether artifacts are present
- manifest validation that rejects empty file-backed manifests and absolute artifact paths
- `HttpModelProvisioner` for explicit source-URL downloads into the Vona cache
- streamed artifact writes through a temporary file followed by atomic rename
- optional expected-size and SHA-256 verification before an artifact is marked ready
- MLX speech manifests for Distil-Whisper Large V3 and Qwen3 TTS model layouts

The default cache environment variable is:

```text
VONA_MODEL_CACHE_DIR
```

The crate can download missing artifacts when the host explicitly calls `HttpModelProvisioner::provision_missing`. It does not download automatically from backend constructors. That keeps model licenses, user prompts, credentials, and spend/network policy under application control while still letting Vona own cache layout and artifact placement.

The MLX speech adapters use this boundary deliberately: loader constructors accept explicit model paths or provisioning-derived cache paths, while the application remains responsible for deciding when to download large artifacts.

Recommended next step:

1. Add broader provider-aware helpers for Hugging Face repositories and Ollama-style local model pulls.
2. Wire `SeamlessM4tLocalConfig::from_env` through a provisioning plan so missing ONNX artifacts produce actionable setup errors.
3. Add complete checksums and size metadata for every published speech manifest artifact.
