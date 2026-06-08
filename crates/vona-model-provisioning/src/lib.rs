use futures_util::StreamExt;
use ring::digest::{Context, SHA256};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub const DEFAULT_CACHE_ENV: &str = "VONA_MODEL_CACHE_DIR";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalModelProvider {
    HuggingFace {
        repo: String,
        revision: Option<String>,
    },
    Ollama {
        model: String,
    },
    LocalFile,
    Custom {
        name: String,
    },
    ProviderManaged {
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub name: String,
    pub relative_path: PathBuf,
    pub source_url: Option<String>,
    pub expected_size_bytes: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub id: String,
    pub provider: LocalModelProvider,
    pub artifacts: Vec<ModelArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCache {
    pub root: PathBuf,
}

impl ModelCache {
    pub fn from_env_or(root: impl Into<PathBuf>) -> Self {
        Self {
            root: std::env::var(DEFAULT_CACHE_ENV)
                .map(PathBuf::from)
                .unwrap_or_else(|_| root.into()),
        }
    }

    pub fn model_dir(&self, manifest: &ModelManifest) -> PathBuf {
        self.root.join(sanitize_model_id(&manifest.id))
    }

    pub fn artifact_path(&self, manifest: &ModelManifest, artifact: &ModelArtifact) -> PathBuf {
        self.model_dir(manifest).join(&artifact.relative_path)
    }

    pub fn inspect(&self, manifest: &ModelManifest) -> ProvisionPlan {
        let mut present = Vec::new();
        let mut missing = Vec::new();
        for artifact in &manifest.artifacts {
            let path = self.artifact_path(manifest, artifact);
            if path.is_file() {
                present.push(PlannedArtifact {
                    artifact: artifact.clone(),
                    path,
                });
            } else {
                missing.push(PlannedArtifact {
                    artifact: artifact.clone(),
                    path,
                });
            }
        }
        ProvisionPlan {
            manifest: manifest.clone(),
            model_dir: self.model_dir(manifest),
            present,
            missing,
        }
    }

    pub fn ensure_dirs(&self, manifest: &ModelManifest) -> Result<(), ProvisioningError> {
        std::fs::create_dir_all(self.model_dir(manifest))
            .map_err(|err| ProvisioningError::Io(err.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvisionPlan {
    pub manifest: ModelManifest,
    pub model_dir: PathBuf,
    pub present: Vec<PlannedArtifact>,
    pub missing: Vec<PlannedArtifact>,
}

impl ProvisionPlan {
    pub fn is_ready(&self) -> bool {
        self.missing.is_empty()
    }

    pub fn missing_urls(&self) -> Vec<&str> {
        self.missing
            .iter()
            .filter_map(|artifact| artifact.artifact.source_url.as_deref())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedArtifact {
    pub artifact: ModelArtifact,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ProvisioningError {
    #[error("model manifest has no artifacts: {0}")]
    EmptyManifest(String),
    #[error("artifact path must be relative: {0}")]
    AbsoluteArtifactPath(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("artifact has no source URL: {0}")]
    MissingSourceUrl(String),
    #[error("download failed for {url}: {message}")]
    Download { url: String, message: String },
    #[error("artifact size mismatch for {name}: expected {expected} bytes, got {actual} bytes")]
    SizeMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },
    #[error("artifact checksum mismatch for {name}: expected sha256 {expected}, got {actual}")]
    ChecksumMismatch {
        name: String,
        expected: String,
        actual: String,
    },
}

#[derive(Debug, Clone)]
pub struct HttpModelProvisioner {
    client: reqwest::Client,
}

impl Default for HttpModelProvisioner {
    fn default() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl HttpModelProvisioner {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }

    pub async fn provision_missing(
        &self,
        cache: &ModelCache,
        manifest: &ModelManifest,
    ) -> Result<ProvisionPlan, ProvisioningError> {
        validate_manifest(manifest)?;
        cache.ensure_dirs(manifest)?;
        let plan = cache.inspect(manifest);
        let mut to_download = plan.missing;
        for planned in plan.present {
            if let Err(err) = verify_artifact_file(&planned).await {
                let _ = tokio::fs::remove_file(&planned.path).await;
                if matches!(
                    err,
                    ProvisioningError::SizeMismatch { .. }
                        | ProvisioningError::ChecksumMismatch { .. }
                ) {
                    to_download.push(planned);
                } else {
                    return Err(err);
                }
            }
        }

        for planned in &to_download {
            self.download_artifact(planned).await?;
        }
        Ok(cache.inspect(manifest))
    }

    async fn download_artifact(&self, planned: &PlannedArtifact) -> Result<(), ProvisioningError> {
        let url =
            planned.artifact.source_url.as_ref().ok_or_else(|| {
                ProvisioningError::MissingSourceUrl(planned.artifact.name.clone())
            })?;
        if let Some(parent) = planned.path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|err| ProvisioningError::Io(err.to_string()))?;
        }

        let temp_path = planned
            .path
            .with_extension(format!("{}.tmp", std::process::id()));
        let mut file = tokio::fs::File::create(&temp_path)
            .await
            .map_err(|err| ProvisioningError::Io(err.to_string()))?;
        let mut response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|err| ProvisioningError::Download {
                url: url.clone(),
                message: err.to_string(),
            })?
            .error_for_status()
            .map_err(|err| ProvisioningError::Download {
                url: url.clone(),
                message: err.to_string(),
            })?
            .bytes_stream();

        let mut hasher = Context::new(&SHA256);
        let mut size = 0_u64;
        while let Some(chunk) = response.next().await {
            let chunk = chunk.map_err(|err| ProvisioningError::Download {
                url: url.clone(),
                message: err.to_string(),
            })?;
            size += chunk.len() as u64;
            hasher.update(&chunk);
            file.write_all(&chunk)
                .await
                .map_err(|err| ProvisioningError::Io(err.to_string()))?;
        }
        file.flush()
            .await
            .map_err(|err| ProvisioningError::Io(err.to_string()))?;
        drop(file);

        verify_size(&planned.artifact, size)?;
        verify_sha256(&planned.artifact, encode_hex(hasher.finish().as_ref()))?;

        tokio::fs::rename(&temp_path, &planned.path)
            .await
            .map_err(|err| ProvisioningError::Io(err.to_string()))?;
        Ok(())
    }
}

pub fn validate_manifest(manifest: &ModelManifest) -> Result<(), ProvisioningError> {
    if manifest.artifacts.is_empty()
        && !matches!(
            manifest.provider,
            LocalModelProvider::Ollama { .. } | LocalModelProvider::ProviderManaged { .. }
        )
    {
        return Err(ProvisioningError::EmptyManifest(manifest.id.clone()));
    }
    for artifact in &manifest.artifacts {
        if artifact.relative_path.is_absolute() {
            return Err(ProvisioningError::AbsoluteArtifactPath(
                artifact.relative_path.display().to_string(),
            ));
        }
    }
    Ok(())
}

pub fn seamless_m4t_onnx_manifest(
    model_id: impl Into<String>,
    onnx_url: impl Into<String>,
) -> ModelManifest {
    ModelManifest {
        id: model_id.into(),
        provider: LocalModelProvider::HuggingFace {
            repo: "facebook/hf-seamless-m4t-medium".to_string(),
            revision: None,
        },
        artifacts: vec![ModelArtifact {
            name: "encoder-decoder-onnx".to_string(),
            relative_path: PathBuf::from("model.onnx"),
            source_url: Some(onnx_url.into()),
            expected_size_bytes: None,
            sha256: None,
        }],
    }
}

pub fn distil_whisper_large_v3_manifest() -> ModelManifest {
    hf_snapshot_manifest(
        "distil-whisper/distil-large-v3",
        &[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "normalizer.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    )
}

pub fn qwen3_tts_12hz_0_6b_base_bf16_manifest() -> ModelManifest {
    hf_snapshot_manifest(
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        &[
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/configuration.json",
            "speech_tokenizer/model.safetensors",
            "speech_tokenizer/preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    )
}

pub fn kokoro_82m_onnx_realtime_manifest() -> ModelManifest {
    ModelManifest {
        id: "onnx-community/Kokoro-82M-ONNX".to_string(),
        provider: LocalModelProvider::ProviderManaged {
            name: "kokoro-82m-onnx".to_string(),
        },
        artifacts: Vec::new(),
    }
}

pub fn piper_low_power_tts_manifest() -> ModelManifest {
    ModelManifest {
        id: "piper/low-power-tts".to_string(),
        provider: LocalModelProvider::ProviderManaged {
            name: "piper".to_string(),
        },
        artifacts: Vec::new(),
    }
}

pub fn realtime_tts_model_manifests() -> [ModelManifest; 3] {
    [
        kokoro_82m_onnx_realtime_manifest(),
        piper_low_power_tts_manifest(),
        qwen3_tts_12hz_0_6b_base_bf16_manifest(),
    ]
}

pub fn mlx_speech_model_manifests() -> [ModelManifest; 2] {
    [
        distil_whisper_large_v3_manifest(),
        qwen3_tts_12hz_0_6b_base_bf16_manifest(),
    ]
}

fn hf_snapshot_manifest(repo: &str, files: &[&str]) -> ModelManifest {
    ModelManifest {
        id: repo.to_string(),
        provider: LocalModelProvider::HuggingFace {
            repo: repo.to_string(),
            revision: None,
        },
        artifacts: files
            .iter()
            .map(|file| ModelArtifact {
                name: (*file).to_string(),
                relative_path: PathBuf::from(file),
                source_url: Some(format!("https://huggingface.co/{repo}/resolve/main/{file}")),
                expected_size_bytes: None,
                sha256: None,
            })
            .collect(),
    }
}

pub fn moshi_server_manifest(model: impl Into<String>) -> ModelManifest {
    let model = model.into();
    ModelManifest {
        id: format!("moshi/{model}"),
        provider: LocalModelProvider::ProviderManaged {
            name: format!("moshi/{model}"),
        },
        artifacts: Vec::new(),
    }
}

async fn verify_artifact_file(planned: &PlannedArtifact) -> Result<(), ProvisioningError> {
    let metadata = tokio::fs::metadata(&planned.path)
        .await
        .map_err(|err| ProvisioningError::Io(err.to_string()))?;
    verify_size(&planned.artifact, metadata.len())?;

    if planned.artifact.sha256.is_some() {
        let mut file = tokio::fs::File::open(&planned.path)
            .await
            .map_err(|err| ProvisioningError::Io(err.to_string()))?;
        let mut hasher = Context::new(&SHA256);
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .await
                .map_err(|err| ProvisioningError::Io(err.to_string()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        verify_sha256(&planned.artifact, encode_hex(hasher.finish().as_ref()))?;
    }
    Ok(())
}

fn verify_size(artifact: &ModelArtifact, actual: u64) -> Result<(), ProvisioningError> {
    if let Some(expected) = artifact.expected_size_bytes
        && actual != expected
    {
        return Err(ProvisioningError::SizeMismatch {
            name: artifact.name.clone(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn verify_sha256(artifact: &ModelArtifact, actual: String) -> Result<(), ProvisioningError> {
    if let Some(expected) = &artifact.sha256
        && !expected.eq_ignore_ascii_case(&actual)
    {
        return Err(ProvisioningError::ChecksumMismatch {
            name: artifact.name.clone(),
            expected: expected.clone(),
            actual,
        });
    }
    Ok(())
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn sanitize_model_id(id: &str) -> String {
    id.chars()
        .map(|ch| match ch {
            '/' | ':' | '\\' => '_',
            ch => ch,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_empty_manifest() {
        let manifest = ModelManifest {
            id: "empty".to_string(),
            provider: LocalModelProvider::LocalFile,
            artifacts: Vec::new(),
        };
        assert_eq!(
            validate_manifest(&manifest),
            Err(ProvisioningError::EmptyManifest("empty".to_string()))
        );
    }

    #[test]
    fn validate_rejects_absolute_artifact_paths() {
        let manifest = ModelManifest {
            id: "bad".to_string(),
            provider: LocalModelProvider::LocalFile,
            artifacts: vec![ModelArtifact {
                name: "bad".to_string(),
                relative_path: PathBuf::from("/tmp/model.onnx"),
                source_url: None,
                expected_size_bytes: None,
                sha256: None,
            }],
        };
        assert!(matches!(
            validate_manifest(&manifest),
            Err(ProvisioningError::AbsoluteArtifactPath(_))
        ));
    }

    #[test]
    fn inspect_splits_present_and_missing_artifacts() {
        let root =
            std::env::temp_dir().join(format!("vona-provisioning-test-{}", std::process::id()));
        let cache = ModelCache { root };
        let manifest = seamless_m4t_onnx_manifest(
            "facebook/hf-seamless-m4t-medium",
            "https://example.test/model.onnx",
        );
        cache.ensure_dirs(&manifest).unwrap();
        std::fs::write(cache.model_dir(&manifest).join("model.onnx"), b"onnx").unwrap();
        let plan = cache.inspect(&manifest);
        assert!(plan.is_ready());
        assert_eq!(plan.present.len(), 1);
        let _ = std::fs::remove_dir_all(cache.root);
    }

    #[test]
    fn moshi_manifest_is_provider_managed_and_valid_without_artifacts() {
        let manifest = moshi_server_manifest("kyutai/moshi");
        assert!(matches!(
            manifest.provider,
            LocalModelProvider::ProviderManaged { .. }
        ));
        assert!(validate_manifest(&manifest).is_ok());
    }

    #[test]
    fn realtime_tts_manifests_are_provider_managed_or_downloadable() {
        let kokoro = kokoro_82m_onnx_realtime_manifest();
        let piper = piper_low_power_tts_manifest();
        assert!(matches!(
            kokoro.provider,
            LocalModelProvider::ProviderManaged { .. }
        ));
        assert!(matches!(
            piper.provider,
            LocalModelProvider::ProviderManaged { .. }
        ));

        let manifests = realtime_tts_model_manifests();
        assert_eq!(manifests.len(), 3);
        for manifest in manifests {
            validate_manifest(&manifest).unwrap();
        }
    }

    #[test]
    fn mlx_speech_manifests_are_valid_hugging_face_snapshots() {
        let manifests = mlx_speech_model_manifests();
        assert_eq!(manifests.len(), 2);
        for manifest in manifests {
            validate_manifest(&manifest).unwrap();
            assert!(matches!(
                manifest.provider,
                LocalModelProvider::HuggingFace { .. }
            ));
            assert!(
                manifest
                    .artifacts
                    .iter()
                    .any(|artifact| artifact.relative_path == std::path::Path::new("config.json"))
            );
            assert!(manifest.artifacts.iter().any(
                |artifact| artifact.relative_path == std::path::Path::new("model.safetensors")
            ));
            assert!(
                manifest
                    .artifacts
                    .iter()
                    .all(|artifact| artifact.source_url.is_some())
            );
        }
    }

    #[test]
    fn sha256_verification_detects_mismatch() {
        let artifact = ModelArtifact {
            name: "model".to_string(),
            relative_path: PathBuf::from("model.bin"),
            source_url: None,
            expected_size_bytes: Some(4),
            sha256: Some("0000".to_string()),
        };
        assert!(matches!(
            verify_sha256(&artifact, "abcd".to_string()),
            Err(ProvisioningError::ChecksumMismatch { .. })
        ));
        assert!(verify_size(&artifact, 4).is_ok());
    }
}
