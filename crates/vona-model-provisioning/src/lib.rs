use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

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
        for planned in &plan.missing {
            let url = planned.artifact.source_url.as_ref().ok_or_else(|| {
                ProvisioningError::MissingSourceUrl(planned.artifact.name.clone())
            })?;
            let bytes = self
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
                .bytes()
                .await
                .map_err(|err| ProvisioningError::Download {
                    url: url.clone(),
                    message: err.to_string(),
                })?;

            if let Some(expected) = planned.artifact.expected_size_bytes {
                let actual = bytes.len() as u64;
                if actual != expected {
                    return Err(ProvisioningError::SizeMismatch {
                        name: planned.artifact.name.clone(),
                        expected,
                        actual,
                    });
                }
            }

            if let Some(parent) = planned.path.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|err| ProvisioningError::Io(err.to_string()))?;
            }
            tokio::fs::write(&planned.path, bytes)
                .await
                .map_err(|err| ProvisioningError::Io(err.to_string()))?;
        }
        Ok(cache.inspect(manifest))
    }
}

pub fn validate_manifest(manifest: &ModelManifest) -> Result<(), ProvisioningError> {
    if manifest.artifacts.is_empty() {
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

pub fn moshi_server_manifest(model: impl Into<String>) -> ModelManifest {
    let model = model.into();
    ModelManifest {
        id: format!("moshi/{model}"),
        provider: LocalModelProvider::HuggingFace {
            repo: model,
            revision: None,
        },
        artifacts: Vec::new(),
    }
}

fn sanitize_model_id(id: &str) -> String {
    id.chars()
        .map(|ch| match ch {
            '/' | ':' | '\\' => '_',
            ch => ch,
        })
        .collect()
}

#[allow(dead_code)]
fn _assert_path_is_relative(path: &Path) -> bool {
    !path.is_absolute()
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
}
