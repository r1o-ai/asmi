//! Local model file discovery — scans known directories for downloaded models.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::debug;

/// A model found on the local filesystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    /// Display name: "org/model-name" (HuggingFace convention)
    pub name: String,
    /// Absolute path on disk
    pub path: PathBuf,
    /// Size in bytes (sum of all files in the directory)
    pub size_bytes: u64,
}

/// Default directories to scan for models (macOS).
/// Creates ~/Models if it doesn't exist (canonical model storage).
pub fn default_model_dirs() -> Vec<PathBuf> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let models_dir = home.join("Models");
    // Ensure ~/Models exists on every node
    if !models_dir.exists() {
        let _ = std::fs::create_dir_all(&models_dir);
    }
    vec![
        models_dir,
        home.join(".cache/huggingface/hub"),
    ]
}

/// Parse a model directory name into a display name.
/// HuggingFace convention: "Qwen--Qwen3-32B-4bit" → "Qwen/Qwen3-32B-4bit"
pub fn parse_model_name(dir_name: &str) -> String {
    dir_name.replacen("--", "/", 1)
}

/// Scan directories for model folders.
/// A "model" is a directory containing at least one `.safetensors` or `.gguf` file, or a `config.json`.
pub fn scan_models(dirs: &[PathBuf]) -> Vec<LocalModel> {
    let mut models = Vec::new();

    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        debug!(dir = %dir.display(), "scanning for models");

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            // Check if this directory contains model files
            let has_model_files = std::fs::read_dir(&path)
                .map(|entries| {
                    entries.flatten().any(|e| {
                        let name = e.file_name();
                        let name = name.to_string_lossy();
                        name.ends_with(".safetensors")
                            || name.ends_with(".gguf")
                            || name == "config.json"
                    })
                })
                .unwrap_or(false);

            if !has_model_files {
                continue;
            }

            let dir_name = path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            // Calculate total size (non-recursive, just top-level files)
            let size_bytes: u64 = std::fs::read_dir(&path)
                .map(|entries| {
                    entries.flatten()
                        .filter_map(|e| e.metadata().ok())
                        .filter(|m| m.is_file())
                        .map(|m| m.len())
                        .sum()
                })
                .unwrap_or(0);

            models.push(LocalModel {
                name: parse_model_name(&dir_name),
                path,
                size_bytes,
            });
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_dir_name() {
        assert_eq!(parse_model_name("Qwen--Qwen3-32B-4bit"), "Qwen/Qwen3-32B-4bit");
    }

    #[test]
    fn test_parse_model_dir_no_separator() {
        assert_eq!(parse_model_name("my-local-model"), "my-local-model");
    }

    #[test]
    fn test_scan_empty_dir() {
        let tmp = std::env::temp_dir().join("asmi-test-empty-models");
        let _ = std::fs::create_dir_all(&tmp);
        let models = scan_models(&[tmp.clone()]);
        assert!(models.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_scan_with_model() {
        let tmp = std::env::temp_dir().join("asmi-test-models");
        let model_dir = tmp.join("TestOrg--TestModel-4bit");
        let _ = std::fs::create_dir_all(&model_dir);
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();
        std::fs::write(model_dir.join("model.safetensors"), "fake").unwrap();

        let models = scan_models(&[tmp.clone()]);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "TestOrg/TestModel-4bit");
        assert!(models[0].size_bytes > 0);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
