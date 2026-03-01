//! Local model file discovery — scans known directories for downloaded models.
//! Also discovers external volumes mounted at `/Volumes/`.

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
    /// Where the model lives: "internal", "hf-cache", or the volume name
    #[serde(default = "default_volume")]
    pub volume: String,
    /// Storage tier: "ssd" for internal/hf-cache, "external" for mounted volumes
    #[serde(default = "default_tier")]
    pub storage_tier: String,
}

fn default_volume() -> String { "internal".into() }
fn default_tier() -> String { "ssd".into() }

/// An external volume mounted on this node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredVolume {
    /// Volume name (e.g. "Samsung T7")
    pub name: String,
    /// Mount point (e.g. "/Volumes/Samsung T7")
    pub mount_point: PathBuf,
    /// Total size in bytes
    pub size_bytes: u64,
    /// Available space in bytes
    pub available_bytes: u64,
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

/// Discover external volumes mounted at `/Volumes/`, excluding "Macintosh HD".
/// Uses `statvfs` for accurate size/available info.
pub fn discover_volumes() -> Vec<DiscoveredVolume> {
    let volumes_dir = PathBuf::from("/Volumes");
    let entries = match std::fs::read_dir(&volumes_dir) {
        Ok(e) => e,
        Err(_) => return vec![],
    };

    let mut volumes = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() { continue; }

        let name = entry.file_name().to_string_lossy().to_string();
        if name == "Macintosh HD" { continue; }

        // Get size info via std::fs metadata on the mount point
        // For accurate free space, shell out to df (statvfs not in std)
        let (size_bytes, available_bytes) = df_bytes(&path);

        volumes.push(DiscoveredVolume {
            name,
            mount_point: path,
            size_bytes,
            available_bytes,
        });
    }

    volumes.sort_by(|a, b| a.name.cmp(&b.name));
    volumes
}

/// Get total/available bytes for a path via `df`.
fn df_bytes(path: &PathBuf) -> (u64, u64) {
    // df -k outputs 1K blocks: Filesystem 1K-blocks Used Available ...
    let output = std::process::Command::new("df")
        .args(["-k", &path.to_string_lossy()])
        .output()
        .ok();

    match output {
        Some(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            // Second line has the data
            if let Some(line) = stdout.lines().nth(1) {
                let cols: Vec<&str> = line.split_whitespace().collect();
                if cols.len() >= 4 {
                    let total_kb: u64 = cols[1].parse().unwrap_or(0);
                    let avail_kb: u64 = cols[3].parse().unwrap_or(0);
                    return (total_kb * 1024, avail_kb * 1024);
                }
            }
            (0, 0)
        }
        _ => (0, 0),
    }
}

/// Collect model directories from external volumes.
/// Scans `models/`, `Models/`, and `LLM-Models/` on each volume.
pub fn external_model_dirs() -> Vec<(String, PathBuf)> {
    let volumes = discover_volumes();
    let mut dirs = Vec::new();
    for vol in &volumes {
        for subdir in &["models", "Models", "LLM-Models"] {
            let dir = vol.mount_point.join(subdir);
            if dir.is_dir() {
                dirs.push((vol.name.clone(), dir));
            }
        }
    }
    dirs
}

/// Scan directories for model folders.
/// A "model" is a directory containing at least one `.safetensors` or `.gguf` file, or a `config.json`.
pub fn scan_models(dirs: &[PathBuf]) -> Vec<LocalModel> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let hf_cache = home.join(".cache/huggingface/hub");

    let mut models = Vec::new();

    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        debug!(dir = %dir.display(), "scanning for models");

        // Determine volume/tier from path
        let (volume, tier) = classify_dir(dir, &home, &hf_cache);

        scan_dir_into(dir, &volume, &tier, &mut models);
    }

    // Also scan external volumes
    for (vol_name, dir) in external_model_dirs() {
        debug!(volume = %vol_name, dir = %dir.display(), "scanning external volume");
        scan_dir_into(&dir, &vol_name, "external", &mut models);
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

/// Classify a scan directory by volume name and storage tier.
fn classify_dir(dir: &PathBuf, home: &PathBuf, hf_cache: &PathBuf) -> (String, String) {
    if dir.starts_with(hf_cache) {
        ("hf-cache".into(), "ssd".into())
    } else if dir.starts_with("/Volumes/") {
        // Extract volume name from path: /Volumes/<name>/...
        let vol_name = dir.components()
            .nth(2)
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .unwrap_or_else(|| "external".into());
        (vol_name, "external".into())
    } else if dir.starts_with(home) {
        ("internal".into(), "ssd".into())
    } else {
        ("internal".into(), "ssd".into())
    }
}

/// Scan a single directory for models and append to the list.
fn scan_dir_into(dir: &PathBuf, volume: &str, tier: &str, models: &mut Vec<LocalModel>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        if has_model_files(&path) {
            let dir_name = path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            models.push(LocalModel {
                name: parse_model_name(&dir_name),
                path: path.clone(),
                size_bytes: dir_size(&path),
                volume: volume.into(),
                storage_tier: tier.into(),
            });
        } else {
            // Check one level deeper (e.g. /Volumes/T7/Models/org/model-name/)
            if let Ok(sub_entries) = std::fs::read_dir(&path) {
                for sub_entry in sub_entries.flatten() {
                    let sub_path = sub_entry.path();
                    if sub_path.is_dir() && has_model_files(&sub_path) {
                        let sub_name = sub_path.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();

                        models.push(LocalModel {
                            name: parse_model_name(&sub_name),
                            path: sub_path.clone(),
                            size_bytes: dir_size(&sub_path),
                            volume: volume.into(),
                            storage_tier: tier.into(),
                        });
                    }
                }
            }
        }
    }
}

/// Check if a directory contains model weight files.
fn has_model_files(path: &PathBuf) -> bool {
    std::fs::read_dir(path)
        .map(|entries| {
            entries.flatten().any(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.ends_with(".safetensors")
                    || name.ends_with(".gguf")
                    || name == "config.json"
            })
        })
        .unwrap_or(false)
}

/// Sum file sizes in a directory (non-recursive, top-level files only).
fn dir_size(path: &PathBuf) -> u64 {
    std::fs::read_dir(path)
        .map(|entries| {
            entries.flatten()
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
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
        // May include external volume models too — filter to our test dir
        let test_models: Vec<_> = models.iter()
            .filter(|m| m.path.starts_with(&tmp))
            .collect();
        assert_eq!(test_models.len(), 1);
        assert_eq!(test_models[0].name, "TestOrg/TestModel-4bit");
        assert!(test_models[0].size_bytes > 0);
        assert_eq!(test_models[0].volume, "internal");
        assert_eq!(test_models[0].storage_tier, "ssd");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
