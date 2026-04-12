//! Local model file discovery — scans known directories for downloaded models.
//! Also discovers external volumes mounted at `/Volumes/`.
//! Enriches each model with metadata from config.json (architecture, params, quant, context).

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use tracing::debug;

/// Metadata extracted from a model's config.json.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture (e.g. "qwen3_5_moe", "mistral3", "nemotron_h")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    /// Architecture class names (e.g. ["Qwen3_5MoeForConditionalGeneration"])
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architectures: Option<Vec<String>>,
    /// Hidden dimension
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<u64>,
    /// Number of transformer layers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_hidden_layers: Option<u64>,
    /// Number of attention heads
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_attention_heads: Option<u64>,
    /// Number of KV heads (GQA if < num_attention_heads)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_key_value_heads: Option<u64>,
    /// Vocabulary size
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<u64>,
    /// Maximum context length
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_position_embeddings: Option<u64>,
    /// Total number of MoE experts (None = dense model)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<u64>,
    /// Active experts per token (MoE routing)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_experts_per_tok: Option<u64>,
    /// Quantization bits (4, 8, 16, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_bits: Option<u64>,
    /// Quantization group size
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_group_size: Option<u64>,
    /// Quantization mode (e.g. "affine", "nvfp4")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_mode: Option<String>,
    /// True if this is a vision-language model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_vlm: Option<bool>,
    /// True if this is a Mixture-of-Experts model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_moe: Option<bool>,
}

/// A model found on the local filesystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    /// Display name: "org/model-name" (HuggingFace convention)
    pub name: String,
    /// Absolute path on disk
    pub path: PathBuf,
    /// Size in bytes (sum of all files in the directory)
    pub size_bytes: u64,
    /// Human-readable size (e.g. "45.2 GB")
    #[serde(default)]
    pub size_human: String,
    /// Where the model lives: "internal", "hf-cache", or the volume name
    #[serde(default = "default_volume")]
    pub volume: String,
    /// Storage tier: "ssd" for internal/hf-cache, "external" for mounted volumes
    #[serde(default = "default_tier")]
    pub storage_tier: String,
    /// Metadata from config.json
    #[serde(default)]
    pub config: ModelConfig,
}

fn default_volume() -> String { "internal".into() }
fn default_tier() -> String { "ssd".into() }

/// Format bytes as human-readable string.
pub fn human_size(bytes: u64) -> String {
    const GB: f64 = 1_073_741_824.0;
    const MB: f64 = 1_048_576.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.0} MB", b / MB)
    } else {
        format!("{} B", bytes)
    }
}

/// Parse config.json from a model directory, extracting key metadata.
/// Handles both top-level configs (text models) and VLM configs where
/// fields may be nested under `text_config`.
fn parse_model_config(model_dir: &std::path::Path) -> ModelConfig {
    let config_path = model_dir.join("config.json");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(c) => c,
        Err(_) => return ModelConfig::default(),
    };
    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return ModelConfig::default(),
    };

    // For VLMs, core LLM fields live under text_config
    let text_cfg = json.get("text_config").unwrap_or(&json);

    let get_u64 = |key: &str| -> Option<u64> {
        text_cfg.get(key).and_then(|v| v.as_u64())
            .or_else(|| json.get(key).and_then(|v| v.as_u64()))
    };
    let get_str = |key: &str| -> Option<String> {
        json.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
            .or_else(|| text_cfg.get(key).and_then(|v| v.as_str()).map(|s| s.to_string()))
    };

    let architectures = json.get("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect());

    // Quantization can be under "quantization" or "quantization_config"
    let quant = json.get("quantization")
        .or_else(|| json.get("quantization_config"));
    let (qbits, qgs, qmode) = match quant {
        Some(q) => (
            q.get("bits").and_then(|v| v.as_u64()),
            q.get("group_size").and_then(|v| v.as_u64()),
            q.get("mode").and_then(|v| v.as_str()).map(String::from),
        ),
        None => (None, None, None),
    };

    // MoE detection: num_local_experts (HF standard) or num_experts
    let num_experts = get_u64("num_local_experts")
        .or_else(|| get_u64("num_experts"));
    let num_active = get_u64("num_experts_per_tok")
        .or_else(|| get_u64("num_active_experts"));

    // VLM detection: has vision_config, image_token_id, or VLM architecture
    let is_vlm = json.get("vision_config").is_some()
        || json.get("image_token_id").is_some()
        || json.get("image_token_index").is_some()
        || architectures.as_ref().map_or(false, |a: &Vec<String>|
            a.iter().any(|s| s.contains("Conditional") || s.contains("Vision") || s.contains("VL")));

    ModelConfig {
        model_type: get_str("model_type"),
        architectures,
        hidden_size: get_u64("hidden_size"),
        num_hidden_layers: get_u64("num_hidden_layers"),
        num_attention_heads: get_u64("num_attention_heads"),
        num_key_value_heads: get_u64("num_key_value_heads"),
        vocab_size: get_u64("vocab_size"),
        max_position_embeddings: get_u64("max_position_embeddings"),
        num_experts,
        num_experts_per_tok: num_active,
        quantization_bits: qbits,
        quantization_group_size: qgs,
        quantization_mode: qmode,
        is_vlm: if is_vlm { Some(true) } else { None },
        is_moe: if num_experts.is_some() { Some(true) } else { Some(false) },
    }
}

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
fn df_bytes(path: &std::path::Path) -> (u64, u64) {
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
/// Deduplicates by canonical path (handles macOS case-insensitive APFS where ~/models == ~/Models).
pub fn scan_models(dirs: &[PathBuf]) -> Vec<LocalModel> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let hf_cache = home.join(".cache/huggingface/hub");

    let mut models = Vec::new();
    let mut seen_paths: HashSet<PathBuf> = HashSet::new();

    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        debug!(dir = %dir.display(), "scanning for models");

        // Determine volume/tier from path
        let (volume, tier) = classify_dir(dir, &hf_cache);

        scan_dir_into(dir, &volume, &tier, &mut models, &mut seen_paths);
    }

    // Also scan external volumes
    for (vol_name, dir) in external_model_dirs() {
        debug!(volume = %vol_name, dir = %dir.display(), "scanning external volume");
        scan_dir_into(&dir, &vol_name, "external", &mut models, &mut seen_paths);
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

/// Classify a scan directory by volume name and storage tier.
fn classify_dir(dir: &std::path::Path, hf_cache: &std::path::Path) -> (String, String) {
    if dir.starts_with(hf_cache) {
        ("hf-cache".into(), "ssd".into())
    } else if dir.starts_with("/Volumes/") {
        // Extract volume name from path: /Volumes/<name>/...
        let vol_name = dir.components()
            .nth(2)
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .unwrap_or_else(|| "external".into());
        (vol_name, "external".into())
    } else {
        ("internal".into(), "ssd".into())
    }
}

/// Scan a single directory for models and append to the list.
/// Uses `seen_paths` (canonical paths) to skip duplicates from symlinks or
/// case-insensitive filesystem aliases (~/models vs ~/Models on APFS).
fn scan_dir_into(
    dir: &PathBuf,
    volume: &str,
    tier: &str,
    models: &mut Vec<LocalModel>,
    seen: &mut HashSet<PathBuf>,
) {
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
            // Canonicalize to dedup case-insensitive aliases
            let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
            if !seen.insert(canonical) {
                continue; // Already scanned this physical directory
            }

            let dir_name = path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let size = dir_size(&path);

            models.push(LocalModel {
                name: parse_model_name(&dir_name),
                path: path.clone(),
                size_bytes: size,
                size_human: human_size(size),
                volume: volume.into(),
                storage_tier: tier.into(),
                config: parse_model_config(&path),
            });
        } else {
            // Check one level deeper (e.g. /Volumes/T7/Models/org/model-name/)
            if let Ok(sub_entries) = std::fs::read_dir(&path) {
                for sub_entry in sub_entries.flatten() {
                    let sub_path = sub_entry.path();
                    if sub_path.is_dir() && has_model_files(&sub_path) {
                        let canonical = sub_path.canonicalize().unwrap_or_else(|_| sub_path.clone());
                        if !seen.insert(canonical) {
                            continue;
                        }

                        let sub_name = sub_path.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();
                        let size = dir_size(&sub_path);

                        models.push(LocalModel {
                            name: parse_model_name(&sub_name),
                            path: sub_path.clone(),
                            size_bytes: size,
                            size_human: human_size(size),
                            volume: volume.into(),
                            storage_tier: tier.into(),
                            config: parse_model_config(&sub_path),
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
/// Uses `std::fs::metadata` (not `DirEntry::metadata`) to follow symlinks,
/// so HF cache snapshots (symlinks → blobs) report their true size.
fn dir_size(path: &PathBuf) -> u64 {
    std::fs::read_dir(path)
        .map(|entries| {
            entries.flatten()
                .filter_map(|e| std::fs::metadata(e.path()).ok())
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
    fn test_human_size() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(500_000_000), "477 MB"); // binary: 500M / 1048576
        assert_eq!(human_size(45_000_000_000), "41.9 GB");
        assert_eq!(human_size(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn test_parse_model_config_with_quant() {
        let tmp = std::env::temp_dir().join("asmi-test-config-parse");
        let _ = std::fs::create_dir_all(&tmp);
        let config = r#"{
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "num_local_experts": 333,
            "num_experts_per_tok": 10,
            "quantization": { "bits": 4, "group_size": 64, "mode": "affine" }
        }"#;
        std::fs::write(tmp.join("config.json"), config).unwrap();

        let cfg = parse_model_config(&tmp);
        assert_eq!(cfg.model_type.as_deref(), Some("qwen3_5_moe"));
        assert_eq!(cfg.hidden_size, Some(4096));
        assert_eq!(cfg.num_experts, Some(333));
        assert_eq!(cfg.num_experts_per_tok, Some(10));
        assert_eq!(cfg.quantization_bits, Some(4));
        assert_eq!(cfg.quantization_mode.as_deref(), Some("affine"));
        assert_eq!(cfg.is_moe, Some(true));
        assert!(cfg.is_vlm.is_none()); // Not a VLM

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_parse_model_config_vlm() {
        let tmp = std::env::temp_dir().join("asmi-test-config-vlm");
        let _ = std::fs::create_dir_all(&tmp);
        let config = r#"{
            "model_type": "qwen3_5_moe",
            "image_token_id": 248056,
            "vision_config": { "hidden_size": 1280 },
            "text_config": {
                "hidden_size": 4096,
                "num_hidden_layers": 60,
                "num_attention_heads": 32
            }
        }"#;
        std::fs::write(tmp.join("config.json"), config).unwrap();

        let cfg = parse_model_config(&tmp);
        assert_eq!(cfg.hidden_size, Some(4096)); // From text_config
        assert_eq!(cfg.is_vlm, Some(true));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_scan_empty_dir() {
        let tmp = std::env::temp_dir().join("asmi-test-empty-models");
        let _ = std::fs::create_dir_all(&tmp);
        let models = scan_models(&[tmp.clone()]);
        // Filter to our test dir since external volumes may add models
        let test_models: Vec<_> = models.iter()
            .filter(|m| m.path.starts_with(&tmp))
            .collect();
        assert!(test_models.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_scan_with_model() {
        let tmp = std::env::temp_dir().join("asmi-test-models-enriched");
        let model_dir = tmp.join("TestOrg--TestModel-4bit");
        let _ = std::fs::create_dir_all(&model_dir);
        let config = r#"{"model_type":"llama","hidden_size":4096,"quantization":{"bits":4}}"#;
        std::fs::write(model_dir.join("config.json"), config).unwrap();
        std::fs::write(model_dir.join("model.safetensors"), "fake").unwrap();

        let models = scan_models(&[tmp.clone()]);
        let test_models: Vec<_> = models.iter()
            .filter(|m| m.path.starts_with(&tmp))
            .collect();
        assert_eq!(test_models.len(), 1);
        assert_eq!(test_models[0].name, "TestOrg/TestModel-4bit");
        assert!(test_models[0].size_bytes > 0);
        assert!(!test_models[0].size_human.is_empty());
        assert_eq!(test_models[0].volume, "internal");
        assert_eq!(test_models[0].storage_tier, "ssd");
        // Config enrichment
        assert_eq!(test_models[0].config.model_type.as_deref(), Some("llama"));
        assert_eq!(test_models[0].config.hidden_size, Some(4096));
        assert_eq!(test_models[0].config.quantization_bits, Some(4));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_dedup_identical_paths() {
        // Simulate scanning the same physical dir twice (~/models + ~/Models on APFS)
        let tmp = std::env::temp_dir().join("asmi-test-dedup");
        let model_dir = tmp.join("SomeModel");
        let _ = std::fs::create_dir_all(&model_dir);
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();
        std::fs::write(model_dir.join("model.safetensors"), "x").unwrap();

        // Scan the same directory twice
        let models = scan_models(&[tmp.clone(), tmp.clone()]);
        let test_models: Vec<_> = models.iter()
            .filter(|m| m.path.starts_with(&tmp))
            .collect();
        // Should only appear once despite scanning twice
        assert_eq!(test_models.len(), 1);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_dir_size_follows_symlinks() {
        // Simulate HF cache layout: model dir contains symlinks → blob files
        let tmp = std::env::temp_dir().join("asmi-test-symlink-size");
        let blobs = tmp.join("blobs");
        let snapshot = tmp.join("snapshot");
        let _ = std::fs::create_dir_all(&blobs);
        let _ = std::fs::create_dir_all(&snapshot);

        // Create a "blob" file with known size
        let payload = vec![0u8; 4096];
        std::fs::write(blobs.join("sha256-abc"), &payload).unwrap();
        std::fs::write(blobs.join("sha256-def"), &payload).unwrap();

        // Symlink from snapshot → blobs (like HF cache does)
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(blobs.join("sha256-abc"), snapshot.join("model.safetensors")).unwrap();
            std::os::unix::fs::symlink(blobs.join("sha256-def"), snapshot.join("config.json")).unwrap();
        }

        let size = dir_size(&snapshot);
        assert_eq!(size, 8192, "dir_size must follow symlinks to get real file sizes");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
