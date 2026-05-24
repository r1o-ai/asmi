use std::fs;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;

use crate::config::QwenConfig;

/// Paths and data for a downloaded Qwen model.
pub struct ModelFiles {
    pub config: QwenConfig,
    pub tokenizer_path: PathBuf,
    /// One or more safetensor shard paths (Qwen may split across shards).
    pub safetensors_paths: Vec<PathBuf>,
}

/// Discover safetensor shard files in a model directory.
///
/// Handles both single-file (`model.safetensors`) and sharded layouts
/// (`model-00001-of-00002.safetensors`, etc.).
fn find_safetensors_in_dir(dir: &Path) -> Vec<PathBuf> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return vec![single];
    }

    // Look for sharded files matching model-NNNNN-of-NNNNN.safetensors
    let mut shards: Vec<PathBuf> = fs::read_dir(dir)
        .expect("failed to read model directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "safetensors")
                && path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .starts_with("model-")
        })
        .collect();

    shards.sort();
    assert!(
        !shards.is_empty(),
        "no safetensors files found in {}",
        dir.display()
    );
    shards
}

/// Load model from a local directory (e.g. ~/models/models--Qwen--Qwen3.5-0.8B).
///
/// The directory should contain `config.json`, `tokenizer.json`, and one or more
/// `.safetensors` files. For HuggingFace cache layout, the actual files live
/// under `snapshots/<hash>/`.
pub fn load_model_local(model_dir: &Path) -> Result<ModelFiles, Box<dyn std::error::Error>> {
    // HuggingFace cache layout: models--org--name/snapshots/<hash>/
    // Also support direct directory with files.
    let resolved_dir = if model_dir.join("config.json").exists() {
        model_dir.to_path_buf()
    } else {
        // Try HF cache layout
        let snapshots_dir = model_dir.join("snapshots");
        if snapshots_dir.exists() {
            // Find the latest snapshot (most recently modified)
            let mut snapshots: Vec<_> = fs::read_dir(&snapshots_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .collect();
            snapshots.sort_by_key(|e| {
                e.metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
            });
            snapshots
                .last()
                .map(|e| e.path())
                .unwrap_or_else(|| panic!("no snapshots found in {}", snapshots_dir.display()))
        } else {
            panic!(
                "cannot find config.json or snapshots/ in {}",
                model_dir.display()
            );
        }
    };

    let config_path = resolved_dir.join("config.json");
    let config: QwenConfig = serde_json::from_reader(fs::File::open(&config_path)?)?;

    let tokenizer_path = resolved_dir.join("tokenizer.json");
    assert!(
        tokenizer_path.exists(),
        "tokenizer.json not found in {}",
        resolved_dir.display()
    );

    let safetensors_paths = find_safetensors_in_dir(&resolved_dir);

    eprintln!(
        "\x1b[1;32m✓\x1b[0m Loaded model from {} ({} shard{})",
        resolved_dir.display(),
        safetensors_paths.len(),
        if safetensors_paths.len() == 1 { "" } else { "s" },
    );

    Ok(ModelFiles {
        config,
        tokenizer_path,
        safetensors_paths,
    })
}

/// Download model files from HuggingFace Hub.
///
/// Uses hf-hub to download `config.json`, `tokenizer.json`, and all safetensor
/// shard files. Files are cached in the default HF cache directory.
pub fn download_model(repo_id: &str) -> Result<ModelFiles, Box<dyn std::error::Error>> {
    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(repo_id.to_string());

    eprintln!("\x1b[1;36m⠋\x1b[0m Downloading config.json");
    let config_path = repo.get("config.json")?;
    let config: QwenConfig = serde_json::from_reader(fs::File::open(&config_path)?)?;

    eprintln!("\x1b[1;36m⠙\x1b[0m Downloading tokenizer.json");
    let tokenizer_path = repo.get("tokenizer.json")?;

    // Try single safetensors file first, then try sharded
    eprintln!("\x1b[1;36m⠹\x1b[0m Downloading safetensors");
    let safetensors_paths = if let Ok(path) = repo.get("model.safetensors") {
        vec![path]
    } else {
        // Sharded model: download the index to find shard filenames
        let index_path = repo.get("model.safetensors.index.json")?;
        let index: serde_json::Value = serde_json::from_reader(fs::File::open(&index_path)?)?;

        // Extract unique shard filenames from the weight_map
        let mut shard_names: Vec<String> = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .expect("model.safetensors.index.json missing weight_map")
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_names.sort();
        shard_names.dedup();

        let mut paths = Vec::new();
        for shard_name in &shard_names {
            eprintln!("\x1b[1;36m⠸\x1b[0m Downloading {shard_name}");
            paths.push(repo.get(shard_name)?);
        }
        paths
    };

    eprintln!(
        "\x1b[1;32m✓\x1b[0m Downloaded model ({} shard{})",
        safetensors_paths.len(),
        if safetensors_paths.len() == 1 { "" } else { "s" },
    );

    Ok(ModelFiles {
        config,
        tokenizer_path,
        safetensors_paths,
    })
}
