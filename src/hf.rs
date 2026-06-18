//! HuggingFace search proxy.
//!
//! Mirrors the shape produced by `web/src/app/api/huggingface/search/route.ts`
//! so the macOS Swift client can decode the same JSON the web client decodes.
//!
//! Endpoint: `GET /hf/search`
//!
//! Query params:
//! * `q`            search text (falls back to `filter` value if absent)
//! * `filter`       HF filter (default `"mlx"`)
//! * `pipeline_tag` optional pipeline tag (e.g. `text-generation`)
//! * `sort`         `downloads` (default), `trendingScore`, `likes`, `lastModified`
//! * `limit`        default 15
//!
//! Response: `{ "models": [HfModel, ...], "error": Optional<String> }`
//!
//! Includes a 60s in-memory response cache keyed by the full canonicalised
//! query string.

use axum::{extract::Query, response::Json};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

const HF_API: &str = "https://huggingface.co/api/models";
const CACHE_TTL: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// Public response shape (matches the web client's decoder exactly)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfModel {
    pub id: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub tags: Vec<String>,
    pub pipeline_tag: Option<String>,
    pub library: Option<String>,
    pub last_modified: String,
    pub trending_score: Option<f64>,
    /// HF returns either a bool (`false`) or a string ("auto" / "manual").
    /// Pass through as raw JSON so the client can decode either form.
    pub gated: serde_json::Value,
    pub architecture: Option<String>,
    pub model_type: Option<String>,
    pub quant_bits: Option<u32>,
    pub base_model: Option<String>,
    pub license: Option<String>,
    pub languages: Vec<String>,
    pub file_count: Option<u32>,
    pub total_size_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct HfSearchResponse {
    pub models: Vec<HfModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct HfSearchQuery {
    pub q: Option<String>,
    pub filter: Option<String>,
    pub pipeline_tag: Option<String>,
    pub sort: Option<String>,
    pub limit: Option<u32>,
}

// ---------------------------------------------------------------------------
// HF upstream raw shape (only the fields we use)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct HfRawModel {
    id: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    downloads: Option<u64>,
    #[serde(default)]
    likes: Option<u64>,
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    library_name: Option<String>,
    #[serde(default, rename = "lastModified")]
    last_modified: Option<String>,
    #[serde(default, rename = "trendingScore")]
    trending_score: Option<f64>,
    #[serde(default)]
    gated: Option<serde_json::Value>,
    #[serde(default)]
    config: Option<HfRawConfig>,
    #[serde(default)]
    siblings: Option<Vec<HfRawSibling>>,
}

#[derive(Debug, Deserialize)]
struct HfRawConfig {
    #[serde(default)]
    architectures: Option<Vec<String>>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    quantization_config: Option<HfRawQuantConfig>,
}

#[derive(Debug, Deserialize)]
struct HfRawQuantConfig {
    #[serde(default)]
    bits: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct HfRawSibling {
    #[serde(default)]
    #[allow(dead_code)]
    rfilename: Option<String>,
    #[serde(default)]
    size: Option<u64>,
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

struct CacheEntry {
    body: HfSearchResponse,
    inserted_at: Instant,
}

fn cache() -> &'static RwLock<HashMap<String, CacheEntry>> {
    static CACHE: OnceLock<RwLock<HashMap<String, CacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent(concat!("asmi/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("failed to build reqwest client")
    })
}

// ---------------------------------------------------------------------------
// Token loading
// ---------------------------------------------------------------------------

/// Read a HuggingFace token from `~/.cache/huggingface/token` if it exists.
/// Trims whitespace. Returns `None` on any error or empty string.
async fn read_hf_token() -> Option<String> {
    let home = dirs::home_dir()?;
    let path = home.join(".cache").join("huggingface").join("token");
    let raw = tokio::fs::read_to_string(&path).await.ok()?;
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

// ---------------------------------------------------------------------------
// Tag parsers (mirror web/src/app/api/huggingface/search/route.ts)
// ---------------------------------------------------------------------------

fn parse_base_model(tags: &[String]) -> Option<String> {
    tags.iter()
        .find_map(|t| t.strip_prefix("base_model:").map(|s| s.to_string()))
}

fn parse_license(tags: &[String]) -> Option<String> {
    tags.iter()
        .find_map(|t| t.strip_prefix("license:").map(|s| s.to_string()))
}

fn parse_languages(tags: &[String]) -> Vec<String> {
    tags.iter()
        .filter_map(|t| t.strip_prefix("language:").map(|s| s.to_string()))
        .collect()
}

// ---------------------------------------------------------------------------
// Mapping
// ---------------------------------------------------------------------------

fn map_raw(raw: HfRawModel) -> HfModel {
    let tags = raw.tags.unwrap_or_default();
    let id = raw.id.clone();
    let author = raw.author.unwrap_or_else(|| {
        id.split_once('/').map(|(a, _)| a.to_string()).unwrap_or_default()
    });
    let quant_bits = raw
        .config
        .as_ref()
        .and_then(|c| c.quantization_config.as_ref())
        .and_then(|q| q.bits);
    let architecture = raw
        .config
        .as_ref()
        .and_then(|c| c.architectures.as_ref())
        .and_then(|arr| arr.first().cloned());
    let model_type = raw.config.as_ref().and_then(|c| c.model_type.clone());
    let (file_count, total_size_bytes) = match raw.siblings.as_ref() {
        Some(arr) => {
            let count = arr.len() as u32;
            let total: u64 = arr.iter().filter_map(|s| s.size).sum();
            (Some(count), Some(total))
        }
        None => (None, None),
    };

    HfModel {
        id,
        author,
        downloads: raw.downloads.unwrap_or(0),
        likes: raw.likes.unwrap_or(0),
        pipeline_tag: raw.pipeline_tag,
        library: raw.library_name,
        last_modified: raw.last_modified.unwrap_or_default(),
        trending_score: raw.trending_score,
        gated: raw.gated.unwrap_or(serde_json::Value::Bool(false)),
        architecture,
        model_type,
        quant_bits,
        base_model: parse_base_model(&tags),
        license: parse_license(&tags),
        languages: parse_languages(&tags),
        file_count,
        total_size_bytes,
        tags,
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// GET /hf/search
pub async fn search_handler(Query(params): Query<HfSearchQuery>) -> Json<HfSearchResponse> {
    let filter = params.filter.unwrap_or_else(|| "mlx".to_string());
    let q = params.q.clone().unwrap_or_default();
    let search = if q.is_empty() { filter.clone() } else { q };
    let sort = params.sort.unwrap_or_else(|| "downloads".to_string());
    let limit = params.limit.unwrap_or(15);
    let pipeline_tag = params.pipeline_tag.unwrap_or_default();

    // Canonical cache key
    let cache_key = format!(
        "search={search}&filter={filter}&sort={sort}&limit={limit}&pipeline={pipeline_tag}"
    );

    {
        let guard = cache().read().await;
        if let Some(entry) = guard.get(&cache_key) {
            if entry.inserted_at.elapsed() < CACHE_TTL {
                return Json(HfSearchResponse {
                    models: entry.body.models.clone(),
                    error: entry.body.error.clone(),
                });
            }
        }
    }

    let mut url = reqwest::Url::parse(HF_API).expect("HF_API parse");
    {
        let mut qp = url.query_pairs_mut();
        qp.append_pair("search", &search);
        qp.append_pair("filter", &filter);
        qp.append_pair("sort", &sort);
        qp.append_pair("direction", "-1");
        qp.append_pair("limit", &limit.to_string());
        qp.append_pair("full", "true");
        qp.append_pair("config", "true");
        if !pipeline_tag.is_empty() {
            qp.append_pair("pipeline_tag", &pipeline_tag);
        }
    }

    let mut req = http_client().get(url).header("Accept", "application/json");
    if let Some(token) = read_hf_token().await {
        req = req.bearer_auth(token);
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            let body = HfSearchResponse {
                models: vec![],
                error: Some(format!("request failed: {e}")),
            };
            return Json(body);
        }
    };

    let status = resp.status();
    if !status.is_success() {
        let body = HfSearchResponse {
            models: vec![],
            error: Some(format!("HuggingFace API returned {status}")),
        };
        return Json(body);
    }

    let raw_models: Vec<HfRawModel> = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            let body = HfSearchResponse {
                models: vec![],
                error: Some(format!("decode failed: {e}")),
            };
            return Json(body);
        }
    };

    let models: Vec<HfModel> = raw_models.into_iter().map(map_raw).collect();
    let body = HfSearchResponse { models, error: None };

    // Insert into cache (clone is shallow — Vec/Strings are reference-counted on heap copies)
    {
        let mut guard = cache().write().await;
        guard.insert(
            cache_key,
            CacheEntry {
                body: HfSearchResponse {
                    models: body.models.clone(),
                    error: body.error.clone(),
                },
                inserted_at: Instant::now(),
            },
        );

        // Opportunistic GC: drop expired entries to keep map bounded.
        guard.retain(|_, v| v.inserted_at.elapsed() < CACHE_TTL * 2);
    }

    Json(body)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_base_model_extracts_value() {
        let tags = vec![
            "text-generation".to_string(),
            "base_model:Qwen/Qwen3-8B".to_string(),
        ];
        assert_eq!(parse_base_model(&tags).as_deref(), Some("Qwen/Qwen3-8B"));
    }

    #[test]
    fn parse_license_extracts_value() {
        let tags = vec!["license:apache-2.0".to_string()];
        assert_eq!(parse_license(&tags).as_deref(), Some("apache-2.0"));
    }

    #[test]
    fn parse_languages_returns_all() {
        let tags = vec![
            "language:en".to_string(),
            "language:zh".to_string(),
            "license:mit".to_string(),
        ];
        let langs = parse_languages(&tags);
        assert_eq!(langs, vec!["en".to_string(), "zh".to_string()]);
    }

    #[test]
    fn map_raw_populates_quant_bits_and_arch() {
        let raw = HfRawModel {
            id: "mlx-community/Qwen3-8B-4bit".to_string(),
            author: None,
            downloads: Some(123),
            likes: Some(7),
            tags: Some(vec![
                "license:apache-2.0".to_string(),
                "language:en".to_string(),
                "base_model:Qwen/Qwen3-8B".to_string(),
            ]),
            pipeline_tag: Some("text-generation".to_string()),
            library_name: Some("mlx".to_string()),
            last_modified: Some("2026-01-01T00:00:00Z".to_string()),
            trending_score: Some(42.0),
            gated: None,
            config: Some(HfRawConfig {
                architectures: Some(vec!["Qwen3ForCausalLM".to_string()]),
                model_type: Some("qwen3".to_string()),
                quantization_config: Some(HfRawQuantConfig { bits: Some(4) }),
            }),
            siblings: Some(vec![
                HfRawSibling { rfilename: Some("a".to_string()), size: Some(100) },
                HfRawSibling { rfilename: Some("b".to_string()), size: Some(200) },
            ]),
        };
        let m = map_raw(raw);
        assert_eq!(m.id, "mlx-community/Qwen3-8B-4bit");
        assert_eq!(m.author, "mlx-community");
        assert_eq!(m.quant_bits, Some(4));
        assert_eq!(m.architecture.as_deref(), Some("Qwen3ForCausalLM"));
        assert_eq!(m.base_model.as_deref(), Some("Qwen/Qwen3-8B"));
        assert_eq!(m.license.as_deref(), Some("apache-2.0"));
        assert_eq!(m.languages, vec!["en".to_string()]);
        assert_eq!(m.file_count, Some(2));
        assert_eq!(m.total_size_bytes, Some(300));
    }

    #[test]
    fn hf_model_serializes_camel_case() {
        let m = HfModel {
            id: "x/y".into(),
            author: "x".into(),
            downloads: 1,
            likes: 1,
            tags: vec![],
            pipeline_tag: None,
            library: None,
            last_modified: "".into(),
            trending_score: None,
            gated: serde_json::Value::Bool(false),
            architecture: None,
            model_type: None,
            quant_bits: Some(4),
            base_model: None,
            license: None,
            languages: vec![],
            file_count: None,
            total_size_bytes: None,
        };
        let v = serde_json::to_value(&m).unwrap();
        assert!(v.get("quantBits").is_some(), "expected camelCase quantBits");
        assert!(v.get("baseModel").is_some(), "expected camelCase baseModel");
        assert!(v.get("totalSizeBytes").is_some(), "expected camelCase totalSizeBytes");
        assert!(v.get("lastModified").is_some(), "expected camelCase lastModified");
        assert!(v.get("pipelineTag").is_some(), "expected camelCase pipelineTag");
        assert!(v.get("trendingScore").is_some(), "expected camelCase trendingScore");
        assert!(v.get("modelType").is_some(), "expected camelCase modelType");
        assert!(v.get("fileCount").is_some(), "expected camelCase fileCount");
    }
}
