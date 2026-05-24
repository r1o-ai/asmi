#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

#![allow(dead_code)]

mod compiled_model;
mod config;
mod deltanet;
mod executables;
mod kv_cache;
mod lm_head;
mod rope;
mod sampling;
mod session;
mod spinner;
mod weights;

use std::io::{self, Write};
use std::iter::repeat_n;
use std::path::PathBuf;
use std::time::Instant;

use safetensors::SafeTensors;
use tokenizers::Tokenizer;

use compiled_model::CompiledModel;
use session::Session;
use spinner::Spinner;

const REPO_ID: &str = "Qwen/Qwen3.5-0.8B";
const PROMPT: &str = "The meaning of life is";
const MAX_NEW_TOKENS: usize = 60;
const MAX_SEQUENCE_LENGTH: usize = 128;
const MIN_SPATIAL_WIDTH: usize = 64;
const TEMPERATURE: f32 = 0.7;
const TOP_P: f32 = 0.9;
const REPETITION_PENALTY: f32 = 1.1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    // Parse CLI args: --model-path <local_dir> or --prompt <text>
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<PathBuf> = None;
    let mut prompt = PROMPT.to_string();
    let mut max_tokens = MAX_NEW_TOKENS;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" | "--model" => {
                i += 1;
                model_path = Some(PathBuf::from(&args[i]));
            }
            "--prompt" | "-p" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--max-tokens" | "-n" => {
                i += 1;
                max_tokens = args[i].parse()?;
            }
            _ => {
                eprintln!("Usage: qwen_forward [--model-path <dir>] [--prompt <text>] [--max-tokens <n>]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Load model files
    let model_files = if let Some(path) = model_path {
        weights::load_model_local(&path)?
    } else {
        eprintln!("\x1b[2mNo --model-path given, downloading from HuggingFace...\x1b[0m");
        weights::download_model(REPO_ID)?
    };

    let config = model_files.config;
    eprintln!(
        "\x1b[2m{} layers ({} full attn, {} DeltaNet), hidden={}, heads={}Q/{}KV, vocab={}\x1b[0m",
        config.num_hidden_layers,
        (0..config.num_hidden_layers).filter(|&i| config.is_full_attention(i)).count(),
        (0..config.num_hidden_layers).filter(|&i| !config.is_full_attention(i)).count(),
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size,
    );

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&model_files.tokenizer_path)
        .map_err(|e| format!("tokenizer: {e}"))?;

    // Encode prompt
    let encoding = tokenizer.encode(prompt.as_str(), false)
        .map_err(|e| format!("encode: {e}"))?;
    let prompt_token_ids = encoding.get_ids();
    let prompt_length = prompt_token_ids.len();

    // Pad to MIN_SPATIAL_WIDTH for ANE
    let padding_token = 0u32; // Use token 0 for padding
    let padded_length = prompt_length.max(MIN_SPATIAL_WIDTH);
    let padded_token_ids: Box<[u32]> = prompt_token_ids
        .iter()
        .copied()
        .chain(repeat_n(padding_token, padded_length - prompt_length))
        .collect();

    // Load safetensors
    let mut spinner = Spinner::new("Loading safetensors");
    let safetensors_bytes: Vec<Vec<u8>> = model_files
        .safetensors_paths
        .iter()
        .map(|path| {
            spinner.update(&format!("Loading {}", path.file_name().unwrap_or_default().to_string_lossy()));
            std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        })
        .collect();

    let safetensors: Vec<SafeTensors> = safetensors_bytes
        .iter()
        .map(|bytes| SafeTensors::deserialize(bytes).expect("deserialize safetensors"))
        .collect();
    spinner.finish("Loaded safetensors");

    // Compile model
    let model = CompiledModel::from_safetensors(
        config, &safetensors, padded_length, MAX_SEQUENCE_LENGTH,
    )?;

    let mut session = Session::new(&model, padded_length);
    let mut rng = rand::rng();

    // Prefill
    {
        let prefill_spinner = Spinner::new("Prefilling prompt");
        let logits = session.prefill(&padded_token_ids, prompt_length);
        let first_token = sampling::sample(
            &logits, TEMPERATURE, TOP_P, REPETITION_PENALTY,
            prompt_token_ids, &mut rng,
        );
        prefill_spinner.finish("Prefill complete");

        let prompt_text = tokenizer.decode(prompt_token_ids, true)
            .map_err(|e| format!("decode: {e}"))?;
        print!("{prompt_text}");
        io::stdout().flush()?;

        let mut generated_tokens: Vec<u32> = prompt_token_ids.to_vec();
        let mut previous_text = prompt_text;

        generated_tokens.push(first_token);
        let current_text = tokenizer.decode(&generated_tokens, true)
            .map_err(|e| format!("decode: {e}"))?;
        if let Some(delta) = current_text.strip_prefix(&previous_text) {
            print!("{delta}");
        }
        io::stdout().flush()?;
        previous_text = current_text;

        let generation_start = Instant::now();
        for _ in 0..max_tokens - 1 {
            let last_token = *generated_tokens.last().unwrap();
            let logits = session.decode_step(last_token);
            let next_token = sampling::sample(
                &logits, TEMPERATURE, TOP_P, REPETITION_PENALTY,
                &generated_tokens, &mut rng,
            );
            generated_tokens.push(next_token);

            let current_text = tokenizer.decode(&generated_tokens, true)
                .map_err(|e| format!("decode: {e}"))?;
            if let Some(delta) = current_text.strip_prefix(&previous_text) {
                print!("{delta}");
            }
            io::stdout().flush()?;
            previous_text = current_text;
        }

        let generation_elapsed = generation_start.elapsed().as_secs_f64();
        println!();
        eprintln!(
            "\n\x1b[2m[{max_tokens} tokens in {generation_elapsed:.1}s ({:.1} tok/s) | total {:.1}s]\x1b[0m",
            max_tokens as f64 / generation_elapsed,
            start.elapsed().as_secs_f64(),
        );
    }

    Ok(())
}
