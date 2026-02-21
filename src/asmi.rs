//! `asmi` — shorthand alias for `apple-smi`.
//! Passes all arguments through.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = std::process::Command::new("apple-smi")
        .args(&args)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("asmi: failed to exec apple-smi: {e}");
            std::process::exit(1);
        });
    std::process::exit(status.code().unwrap_or(1));
}
