//! `asmi` — shorthand alias for `mlx-top`.
//! Passes all arguments through.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = std::process::Command::new("mlx-top")
        .args(&args)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("asmi: failed to exec mlx-top: {e}");
            std::process::exit(1);
        });
    std::process::exit(status.code().unwrap_or(1));
}
