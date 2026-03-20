fn main() {
    #[cfg(feature = "ane")]
    {
        let bridge_path = std::path::Path::new("src/ane_bridge/bridge.m");
        if bridge_path.exists() {
            cc::Build::new()
                .file(bridge_path)
                .flag("-fobjc-arc")
                .flag("-framework").flag("Foundation")
                .flag("-framework").flag("IOSurface")
                .flag("-ldl")
                .compile("ane_bridge");

            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=IOSurface");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rerun-if-changed=src/ane_bridge/bridge.m");
        }
    }
}