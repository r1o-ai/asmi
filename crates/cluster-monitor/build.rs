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

    #[cfg(feature = "jaccl")]
    {
        let vendor_dir = std::path::Path::new("../../vendor/jaccl");
        // Only compile if vendor files are present
        if vendor_dir.join("jaccl_shim.cpp").exists() {
            let cpp_files = [
                "rdma.cpp",
                "tcp.cpp",
                "jaccl.cpp",
                "mesh.cpp",
                "ring.cpp",
                "jaccl_shim.cpp",
            ];

            let mut build = cc::Build::new();
            build
                .cpp(true)
                .std("c++20")
                // "jaccl/rdma.h" etc. resolve relative to vendor/
                .include("../../vendor")
                // <json.hpp> resolves to vendor/jaccl/json.hpp
                .include("../../vendor/jaccl");

            for f in &cpp_files {
                let path = vendor_dir.join(f);
                build.file(&path);
                println!("cargo:rerun-if-changed={}", path.display());
            }

            // Rerun if any header changes
            for entry in std::fs::read_dir(vendor_dir).expect("read vendor/jaccl") {
                let entry = entry.expect("dir entry");
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "h" || e == "hpp") {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }

            build.compile("jaccl_shim");

            // libibverbs is loaded at runtime via dlsym, but we need libdl
            println!("cargo:rustc-link-lib=dylib=dl");
        }
    }
}