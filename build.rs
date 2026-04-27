use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let faiss_src = manifest_dir.join("third_party").join("faiss");
    let faiss_build = manifest_dir.join("build-faiss-c");
    let faiss_c = faiss_build.join("c_api");
    let faiss = faiss_build.join("faiss");
    let lib_path = faiss_c.join("libfaiss_c.so");

    if !lib_path.exists() {
        if !faiss_src.join("CMakeLists.txt").exists() {
            panic!(
                "FAISS submodule not found at {}. Run: git submodule update --init --recursive",
                faiss_src.display()
            );
        }
        build_faiss_c(&faiss_src, &faiss_build);
        if !lib_path.exists() {
            panic!(
                "FAISS build finished but {} is still missing — \
                 check the build output above.",
                lib_path.display()
            );
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-search=native={}", faiss_c.display());
    println!("cargo:rustc-link-search=native={}", faiss.display());
    // Embed rpath so the binary finds libfaiss_c.so / libfaiss.so without
    // needing LD_LIBRARY_PATH at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", faiss_c.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", faiss.display());
}

fn build_faiss_c(src: &Path, build_dir: &Path) {
    eprintln!("=== arango-vector-inspector: building libfaiss_c.so (one-time) ===");
    std::fs::create_dir_all(build_dir).expect("creating build-faiss-c");

    let cxx = std::env::var("CXX").unwrap_or_else(|_| "/usr/bin/g++".into());
    let cc = std::env::var("CC").unwrap_or_else(|_| "/usr/bin/gcc".into());

    let configure = Command::new("cmake")
        .arg(src)
        .arg("-DFAISS_ENABLE_C_API=ON")
        .arg("-DBUILD_SHARED_LIBS=ON")
        .arg("-DFAISS_ENABLE_GPU=OFF")
        .arg("-DFAISS_ENABLE_PYTHON=OFF")
        .arg("-DBUILD_TESTING=OFF")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCMAKE_C_COMPILER={cc}"))
        .arg(format!("-DCMAKE_CXX_COMPILER={cxx}"))
        .current_dir(build_dir)
        .status()
        .expect("running cmake configure for FAISS");
    if !configure.success() {
        panic!(
            "FAISS cmake configure failed. Make sure cmake, a C++ compiler, \
             BLAS, and OpenMP are installed."
        );
    }

    let parallel = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".into());
    let build = Command::new("cmake")
        .arg("--build")
        .arg(".")
        .arg("--target")
        .arg("faiss_c")
        .arg("--parallel")
        .arg(&parallel)
        .current_dir(build_dir)
        .status()
        .expect("running cmake --build for FAISS");
    if !build.success() {
        panic!("FAISS cmake build failed.");
    }
    eprintln!("=== libfaiss_c.so built ===");
}
