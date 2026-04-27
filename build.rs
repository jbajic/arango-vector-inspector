use std::path::PathBuf;

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let faiss_build = manifest_dir.join("build-faiss-c");
    let faiss_c = faiss_build.join("c_api");
    let faiss = faiss_build.join("faiss");

    if !faiss_c.join("libfaiss_c.so").exists() {
        panic!(
            "libfaiss_c.so not found at {}. Run the FAISS C API build first \
             (see README).",
            faiss_c.display()
        );
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-search=native={}", faiss_c.display());
    println!("cargo:rustc-link-search=native={}", faiss.display());
    // Embed rpath so the binary finds libfaiss_c.so / libfaiss.so without
    // needing LD_LIBRARY_PATH at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", faiss_c.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", faiss.display());
}
