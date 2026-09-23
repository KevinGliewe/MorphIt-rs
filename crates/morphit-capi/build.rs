//! Regenerates `include/morphit.h` from the Rust sources with cbindgen.
//! The file is only rewritten when its content changes.

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    let config =
        cbindgen::Config::from_file(format!("{crate_dir}/cbindgen.toml")).expect("valid cbindgen.toml");
    match cbindgen::Builder::new().with_crate(&crate_dir).with_config(config).generate() {
        Ok(bindings) => {
            bindings.write_to_file(format!("{crate_dir}/include/morphit.h"));
        }
        Err(e) => println!("cargo:warning=cbindgen could not generate include/morphit.h: {e}"),
    }
}
