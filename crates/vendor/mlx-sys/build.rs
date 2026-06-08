extern crate cmake;

use bindgen::RustTarget;
use cmake::Config;
use std::{
    env,
    path::PathBuf,
};

fn build_and_link_mlx_c() {
    configure_clang_module_cache();
    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");
    config.define("MLX_SOURCE_DIR", mlx_source_dir());
    config.define("FETCHCONTENT_SOURCE_DIR_JSON", json_source_dir());
    config.define("FETCHCONTENT_SOURCE_DIR_FMT", fmt_source_dir());
    config.define("FETCHCONTENT_SOURCE_DIR_GGUFLIB", gguf_source_dir());
    #[cfg(feature = "metal")]
    {
        config.define("FETCHCONTENT_SOURCE_DIR_METAL_CPP", metal_cpp_source_dir());
    }

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}

fn configure_clang_module_cache() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cache_dir = out_dir.join("clang-module-cache");
    std::fs::create_dir_all(&cache_dir).expect("failed to create clang module cache dir");
    env::set_var("CLANG_MODULE_CACHE_PATH", cache_dir);
}

fn mlx_source_dir() -> String {
    if let Ok(source_dir) = env::var("MLX_SOURCE_DIR") {
        let path = resolve_mlx_source_path(&PathBuf::from(source_dir));
        if path.join("CMakeLists.txt").is_file() {
            return path.display().to_string();
        }
        panic!(
            "MLX_SOURCE_DIR={} does not look like a prefetched MLX source checkout",
            path.display()
        );
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let default_source = manifest_dir.join("../../../resources/vendor/mlx");
    if default_source.join("CMakeLists.txt").is_file() {
        return default_source.display().to_string();
    }

    panic!(
        "missing prefetched MLX source. Run `MLX_CACHE_DIR=resources/vendor/mlx ./scripts/prefetch_mlx_artifacts.sh` and set MLX_SOURCE_DIR=resources/vendor/mlx before building native MLX targets"
    );
}

fn resolve_mlx_source_path(path: &PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path.clone();
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_relative = manifest_dir.join("../../..").join(path);
    if workspace_relative.join("CMakeLists.txt").is_file() {
        return workspace_relative;
    }
    path.clone()
}

#[cfg(feature = "metal")]
fn metal_cpp_source_dir() -> String {
    if let Ok(source_dir) = env::var("METAL_CPP_SOURCE_DIR") {
        let path = resolve_workspace_relative_path(&PathBuf::from(source_dir));
        if looks_like_metal_cpp_source(&path) {
            return path.display().to_string();
        }
        panic!(
            "METAL_CPP_SOURCE_DIR={} does not look like a prefetched metal-cpp source checkout",
            path.display()
        );
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let default_source = manifest_dir.join("../../../resources/vendor/metal-cpp");
    if looks_like_metal_cpp_source(&default_source) {
        return default_source.display().to_string();
    }

    panic!(
        "missing prefetched Metal C++ source. Run `./scripts/prefetch_mlx_artifacts.sh` and set METAL_CPP_SOURCE_DIR=resources/vendor/metal-cpp before building native MLX targets"
    );
}

#[cfg(feature = "metal")]
fn looks_like_metal_cpp_source(path: &PathBuf) -> bool {
    path.join("Metal").is_dir() || path.join("SingleHeader/Metal.hpp").is_file()
}

fn resolve_workspace_relative_path(path: &PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path.clone();
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_relative = manifest_dir.join("../../..").join(path);
    if workspace_relative.exists() {
        return workspace_relative;
    }
    path.clone()
}

fn json_source_dir() -> String {
    source_dir_from_env_or_default(
        "JSON_SOURCE_DIR",
        "resources/vendor/nlohmann-json",
        looks_like_json_source,
        "nlohmann/json",
    )
}

fn fmt_source_dir() -> String {
    source_dir_from_env_or_default(
        "FMT_SOURCE_DIR",
        "resources/vendor/fmt",
        looks_like_fmt_source,
        "fmtlib/fmt",
    )
}

fn gguf_source_dir() -> String {
    source_dir_from_env_or_default(
        "GGUF_SOURCE_DIR",
        "resources/vendor/gguf-tools",
        looks_like_gguf_source,
        "antirez/gguf-tools",
    )
}

fn source_dir_from_env_or_default(
    env_name: &str,
    default_workspace_path: &str,
    validate: fn(&PathBuf) -> bool,
    label: &str,
) -> String {
    if let Ok(source_dir) = env::var(env_name) {
        let path = resolve_workspace_relative_path(&PathBuf::from(source_dir));
        if validate(&path) {
            return path.display().to_string();
        }
        panic!(
            "{env_name}={} does not look like a prefetched {label} source checkout",
            path.display()
        );
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let default_source = manifest_dir.join("../../..").join(default_workspace_path);
    if validate(&default_source) {
        return default_source.display().to_string();
    }

    panic!(
        "missing prefetched {label} source. Run `./scripts/prefetch_mlx_artifacts.sh` before building native MLX targets"
    );
}

fn looks_like_json_source(path: &PathBuf) -> bool {
    path.join("single_include/nlohmann/json.hpp").is_file()
}

fn looks_like_fmt_source(path: &PathBuf) -> bool {
    path.join("include/fmt/core.h").is_file()
}

fn looks_like_gguf_source(path: &PathBuf) -> bool {
    path.join("gguflib.c").is_file() || path.join("gguflib.h").is_file()
}

fn main() {
    build_and_link_mlx_c();

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target(RustTarget::Stable_1_73)
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
