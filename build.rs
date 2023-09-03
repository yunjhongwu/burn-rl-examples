fn main() {
    println!("cargo:rustc-link-lib=blas");
    println!("cargo:rustc-link-lib=lapack");
}
