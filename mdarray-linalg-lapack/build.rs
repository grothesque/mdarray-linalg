fn main() {
    #[cfg(feature = "lapack-sys-backend")]
    {
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-search=native=/usr/lib");
    }
}
