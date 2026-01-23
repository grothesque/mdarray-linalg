fn main() {
    #[cfg(feature = "cblas-sys-backend")]
    {
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-search=native=/usr/lib");
    }
}
