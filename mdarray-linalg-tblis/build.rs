fn main() {
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=tblis");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib");
}
