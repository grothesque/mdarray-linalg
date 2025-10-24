# mdarray-linalg: linear algebra bindings for Rust mdarray

Efficient, flexible, and idiomatic linear algebra bindings (BLAS, LAPACK, etc.)
to the Rust [mdarray](https://crates.io/crates/mdarray) crate.

## Usage
These crates are released on crates.io:
```bash
cargo add mdarray-linalg
```
and if you need a backend:
```bash
cargo add mdarray-linalg-blas 
```

**Important notes:**
- Use the latest version of `mdarray`.
- When using the BLAS backend, include `openblas-src` to avoid linkage errors
- When running doctests with Blas or Lapack, linking issues may occur
 due to this Rust issue:
 [rust-lang/rust#125657](https://github.com/rust-lang/rust/issues/125657). In
 that case, run the doctests with: `RUSTDOCFLAGS="-L native=/usr/lib
 -C link-arg=-lopenblas" cargo test --doc`

If you encounter linking issues with BLAS or LAPACK on Linux, one solution is to add a build.rs file and configure it to link the libraries manually.
In your Cargo.toml, add:

```toml
[package]
build = "build.rs"
```
Then, create a `build.rs` file with the following content:

```rust
fn main() {
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-search=native=/usr/lib");
}
```

See [docs.rs](https://docs.rs/mdarray/latest/mdarray-linalg/) documentation for code examples and usage instructions.

## License
Dual-licensed (Apache and MIT) to be compatible with the Rust project.
See the file LICENSE.md in this directory.
