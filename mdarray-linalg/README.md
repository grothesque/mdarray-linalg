# mdarray-linalg: linear algebra bindings for Rust mdarray

Efficient, flexible, and idiomatic linear algebra bindings (BLAS, LAPACK, etc.)
to the Rust [mdarray](https://crates.io/crates/mdarray) crate.

## Usage
These crates are released on crates.io. To depend on the main crate
and one of the backends (for example `mdarray-linalg-blas`) add the following
to your `Cargo.toml`:
```toml
[dependencies]
mdarray = "0.7.1"
mdarray-linalg = "0.1"
mdarray-linalg-blas = "0.1"
openblas-src = { version = "0.10", features = ["system"] }
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

See the tests and doc for a “tutorial”.

### Example: Matrix Multiplication
```rust
use mdarray::tensor;
use mdarray_linalg::{MatMul, MatMulBuilder};
use mdarray_linalg_blas::Blas;

use openblas_src as _;

fn main() {
    let a = tensor![[1., 2.], [3., 4.]];
    let b = tensor![[5., 6.], [7., 8.]];
    let c = Blas.matmul(&a, &b).eval();
    println!("{:?}", c);
}
```

Should output: `[[19.0, 22.0], [43.0, 50.0]]`
```

## License
Dual-licensed (Apache and MIT) to be compatible with the Rust project.
See the file LICENSE.md in this directory.
