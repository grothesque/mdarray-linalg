# mdarray-linalg: linear algebra bindings for Rust mdarray

Efficient, flexible, and idiomatic linear algebra bindings (BLAS, LAPACK, etc.)
to the Rust [mdarray](https://github.com/fre-hu/mdarray) crate.

## Usage
These crates are not yet released on crates.io. To depend on the main crate
and one of the backends (for example `mdarray-linalg-blas`) add the following
to your `Cargo.toml`:
```toml
[dependencies]
mdarray = { git = "https://github.com/fre-hu/mdarray.git", rev = "2a67f2ec7ed326fad9dc95a94f772b7e2140c8eb" }
mdarray-linalg = { git = "https://github.com/grothesque/mdarray-linalg" }
mdarray-linalg-blas = { git = "https://github.com/grothesque/mdarray-linalg" }
openblas-src = { version = "0.10.11", features = ["system"] }
```

**Important notes:**
- Use the latest GitHub version of `mdarray`, not the crates.io version
- When using the BLAS backend, include `openblas-src` to avoid linkage errors

See the tests for a “tutorial”.

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
