//! Linear algebra backends for [`mdarray`](https://crates.io/crates/mdarray).
//!
//! This crate defines a set of traits (`MatVec`, `MatMul`, `Eig`, `SVD`, …) that are
//! implemented by different backends, allowing users to switch between them depending
//! on their needs (performance, portability, or debugging).
//!
//! # Backends
//!
//! - `Blas`: bindings to [BLAS](https://www.netlib.org/blas/)
//! - `Lapack`: bindings to [LAPACK](https://www.netlib.org/lapack/)
//! - `Faer`: bindings to [Faer](https://github.com/sarah-ek/faer-rs)
//! - `Naive`: a simple backend with textbook implementations of some algorithms
//!   useful for testing or when other backends do not provide them.
//! > **Note:** Not all backends support all functionalities.
//!
// ! <details>
// ! <summary>Click to expand the feature support table</summary>
//!
//! | Functionality | BLAS | LAPACK | Naive | Faer |
//! |---------------|:----:|:------:|:-----:|:----:|
//! | `copy/swap`   | 🔧   | ⬜     | 🔧    | ⬜   |
//! | **▶︎ Basic vector/matrix ops** |||| |
//! | `norm1`       | ✅   | ⬜     | 🔧    | 🔧   |
//! | `norm2`       | ✅   | ⬜     | 🔧    | 🔧   |
//! | `dot`         | ✅   | ⬜     | 🔧    | 🔧   |
//! | `α·x + y`     | ✅   | ⬜     | 🔧    | 🔧   |
//! | `α·A·x + y`   | ✅   | ⬜     | 🔧    | 🔧   |
//! | `matmul`      | ✅   | ⬜     | 🔧    | ✅   |
//! | `rank1 update`| ✅   | ⬜     | 🔧    | 🔧   |
//! | `argmax_abs`  | 🔧   | ⬜     | 🔧    | ⬜   |
//! | `argmax`      | ⬜   | ⬜     | ✅    | ⬜   |
//! | **▶︎ Linear algebra** |||| |
//! | `eigen`       | ⬜   | ✅     | ⬜    | ✅   |
//! | `SVD`         | ⬜   | ✅     | ⬜    | ✅   |
//! | `LU`          | ⬜   | ✅     | ⬜    | ✅   |
//! | `solve/inv`   | ⬜   | ✅     | ⬜    | ✅   |
//! | `QR`          | ⬜   | ✅     | ⬜    | ✅   |
//! | `Cholesky`    | ⬜   | ✅     | ⬜    | 🔧   |
//! | `Schur`       | ⬜   | ✅     | ⬜    | 🔧   |
//! | **▶︎ Advanced** |||| |
//! | `givens rot`  | 🔧   | ⬜     | 🔧    | 🔧   |
//! | `prrlu`       | ⬜   | ⬜     | ✅    | ⬜   |
//! | `tensordot`   | ✅   | ⬜     | ✅    |🔧    |
//!
//! ✅ = implemented
//! 🔧 = not implemented yet
//! ⬜ = not applicable / not part of the backend’s scope
//!
// </details>
//!
//! # Example
//!
//! > **Note:**
//! > When running doctests with Blas or Lapack, linking issues may occur due to this Rust issue:
//! > [rust-lang/rust#125657](https://github.com/rust-lang/rust/issues/125657). In that case, run the doctests with:
//! > `RUSTDOCFLAGS="-L native=/usr/lib -C link-arg=-lopenblas" cargo test --doc`
//! >
//! > See also the section **Troubleshoot** below.
//!
//! The following example demonstrates basic functionality:
//!
//! ```rust
//! use mdarray::{DTensor, tensor};
//! use mdarray_linalg::prelude::*; // Imports only traits
//!
//! use mdarray_linalg::Naive;
//! // Use other backends for improved performance and more extensive functionality.
//!
//! fn main() {
//!     // Declare two vectors
//!     let x = tensor![1., 2.];
//!     let y = tensor![2., 4.];
//!
//!     // Declare two matrices
//!     let a = tensor![[1., 2.], [3., 4.]];
//!     let b = tensor![[5., 6.], [7., 8.]];
//!
//!     // ----- Vector operations -----
//!     let dot_result = Naive.dot(&x, &y);
//!     println!("dot(x, y) = {}", dot_result);
//!
//!     // ----- Matrix multiplication -----
//!     let c = Naive.matmul(&a, &b).eval();
//!     println!("A * B = {:?}", c);
//! }
//! ```
//!Some notes:
//!
//! - **Memory usage**: Each trait provides a method returning new
//!   matrices and an overwrite variant using user-allocated buffers.
//!   In that last case, output shapes must match exactly.
//!
//! - **Backend configuration**: Some accept parameters; for example, SVD
//!   may choose an optimal algorithm by default, but the user can
//!   select a specific one if desired.
//!
//! - **Errors**: Convergence issues return a Result; other problems
//!   (dimension mismatch) may panic.
//!
//! # Troubleshoot
//!
//! If you encounter linking issues with BLAS or LAPACK on Linux,
//! one solution is to add a `build.rs` file and configure it to link the libraries manually.
//! In your `Cargo.toml`, add:
//!
//! ```toml
//! [package]
//! build = "build.rs"
//! ```
//!
//! Then, create a `build.rs` file with the following content:
//!
//! ```rust
//! fn main() {
//!     println!("cargo:rustc-link-lib=openblas");
//!     println!("cargo:rustc-link-search=native=/usr/lib");
//! }
//! ```

pub mod prelude;

pub mod eig;
pub mod lu;
pub mod matmul;
pub mod matvec;
pub mod prrlu;
pub mod qr;
pub mod solve;
pub mod svd;

pub mod utils;
pub use utils::*;

mod naive;
pub use naive::Naive;

pub mod testing;
