//! Linear algebra backends for [`mdarray`](https://crates.io/crates/mdarray)
//!
//! This crate defines a set of traits (`MatVec`, `MatMul`, `Eig`, `SVD`, …) that are
//! implemented by different backends, allowing users to switch between them depending
//! on their needs (performance, portability, or debugging).
//!
//! # Backends
//!
//! - [`Blas`](https://docs.rs/mdarray-linalg-blas): bindings to [BLAS](https://www.netlib.org/blas/)
//! - [`Lapack`](https://docs.rs/mdarray-linalg-lapack): bindings to [LAPACK](https://www.netlib.org/lapack/)
//! - [`Faer`](https://docs.rs/mdarray-linalg-faer): bindings to [faer](https://faer.veganb.tw/)
//! - [`Nalgebra`](https://docs.rs/mdarray-linalg-nalgebra): bindings to [nalgebra](https://nalgebra.rs/)
//! - [`Tblis`](https://docs.rs/mdarray-linalg-tblis): bindings to [TBLIS](https://github.com/MatthewsResearchGroup/tblis)
//! - `Naive`: simple demo backend, integrated into this crate
//! > **Note:** Not all backends support all functionalities.
//!
// ! <details>
// ! <summary>Click to expand the feature support table</summary>
//!
//! | Functionality                                     | BLAS | LAPACK | Naive | Faer | Nalgebra | TBLIS |
//! |---------------------------------------------------|:----:|:------:|:-----:|:----:|:--------:|:-----:|
//! | **▶︎ Basic vector/matrix operations**              |||||||
//! | [Matrix-vector multiplications](crate::matvec#matrix-vector-operations) | ✅ | ⬜ | ✅ | ✅ | ✅ | ⬜ |
//! | [Operations on vectors](crate::matvec#vector-operations)     | ✅ | ⬜ | ✅ | ✅ | ✅ | ⬜ |
//! | [Matrix multiplication](mod@crate::matmul)     | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ |
//! | [Argmax](crate::matvec#argmax)                    | ✅ | ⬜ | ✅ | ⬜ | ✅ | ⬜ |
//! | **▶︎ Decomposition and solving**                              |||||||
//! | [Eigen decomposition](crate::eig)             | ⬜ | ✅ | ⬜ | ✅ | ✅ | ⬜ |
//! | [SVD decomposition](crate::svd)               | ⬜ | ✅ | ⬜ | ✅ | ✅ | ⬜ |
//! | [LU decomposition and inverse](crate::lu)                  | ⬜ | ✅ | ⬜ | ✅ | ✅ | ⬜ |
//! | [Solve](crate::solve)           | ⬜ | ✅ | ⬜ | ✅ | ✅ | ⬜ |
//! | [QR decomposition](crate::qr)                  | ⬜ | ✅ | ✅ | ✅ | ✅ | ⬜ |
//! | [Cholesky decomposition](crate::lu)| ⬜ | ✅ | ⬜ | ✅ |✅ | ⬜ |
//! | [Schur decomposition](crate::eig)         | ⬜ | ✅ | ⬜ | ✅ | ✅ | ⬜ |
//! | **▶︎ Advanced**                                   |||||||
//! | [Tensor contraction](mod@crate::matmul)        | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ |
//!
//! ✅ = implemented
//! 🔧 = not implemented yet / partially implemented
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
//! > See also the section **Troubleshooting** below.
//!
//! The following example demonstrates basic functionality:
//!
//! ```rust
//! use mdarray::array;
//!
//! // The prelude does not expose any names.  It only provides traits as _.
//! use mdarray_linalg::prelude::*;
//!
//! // Backends are provided in partner crates (e.g. mdarray-linalg-blas or mdarray-linalg-faer),
//! // the naive backend exists mostly as a demonstration.
//! use mdarray_linalg::Naive;
//!
//! fn main() {
//!     // Declare two vectors
//!     let x = array![1., 2.];
//!     let y = array![2., 4.];
//!
//!     // Declare two matrices
//!     let a = array![[1., 2.], [3., 4.]];
//!     let b = array![[5., 6.], [7., 8.]];
//!
//!     // ----- Scalar product -----
//!     let dot_result = Naive.dot(&x, &y);
//!     println!("dot(x, y) = {}", dot_result); // x · y
//!
//!     // ----- Matrix multiplication -----
//!     let mut c = Naive.matmul(&a, &b).eval(); // C ← A ✕ B
//!     Naive.matmul(&b, &a).add_to(&mut c);     // C ← B ✕ A + C
//!     println!("A * B + B * A = {:?}", c);
//!
//!     let tmp = Naive.matmul(&b, &c).eval();
//!     let d = Naive.matmul(&a, &tmp).eval();
//! }
//! ```
//!
//! # Troubleshooting
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
pub mod qr;
pub mod solve;
pub mod svd;

pub mod utils;
pub use utils::*;

mod naive;
pub use naive::Naive;

pub mod testing;
