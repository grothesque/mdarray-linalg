//! Linear algebra backends for [`mdarray`](https://crates.io/crates/mdarray)
//!
//! This crate defines a set of generic traits (`MatVec<T>`, `MatMul<T>`, `Eig<T>`, `SVD<T>`, …)
//! that expose common linear algebra operations. The traits are implemented by backends that
//! delegate the work to specialized linear algebra libraries. The backend design allows users to
//! mix and switch between them depending on their needs (performance, portability, or debugging).
//!
//! Each backend (except `Naive`) lives in a separate crate with specific dependencies.
//!
//! # Backend crates
//!
//! - [`mdarray_linalg_blas`][blas-docs]: bindings to [BLAS](https://www.netlib.org/blas/)
//! - [`mdarray_linalg_lapack`][lapack-docs]: bindings to [LAPACK](https://www.netlib.org/lapack/)
//! - [`mdarray_linalg_faer`][faer-docs]: bindings to [faer](https://faer.veganb.tw/)
//! - [`mdarray_linalg_nalgebra`][nalgebra-docs]: bindings to [nalgebra](https://nalgebra.rs/)
//! - [`mdarray_linalg_tblis`][tblis-docs]: bindings to [TBLIS](https://github.com/MatthewsResearchGroup/tblis)
//! - `Naive`: simple demo backend, integrated into this crate
//!
//! # Backend functionality
//!
//! Backends support functionality based on the capabilities of the underlying libraries.
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
//! // Declare two vectors
//! let x = array![1., 2.];
//! let y = array![2., 4.];
//!
//! // Declare two matrices
//! let a = array![[1., 2.], [3., 4.]];
//! let b = array![[5., 6.], [7., 8.]];
//!
//! // ----- Scalar product -----
//! let dot_result = Naive.dot(&x, &y);
//! println!("dot(x, y) = {}", dot_result); // x · y
//!
//! // ----- Matrix multiplication -----
//! let mut c = Naive.matmul(&a, &b).eval(); // C ← A ✕ B
//! Naive.matmul(&b, &a).add_to(&mut c);     // C ← B ✕ A + C
//! println!("A * B + B * A = {:?}", c);
//!
//! let tmp = Naive.matmul(&b, &c).eval();
//! let d = Naive.matmul(&a, &tmp).eval();
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
//! println!("cargo:rustc-link-lib=openblas");
//! println!("cargo:rustc-link-search=native=/usr/lib");
//! ```

#![cfg_attr(docrs, doc = concat!(
    "[blas-docs]: https://docs.rs/mdarray-linalg-blas/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_blas/\n",
    "[lapack-docs]: https://docs.rs/mdarray-linalg-lapack/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_lapack/\n",
    "[faer-docs]: https://docs.rs/mdarray-linalg-faer/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_faer/\n",
    "[nalgebra-docs]: https://docs.rs/mdarray-linalg-nalgebra/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_nalgebra/\n",
    "[tblis-docs]: https://docs.rs/mdarray-linalg-tblis/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_tblis/",
))]
#![cfg_attr(not(docrs), doc = "\
[blas-docs]: ../mdarray_linalg_blas/index.html
[lapack-docs]: ../mdarray_linalg_lapack/index.html
[faer-docs]: ../mdarray_linalg_faer/index.html
[nalgebra-docs]: ../mdarray_linalg_nalgebra/index.html
[tblis-docs]: ../mdarray_linalg_tblis/index.html
")]

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
