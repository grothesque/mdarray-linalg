#![doc(html_logo_url = "logo.png")]
//! Linear algebra backends for [`mdarray`](https://crates.io/crates/mdarray)
//!
//! This crate defines traits for linear algebra operations on `mdarray` arrays. Whole-array
//! operations including tensor contraction, matrix multiplication, decompositions and
//! factorizations are exposed as trait methods. Crates such as [`mdarray_linalg_blas`] and
//! [`mdarray_linalg_faer`] provide library-specific backends, i.e., Rust types that implement these
//! traits.
//!
//! This backend-based approach is more than just a unified interface to multiple libraries.
//! Backends are Rust values, so they can carry configuration such as threading settings or
//! library-specific context.
//!
//! The operation traits are deliberately generic over the scalar type. This allows each backend to
//! choose the scalar types it supports: BLAS/LAPACK backends naturally cover the classic BLAS
//! scalar types, while other backends may be generic over broader families of scalars.
//!
//! User code can be generic over both the scalar type and the backend that provides whole-array
//! operations. This allows user code to be written in any of the following ways:
//!
//! - tied to a concrete combination of backend and scalar type;
//! - generic over the backend for a particular concrete scalar type;
//! - generic over the scalar type for a concrete backend;
//! - generic over both the backend and the scalar type.
//!
//! In the most general case, trait bounds for the scalar type and the backend are expressed independently:
//! ```text
//! T: ...          // Require certain operations for the scalar type T.
//! B: Contract<T>  // Require a backend that can contract arrays of T.
//! ```
//!
//! This separation also leaves room for backend implementations optimized for particular scalar
//! types, e.g., matrix multiplication for double-double scalars. These can outperform
//! implementations built from generic scalar operations of that type.
//!
//! Each backend (except `Naive`) lives in a separate crate with specific dependencies.
//!
//! # Backend crates
//!
//! - [`mdarray_linalg_blas`]: bindings to [BLAS](https://www.netlib.org/blas/)
//! - [`mdarray_linalg_lapack`]: bindings to [LAPACK](https://www.netlib.org/lapack/)
//! - [`mdarray_linalg_faer`]: bindings to [faer](https://faer.veganb.tw/)
//! - [`mdarray_linalg_nalgebra`]: bindings to [nalgebra](https://nalgebra.rs/)
//! - [`mdarray_linalg_tblis`]: bindings to [TBLIS](https://github.com/MatthewsResearchGroup/tblis)
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
//! | [Matrix multiplication](mod@crate::contract)     | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ |
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
//! | [Tensor contraction](mod@crate::contract)        | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ |
//!
//! ✅ = implemented
//! 🔧 = not implemented yet / partially implemented
//! ⬜ = not applicable / not part of the backend’s scope
//!
// </details>
//!
//! # Example
//!
//! The following example demonstrates basic functionality:
//!
//! ```rust
//! use mdarray::array;
//!
//! // The prelude does not expose any names.  It only provides traits as _.
//! use mdarray_linalg::prelude::*;
//!
//! // Backends are provided in partner crates (e.g. mdarray_linalg_blas or mdarray_linalg_faer),
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
//! # Dependencies on non-Rust libraries
//!
//! Backend crates that bind non-Rust libraries do not impose a concrete library to link against.
//! This choice is left to the user.  For example, users of [`mdarray_linalg_blas`] may add
//! a provider crate such as `openblas-src`, users of [`mdarray_linalg_lapack`] may add
//! `lapack-src`, and users of [`mdarray_linalg_tblis`] may add `tblis-src` or arrange
//! to link TBLIS differently.  The provider crate must be referenced from Rust code,
//! e.g. by adding `extern crate openblas_src as _;`, so that appropriate link
//! directives are used.
//!
//! See the documentation of the individual backend crates for further information.
//!
// Keep the doc-comment blank line above: these reference definitions must start
// a separate Markdown block from the preceding paragraph.
#![cfg_attr(docsrs, doc = concat!(
    "[`mdarray_linalg_blas`]: https://docs.rs/mdarray-linalg-blas/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_blas/\n",
    "[`mdarray_linalg_lapack`]: https://docs.rs/mdarray-linalg-lapack/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_lapack/\n",
    "[`mdarray_linalg_faer`]: https://docs.rs/mdarray-linalg-faer/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_faer/\n",
    "[`mdarray_linalg_nalgebra`]: https://docs.rs/mdarray-linalg-nalgebra/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_nalgebra/\n",
    "[`mdarray_linalg_tblis`]: https://docs.rs/mdarray-linalg-tblis/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_tblis/",
))]
#![cfg_attr(not(docsrs), doc = "\
[`mdarray_linalg_blas`]: ../mdarray_linalg_blas/index.html
[`mdarray_linalg_lapack`]: ../mdarray_linalg_lapack/index.html
[`mdarray_linalg_faer`]: ../mdarray_linalg_faer/index.html
[`mdarray_linalg_nalgebra`]: ../mdarray_linalg_nalgebra/index.html
[`mdarray_linalg_tblis`]: ../mdarray_linalg_tblis/index.html
")]

pub mod prelude;

pub mod eig;
pub mod lu;
pub mod contract;
pub mod matvec;
pub mod qr;
pub mod solve;
pub mod svd;

pub mod utils;

mod naive;
pub use naive::Naive;

#[doc(hidden)]
pub mod testing;
