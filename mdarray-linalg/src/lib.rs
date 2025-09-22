//! Linear algebra backends for [`mdarray`](https://crates.io/crates/mdarray).
//!
//! This crate defines a set of traits (`MatVec`, `MatMul`, `Eig`, `SVD`, …) that are
//! implemented by different backends, allowing users to switch between them depending
//! on their needs (performance, portability, or debugging).
//!
//! # Backends
//!
//! - `BLAS` bindings to [BLAS](https://www.netlib.org/blas/)
//! - `LAPACK` bindings to [LAPACK](https://www.netlib.org/lapack/)
//! - `Faer` bindings to [Faer](https://github.com/sarah-ek/faer-rs)
//! - `Naive` a simple backend with textbook implementations of some algorithms (e.g., `PRRLU`)
//!   useful for testing or when other backends do not provide them.
//! > **Note:** Not all backends support all functionalities.
//!
//! <details>
//! <summary>Click to expand the feature support table</summary>
//!
//! | Functionality | BLAS | LAPACK | Naive | Faer |
//! |---------------|:----:|:------:|:-----:|:----:|
//! | `copy/swap`   | ❌   | ⬜     | ❌    | ⬜   |
//! | **▶︎ Basic vector/matrix ops** |||| |
//! | `norm1`       | ✅   | ⬜     | ❌    | ❌   |
//! | `norm2`       | ✅   | ⬜     | ❌    | ❌   |
//! | `dot`         | ✅   | ⬜     | ❌    | ❌   |
//! | `α·x + y`     | ✅   | ⬜     | ❌    | ❌   |
//! | `α·A·x + y`   | ✅   | ⬜     | ❌    | ❌   |
//! | `matmul`      | ✅   | ⬜     | ❌    | ✅   |
//! | `rank1 update`| ✅   | ⬜     | ❌    | ❌   |
//! | `argmax_abs`  | ❌   | ⬜     | ❌    | ⬜   |
//! | `argmax`      | ⬜   | ⬜     | ✅    | ⬜   |
//! | **▶︎ Linear algebra** |||| |
//! | `eigen`       | ⬜   | ✅     | ⬜    | ❌   |
//! | `SVD`         | ⬜   | ✅     | ⬜    | ✅   |
//! | `LU`          | ⬜   | ✅     | ⬜    | ❌   |
//! | `solve/inv`   | ⬜   | ❌     | ⬜    | ❌   |
//! | `QR`          | ⬜   | ✅     | ⬜    | ❌   |
//! | `Cholesky`    | ⬜   | ❌     | ⬜    | ❌   |
//! | `Schur`       | ⬜   | ❌     | ⬜    | ❌   |
//! | **▶︎ Advanced** |||| |
//! | `givens rot`  | ❌   | ⬜     | ❌    | ❌   |
//! | `prrlu`       | ⬜   | ⬜     | ✅    | ⬜   |
//!
//! ✅ = implemented  
//! ❌ = not implemented yet  
//! ⬜ = not applicable / not part of the backend’s scope
//!
//! </details>
//!
//! # How it works
//!
//! Each backend implements the same set of traits defined in this crate. This allows
//! code written against `mdarray-linalg` to run seamlessly with BLAS, LAPACK, Faer,
//! or the Naive backend by simply swapping the imported backend type.
//!
//! # Example
//!
//! The following example demonstrates core functionality:
//!
//! - vector operations (dot product, matrix-vector multiplication),
//! - matrix multiplication,
//! - eigenvalue decomposition.
//!
//! ```rust
//! use mdarray::DTensor;
//! use mdarray_linalg::prelude::*;
//!
//! // Backends
//! use mdarray_linalg_blas::Blas;
//! use mdarray_linalg_naive::Naive;
//! use mdarray_linalg_lapack::Lapack;
//!
//! fn main() {
//!     // ----- Vector operations -----
//!     let n = 3;
//!     let x = DTensor::<f64, 1>::from_fn([n], |i| (i[0] + 1) as f64); // [1., 2., 3.]
//!     let y = DTensor::<f64, 1>::from_fn([n], |i| (2 * (i[0] + 1)) as f64); // [2., 4., 6.]
//!
//!     // Dot product using BLAS
//!     let dot_result = Blas.dot(&x, &y);
//!     println!("dot(x, y) = {}", dot_result); // 28
//!
//!     // Matrix-vector multiplication with scaling
//!     let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * n + i[1] + 1) as f64);
//!     let y_result = Blas.matvec(&a, &x).scale(2.).eval();
//!     println!("A * x * 2 = {:?}", y_result);
//!
//!     // ----- Matrix multiplication -----
//!     let a = mdarray::tensor![[1., 2.], [3., 4.]];
//!     let b = mdarray::tensor![[5., 6.], [7., 8.]];
//!
//!     // Using Faer backend
//!     use mdarray_linalg_faer::Faer;
//!     let c = Faer.matmul(&a, &b).eval();
//!     println!("A * B = {:?}", c);
//!
//!     // ----- Eigenvalue decomposition -----
//!     let mut m = mdarray::tensor![[1., 2.], [2., 3.]];
//!
//!     let decomp = Lapack
//!         .eig(&mut m)
//!         .expect("Eigenvalue decomposition failed");
//!
//!     println!("Eigenvalues: {:?}", decomp.eigenvalues);
//!     if let Some(vectors) = decomp.right_eigenvectors {
//!         println!("Right eigenvectors: {:?}", vectors);
//!     }
//!
//!     // ----- Naive backend -----
//!     // The Naive backend provides fallback implementations for algorithms
//!     // not available in other libraries, such as PRRLU.
//!     let _naive_result = Naive.matmul(&a, &b).eval();
//! }
//! ```
pub mod prelude;

mod matmul;
pub use matmul::*;

mod qr;
pub use qr::*;

mod svd;
pub use svd::*;

mod utils;
pub use utils::*;

mod matvec;
pub use matvec::*;

mod prrlu;
pub use prrlu::*;

mod lu;
pub use lu::*;

mod eig;
pub use eig::*;
