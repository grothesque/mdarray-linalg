//! Linear algebra backends for [`mdarray`](https://crates.io/crates/mdarray).
//!
//! This crate defines a set of traits (`MatVec`, `MatMul`, `Eig`, `SVD`, …) that are
//! implemented by different backends, allowing users to switch between them depending
//! on their needs (performance, portability, or debugging).
//!
//! # Backends
//!
//! - `Blas` bindings to [BLAS](https://www.netlib.org/blas/)
//! - `Lapack` bindings to [LAPACK](https://www.netlib.org/lapack/)
//! - `Faer` bindings to [Faer](https://github.com/sarah-ek/faer-rs)
//! - `Naive` a simple backend with textbook implementations of some algorithms
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
//! | `solve/inv`   | ⬜   | ✅     | ⬜    | ❌   |
//! | `QR`          | ⬜   | ✅     | ⬜    | ❌   |
//! | `Cholesky`    | ⬜   | ✅     | ⬜    | ❌   |
//! | `Schur`       | ⬜   | ✅     | ⬜    | ❌   |
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
//! # Example
//!
//! > **Note:**
//! > When running doctests with Blas or Lapack, linking issues may occur due to this Rust issue:
//! > [rust-lang/rust#125657](https://github.com/rust-lang/rust/issues/125657). In that case, run the doctests with:
//! > `RUSTDOCFLAGS="-L native=/usr/lib -C link-arg=-lopenblas" cargo test --doc`
//!
//! The following example demonstrates core functionality:
//!
//! ```rust
//! use mdarray::{DTensor, tensor};
//! use mdarray_linalg::prelude::*;
//!
//! // Backends
//! use mdarray_linalg_blas::Blas;
//! use mdarray_linalg_faer::Faer;
//! use mdarray_linalg_naive::Naive;
//! use mdarray_linalg_lapack::Lapack;
//! use mdarray_linalg_lapack::svd::SVDConfig;
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
//!     let dot_result = Blas.dot(&x, &y);
//!     println!("dot(x, y) = {}", dot_result);
//!
//!     let y_result = Blas.matvec(&a, &x).scale(2.).eval();
//!     println!("A * x * 2 = {:?}", y_result);
//!
//!     // ----- Matrix multiplication -----
//!     let c = Faer.matmul(&a, &b).eval();
//!     println!("A * B = {:?}", c);
//!
//!     // ----- Eigenvalue decomposition -----
//!     // Note: we must clone `a` here because decomposition routines destroy the input.
//!     let bd = Lapack::default();
//!     let EigDecomp {
//!        eigenvalues,
//!        right_eigenvectors,
//!        ..
//!      } = bd.eig(&mut a.clone()).expect("Eigenvalue decomposition failed");
//!
//!     println!("Eigenvalues: {:?}", eigenvalues);
//!     if let Some(vectors) = right_eigenvectors {
//!         println!("Right eigenvectors: {:?}", vectors);
//!     }
//!
//!     // ----- Singular Value Decomposition (SVD) -----
//!     let bd = Lapack::default().config_svd(SVDConfig::DivideConquer);
//!     let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");
//!     println!("Singular values: {:?}", s);
//!     println!("Left singular vectors U: {:?}", u);
//!     println!("Right singular vectors V^T: {:?}", vt);
//!
//!     // ----- QR Decomposition -----
//!     let (m, n) = *a.shape();
//!     let mut q = DTensor::<f64, 2>::zeros([m, m]);
//!     let mut r = DTensor::<f64, 2>::zeros([m, n]);
//!
//!     let bd = Lapack::default();
//!     bd.qr_overwrite(&mut a.clone(), &mut q, &mut r);
//!     println!("Q: {:?}", q);
//!     println!("R: {:?}", r);
//!
//!     // ----- Naive backend -----
//!     let PRRLUDecomp { p, l, u, q, rank } = Naive.prrlu(&mut a.clone());
//!     println!("PRRLU decomposition done (Naive backend)");
//!     println!(                                                                                                 
//!         "p: {:?}, l: {:?}, u: {:?}, q: {:?}, rank: {:?}",                                                     
//!         p, l, u, q, rank                                                                                      
//!     );
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

mod solve;
pub use solve::*;
