//! # mdarray-linalg-blas
//!
//! BLAS backend for [`mdarray_linalg`].
//!
//! This crate provides the [`Blas`] struct that implements the linear algebra traits
//! defined by `mdarray-linalg`, delegating computations to a BLAS implementation
//! (e.g. OpenBLAS) via the `cblas-sys` crate.
//!
//! ## Scope
//!
//! The BLAS backend covers:
//!
//! - **Level 1** — vector operations: `dot`, `dotc`, `norm2`, `norm1`, `add_to_scaled`
//! - **Level 2** — matrix-vector & outer product: `matvec`, `outer`
//! - **Level 3** — matrix multiplication: `matmul`, `symm`, `hemm`, `trmm`
//! - **Tensor contraction** — `contract_all`, `contract_n`, `contract_pairs`, `contract`
//! - **Argmax** — `argmax`, `argmax_abs`
//!
//! For decompositions (Eig, SVD, LU, QR, Cholesky, Schur) and solving linear systems,
//! use the `mdarray-linalg-lapack` or `mdarray-linalg-faer` backends instead.
//!
//!
//! ## Example
//!
//! All operations are accessed through the [`Blas`] backend via the traits from
//! `mdarray_linalg::prelude::*`:
//!
//! ```rust
//! use mdarray::array;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg_blas::Blas;
//!
//! // ----- Vector operations (Level 1) -----
//! let x = array![1., 2., 3.];
//! let y = array![4., 5., 6.];
//!
//! let d = Blas.dot(&x, &y);
//! assert_eq!(d, 32.0);  // 1·4 + 2·5 + 3·6
//!
//! // ----- Matrix-vector multiplication (Level 2) -----
//! let a = array![[1., 2., 3.], [4., 5., 6.]];
//! let v = array![1., 1., 1.];
//!
//! let av = Blas.matvec(&a, &v).eval();
//! assert_eq!(av, array![6., 15.]);  // A·v
//!
//! // ----- Matrix multiplication (Level 3) -----
//! let b = array![[1., 2.], [3., 4.], [5., 6.]];
//!
//! let c = Blas.matmul(&a, &b).eval();
//! assert_eq!(c, array![[22., 28.], [49., 64.]]);  // (2×3)·(3×2) = (2×2)
//!
//! // Scaled addition: C = α·A·B + β·C
//! let mut c = array![[1., 1.], [1., 1.]];
//! Blas.matmul(&a, &b).add_to_scaled(&mut c, 2.0);
//!
//! // ----- Tensor contraction -----
//! let t1 = array![[1., 2.], [3., 4.]].into_dyn();
//! let t2 = array![[5., 6.], [7., 8.]].into_dyn();
//!
//! // Full contraction over all axes
//! let scalar = Blas.contract_all(&t1, &t2);
//! assert_eq!(scalar, 70.0);  // 1·5 + 2·6 + 3·7 + 4·8
//!
//! // Contract last n axes (n=1 → standard matmul)
//! let tensordot = Blas.contract_n(&t1, &t2, 1).eval();
//! assert_eq!(tensordot, array![[19., 22.], [43., 50.]].into_dyn());
//! ```
//!
//! ## Supported types
//!
//! `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
//!
//! ## Troubleshooting
//!
//! If you encounter linking issues, the included `build.rs` links against
//! `libopenblas` in `/usr/lib`.  For other BLAS implementations or custom
//! install paths, you may need to override `build.rs`.
//! See also the `mdarray-linalg` crate documentation.

// #![cfg_attr(docsrs, doc = concat!(
//     "[mdarray_linalg]: https://docs.rs/mdarray-linalg/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg/",
// ))]
// #![cfg_attr(
//     not(docsrs),
//     doc = "\
// [mdarray_linalg]: ../mdarray_linalg/index.html
// "
// )]

pub mod matmul;
pub mod matvec;

/// BLAS backend.
///
/// Implements the linear algebra traits from [`mdarray_linalg`] by delegating
/// to BLAS routines.  The struct is a zero-sized marker — all state is managed
/// by the underlying BLAS library.
#[derive(Default)]
pub struct Blas;
