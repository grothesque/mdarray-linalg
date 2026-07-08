//! # mdarray_linalg_blas
//!
//! BLAS backend for [`mdarray_linalg`].
//!
//! This crate provides the [`Blas`] struct that implements the linear algebra traits
//! defined by [`mdarray_linalg`], delegating computations to a BLAS implementation
//! (e.g. OpenBLAS) via the `cblas-sys` crate.
//!
//! ## Scope
//!
//! The BLAS backend covers:
//!
//! - **Level 1** — vector operations: `dot`, `dotc`, `norm2`, `norm1`, `add_to_scaled`
//! - **Level 2** — matrix-vector & outer product: `matvec`, `outer`
//! - **Level 3** — matrix multiplication: `matmul`
//! - **Tensor contraction** — `contract_all`, `contract_n`, `contract_pairs`, `contract`
//! - **Argmax** — `argmax`, `argmax_abs`
//!
//! For decompositions (Eig, SVD, LU, QR, Cholesky, Schur) and solving linear systems,
//! use the [`mdarray_linalg_lapack`] or [`mdarray_linalg_faer`] backends instead.
//!
//! ## Setup
//!
//! This crate binds to the CBLAS ABI but does not choose a native BLAS library to link against.
//! This is left to the user. For example, to use a system OpenBLAS installation:
//!
//! ```bash
//! cargo add mdarray mdarray-linalg mdarray-linalg-blas
//! cargo add openblas-src --features system
//! ```
//!
//! In one of your Rust crates, reference the CBLAS provider so its link directives are included:
//!
//! ```rust
//! extern crate openblas_src as _;
//! ```
//!
//! Other BLAS providers may be used if they expose the CBLAS symbols required by
//! `cblas-sys`.
//!
//! ## Example
//!
//! All operations are accessed through the [`Blas`] backend via the traits from
//! `mdarray_linalg::prelude::*`:
//!
//! ```rust
//! # extern crate openblas_src as _;
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
//! let contracted = Blas.contract_n(&t1, &t2, 1).eval();
//! assert_eq!(contracted, array![[19., 22.], [43., 50.]].into_dyn());
//! ```
//!
//! ## Supported types
//!
//! `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
//!
//! ## Troubleshooting
//!
//! Linking errors usually mean that no BLAS library was linked into the final
//! binary, or that the selected library is not in the linker/runtime search
//! path.  Add a source crate such as `openblas-src`, reference it from Rust code,
//! or provide equivalent link flags from your application `build.rs`.
//!
// Keep the doc-comment blank line above: these reference definitions must start
// a separate Markdown block from the preceding paragraph.
#![cfg_attr(docsrs, doc = concat!(
    "[`mdarray_linalg`]: https://docs.rs/mdarray-linalg/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg/\n",
    "[`mdarray_linalg_lapack`]: https://docs.rs/mdarray-linalg-lapack/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_lapack/\n",
    "[`mdarray_linalg_faer`]: https://docs.rs/mdarray-linalg-faer/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg_faer/",
))]
#![cfg_attr(not(docsrs), doc = "\
[`mdarray_linalg`]: ../mdarray_linalg/index.html
[`mdarray_linalg_lapack`]: ../mdarray_linalg_lapack/index.html
[`mdarray_linalg_faer`]: ../mdarray_linalg_faer/index.html
")]

#[cfg(test)]
extern crate openblas_src as _;

mod contract;
mod matvec;

/// BLAS backend.
///
/// Implements the linear algebra traits from [`mdarray_linalg`] by delegating
/// to BLAS routines.  The struct is a zero-sized marker — all state is managed
/// by the underlying BLAS library.
#[derive(Default)]
pub struct Blas;

pub(crate) fn trans_stride<T, D0, D1, L>(
    x: &mdarray::Slice<T, (D0, D1), L>,
    same_order: cblas_sys::CBLAS_TRANSPOSE,
    other_order: cblas_sys::CBLAS_TRANSPOSE,
) -> (cblas_sys::CBLAS_TRANSPOSE, i32)
where
    D0: mdarray::Dim,
    D1: mdarray::Dim,
    L: mdarray::Layout,
{
    if x.stride(1) == 1 {
        (same_order, mdarray_linalg::utils::into_i32(x.stride(0)))
    } else {
        assert!(
            x.stride(0) == 1,
            "matrix must be contiguous in one dimension"
        );
        (other_order, mdarray_linalg::utils::into_i32(x.stride(1)))
    }
}
