//! # mdarray-linalg-faer
//!
//! [faer](https://crates.io/crates/faer) backend for [`mdarray_linalg`].
//!
//! This crate provides the [`Faer`] backend that implements the linear algebra traits
//! defined by `mdarray-linalg`, delegating computations to the pure-Rust `faer` library.
//!
//! ## Scope
//!
//! The Faer backend covers:
//!
//! - **Level 1** — vector operations: `dot`, `dotc`, `norm2`, `norm1`, `add_to_scaled`
//! - **Level 2** — matrix-vector & outer product: `matvec`, `outer`
//! - **Level 3** — matrix multiplication: `matmul`
//! - **Tensor contraction** — `contract_all`, `contract_n`, `contract_pairs`, `contract`
//! - **Eigenvalue decomposition** — `eig`, `eig_full`, `eig_values`, `eigh`, `eigs`
//! - **Schur decomposition** — `schur`, `schur_complex`
//! - **SVD** — `svd`, `svd_thin`, `svd_s`
//! - **LU decomposition** — `lu`, `det`, `inv`
//! - **Cholesky decomposition** — `choleski`
//! - **QR decomposition** — `qr`
//! - **Linear system solving** — `solve`
//!
//!
//! ## Example
//!
//! All operations are accessed through the [`Faer`] backend via the traits from
//! `mdarray_linalg::prelude::*`:
//!
//! ```rust
//! use mdarray::array;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg::eig::EigDecomp;
//! use mdarray_linalg::svd::SVDDecomp;
//! use mdarray_linalg_faer::Faer;
//!
//! // ----- Matrix multiplication (Level 3) -----
//! let a = array![[1., 2.], [3., 4.]];
//! let b = array![[5., 6.], [7., 8.]];
//!
//! let c = Faer::default().matmul(&a, &b).eval();
//! assert_eq!(c, array![[19., 22.], [43., 50.]]);
//!
//! // ----- Eigenvalue decomposition -----
//! let mut a = array![[1., 2.], [3., 4.]];
//! let EigDecomp {
//!     eigenvalues: lambda,
//!     right_eigenvectors,
//!     ..
//! } = Faer::default().eig(&mut a.clone()).expect("Eigenvalue decomposition failed");
//!
//! println!("Eigenvalues: {:?}", lambda);
//! if let Some(v) = right_eigenvectors {
//!     println!("Right eigenvectors: {:?}", v);
//! }
//!
//! // ----- SVD -----
//! let mut a = array![[1., 2.], [3., 4.]];
//! let SVDDecomp { s, u, vt } = Faer::default().svd_thin(&mut a).expect("SVD failed");
//! println!("Singular values: {:?}", s);
//!
//! // ----- QR decomposition -----
//! let mut a = array![[12., -51., 4.], [6., 167., -68.], [-4., 24., -41.]];
//! let (q, r) = Faer::default().qr(&mut a);
//! println!("Q: {:?}", q);
//! println!("R: {:?}", r);
//!
//! // ----- Tensor contraction -----
//! let t1 = array![[1., 2.], [3., 4.]].into_dyn();
//! let t2 = array![[5., 6.], [7., 8.]].into_dyn();
//!
//! let scalar = Faer::default().contract_all(&t1, &t2);
//! assert_eq!(scalar, 70.0);
//! ```
//!
//! > **Note:** Decomposition routines (eig, svd, lu, etc.) **destroy the input matrix**.
//! > Always pass a clone if you need the original data.
//!
//! ## Supported types
//!
//! `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.

// #![cfg_attr(docsrs, doc = concat!(
//     "[mdarray_linalg]: https://docs.rs/mdarray-linalg/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg/",
// ))]
// #![cfg_attr(
//     not(docsrs),
//     doc = "\
// [mdarray_linalg]: ../mdarray_linalg/index.html
// "
// )]

pub mod eig;
pub mod lu;
pub mod matmul;
pub mod matvec;
pub mod qr;
pub mod solve;
pub mod svd;

/// Configuration for the QR decomposition.
#[derive(Default, Debug, Clone, Copy)]
pub enum QRConfig {
    /// Reduced QR: Q is M×K, R is K×N (where K = min(M, N)).
    #[default]
    Reduced,
    /// Complete QR: Q is M×M, R is M×N.
    Complete,
}

/// Faer backend.
///
/// Implements the linear algebra traits from [`mdarray_linalg`] by delegating
/// to the pure-Rust `faer` library.  This backend supports the broadest range of
/// operations — from basic BLAS to full decompositions and tensor contractions —
/// without requiring any system BLAS/LAPACK installation.
///
/// By default, multithreading is enabled (via `rayon`).
pub struct Faer {
    parallelize: bool,
    qr_config: QRConfig,
}

impl Default for Faer {
    fn default() -> Self {
        Self {
            parallelize: true,
            qr_config: QRConfig::Reduced,
        }
    }
}

use mdarray::{Dim, Layout, Shape, Slice};

/// Converts a `Slice<T, (_, _), L>` (from `mdarray`) into a `faer::MatRef<'a, T>`.
/// This function **does not copy** any data.
pub fn into_faer<'a, T, L: Layout, D0: Dim, D1: Dim>(
    mat: &'a Slice<T, (D0, D1), L>,
) -> faer::mat::MatRef<'a, T> {
    let (nrows, ncols) = *mat.shape();
    let strides = (mat.stride(0), mat.stride(1));

    // SAFETY:
    // We are constructing a MatRef from raw parts. This requires that:
    // - `mat.as_ptr()` points to a valid matrix of size `nrows x ncols`
    // - The given strides correctly describe the memory layout
    unsafe {
        faer::MatRef::from_raw_parts(
            mat.as_ptr(),
            nrows.size(),
            ncols.size(),
            strides.0,
            strides.1,
        )
    }
}

/// Converts a `Slice<T, (_, _), L>` (from `mdarray`) into a `faer::MatMut<'a, T>`.
/// This function **does not copy** any data.
pub fn into_faer_mut<'a, T, L: Layout, D0: Dim, D1: Dim>(
    mat: &'a mut Slice<T, (D0, D1), L>,
) -> faer::mat::MatMut<'a, T> {
    let (nrows, ncols) = *mat.shape();
    let strides = (mat.stride(0), mat.stride(1));

    // SAFETY:
    // We are constructing a MatMut from raw parts. This requires that:
    // - `mat.as_mut_ptr()` points to a valid mutable matrix of size `nrows x ncols`
    // - The given strides correctly describe the memory layout
    unsafe {
        faer::MatMut::from_raw_parts_mut(
            mat.as_mut_ptr() as *mut _,
            nrows.size(),
            ncols.size(),
            strides.0,
            strides.1,
        )
    }
}

/// Converts a `Slice<T, (D0, D1), L>` (from `mdarray`) into a
/// `faer::MatMut<'a, T>` and transposes data.  This function
/// **does not copy** any data.
pub fn into_faer_mut_transpose<'a, T, D0: Dim, D1: Dim, L: Layout>(
    mat: &'a mut Slice<T, (D0, D1), L>,
) -> faer::mat::MatMut<'a, T> {
    let matsh = *mat.shape();
    let (nrows, ncols) = (matsh.dim(0), matsh.dim(1));
    let strides = (mat.stride(0), mat.stride(1));

    // SAFETY:
    // We are constructing a MatMut from raw parts. This requires that:
    // - `mat.as_mut_ptr()` points to a valid mutable matrix of size `nrows x ncols`
    // - The given strides correctly describe the memory layout
    unsafe {
        faer::MatMut::from_raw_parts_mut(
            mat.as_mut_ptr() as *mut _,
            nrows,
            ncols,
            strides.1,
            strides.0,
        )
    }
}

/// Converts a `Slice<T, (D0,), L>` (from `mdarray`) into a `faer::ColRef<'a, T>`.
/// This function **does not copy** any data.
pub fn into_faer_col<'a, T, D0: Dim, L: Layout>(
    vec: &'a Slice<T, (D0,), L>,
) -> faer::col::ColRef<'a, T> {
    let n = vec.shape().dim(0);

    // SAFETY:
    // - `vec.as_ptr()` points to a valid vector with `n` elements.
    // - `vec.stride(0)` describes the spacing between consecutive elements.
    unsafe { faer::col::ColRef::from_raw_parts(vec.as_ptr(), n, vec.stride(0)) }
}

/// Converts a `Slice<T, (D0,), L>` (from `mdarray`) into a `faer::ColMut<'a, T>`.
/// This function **does not copy** any data.
pub fn into_faer_col_mut<'a, T, D0: Dim, L: Layout>(
    vec: &'a mut Slice<T, (D0,), L>,
) -> faer::col::ColMut<'a, T> {
    let n = vec.shape().dim(0);

    // SAFETY:
    // - `vec.as_mut_ptr()` points to a valid mutable vector with `n` elements.
    // - `vec.stride(0)` describes the spacing between consecutive elements.
    unsafe { faer::col::ColMut::from_raw_parts_mut(vec.as_mut_ptr() as *mut _, n, vec.stride(0)) }
}

/// Converts a `Slice<T, (D0,), L>` (from `mdarray`) into a `faer::RowRef<'a, T>`.
/// This function **does not copy** any data.
pub fn into_faer_row<'a, T, D0: Dim, L: Layout>(
    vec: &'a Slice<T, (D0,), L>,
) -> faer::row::RowRef<'a, T> {
    let n = vec.shape().dim(0);

    // SAFETY:
    // - `vec.as_ptr()` points to a valid vector with `n` elements.
    // - `vec.stride(0)` describes the spacing between consecutive elements.
    unsafe { faer::row::RowRef::from_raw_parts(vec.as_ptr(), n, vec.stride(0)) }
}

/// Converts a mutable `Slice<T, (D0, D1), L>` (from `mdarray`) into a `faer::diag::DiagMut<'a, T>`,
/// which is a mutable view over the diagonal elements of a matrix in Faer.
///
/// # Important Notes for Users:
/// - This function **does not copy** any data. It gives direct mutable access to
///   the diagonal values of the matrix represented by `mat`.
/// - The stride along the **Y-axis (i.e., column stride)** is chosen to be consistent
///   with LAPACK-style storage, where singular values are typically stored in the first row.
/// - This function is unsafe internally and assumes that `mat` contains at least `n` elements
///   in memory laid out consistently with the given stride.
pub fn into_faer_diag_mut<'a, T, D0: Dim, L: Layout>(
    mat: &'a mut Slice<T, (D0,), L>,
) -> faer::diag::DiagMut<'a, T> {
    let n = mat.shape().dim(0);

    // SAFETY:
    // - `mat.as_mut_ptr()` must point to a buffer with at least `n` diagonal elements.
    // - `mat.stride(1)` is used as the step between diagonal elements, assuming storage
    //   along the first row for compatibility with LAPACK convention.
    unsafe { faer::diag::DiagMut::from_raw_parts_mut(mat.as_mut_ptr() as *mut _, n, mat.stride(0)) }
}
