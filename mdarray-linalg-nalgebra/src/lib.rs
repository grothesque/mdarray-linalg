//! # mdarray_linalg_nalgebra
//!
//! [nalgebra](https://crates.io/crates/nalgebra) backend for [`mdarray_linalg`].
//!
//! This crate provides the [`Nalgebra`] struct that implements the linear algebra traits
//! defined by [`mdarray_linalg`], delegating computations to the pure-Rust `nalgebra` library.
//! Nalgebra is particularly efficient for **small matrices** thanks to its extensive use of
//! compile-time dimension optimizations.
//!
//! Backend implementation modules are private.  Use [`Nalgebra`] together with
//! the operation traits from `mdarray_linalg::prelude::*`.
//!
//! ## Scope
//!
//! The Nalgebra backend covers:
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
//! - **Argmax** — `argmax`, `argmax_abs`
//!
//! ## Setup
//!
//! Add the dependencies to your project:
//!
//! ```bash
//! cargo add mdarray mdarray-linalg mdarray-linalg-nalgebra
//! ```
//!
//! ## Example
//!
//! All operations are accessed through the [`Nalgebra`] backend via the traits from
//! `mdarray_linalg::prelude::*`:
//!
//! ```rust
//! use mdarray::array;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg::eig::EigDecomp;
//! use mdarray_linalg::svd::SVDDecomp;
//! use mdarray_linalg_nalgebra::Nalgebra;
//!
//! // ----- Matrix multiplication -----
//! let a = array![[1., 2.], [3., 4.]];
//! let b = array![[5., 6.], [7., 8.]];
//!
//! let c = Nalgebra::default().matmul(&a, &b).eval();
//! assert_eq!(c, array![[19., 22.], [43., 50.]]);
//!
//! // ----- Eigenvalue decomposition -----
//! let mut a = array![[1., 2.], [3., 4.]];
//! let EigDecomp {
//!     eigenvalues: lambda,
//!     right_eigenvectors,
//!     ..
//! } = Nalgebra::default().eig(&mut a.clone()).expect("Eigenvalue decomposition failed");
//!
//! println!("Eigenvalues: {:?}", lambda);
//!
//! // ----- SVD -----
//! let mut a = array![[1., 2.], [3., 4.]];
//! let SVDDecomp { s, u, vt } = Nalgebra::default().svd(&mut a).expect("SVD failed");
//! println!("Singular values: {:?}", s);
//!
//! // ----- Argmax -----
//! let x = array![1., 5., 3., 8., 2.];
//! let idx = Nalgebra::default().argmax(&x).unwrap();
//! assert_eq!(idx, vec![3]);
//!
//! // ----- Tensor contraction -----
//! let t1 = array![[1., 2.], [3., 4.]].into_dyn();
//! let t2 = array![[5., 6.], [7., 8.]].into_dyn();
//!
//! let scalar = Nalgebra::default().contract_all(&t1, &t2);
//! assert_eq!(scalar, 70.0);
//! ```
//!
//! > **Note:** Decomposition routines (eig, svd, lu, etc.) **destroy the input matrix**.
//! > Always pass a clone if you need the original data.
//!
//! ## Supported types
//!
//! `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
//!
// Keep the doc-comment blank line above: these reference definitions must start
// a separate Markdown block from the preceding paragraph.
#![cfg_attr(docsrs, doc = concat!(
    "[`mdarray_linalg`]: https://docs.rs/mdarray-linalg/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg/",
))]
#![cfg_attr(not(docsrs), doc = "\
[`mdarray_linalg`]: ../mdarray_linalg/index.html
")]

mod eig;
mod lu;
mod contract;
mod matvec;
mod qr;
mod solve;
mod svd;

/// Nalgebra backend.
///
/// Implements the linear algebra traits from [`mdarray_linalg`] by delegating
/// to the pure-Rust `nalgebra` library.  Nalgebra is particularly well-suited
/// for **small matrices** where its compile-time dimension optimizations
/// provide excellent performance without the overhead of system BLAS/LAPACK.
#[derive(Default)]
pub struct Nalgebra;

use mdarray::{Dim, Layout, Shape, Slice};
use num_complex::{Complex, ComplexFloat};

/// Copy an mdarray matrix into a dense nalgebra matrix.
pub(crate) fn to_dmatrix<T, D0, D1, L>(a: &Slice<T, (D0, D1), L>) -> nalgebra::DMatrix<T>
where
    T: nalgebra::Scalar + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let rows = a.shape().dim(0);
    let cols = a.shape().dim(1);
    let mut data = Vec::with_capacity(rows * cols);

    // nalgebra stores dense matrices in column-major order.
    for j in 0..cols {
        for i in 0..rows {
            data.push(a[[i, j]]);
        }
    }

    nalgebra::DMatrix::from_vec(rows, cols, data)
}

/// Copy a dense nalgebra matrix back into an mdarray slice.
pub(crate) fn write_dmatrix<T, D0, D1, L>(src: &nalgebra::DMatrix<T>, dst: &mut Slice<T, (D0, D1), L>)
where
    T: nalgebra::Scalar + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    assert_eq!(src.nrows(), dst.shape().dim(0));
    assert_eq!(src.ncols(), dst.shape().dim(1));

    for i in 0..src.nrows() {
        for j in 0..src.ncols() {
            dst[[i, j]] = src[(i, j)];
        }
    }
}

/// Copy an mdarray matrix into a dense complex nalgebra matrix.
pub(crate) fn to_complex_dmatrix<T, D0, D1, L>(
    a: &Slice<T, (D0, D1), L>,
) -> nalgebra::DMatrix<Complex<T::Real>>
where
    T: ComplexFloat,
    T::Real: nalgebra::RealField + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let rows = a.shape().dim(0);
    let cols = a.shape().dim(1);
    let mut data = Vec::with_capacity(rows * cols);

    // nalgebra stores dense matrices in column-major order.
    for j in 0..cols {
        for i in 0..rows {
            data.push(Complex::new(a[[i, j]].re(), a[[i, j]].im()));
        }
    }

    nalgebra::DMatrix::from_vec(rows, cols, data)
}

/// Copy an mdarray slice into a dense nalgebra vector using logical iteration order.
pub(crate) fn to_dvector<T, S, L>(x: &Slice<T, S, L>) -> nalgebra::DVector<T>
where
    T: nalgebra::Scalar + Copy,
    S: Shape,
    L: Layout,
{
    nalgebra::DVector::from_iterator(x.len(), x.iter().copied())
}

/// Create a borrowed nalgebra vector view from an mdarray slice without copying.
///
/// # Panics
///
/// Panics if the slice has rank > 1 and is not contiguous, or if rank == 1 and
/// the stride is non-positive.
pub(crate) fn to_dvector_view<T, S, L>(
    x: &Slice<T, S, L>,
) -> nalgebra::DVectorView<'_, T, nalgebra::Dyn, nalgebra::Dyn>
where
    T: nalgebra::Scalar + Copy,
    S: Shape,
    L: Layout,
{
    let len = x.len();

    if len == 0 {
        let data: &[T] = &[];
        return nalgebra::MatrixView::from_slice_with_strides_generic(
            data,
            nalgebra::Dyn(0),
            nalgebra::Const::<1>,
            nalgebra::Dyn(1),
            nalgebra::Dyn(0),
        );
    }

    let rstride = if x.rank() == 1 {
        let stride = x.stride(0);
        assert!(stride > 0, "to_dvector_view: negative strides not supported");
        stride as usize
    } else {
        assert!(
            x.is_contiguous(),
            "to_dvector_view: non-contiguous multi-dimensional slices not supported"
        );
        1
    };

    let data_len = (len - 1) * rstride + 1;
    // SAFETY: x.as_ptr() points to the first element in logical order.
    // For rank-1, stride gives the spacing between consecutive logical elements,
    // so the accessed range spans data_len elements from the base pointer.
    // For rank>1 contiguous, the logical order matches memory order with stride 1.
    let data = unsafe { std::slice::from_raw_parts(x.as_ptr(), data_len) };

    nalgebra::MatrixView::from_slice_with_strides_generic(
        data,
        nalgebra::Dyn(len),
        nalgebra::Const::<1>,
        nalgebra::Dyn(rstride),
        nalgebra::Dyn(len),
    )
}

/// Copy a dense nalgebra vector back into an mdarray vector slice.
pub(crate) fn write_dvector<T, D1, L>(src: &nalgebra::DVector<T>, dst: &mut Slice<T, (D1,), L>)
where
    T: nalgebra::Scalar + Copy,
    D1: Dim,
    L: Layout,
{
    assert_eq!(src.len(), dst.len());

    for (dsti, srci) in dst.iter_mut().zip(src.iter()) {
        *dsti = *srci;
    }
}

/// Copy a dense complex nalgebra matrix back into an mdarray slice.
pub(crate) fn write_complex_dmatrix<R, D0, D1, L>(
    src: &nalgebra::DMatrix<Complex<R>>,
    dst: &mut Slice<Complex<R>, (D0, D1), L>,
) where
    R: nalgebra::RealField + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    assert_eq!(src.nrows(), dst.shape().dim(0));
    assert_eq!(src.ncols(), dst.shape().dim(1));

    for i in 0..src.nrows() {
        for j in 0..src.ncols() {
            dst[[i, j]] = src[(i, j)];
        }
    }
}

/// Copy a dense complex nalgebra vector back into an mdarray slice.
pub(crate) fn write_complex_dvector<R, D1, L>(
    src: &nalgebra::DVector<Complex<R>>,
    dst: &mut Slice<Complex<R>, (D1,), L>,
) where
    R: nalgebra::RealField + Copy,
    D1: Dim,
    L: Layout,
{
    assert_eq!(src.len(), dst.len());

    for (dsti, srci) in dst.iter_mut().zip(src.iter()) {
        *dsti = *srci;
    }
}
