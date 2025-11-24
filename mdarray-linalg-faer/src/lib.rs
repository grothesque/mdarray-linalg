//! ```rust
//! use mdarray::{DTensor, tensor};
//! use mdarray_linalg::prelude::*; // Imports traits anonymously
//! use mdarray_linalg::eig::EigDecomp;
//! use mdarray_linalg::svd::SVDDecomp;
//!
//! use mdarray_linalg_faer::Faer;
//!
//! // Declare two matrices
//! let a = tensor![[1., 2.], [3., 4.]];
//! let b = tensor![[5., 6.], [7., 8.]];
//!
//! // ----- Matrix multiplication -----
//! let c = Faer.matmul(&a, &b).eval();
//! println!("A * B = {:?}", c);
//!
//! // ----- Eigenvalue decomposition -----
//! // Note: we must clone `a` here because decomposition routines destroy the input.
//! let bd = Faer;
//! let EigDecomp {
//!     eigenvalues,
//!     right_eigenvectors,
//!     ..
//! } = bd.eig(&mut a.clone()).expect("Eigenvalue decomposition failed");
//!
//! println!("Eigenvalues: {:?}", eigenvalues);
//! if let Some(vectors) = right_eigenvectors {
//!     println!("Right eigenvectors: {:?}", vectors);
//! }
//!
//! // ----- Singular Value Decomposition (SVD) -----
//! let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");
//! println!("Singular values: {:?}", s);
//! println!("Left singular vectors U: {:?}", u);
//! println!("Right singular vectors V^T: {:?}", vt);
//!
//! // ----- QR Decomposition -----
//! let (m, n) = *a.shape();
//! let mut q = DTensor::<f64, 2>::zeros([m, m]);
//! let mut r = DTensor::<f64, 2>::zeros([m, n]);
//!
//! bd.qr_write(&mut a.clone(), &mut q, &mut r); //
//! println!("Q: {:?}", q);
//! println!("R: {:?}", r);
//! ```

pub mod eig;
pub mod lu;
pub mod matmul;
pub mod qr;
pub mod solve;
pub mod svd;

#[derive(Default)]
pub struct Faer;

use std::mem::ManuallyDrop;

use mdarray::{DSlice, DTensor, Layout, Strided, StridedMapping, View};

/// Converts a `DSlice<T, 2, L>` (from `mdarray`) into a `faer::MatRef<'static, T>`.
/// This function **does not copy** any data.
pub fn into_faer<T, L: Layout>(mat: &DSlice<T, 2, L>) -> faer::mat::MatRef<'static, T> {
    let (nrows, ncols) = *mat.shape();
    let strides = (mat.stride(0), mat.stride(1));

    // SAFETY:
    // We are constructing a MatRef from raw parts. This requires that:
    // - `mat.as_ptr()` points to a valid matrix of size `nrows x ncols`
    // - The given strides correctly describe the memory layout
    unsafe { faer::MatRef::from_raw_parts(mat.as_ptr(), nrows, ncols, strides.0, strides.1) }
}

/// Converts a `DSlice<T, 2, L>` (from `mdarray`) into a `faer::MatMut<'static, T>`.
/// This function **does not copy** any data.
pub fn into_faer_mut<T, L: Layout>(mat: &mut DSlice<T, 2, L>) -> faer::mat::MatMut<'static, T> {
    let (nrows, ncols) = *mat.shape();
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
            strides.0,
            strides.1,
        )
    }
}

/// Converts a `faer::Mat<T>` into a `DTensor<T, 2>` (from `mdarray`) by constructing
/// a strided view over the matrix memory. This function **does not copy** any data.
pub fn into_mdarray<T: std::clone::Clone>(mat: faer::Mat<T>) -> DTensor<T, 2> {
    // Manually dropping to avoid a double free: DTensor will take ownership of the data,
    // so we must prevent Rust from automatically dropping the original matrix.
    let mut mat = ManuallyDrop::new(mat);

    let (nrows, ncols) = (mat.nrows(), mat.ncols());

    // faer and mdarray have different memory layouts; we need to construct a
    // strided mapping explicitly to describe the layout of `mat` to mdarray.
    let mapping = StridedMapping::new((nrows, ncols), &[mat.row_stride(), mat.col_stride()]);

    // SAFETY:
    // We use `new_unchecked` because the memory layout in faer isn't guaranteed
    // to satisfy mdarray's internal invariants automatically.
    // `from_raw_parts` isn't usable here due to layout incompatibilities.
    let view_strided: View<'_, _, (usize, usize), Strided> =
        unsafe { mdarray::View::new_unchecked(mat.as_ptr_mut(), mapping) };

    DTensor::<T, 2>::from(view_strided)
}

/// Converts a `DSlice<T, 2, L>` (from `mdarray`) into a
/// `faer::MatMut<'static, T>` and transposes data.  This function
/// **does not copy** any data.
pub fn into_faer_mut_transpose<T, L: Layout>(
    mat: &mut DSlice<T, 2, L>,
) -> faer::mat::MatMut<'static, T> {
    let (nrows, ncols) = *mat.shape();
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

/// Converts a mutable `DSlice<T, 2, L>` (from `mdarray`) into a `faer::diag::DiagMut<'static, T>`,
/// which is a mutable view over the diagonal elements of a matrix in Faer.
///
/// # Important Notes for Users:
/// - This function **does not copy** any data. It gives direct mutable access to
///   the diagonal values of the matrix represented by `mat`.
/// - The stride along the **Y-axis (i.e., column stride)** is chosen to be consistent
///   with LAPACK-style storage, where singular values are typically stored in the first row.
/// - This function is unsafe internally and assumes that `mat` contains at least `n` elements
///   in memory laid out consistently with the given stride.
pub fn into_faer_diag_mut<T, L: Layout>(
    mat: &mut DSlice<T, 2, L>,
) -> faer::diag::DiagMut<'static, T> {
    let (n, _) = *mat.shape();

    // SAFETY:
    // - `mat.as_mut_ptr()` must point to a buffer with at least `n` diagonal elements.
    // - `mat.stride(1)` is used as the step between diagonal elements, assuming storage
    //   along the first row for compatibility with LAPACK convention.
    unsafe { faer::diag::DiagMut::from_raw_parts_mut(mat.as_mut_ptr() as *mut _, n, mat.stride(1)) }
}
