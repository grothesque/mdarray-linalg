//! LU, Cholesky, matrix inversion, and determinant computation utilities
//!
//! ```rust,ignore
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg_backend::Backend;
//!
//! let bd = Backend::default();
//!
//! let a = Array::from_fn([3, 3], |i| {
//!     (i[0] + 1) as f64 + 2. * (i[1] + 1) as f64 + if i[0] == i[1] { 3. } else { 0. }
//! }); // invertible matrix
//!
//! // ----- LU decomposition -----
//! // P * A = L * U  where P is a permutation matrix.
//! let (l, u, p) = bd.lu(&mut a.clone());
//!
//! // ----- Determinant and inverse -----
//! let d = bd.det(&mut a.clone());
//! let a_inv = bd.inv(&mut a.clone()).expect("Can't compute inverse");
//!
//! // ----- Cholesky decomposition -----
//! // For a symmetric positive-definite matrix: A = L * L^T
//! let s = a.clone() + a.permute([1, 0]); // symmetric matrix
//! let l = bd.cholesky(&mut s).unwrap();
//! // Reconstruct: A ≈ L * L^T
//! let a_reconstructed = l.dot(&l.transpose());
//! ```

use mdarray::{Array, Dim, Layout, Slice};
use thiserror::Error;

/// Error types related to matrix inversion
#[derive(Debug, Error)]
pub enum InvError {
    /// The input or output matrix is not square
    #[error("Matrix must be square: got {rows}x{cols}")]
    NotSquare { rows: i32, cols: i32 },

    /// Backend returned a non-zero error code
    #[error("Backend error code: {0}")]
    BackendError(i32),

    /// Matrix is singular: U(i,i) is exactly zero
    #[error("Matrix is singular: zero pivot at position {pivot}")]
    Singular { pivot: i32 },

    /// The leading principal minor is not positive (Cholesky decomp)
    #[error("The leading principal minor is not positive")]
    NotPositiveDefinite { lpm: i32 },
}

///  LU decomposition and matrix inversion
pub trait LU<T, D0: Dim, D1: Dim> {
    /// Computes LU decomposition overwriting existing matrices
    fn lu_write<L: Layout, Ll: Layout, Lu: Layout, Lp: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        l: &mut Slice<T, (D0, D0), Ll>,
        u: &mut Slice<T, (D0, D1), Lu>,
        p: &mut Slice<T, (D0, D0), Lp>,
    );

    /// Computes LU decomposition with new allocated matrices: L, U, P (permutation matrix)
    fn lu<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Array<T, (D0, D0)>, Array<T, (D0, D1)>, Array<T, (D0, D0)>);

    /// Computes inverse overwriting the input matrix
    fn inv_write<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<(), InvError>;

    /// Computes inverse with new allocated matrix
    fn inv<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> Result<Array<T, (D0, D1)>, InvError>;

    /// Computes the determinant of a square matrix. Panics if the
    /// matrix is non-square.
    fn det<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> T;

    /// Computes the Cholesky decomposition, returning a lower-triangular matrix
    fn cholesky<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> Result<Array<T, (D0, D1)>, InvError>;

    /// Computes the Cholesky decomposition in-place, overwriting the input matrix
    fn cholesky_write<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<(), InvError>;
}
