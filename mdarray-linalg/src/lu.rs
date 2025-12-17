//! LU, Cholesky, matrix inversion, and determinant computation utilities
use mdarray::{DSlice, DTensor, Dim, Layout, Slice, Tensor};
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

/// Result type for matrix inversion
pub type InvResult<T, D0: Dim, D1: Dim> = Result<Tensor<T, (D0, D1)>, InvError>;

///  LU decomposition and matrix inversion
pub trait LU<T, D0: Dim, D1: Dim> {
    /// Computes LU decomposition overwriting existing matrices
    fn lu_write<L: Layout, Ll: Layout, Lu: Layout, Lp: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        l: &mut DSlice<T, 2, Ll>,
        u: &mut DSlice<T, 2, Lu>,
        p: &mut DSlice<T, 2, Lp>,
    );

    /// Computes LU decomposition with new allocated matrices: L, U, P (permutation matrix)
    fn lu<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (DTensor<T, 2>, DTensor<T, 2>, DTensor<T, 2>);

    /// Computes inverse overwriting the input matrix
    fn inv_write<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<(), InvError>;

    /// Computes inverse with new allocated matrix
    fn inv<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> InvResult<T, D0, D1>;

    /// Computes the determinant of a square matrix. Panics if the
    /// matrix is non-square.
    fn det<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> T;

    /// Computes the Cholesky decomposition, returning a lower-triangular matrix
    fn choleski<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> InvResult<T, D0, D1>;

    /// Computes the Cholesky decomposition in-place, overwriting the input matrix
    fn choleski_write<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<(), InvError>;
}
