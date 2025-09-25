use mdarray::{DSlice, DTensor, Layout};
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

    /// The matrix is singular: U(i,i) is exactly zero
    #[error("Matrix is singular: zero pivot at position {pivot}")]
    Singular { pivot: i32 },
}

/// Result type for matrix inversion
pub type InvResult<T> = Result<DTensor<T, 2>, InvError>;

///  LU decomposition and matrix inversion
pub trait LU<T> {
    /// Computes LU decomposition overwriting existing matrices
    fn lu_overwrite<L: Layout, Ll: Layout, Lu: Layout, Lp: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        l: &mut DSlice<T, 2, Ll>,
        u: &mut DSlice<T, 2, Lu>,
        p: &mut DSlice<T, 2, Lp>,
    );

    /// Computes LU decomposition with new allocated matrices: L, U, P (permutation matrix)
    fn lu<L: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
    ) -> (DTensor<T, 2>, DTensor<T, 2>, DTensor<T, 2>);

    /// Computes inverse overwriting the input matrix
    fn inv_overwrite<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> Result<(), InvError>;

    /// Computes inverse with new allocated matrix
    fn inv<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> InvResult<T>;
}
