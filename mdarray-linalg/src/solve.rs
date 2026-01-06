//! Linear system solving utilities for equations of the form Ax = B
use mdarray::{Dim, Layout, Slice, Tensor};
use thiserror::Error;

/// Error types related to linear system solving
#[derive(Debug, Error)]
pub enum SolveError {
    #[error("Backend error code: {0}")]
    BackendError(i32),

    #[error("Matrix is singular: U({diagonal},{diagonal}) is exactly zero")]
    SingularMatrix { diagonal: i32 },

    #[error("Invalid matrix dimensions")]
    InvalidDimensions,
}

/// Holds the results of a linear system solve, including
/// the solution matrix and permutation matrix
pub struct SolveResult<T, D0: Dim, D1: Dim> {
    pub x: Tensor<T, (D0, D1)>,
    pub p: Tensor<T, (D0, D1)>,
}

/// Result type for linear system solving, returning either a
/// `SolveResult` or a `SolveError`
pub type SolveResultType<T, D0, D1> = Result<SolveResult<T, D0, D1>, SolveError>;

/// Linear system solver using LU decomposition
pub trait Solve<T, D0: Dim, D1: Dim> {
    /// Solves linear system AX = b overwriting existing matrices
    /// A is overwritten with its LU decomposition
    /// B is overwritten with the solution X
    /// P is filled with the permutation matrix such that A = P*L*U
    /// Returns Ok(()) on success, Err(SolveError) on failure
    fn solve_write<La: Layout, Lb: Layout, Lp: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), La>,
        b: &mut Slice<T, (D0, D1), Lb>,
        p: &mut Slice<T, (D0, D1), Lp>,
    ) -> Result<(), SolveError>;

    /// Solves linear system AX = B with new allocated solution matrix
    /// A is modified (overwritten with LU decomposition)
    /// Returns the solution X and P the permutation matrix, or error
    fn solve<La: Layout, Lb: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), La>,
        b: &Slice<T, (D0, D1), Lb>,
    ) -> SolveResultType<T, D0, D1>;
}
