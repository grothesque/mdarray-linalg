//! Linear system solving utilities for equations of the form Ax = B.
use mdarray::{DSlice, DTensor, Layout};
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
pub struct SolveResult<T> {
    pub x: DTensor<T, 2>,
    pub p: DTensor<T, 2>,
}

/// Result type for linear system solving, returning either a
/// `SolveResult` or a `SolveError`
pub type SolveResultType<T> = Result<SolveResult<T>, SolveError>;

/// Linear system solver using LU decomposition
pub trait Solve<T> {
    /// Solves linear system AX = b overwriting existing matrices
    /// A is overwritten with its LU decomposition
    /// B is overwritten with the solution X
    /// P is filled with the permutation matrix such that A = P*L*U
    /// Returns Ok(()) on success, Err(SolveError) on failure
    fn solve_overwrite<La: Layout, Lb: Layout, Lp: Layout>(
        &self,
        a: &mut DSlice<T, 2, La>,
        b: &mut DSlice<T, 2, Lb>,
        p: &mut DSlice<T, 2, Lp>,
    ) -> Result<(), SolveError>;

    /// Solves linear system AX = B with new allocated solution matrix
    /// A is modified (overwritten with LU decomposition)
    /// Returns the solution X and P the permutation matrix, or error
    fn solve<La: Layout, Lb: Layout>(
        &self,
        a: &mut DSlice<T, 2, La>,
        b: &DSlice<T, 2, Lb>,
    ) -> SolveResultType<T>;
}
