//! Linear system solving utilities for equations of the form Ax = B
//!```rust, ignore
//!use mdarray_linalg_backend::Backend; // Use the real backend here, Lapack, Faer, ...
//!let bd = Backend::default();
//!use mdarray_linalg::solve::Solve;
//!
//!let a = darray![[2.0_f64, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
//!let b = darray![[1.0_f64], [2.0], [1.0]];
//!
//!let xr = Lapack::default().solve(&mut a.clone(), &b);
//!
//!let SolveResult { x, .. } = xr.unwrap();
//!let ax = Naive.matvec(&a, &x.view(.., 0)).eval(); // Ax = b
//!```
use mdarray::{Array, Dim, Layout, Slice};
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
    pub x: Array<T, (D0, D1)>,
    pub p: Array<T, (D0, D1)>,
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
