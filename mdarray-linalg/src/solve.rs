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
//!let x = xr.unwrap();
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

/// Linear system solver.
pub trait Solve<T, D: Dim> {
    /// Solves linear system AX = B, overwriting B with the solution X.
    ///
    /// The backend may also overwrite A with intermediate factorization data;
    /// callers should not rely on A's contents after this call.
    fn solve_write<R: Dim, La: Layout, Lb: Layout>(
        &self,
        a: &mut Slice<T, (D, D), La>,
        b: &mut Slice<T, (D, R), Lb>,
    ) -> Result<(), SolveError>;

    /// Solves linear system AX = B with a newly allocated solution matrix.
    ///
    /// The backend may overwrite A with intermediate factorization data;
    /// callers should not rely on A's contents after this call.
    fn solve<R: Dim, La: Layout, Lb: Layout>(
        &self,
        a: &mut Slice<T, (D, D), La>,
        b: &Slice<T, (D, R), Lb>,
    ) -> Result<Array<T, (D, R)>, SolveError>;
}
