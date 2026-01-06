//! Singular Value Decomposition (SVD)
use mdarray::{Dim, Layout, Slice, Tensor};
use thiserror::Error;

/// Error types related to singular value decomposition
#[derive(Debug, Error)]
pub enum SVDError {
    #[error("Backend error code: {0}")]
    BackendError(i32),

    #[error("Inconsistent U and VT: must be both Some or both None")]
    InconsistentUV,

    #[error("Backend failed to converge: {superdiagonals} superdiagonals did not converge to zero")]
    BackendDidNotConverge { superdiagonals: i32 },
}

/// Holds the results of a singular value decomposition, including
/// singular values and the left and right singular vectors
pub struct SVDDecomp<T, D: Dim> {
    pub s: Tensor<T, (D, D)>,
    pub u: Tensor<T, (D, D)>,
    pub vt: Tensor<T, (D, D)>,
}

/// Result type for singular value decomposition, returning either an
/// `SVDDecomp` or an `SVDError`
pub type SVDResult<T, D> = Result<SVDDecomp<T, D>, SVDError>;

/// Singular value decomposition for matrix factorization and analysis
pub trait SVD<T, D: Dim, L: Layout> {
    /// Compute full SVD with new allocated matrices
    fn svd(&self, a: &mut Slice<T, (D, D), L>) -> SVDResult<T, D>;

    /// Compute only singular values with new allocated matrix
    fn svd_s(&self, a: &mut Slice<T, (D, D), L>) -> Result<Tensor<T, (D, D)>, SVDError>;

    /// Compute full SVD, overwriting existing matrices
    /// The matrix A is decomposed as A = U * S * V^T where:
    /// - `s` contains the singular values (diagonal matrix S)
    /// - `u` contains the left singular vectors (matrix U)
    /// - `vt` contains the transposed right singular vectors (matrix V^T)
    fn svd_write<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<T, (D, D), Ls>,
        u: &mut Slice<T, (D, D), Lu>,
        vt: &mut Slice<T, (D, D), Lvt>,
    ) -> Result<(), SVDError>;

    /// Compute only singular values, overwriting existing matrix
    /// Computes only the diagonal elements of the S matrix from the SVD decomposition.
    fn svd_write_s<Ls: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<T, (D, D), Ls>,
    ) -> Result<(), SVDError>;
}
