use mdarray::{DSlice, DTensor, Layout};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SVDError {
    #[error("Backend error code: {0}")]
    BackendError(i32),

    #[error("Inconsistent U and VT: must be both Some or both None")]
    InconsistentUV,

    #[error("Backend failed to converge: {superdiagonals} superdiagonals did not converge to zero")]
    BackendDidNotConverge { superdiagonals: i32 },
}

pub type SVDResult<T> = Result<(DTensor<T, 2>, DTensor<T, 2>, DTensor<T, 2>), SVDError>;

pub trait SVD<T> {
    /// Print the name of the current Backend for debug purpose.
    fn print_name(&self);
    fn svd<'a, L: Layout>(&self, a: &'a mut DSlice<T, 2, L>) -> impl SVDBuilder<'a, T, L>;
}

pub trait SVDBuilder<'a, T, L> {
    /// Overwrites the provided slices with the complete SVD decomposition result.
    /// The matrix A is decomposed as A = U * S * V^T where:
    /// - `s` contains the singular values (diagonal matrix S)
    /// - `u` contains the left singular vectors (matrix U)
    /// - `vt` contains the transposed right singular vectors (matrix V^T)
    fn overwrite_suvt<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &mut self,
        s: &'a mut DSlice<T, 2, Ls>,
        u: &'a mut DSlice<T, 2, Lu>,
        vt: &'a mut DSlice<T, 2, Lvt>,
    ) -> Result<(), SVDError>;

    /// Overwrites the provided slice with only the singular values.
    /// Computes only the diagonal elements of the S matrix from the SVD decomposition.
    fn overwrite_s<Ls: Layout>(&mut self, s: &'a mut DSlice<T, 2, Ls>) -> Result<(), SVDError>;

    /// Returns new owned tensors containing the complete SVD decomposition result.
    /// Returns a tuple (S, U, V^T) where the matrix A = U * S * V^T.
    #[allow(clippy::type_complexity)]
    fn eval<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &mut self,
    ) -> Result<(DTensor<T, 2>, DTensor<T, 2>, DTensor<T, 2>), SVDError>;

    /// Returns a new owned tensor containing only the singular values.
    /// Computes and returns only the diagonal elements of the S matrix.
    fn eval_s<Ls: Layout, Lu: Layout, Lvt: Layout>(&mut self) -> Result<DTensor<T, 2>, SVDError>;
}
