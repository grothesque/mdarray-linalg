use mdarray::{DSlice, DTensor, Dense, Layout};
use num_complex::{Complex, ComplexFloat};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EigError {
    #[error("Backend error code: {0}")]
    BackendError(i32),

    #[error("Backend failed to converge: {iterations} iterations exceeded")]
    BackendDidNotConverge { iterations: i32 },

    #[error("Matrix must be square for eigenvalue decomposition")]
    NotSquareMatrix,
}

pub struct EigDecomp<T: ComplexFloat> {
    pub eigenvalues_real: DTensor<T, 2>,
    pub eigenvalues_imag: Option<DTensor<T, 2>>,
    pub left_eigenvectors: Option<DTensor<Complex<T::Real>, 2>>,
    pub right_eigenvectors: Option<DTensor<Complex<T::Real>, 2>>,
}

pub type EigResult<T> = Result<EigDecomp<T>, EigError>;

/// Eigenvalue decomposition operations of general and Hermitian/symmetric matrices
pub trait Eig<T: ComplexFloat> {
    /// Compute eigenvalues and right eigenvectors with new allocated matrices
    /// The matrix `A` satisfies: `A * v = λ * v` where v are the right eigenvectors
    fn eig<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T>;

    /// Compute eigenvalues and both left/right eigenvectors with new allocated matrices
    /// The matrix A satisfies: `A * vr = λ * vr` and `vl^H * A = λ * vl^H`
    /// where `vr` are right eigenvectors and `vl` are left eigenvectors
    fn eig_full<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T>;

    /// Compute only eigenvalues with new allocated vectors
    fn eig_values<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T>;

    /// Compute eigenvalues and right eigenvectors, overwriting existing matrices
    fn eig_overwrite<L: Layout, Lr: Layout, Li: Layout, Lv: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        eigenvalues_real: &mut DSlice<T, 2, Dense>,
        eigenvalues_imag: &mut DSlice<T, 2, Dense>,
        right_eigenvectors: &mut DSlice<T, 2, Dense>,
    ) -> Result<(), EigError>;

    /// Compute eigenvalues and eigenvectors of a Hermitian/symmetric matrix
    fn eigh<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T>;
}
