use mdarray::{DSlice, DTensor, Dense, Layout};
use num_complex::{Complex, ComplexFloat};
use thiserror::Error;

/// Error types related to eigenvalue decomposition
#[derive(Debug, Error)]
pub enum EigError {
    #[error("Backend error code: {0}")]
    BackendError(i32),

    #[error("Backend failed to converge: {iterations} iterations exceeded")]
    BackendDidNotConverge { iterations: i32 },

    #[error("Matrix must be square for eigenvalue decomposition")]
    NotSquareMatrix,
}

/// Holds the results of an eigenvalue decomposition, including
/// eigenvalues (complex) and optionally left and right eigenvectors
pub struct EigDecomp<T: ComplexFloat> {
    pub eigenvalues: DTensor<Complex<T::Real>, 2>,
    pub left_eigenvectors: Option<DTensor<Complex<T::Real>, 2>>,
    pub right_eigenvectors: Option<DTensor<Complex<T::Real>, 2>>,
}

/// Result type for eigenvalue decomposition, returning either an
/// `EigDecomp` or an `EigError`
pub type EigResult<T> = Result<EigDecomp<T>, EigError>;

/// Error types related to Schur decomposition
#[derive(Debug, Error)]
pub enum SchurError {
    #[error("Backend error code: {0}")]
    BackendError(i32),

    #[error("Backend failed to converge: {iterations} iterations exceeded")]
    BackendDidNotConverge { iterations: i32 },

    #[error("Matrix must be square for Schur decomposition")]
    NotSquareMatrix,
}

/// Holds the results of a Schur decomposition: A = Z * T * Z^H
/// where Z is unitary and T is upper-triangular (complex) or quasi-upper triangular (real)
pub struct SchurDecomp<T: ComplexFloat> {
    /// Schur form T (upper-triangular for complex, quasi-upper triangular for real)
    pub t: DTensor<T, 2>,
    /// Unitary Schur transformation matrix Z
    pub z: DTensor<T, 2>,
}

/// Result type for Schur decomposition, returning either a
/// `SchurDecomp` or a `SchurError`
pub type SchurResult<T> = Result<SchurDecomp<T>, SchurError>;

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

    // Note of october 2025: this method has been
    // temporary removed as it was very hard to provide a coherent
    // interface for the user that deals with all the cases and all
    // the backends.

    // // Compute eigenvalues and right eigenvectors, overwriting
    // existing matrices
    //fn eig_overwrite<L: Layout, Lr: Layout, Li:
    // Layout, Lv: Layout>( &self, a: &mut DSlice<T, 2, L>,
    // eigenvalues: &mut DSlice<Complex<T>, 2, Dense>,
    // right_eigenvectors: &mut DSlice<Complex<T>, 2, Dense>, ) ->
    // Result<(), EigError>;

    /// Compute eigenvalues and eigenvectors of a Hermitian matrix (input should be complex)
    fn eigh<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T>;

    /// Compute eigenvalues and eigenvectors of a symmetric matrix (input should be real)
    fn eigs<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T>;

    /// Compute Schur decomposition with new allocated matrices
    fn schur<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> SchurResult<T>;

    /// Compute Schur decomposition overwriting existing matrices
    fn schur_overwrite<L: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        t: &mut DSlice<T, 2, Dense>,
        z: &mut DSlice<T, 2, Dense>,
    ) -> Result<(), SchurError>;

    /// Compute Schur (complex) decomposition with new allocated matrices
    fn schur_complex<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> SchurResult<T>;

    /// Compute Schur complex) decomposition overwriting existing matrices
    fn schur_complex_overwrite<L: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        t: &mut DSlice<T, 2, Dense>,
        z: &mut DSlice<T, 2, Dense>,
    ) -> Result<(), SchurError>;
}
