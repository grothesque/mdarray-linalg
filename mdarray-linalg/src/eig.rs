//! Eigenvalue, eigenvector, and Schur decomposition utilities for general and self-adjoint matrices
//!
//! ```rust,ignore
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg_backend::Backend;
//!
//! // ----- Eigenvalue decomposition -----
//! // Note: we must clone `a` here because decomposition routines destroy the input.
//! let bd = Backend::default();
//! let EigDecomp { eigenvalues, right_eigenvectors, .. } = bd
//!     .eig(&mut a.clone())
//!     .expect("Eigenvalue decomposition failed");
//!
//! // Or...
//! let EigDecomp { eigenvalues: lambda, right_eigenvectors: Some(v), .. } = bd
//!     .eig(&mut a.clone())
//!     .expect("Eigenvalue decomposition failed");
//!
//! // Full decomposition with left and right eigenvectors.
//! let EigDecomp { eigenvalues, left_eigenvectors, right_eigenvectors } = bd
//!     .eig_full(&mut a.clone())
//!     .expect("Full eigenvalue decomposition failed");
//! let left = left_eigenvectors.expect("Left eigenvectors were not computed");
//! let right = right_eigenvectors.expect("Right eigenvectors were not computed");
//!
//! // ----- Schur decomposition -----
//! // A = Z * T * Z^H, with `Z^H` reducing to `Z^T` for real Schur decompositions.
//! let SchurDecomp { t, z } = bd
//!     .schur(&mut a.clone())
//!     .expect("Schur decomposition failed");
//!
//! // Reconstruct A from the decomposition with the conjugate transpose Z^H:
//! // A ≈ Z * T * Z^H
//! ```

use mdarray::{Array, Dense, Dim, Layout, Slice};
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

/// Holds the results of a general eigenvalue decomposition.
///
/// The scalar type `S` is the backend's spectral scalar for the input matrix
/// scalar. For a real matrix this is typically a complex scalar, while for a
/// complex matrix it is usually the matrix scalar itself.
pub struct EigDecomp<S, D0: Dim, D1: Dim> {
    pub eigenvalues: Array<S, (D0,)>,
    pub left_eigenvectors: Option<Array<S, (D0, D1)>>,
    pub right_eigenvectors: Option<Array<S, (D0, D1)>>,
}

/// Holds the results of a self-adjoint eigenvalue decomposition.
///
/// Self-adjoint eigenvalues are real, while eigenvectors live in the input
/// matrix scalar field.
pub struct EighDecomp<T, R, D0: Dim, D1: Dim> {
    pub eigenvalues: Array<R, (D0,)>,
    pub eigenvectors: Array<T, (D0, D1)>,
}

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
pub struct SchurDecomp<T, D0: Dim, D1: Dim> {
    /// Schur form T (upper-triangular for complex, quasi-upper triangular for real)
    pub t: Array<T, (D0, D1)>,
    /// Unitary Schur transformation matrix Z
    pub z: Array<T, (D0, D1)>,
}

/// Eigenvalue decomposition operations of general and self-adjoint matrices.
///
/// Backends choose the spectral scalar model through associated types.
/// General eigendecompositions and complex Schur decompositions use
/// [`Self::SpectralScalar`], while self-adjoint eigendecompositions use
/// [`Self::RealScalar`] for eigenvalues and the input scalar `T` for
/// eigenvectors.
pub trait Eig<T, D0: Dim, D1: Dim> {
    /// Spectral scalar type used for general eigenvalues/eigenvectors and complex Schur decompositions.
    type SpectralScalar;

    /// Real scalar type used for self-adjoint eigenvalues.
    type RealScalar;

    /// Compute eigenvalues and right eigenvectors with new allocated matrices.
    /// The matrix `A` satisfies: `A * v = λ * v` where v are the right eigenvectors.
    fn eig<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> Result<EigDecomp<Self::SpectralScalar, D0, D1>, EigError>;

    /// Compute eigenvalues and both left/right eigenvectors with new allocated matrices.
    /// The matrix A satisfies: `A * vr = λ * vr` and `vl^H * A = λ * vl^H`.
    fn eig_full<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> Result<EigDecomp<Self::SpectralScalar, D0, D1>, EigError>;

    /// Compute only eigenvalues with a newly allocated vector.
    fn eig_values<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> Result<Array<Self::SpectralScalar, (D0,)>, EigError>;

    /// Compute eigenvalues and eigenvectors of a self-adjoint matrix.
    fn eigh<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> Result<EighDecomp<T, Self::RealScalar, D0, D1>, EigError>;

    /// Compute Schur decomposition over the input scalar field.
    fn schur<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> Result<SchurDecomp<T, D0, D1>, SchurError>;

    /// Compute Schur decomposition overwriting existing matrices.
    fn schur_write<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        t: &mut Slice<T, (D0, D1), Dense>,
        z: &mut Slice<T, (D0, D1), Dense>,
    ) -> Result<(), SchurError>;

    /// Compute Schur decomposition over the spectral scalar field.
    fn schur_complex<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> Result<SchurDecomp<Self::SpectralScalar, D0, D1>, SchurError>;

    /// Compute Schur decomposition over the spectral scalar field, overwriting existing matrices.
    fn schur_complex_write<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        t: &mut Slice<Self::SpectralScalar, (D0, D1), Dense>,
        z: &mut Slice<Self::SpectralScalar, (D0, D1), Dense>,
    ) -> Result<(), SchurError>;
}
