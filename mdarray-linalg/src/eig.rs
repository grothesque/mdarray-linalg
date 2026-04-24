//! Eigenvalue, eigenvector, and Schur decomposition utilities for general and Hermitian matrices
//!
//!```rust,ignore
//!// ----- Eigenvalue decomposition -----
//!use mdarray_linalg::prelude::*; // Import traits anonymously
//!use mdarray_linalg_backend::{Backend, eig}; // Use the real backend here, Lapack, Faer, ...
//!// Note: we must clone `a` here because decomposition routines destroy the input.
//!let bd = Backend::default();
//!let EigDecomp {
//!    eigenvalues,
//!    right_eigenvectors,
//!    ..
//!} = bd.eig(&mut a.clone()).expect("Eigenvalue decomposition failed");
//!
//!// Or...
//!let (lambda, v) = eig!(&mut a.clone());
//!```

use mdarray::{Array, Dense, Dim, Layout, Slice};
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
pub struct EigDecomp<T: ComplexFloat, D0: Dim, D1: Dim> {
    pub eigenvalues: Array<Complex<T::Real>, (D0,)>,
    pub left_eigenvectors: Option<Array<Complex<T::Real>, (D0, D1)>>,
    pub right_eigenvectors: Option<Array<Complex<T::Real>, (D0, D1)>>,
}

/// Result type for eigenvalue decomposition, returning either an
/// `EigDecomp` or an `EigError`
pub type EigResult<T, D0, D1> = Result<EigDecomp<T, D0, D1>, EigError>;

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
pub struct SchurDecomp<T: ComplexFloat, D0: Dim, D1: Dim> {
    /// Schur form T (upper-triangular for complex, quasi-upper triangular for real)
    pub t: Array<T, (D0, D1)>,
    /// Unitary Schur transformation matrix Z
    pub z: Array<T, (D0, D1)>,
}

/// Result type for Schur decomposition, returning either a
/// `SchurDecomp` or a `SchurError`
pub type SchurResult<T, D0, D1> = Result<SchurDecomp<T, D0, D1>, SchurError>;

/// Eigenvalue decomposition operations of general and Hermitian/symmetric matrices
pub trait Eig<T: ComplexFloat, D0: Dim, D1: Dim> {
    /// Compute eigenvalues and right eigenvectors with new allocated matrices
    /// The matrix `A` satisfies: `A * v = λ * v` where v are the right eigenvectors
    fn eig<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> EigResult<T, D0, D1>;

    /// Compute eigenvalues and both left/right eigenvectors with new allocated matrices
    /// The matrix A satisfies: `A * vr = λ * vr` and `vl^H * A = λ * vl^H`
    /// where `vr` are right eigenvectors and `vl` are left eigenvectors
    fn eig_full<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> EigResult<T, D0, D1>;

    /// Compute only eigenvalues with new allocated vectors
    fn eig_values<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> EigResult<T, D0, D1>;

    // Note of october 2025: this method has been
    // temporary removed as it was very hard to provide a coherent
    // interface for the user that deals with all the cases and all
    // the backends.

    // // Compute eigenvalues and right eigenvectors, overwriting existing matrices
    // fn eig_write<L: Layout, Lr: Layout, Li: Layout, Lv: Layout>(
    //     &self,
    //     a: &mut Slice<T, (D0, D1), L>,
    //     eigenvalues: &mut Slice<Complex<T::Real>, (D0, D1), Dense>,
    //     right_eigenvectors: &mut Slice<Complex<T::Real>, (D0, D1), Dense>,
    // ) -> Result<(), EigError>;

    /// Compute eigenvalues and eigenvectors of a Hermitian matrix (input should be complex)
    fn eigh<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> EigResult<T, D0, D1>;

    /// Compute eigenvalues and eigenvectors of a symmetric matrix (input should be real)
    fn eigs<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> EigResult<T, D0, D1>;

    /// Compute Schur decomposition with new allocated matrices
    fn schur<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> SchurResult<T, D0, D1>;

    /// Compute Schur decomposition overwriting existing matrices
    fn schur_write<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        t: &mut Slice<T, (D0, D1), Dense>,
        z: &mut Slice<T, (D0, D1), Dense>,
    ) -> Result<(), SchurError>;

    /// Compute Schur (complex) decomposition with new allocated matrices
    fn schur_complex<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> SchurResult<T, D0, D1>;

    /// Compute Schur (complex) decomposition overwriting existing matrices
    fn schur_complex_write<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        t: &mut Slice<T, (D0, D1), Dense>,
        z: &mut Slice<T, (D0, D1), Dense>,
    ) -> Result<(), SchurError>;
}
