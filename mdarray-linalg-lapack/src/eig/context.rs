//! Eigenvalue Decomposition (EIG):
//!     A * v = λ * v (right eigenvectors)
//!     u^H * A = λ * u^H (left eigenvectors)
//! where:
//!     - A is n × n         (input square matrix)
//!     - λ are eigenvalues  (can be complex)
//!     - v are right eigenvectors
//!     - u are left eigenvectors
//!
//! For Hermitian/symmetric matrices (EIGH):
//!     A * v = λ * v
//! where:
//!     - A is n × n Hermitian/symmetric matrix
//!     - λ are real eigenvalues
//!     - v are orthonormal eigenvectors

use super::simple::{geig, geigh};
use mdarray_linalg::{get_dims, into_i32};

use mdarray::{DSlice, DTensor, Dense, Layout, tensor};

use super::scalar::{LapackScalar, NeedsRwork, ToComplex};
use mdarray_linalg::{Eig, EigDecomp, EigError, EigResult};
use num_complex::{Complex, ComplexFloat};

use crate::Lapack;

impl<T> Eig<T> for Lapack
where
    T: ComplexFloat + Default + LapackScalar + NeedsRwork<Elem = T>,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    /// Compute eigenvalues and right eigenvectors with new allocated matrices
    fn eig<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T>
    where
        <T as ComplexFloat>::Real: From<T>,
    {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues_imag = tensor![[T::default(); n as usize]; 1];

        let mut right_eigenvectors_tmp = tensor![[T::default();n as usize]; n as usize];

        let mut right_eigenvectors = tensor![[Complex::<T::Real>::new(T::Real::from(0.into().into()), T::Real::from(0.into().into()));
             n as usize]; n as usize];

        match geig::<L, Dense, Dense, Dense, Dense, T>(
            a,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            None, // no left eigenvectors
            Some(&mut right_eigenvectors_tmp),
        ) {
            Ok(_) => {
                let mut j = 0_usize;
                while j < n as usize {
                    let imag = eigenvalues_imag[[0, j]];
                    if imag == T::default() {
                        for i in 0..(n as usize) {
                            let re = right_eigenvectors_tmp[[i, j]];
                            right_eigenvectors[[i, j]] = Complex::new(
                                T::Real::from(re.into()),
                                T::Real::from(0.into().into()),
                            );
                        }
                        j += 1;
                    } else {
                        for i in 0..(n as usize) {
                            let re = right_eigenvectors_tmp[[i, j]];
                            let im = right_eigenvectors_tmp[[i, j + 1]];
                            right_eigenvectors[[i, j]] =
                                Complex::new(T::Real::from(re.into()), T::Real::from(im.into())); // v = Re + i Im
                            right_eigenvectors[[i, j + 1]] =
                                ComplexFloat::conj(Complex::new(re.into(), im.into())); // v̄ = Re - i Im
                        }
                        j += 2;
                    }
                }

                Ok(EigDecomp {
                    eigenvalues_real,
                    eigenvalues_imag: Some(eigenvalues_imag),
                    left_eigenvectors: None,
                    right_eigenvectors: Some(right_eigenvectors),
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Compute eigenvalues and both left/right eigenvectors with new allocated matrices
    fn eig_full<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues_imag = tensor![[T::default(); n as usize]; 1];
        let mut left_eigenvectors = tensor![[T::default(); n as usize]; n as usize];
        let mut right_eigenvectors = tensor![[T::default(); n as usize]; n as usize];

        match geig(
            a,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            Some(&mut left_eigenvectors),
            Some(&mut right_eigenvectors),
        ) {
            Ok(_) => Ok(EigDecomp {
                eigenvalues_real,
                eigenvalues_imag: Some(eigenvalues_imag),
                left_eigenvectors: Some(left_eigenvectors),
                right_eigenvectors: Some(right_eigenvectors),
            }),
            Err(e) => Err(e),
        }
    }

    /// Compute only eigenvalues with new allocated vectors
    fn eig_values<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues_imag = tensor![[T::default(); n as usize]; 1];

        match geig::<L, Dense, Dense, Dense, Dense, T>(
            a,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            None, // no left eigenvectors
            None, // no right eigenvectors
        ) {
            Ok(_) => Ok(EigDecomp {
                eigenvalues_real,
                eigenvalues_imag: Some(eigenvalues_imag),
                left_eigenvectors: None,
                right_eigenvectors: None,
            }),
            Err(e) => Err(e),
        }
    }

    /// Compute eigenvalues and right eigenvectors, overwriting existing matrices
    fn eig_overwrite<L: Layout, Lr: Layout, Li: Layout, Lv: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        eigenvalues_real: &mut DSlice<T, 2, Dense>,
        eigenvalues_imag: &mut DSlice<T, 2, Dense>,
        right_eigenvectors: &mut DSlice<T, 2, Dense>,
    ) -> Result<(), EigError> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        geig::<L, Dense, Dense, Dense, Dense, T>(
            a,
            eigenvalues_real,
            eigenvalues_imag,
            None, // no left eigenvectors
            Some(right_eigenvectors),
        )
    }

    /// Compute eigenvalues and eigenvectors of a Hermitian/symmetric matrix
    fn eigh<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        let mut right_eigenvectors = tensor![[T::default(); n as usize]; n as usize];

        // For Hermitian matrices, we use a specialized routine that only computes real eigenvalues
        match geigh(a, &mut eigenvalues_real, &mut right_eigenvectors) {
            Ok(_) => Ok(EigDecomp {
                eigenvalues_real,
                eigenvalues_imag: None, // Hermitian matrices have real eigenvalues only
                left_eigenvectors: None,
                right_eigenvectors: Some(right_eigenvectors),
            }),
            Err(e) => Err(e),
        }
    }
}
