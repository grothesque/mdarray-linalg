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

use super::simple::{gees, gees_complex, geig, geigh};
use mdarray_linalg::{get_dims, into_i32, transpose_in_place};

use mdarray::{DSlice, Dense, Layout, tensor};

use super::scalar::{LapackScalar, NeedsRwork};
use mdarray_linalg::eig::{
    Eig, EigDecomp, EigError, EigResult, SchurDecomp, SchurError, SchurResult,
};
use num_complex::{Complex, ComplexFloat};
use num_traits::identities::Zero;

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
        T: ComplexFloat,
    {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let x = T::default();

        let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues_imag = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues = tensor![[Complex::new(x.re(), x.re()); n as usize]; 1];

        let mut right_eigenvectors_tmp = tensor![[T::default();n as usize]; n as usize];
        let mut right_eigenvectors = tensor![[Complex::new(x.re(), x.re());n as usize]; n as usize];

        match geig::<L, Dense, Dense, Dense, Dense, T>(
            a,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            None, // no left eigenvectors
            Some(&mut right_eigenvectors_tmp),
        ) {
            Ok(_) => {
                for i in 0..(n as usize) {
                    eigenvalues[[0, i]] = if !eigenvalues_real[[0, i]].im().is_zero() {
                        Complex::new(eigenvalues_real[[0, i]].re(), eigenvalues_real[[0, i]].im())
                    } else {
                        Complex::new(eigenvalues_real[[0, i]].re(), eigenvalues_imag[[0, i]].re())
                    }
                }
                let mut j = 0_usize;
                while j < n as usize {
                    let imag = eigenvalues_imag[[0, j]];
                    if imag == T::default() {
                        for i in 0..(n as usize) {
                            let re = right_eigenvectors_tmp[[i, j]];
                            right_eigenvectors[[i, j]] = Complex::new(re.re(), re.im());
                        }
                        j += 1;
                    } else {
                        for i in 0..(n as usize) {
                            let re = right_eigenvectors_tmp[[i, j]];
                            let im = right_eigenvectors_tmp[[i, j + 1]];
                            right_eigenvectors[[i, j]] = Complex::new(re.re(), im.re()); // v = Re + i Im
                            right_eigenvectors[[i, j + 1]] =
                                ComplexFloat::conj(Complex::new(re.re(), im.re())); // v̄ = Re - i Im
                        }
                        j += 2;
                    }
                }

                Ok(EigDecomp {
                    eigenvalues,
                    left_eigenvectors: None,
                    right_eigenvectors: Some(right_eigenvectors),
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Compute eigenvalues and both left/right eigenvectors with new allocated matrices
    fn eig_full<L: Layout>(&self, _a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        todo!(); // fix bug in complex case
        // let (m, n) = get_dims!(a);
        // if m != n {
        //     return Err(EigError::NotSquareMatrix);
        // }

        // let x = T::default();

        // let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        // let mut eigenvalues_imag = tensor![[T::default(); n as usize]; 1];
        // let mut eigenvalues = tensor![[Complex::new(x.re(), x.re()); n as usize]; 1];

        // let mut left_eigenvectors_tmp = tensor![[T::default(); n as usize]; n as usize];
        // let mut right_eigenvectors_tmp = tensor![[T::default(); n as usize]; n as usize];

        // let x = T::default();
        // let mut left_eigenvectors = tensor![[Complex::new(x.re(), x.re()); n as usize]; n as usize];
        // let mut right_eigenvectors =
        //     tensor![[Complex::new(x.re(), x.re()); n as usize]; n as usize];

        // match geig(
        //     a,
        //     &mut eigenvalues_real,
        //     &mut eigenvalues_imag,
        //     Some(&mut left_eigenvectors_tmp),
        //     Some(&mut right_eigenvectors_tmp),
        // ) {
        //     Ok(_) => {
        //         for i in 0..(n as usize) {
        //             eigenvalues[[0, i]] =
        //                 Complex::new(eigenvalues_real[[0, i]].re(), eigenvalues_imag[[0, i]].re());
        //         }

        //         // Process right eigenvectors
        //         let mut j = 0_usize;
        //         while j < n as usize {
        //             let imag = eigenvalues_imag[[0, j]];
        //             if imag == T::default() {
        //                 // Real eigenvalue
        //                 for i in 0..(n as usize) {
        //                     let re_right = right_eigenvectors_tmp[[i, j]];
        //                     let re_left = left_eigenvectors_tmp[[i, j]];
        //                     right_eigenvectors[[i, j]] = Complex::new(re_right.re(), re_right.im());
        //                     left_eigenvectors[[i, j]] = Complex::new(re_left.re(), re_left.im());
        //                 }
        //                 j += 1;
        //             } else {
        //                 // Complex conjugate pair
        //                 for i in 0..(n as usize) {
        //                     let re_right = right_eigenvectors_tmp[[i, j]];
        //                     let im_right = right_eigenvectors_tmp[[i, j + 1]];
        //                     let re_left = left_eigenvectors_tmp[[i, j]];
        //                     let im_left = left_eigenvectors_tmp[[i, j + 1]];

        //                     right_eigenvectors[[i, j]] = Complex::new(re_right.re(), im_right.re());
        //                     right_eigenvectors[[i, j + 1]] =
        //                         ComplexFloat::conj(Complex::new(re_right.re(), im_right.re()));

        //                     left_eigenvectors[[i, j]] = Complex::new(re_left.re(), im_left.re());
        //                     left_eigenvectors[[i, j + 1]] =
        //                         ComplexFloat::conj(Complex::new(re_left.re(), im_left.re()));
        //                 }
        //                 j += 2;
        //             }
        //         }

        //         Ok(EigDecomp {
        //             eigenvalues,
        //             left_eigenvectors: Some(left_eigenvectors),
        //             right_eigenvectors: Some(right_eigenvectors),
        //         })
        //     }
        //     Err(e) => Err(e),
        // }
    }

    /// Compute only eigenvalues with new allocated vectors
    fn eig_values<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }
        let x = T::default();

        let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues_imag = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues = tensor![[Complex::new(x.re(), x.re()); n as usize]; 1];

        match geig::<L, Dense, Dense, Dense, Dense, T>(
            a,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            None,
            None,
        ) {
            Ok(_) => {
                for i in 0..(n as usize) {
                    eigenvalues[[0, i]] =
                        Complex::new(eigenvalues_real[[0, i]].re(), eigenvalues_imag[[0, i]].re());
                }

                Ok(EigDecomp {
                    eigenvalues,
                    left_eigenvectors: None,
                    right_eigenvectors: None,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Compute eigenvalues and eigenvectors of a Hermitian matrix
    fn eigh<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let x = T::default();

        let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues = tensor![[Complex::new(x.re(), x.re()); n as usize]; 1];
        // let mut eigenvalues = tensor![[x.re()]; 1];

        let mut right_eigenvectors_tmp = tensor![[T::default(); n as usize]; n as usize];

        let x = T::default();
        let mut right_eigenvectors =
            tensor![[Complex::new(x.re(), x.re()); n as usize]; n as usize];

        match geigh(a, &mut eigenvalues_real, &mut right_eigenvectors_tmp) {
            Ok(_) => {
                println!("{}", n / 2);
                for i in 0..((n / 2 + 1) as usize) {
                    eigenvalues[[0, 2 * i]] = Complex::new(eigenvalues_real[[0, i]].re(), x.re());
                    if 2 * i + 1 < n as usize {
                        eigenvalues[[0, 2 * i + 1]] =
                            Complex::new(eigenvalues_real[[0, i]].im(), x.re());
                    }
                }

                for j in 0..(n as usize) {
                    for i in 0..(n as usize) {
                        let re = a[[j, i]];
                        right_eigenvectors[[i, j]] = Complex::new(re.re(), re.im());
                    }
                }

                Ok(EigDecomp {
                    eigenvalues,
                    left_eigenvectors: None,
                    right_eigenvectors: Some(right_eigenvectors),
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Compute eigenvalues and eigenvectors of a Hermitian matrix
    fn eigs<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let x = T::default();

        let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues = tensor![[Complex::new(x.re(), x.re()); n as usize]; 1];
        // let mut eigenvalues = tensor![[x.re()]; 1];

        let mut right_eigenvectors_tmp = tensor![[T::default(); n as usize]; n as usize];

        let x = T::default();
        let mut right_eigenvectors =
            tensor![[Complex::new(x.re(), x.re()); n as usize]; n as usize];

        match geigh(a, &mut eigenvalues_real, &mut right_eigenvectors_tmp) {
            Ok(_) => {
                for i in 0..n as usize {
                    eigenvalues[[0, i]] = Complex::new(eigenvalues_real[[0, i]].re(), x.re());
                }

                for j in 0..(n as usize) {
                    for i in 0..(n as usize) {
                        let re = a[[j, i]];
                        right_eigenvectors[[i, j]] = Complex::new(re.re(), re.im());
                    }
                }

                Ok(EigDecomp {
                    eigenvalues,
                    left_eigenvectors: None,
                    right_eigenvectors: Some(right_eigenvectors),
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Compute Schur decomposition with new allocated matrices
    fn schur<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> SchurResult<T> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues_imag = tensor![[T::default(); n as usize]; 1];
        let mut schur_vectors = tensor![[T::default(); n as usize]; n as usize];

        match gees::<L, Dense, Dense, Dense, T>(
            a,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            &mut schur_vectors,
        ) {
            Ok(_) => {
                let mut t = tensor![[T::default(); n as usize]; n as usize];
                for j in 0..(n as usize) {
                    for i in 0..(n as usize) {
                        t[[i, j]] = a[[j, i]];
                    }
                }

                transpose_in_place(&mut schur_vectors);

                Ok(SchurDecomp {
                    t,
                    z: schur_vectors,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Compute Schur decomposition overwriting existing matrices
    fn schur_write<L: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        t: &mut DSlice<T, 2, Dense>,
        z: &mut DSlice<T, 2, Dense>,
    ) -> Result<(), SchurError> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        for j in 0..(n as usize) {
            for i in 0..(n as usize) {
                t[[i, j]] = a[[i, j]];
            }
        }

        let mut eigenvalues_real = tensor![[T::default(); n as usize]; 1];
        let mut eigenvalues_imag = tensor![[T::default(); n as usize]; 1];

        let result = gees::<Dense, Dense, Dense, Dense, T>(
            t,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            z,
        );
        transpose_in_place(z);
        transpose_in_place(t);
        result
    }

    /// Compute Schur (complex) decomposition with new allocated matrices
    fn schur_complex<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> SchurResult<T> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        let mut eigenvalues = tensor![T::default(); n as usize];
        let mut schur_vectors = tensor![[T::default(); n as usize]; n as usize];

        match gees_complex::<L, Dense, Dense, T>(a, &mut eigenvalues, &mut schur_vectors) {
            Ok(_) => {
                let mut t = tensor![[T::default(); n as usize]; n as usize];
                for j in 0..(n as usize) {
                    for i in 0..(n as usize) {
                        t[[i, j]] = a[[j, i]];
                    }
                }

                transpose_in_place(&mut schur_vectors);

                Ok(SchurDecomp {
                    t,
                    z: schur_vectors,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Compute Schur (complex) decomposition overwriting existing matrices
    fn schur_complex_write<L: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        t: &mut DSlice<T, 2, Dense>,
        z: &mut DSlice<T, 2, Dense>,
    ) -> Result<(), SchurError> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        for j in 0..(n as usize) {
            for i in 0..(n as usize) {
                t[[i, j]] = a[[i, j]];
            }
        }

        let mut eigenvalues = tensor![T::default(); n as usize];
        let result = gees_complex::<Dense, Dense, Dense, T>(t, &mut eigenvalues, z);

        transpose_in_place(z);
        transpose_in_place(t);

        result
    }
}
