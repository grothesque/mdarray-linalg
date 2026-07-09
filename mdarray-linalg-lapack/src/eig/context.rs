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

use mdarray::{Array, Dense, Dim, Layout, Shape, Slice};
use mdarray_linalg::{
    eig::{Eig, EigDecomp, EigError, EighDecomp, SchurDecomp, SchurError},
    utils::transpose_in_place,
};
use num_complex::{Complex, ComplexFloat};
use num_traits::identities::Zero;

use super::{
    scalar::{LapackScalar, NeedsRwork},
    simple::{gees, gees_complex, geig, geigh},
};
use crate::Lapack;

impl<T, D0: Dim, D1: Dim> Eig<T, D0, D1> for Lapack
where
    T: ComplexFloat + Default + LapackScalar + NeedsRwork<Elem = T>,
    Complex<T::Real>: ComplexFloat + Default + LapackScalar + NeedsRwork<Elem = Complex<T::Real>>,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    type SpectralScalar = Complex<T::Real>;
    type RealScalar = T::Real;

    /// Compute eigenvalues and right eigenvectors with new allocated matrices
    fn eig<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<EigDecomp<Self::SpectralScalar, D0, D1>, EigError>
    where
        T: ComplexFloat,
    {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let x = T::default();
        let ash1 = <(D0,) as Shape>::from_dims(&[n]);

        let mut eigenvalues_real = Array::from_elem(ash1, T::default());
        let mut eigenvalues_imag = Array::from_elem(ash1, T::default());
        let mut eigenvalues = Array::from_elem(ash1, Complex::new(x.re(), x.re()));

        let mut right_eigenvectors_tmp = Array::from_elem(ash, T::default());
        let mut right_eigenvectors = Array::from_elem(ash, Complex::new(x.re(), x.re()));

        match geig::<L, Dense, Dense, Dense, Dense, T, D0, D1>(
            a,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            None, // no left eigenvectors
            Some(&mut right_eigenvectors_tmp),
        ) {
            Ok(_) => {
                for i in 0..n {
                    eigenvalues[i] = if !eigenvalues_real[i].im().is_zero() {
                        Complex::new(eigenvalues_real[i].re(), eigenvalues_real[i].im())
                    } else {
                        Complex::new(eigenvalues_real[i].re(), eigenvalues_imag[i].re())
                    }
                }
                let mut j = 0_usize;
                while j < n {
                    let imag = eigenvalues_imag[[j]];
                    if imag == T::default() {
                        for i in 0..n {
                            let re = right_eigenvectors_tmp[[i, j]];
                            right_eigenvectors[[i, j]] = Complex::new(re.re(), re.im());
                        }
                        j += 1;
                    } else {
                        for i in 0..n {
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
    fn eig_full<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<EigDecomp<Self::SpectralScalar, D0, D1>, EigError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let x = T::default();
        let ash1 = <(D0,) as Shape>::from_dims(&[n]);

        let mut eigenvalues_real = Array::from_elem(ash1, T::default());
        let mut eigenvalues_imag = Array::from_elem(ash1, T::default());
        let mut eigenvalues = Array::from_elem(ash1, Complex::new(x.re(), x.re()));

        let mut left_eigenvectors_tmp = Array::from_elem(ash, T::default());
        let mut right_eigenvectors_tmp = Array::from_elem(ash, T::default());
        let mut left_eigenvectors = Array::from_elem(ash, Complex::new(x.re(), x.re()));
        let mut right_eigenvectors = Array::from_elem(ash, Complex::new(x.re(), x.re()));

        match geig::<L, Dense, Dense, Dense, Dense, T, D0, D1>(
            a,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            Some(&mut left_eigenvectors_tmp),
            Some(&mut right_eigenvectors_tmp),
        ) {
            Ok(_) => {
                for i in 0..n {
                    eigenvalues[i] = if !eigenvalues_real[i].im().is_zero() {
                        Complex::new(eigenvalues_real[i].re(), eigenvalues_real[i].im())
                    } else {
                        Complex::new(eigenvalues_real[i].re(), eigenvalues_imag[i].re())
                    };
                }

                let mut j = 0_usize;
                while j < n {
                    let imag = eigenvalues_imag[[j]];
                    if imag == T::default() {
                        for i in 0..n {
                            let re_right = right_eigenvectors_tmp[[i, j]];
                            let re_left = left_eigenvectors_tmp[[i, j]];
                            right_eigenvectors[[i, j]] =
                                Complex::new(re_right.re(), re_right.im());
                            left_eigenvectors[[i, j]] = Complex::new(re_left.re(), re_left.im());
                        }
                        j += 1;
                    } else {
                        for i in 0..n {
                            let re_right = right_eigenvectors_tmp[[i, j]];
                            let im_right = right_eigenvectors_tmp[[i, j + 1]];
                            let re_left = left_eigenvectors_tmp[[i, j]];
                            let im_left = left_eigenvectors_tmp[[i, j + 1]];

                            right_eigenvectors[[i, j]] =
                                Complex::new(re_right.re(), im_right.re());
                            right_eigenvectors[[i, j + 1]] =
                                ComplexFloat::conj(Complex::new(re_right.re(), im_right.re()));

                            left_eigenvectors[[i, j]] = Complex::new(re_left.re(), im_left.re());
                            left_eigenvectors[[i, j + 1]] =
                                ComplexFloat::conj(Complex::new(re_left.re(), im_left.re()));
                        }
                        j += 2;
                    }
                }

                Ok(EigDecomp {
                    eigenvalues,
                    left_eigenvectors: Some(left_eigenvectors),
                    right_eigenvectors: Some(right_eigenvectors),
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Compute only eigenvalues with new allocated vectors
    fn eig_values<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<Array<Self::SpectralScalar, (D0,)>, EigError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let x = T::default();
        let ash1 = <(D0,) as Shape>::from_dims(&[n]);

        let mut eigenvalues_real = Array::from_elem(ash1, T::default());
        let mut eigenvalues_imag = Array::from_elem(ash1, T::default());
        let mut eigenvalues = Array::from_elem(ash1, Complex::new(x.re(), x.re()));

        match geig::<L, Dense, Dense, Dense, Dense, T, D0, D1>(
            a,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            None,
            None,
        ) {
            Ok(_) => {
                for i in 0..n {
                    eigenvalues[i] =
                        Complex::new(eigenvalues_real[i].re(), eigenvalues_imag[i].re());
                }

                Ok(eigenvalues)
            }
            Err(e) => Err(e),
        }
    }

    /// Compute eigenvalues and eigenvectors of a self-adjoint matrix
    fn eigh<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<EighDecomp<T, Self::RealScalar, D0, D1>, EigError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let ash1 = <(D0,) as Shape>::from_dims(&[n]);
        let mut eigenvalues = Array::from_elem(ash1, T::Real::zero());
        let mut eigenvectors = Array::from_elem(ash, T::default());

        match geigh(a, &mut eigenvalues) {
            Ok(_) => {
                for j in 0..n {
                    for i in 0..n {
                        eigenvectors[[i, j]] = a[[j, i]];
                    }
                }

                Ok(EighDecomp {
                    eigenvalues,
                    eigenvectors,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Compute Schur decomposition with new allocated matrices
    fn schur<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<SchurDecomp<T, D0, D1>, SchurError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        let ash1 = <(D0,) as Shape>::from_dims(&[n]);

        let mut eigenvalues_real = Array::from_elem(ash1, T::default());
        let mut eigenvalues_imag = Array::from_elem(ash1, T::default());
        let mut schur_vectors = Array::from_elem(ash, T::default());

        match gees::<L, Dense, Dense, Dense, T, D0, D1>(
            a,
            &mut eigenvalues_real,
            &mut eigenvalues_imag,
            &mut schur_vectors,
        ) {
            Ok(_) => {
                let mut t = Array::from_elem(ash, T::default());
                for j in 0..n {
                    for i in 0..n {
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
        a: &mut Slice<T, (D0, D1), L>,
        t: &mut Slice<T, (D0, D1), Dense>,
        z: &mut Slice<T, (D0, D1), Dense>,
    ) -> Result<(), SchurError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        for j in 0..n {
            for i in 0..n {
                t[[i, j]] = a[[i, j]];
            }
        }

        let ash1 = <(D0,) as Shape>::from_dims(&[n]);
        let mut eigenvalues_real = Array::from_elem(ash1, T::default());
        let mut eigenvalues_imag = Array::from_elem(ash1, T::default());

        let result = gees::<Dense, Dense, Dense, Dense, T, D0, D1>(
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
    fn schur_complex<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<SchurDecomp<Self::SpectralScalar, D0, D1>, SchurError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        let zero = T::Real::zero();
        let ash1 = <(D0,) as Shape>::from_dims(&[n]);
        let mut eigenvalues = Array::from_elem(ash1, Complex::new(zero, zero));
        let mut a_complex = Array::from_fn(ash, |idx| {
            let x = a[idx];
            Complex::new(x.re(), x.im())
        });
        let mut schur_vectors = Array::from_elem(ash, Complex::new(zero, zero));

        match gees_complex::<Dense, Dense, Dense, Self::SpectralScalar, D0, D1>(
            &mut a_complex,
            &mut eigenvalues,
            &mut schur_vectors,
        ) {
            Ok(_) => {
                let mut t = Array::from_elem(ash, Complex::new(zero, zero));
                for j in 0..n {
                    for i in 0..n {
                        t[[i, j]] = a_complex[[j, i]];
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
        a: &mut Slice<T, (D0, D1), L>,
        t: &mut Slice<Self::SpectralScalar, (D0, D1), Dense>,
        z: &mut Slice<Self::SpectralScalar, (D0, D1), Dense>,
    ) -> Result<(), SchurError> {
        let SchurDecomp { t: t_result, z: z_result } = self.schur_complex(a)?;
        for (dst, src) in t.iter_mut().zip(t_result.iter()) {
            *dst = *src;
        }
        for (dst, src) in z.iter_mut().zip(z_result.iter()) {
            *dst = *src;
        }
        Ok(())
    }
}
