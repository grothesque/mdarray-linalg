use mdarray::{Array, Dense, Dim, Layout, Shape, Slice};
use mdarray_linalg::eig::{Eig, EigDecomp, EigError, EigResult, SchurError, SchurResult};
use num_complex::{Complex, ComplexFloat};
use num_traits::Zero;

use super::simple::{eig_values_from_complex_matrix, eig_vectors_from_complex_matrix, eigendecomp, schur};
use crate::{Nalgebra, to_complex_dmatrix, to_dmatrix, write_dmatrix};

impl<T, D0, D1> Eig<T, D0, D1> for Nalgebra
where
    T: ComplexFloat + Copy + Zero + nalgebra::ComplexField<RealField = T::Real>,
    T::Real: nalgebra::RealField + Copy,
    D0: Dim,
    D1: Dim,
{
    fn eig<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> EigResult<T, D0, D1> {
        let (m, n) = (a.shape().dim(0), a.shape().dim(1));
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let (eigenvalues_nalgebra, right_nalgebra) = eigendecomp(a)?;

        Ok(EigDecomp {
            eigenvalues: eig_values_from_complex_matrix::<T::Real, D0>(&eigenvalues_nalgebra),
            left_eigenvectors: None,
            right_eigenvectors: Some(eig_vectors_from_complex_matrix::<T::Real, D0, D1>(
                &right_nalgebra,
            )),
        })
    }

    fn eig_full<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> EigResult<T, D0, D1> {
        let (m, n) = (a.shape().dim(0), a.shape().dim(1));
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let (eigenvalues_nalgebra, right_nalgebra) = eigendecomp(a)?;
        let left_nalgebra = right_nalgebra
            .clone()
            .try_inverse()
            .ok_or(EigError::BackendError(-1))?
            .adjoint();

        Ok(EigDecomp {
            eigenvalues: eig_values_from_complex_matrix::<T::Real, D0>(&eigenvalues_nalgebra),
            left_eigenvectors: Some(eig_vectors_from_complex_matrix::<T::Real, D0, D1>(
                &left_nalgebra,
            )),
            right_eigenvectors: Some(eig_vectors_from_complex_matrix::<T::Real, D0, D1>(
                &right_nalgebra,
            )),
        })
    }

    fn eig_values<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> EigResult<T, D0, D1> {
        let (m, n) = (a.shape().dim(0), a.shape().dim(1));
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let schur = to_complex_dmatrix(a).schur();
        let (_, t_nalgebra) = schur.unpack();
        let eigenvalues_nalgebra = t_nalgebra.diagonal();

        Ok(EigDecomp {
            eigenvalues: eig_values_from_complex_matrix::<T::Real, D0>(&eigenvalues_nalgebra),
            left_eigenvectors: None,
            right_eigenvectors: None,
        })
    }

    fn eigh<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> EigResult<T, D0, D1> {
        let (m, n) = (a.shape().dim(0), a.shape().dim(1));
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let eig = to_dmatrix(a).symmetric_eigen();
        let mut eigenvalues = Array::from_elem(
            <(D0,) as Shape>::from_dims(&[n]),
            Complex::new(T::Real::zero(), T::Real::zero()),
        );
        let mut right = Array::from_elem(
            <(D0, D1) as Shape>::from_dims(&[m, n]),
            Complex::new(T::Real::zero(), T::Real::zero()),
        );

        for (dst, src) in eigenvalues.iter_mut().zip(eig.eigenvalues.iter()) {
            *dst = Complex::new(*src, T::Real::zero());
        }

        for i in 0..eig.eigenvectors.nrows() {
            for j in 0..eig.eigenvectors.ncols() {
                let val = eig.eigenvectors[(i, j)];
                right[[i, j]] = Complex::new(val.re(), val.im());
            }
        }

        Ok(EigDecomp {
            eigenvalues,
            left_eigenvectors: None,
            right_eigenvectors: Some(right),
        })
    }

    fn eigs<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> EigResult<T, D0, D1> {
        self.eigh(a)
    }

    fn schur<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> SchurResult<T, D0, D1> {
        let (m, n) = (a.shape().dim(0), a.shape().dim(1));
        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        schur(a)
    }

    fn schur_write<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        t: &mut Slice<T, (D0, D1), Dense>,
        z: &mut Slice<T, (D0, D1), Dense>,
    ) -> Result<(), SchurError> {
        let (m, n) = (a.shape().dim(0), a.shape().dim(1));
        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        let schur = to_dmatrix(a).schur();
        let (z_nalgebra, t_nalgebra) = schur.unpack();
        write_dmatrix(&t_nalgebra, t);
        write_dmatrix(&z_nalgebra, z);
        Ok(())
    }

    fn schur_complex<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> SchurResult<T, D0, D1> {
        let (m, n) = (a.shape().dim(0), a.shape().dim(1));
        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        schur(a)
    }

    fn schur_complex_write<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        t: &mut Slice<T, (D0, D1), Dense>,
        z: &mut Slice<T, (D0, D1), Dense>,
    ) -> Result<(), SchurError> {
        self.schur_write(a, t, z)
    }
}
