use mdarray::{Array, Dense, Dim, Layout, Shape, Slice};
use mdarray_linalg::eig::{Eig, EigDecomp, EigError, EighDecomp, SchurDecomp, SchurError};
use num_complex::{Complex, ComplexFloat};
use num_traits::Zero;

use super::simple::{eig_values_from_complex_matrix, eig_vectors_from_complex_matrix, eigendecomp, schur, schur_complex};
use crate::{Nalgebra, to_complex_dmatrix, to_dmatrix, write_dmatrix};

impl<T, D0, D1> Eig<T, D0, D1> for Nalgebra
where
    T: ComplexFloat + Copy + Zero + nalgebra::ComplexField<RealField = T::Real>,
    T::Real: nalgebra::RealField + Copy,
    D0: Dim,
    D1: Dim,
{
    type SpectralScalar = Complex<T::Real>;
    type RealScalar = T::Real;

    fn eig<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<EigDecomp<Self::SpectralScalar, D0, D1>, EigError> {
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

    fn eig_full<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<EigDecomp<Self::SpectralScalar, D0, D1>, EigError> {
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

    fn eig_values<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<Array<Self::SpectralScalar, (D0,)>, EigError> {
        let (m, n) = (a.shape().dim(0), a.shape().dim(1));
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let schur = to_complex_dmatrix(a).schur();
        let (_, t_nalgebra) = schur.unpack();
        let eigenvalues_nalgebra = t_nalgebra.diagonal();

        Ok(eig_values_from_complex_matrix::<T::Real, D0>(&eigenvalues_nalgebra))
    }

    fn eigh<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<EighDecomp<T, Self::RealScalar, D0, D1>, EigError> {
        let (m, n) = (a.shape().dim(0), a.shape().dim(1));
        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let eig = to_dmatrix(a).symmetric_eigen();
        let mut eigenvalues = Array::from_elem(<(D0,) as Shape>::from_dims(&[n]), T::Real::zero());
        let mut eigenvectors = Array::from_elem(<(D0, D1) as Shape>::from_dims(&[m, n]), T::zero());

        for (dst, src) in eigenvalues.iter_mut().zip(eig.eigenvalues.iter()) {
            *dst = *src;
        }

        for i in 0..eig.eigenvectors.nrows() {
            for j in 0..eig.eigenvectors.ncols() {
                eigenvectors[[i, j]] = eig.eigenvectors[(i, j)];
            }
        }

        Ok(EighDecomp {
            eigenvalues,
            eigenvectors,
        })
    }

    fn schur<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<SchurDecomp<T, D0, D1>, SchurError> {
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

    fn schur_complex<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<SchurDecomp<Self::SpectralScalar, D0, D1>, SchurError> {
        let (m, n) = (a.shape().dim(0), a.shape().dim(1));
        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        schur_complex(a)
    }

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
