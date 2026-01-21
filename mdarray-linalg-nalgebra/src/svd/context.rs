// Singular Value Decomposition (SVD):
//     A = U * Σ * V^T
// where:
//     - A is m × n         (input matrix)
//     - U is m × m        (left singular vectors, orthogonal)
//     - Σ is µ × µ         (diagonal matrix with singular values on the diagonal, µ = min(m,n))
//     - V^T is n × n      (transpose of right singular vectors, orthogonal)
//     - s (Σ) contains min(m, n) singular values (non-negative, sorted in descending order)
use std::fmt::Debug;

use mdarray::{Dim, Layout, Shape, Slice, Tensor};
use mdarray_linalg::svd::{SVD, SVDDecomp, SVDError, SVDResult};

use matamorph::ref_::MataConvertRef;

use crate::Nalgebra;

impl<T, D, L> SVD<T, D, L> for Nalgebra
where
    T: Default
        + num_complex::ComplexFloat
        + Debug
        + nalgebra::ComplexField<RealField = <T as num_complex::ComplexFloat>::Real>
        + 'static,
    <T as num_complex::ComplexFloat>::Real: nalgebra::RealField + Default + Copy,
    D: Dim,
    L: Layout,
    for<'a> mdarray::View<'a, T, (D, D), L>: MataConvertRef<'a, T>,
{
    /// Compute full SVD with new allocated matrices
    fn svd(&self, a: &mut Slice<T, (D, D), L>) -> SVDResult<T, D> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);

        let a_nalgebra = nalgebra::DMatrix::<T>::from_fn(m, n, |i, j| a[[i, j]]);
        // Here we do a dumb copy of the matrix. This copy takes time
        // and memory but the nalgebra backend is intended to be used
        // with small matrices and the gain of using nalgebra/SVD in
        // the case of small matrices exceeds the time of copy.

        let svd_result = a_nalgebra.svd(true, true);

        let singular_values = svd_result.singular_values;
        let u = svd_result.u.ok_or(SVDError::BackendError(-1))?;
        let v_t = svd_result.v_t.ok_or(SVDError::BackendError(-1))?;

        let s_shape = <(D, D) as Shape>::from_dims(&[min_mn, min_mn]);
        let u_shape = <(D, D) as Shape>::from_dims(&[m, m]);
        let vt_shape = <(D, D) as Shape>::from_dims(&[n, n]);

        let mut s_mda = Tensor::<T, (D, D)>::from_elem(s_shape, T::default());
        let mut u_mda = Tensor::<T, (D, D)>::from_elem(u_shape, T::default());
        let mut vt_mda = Tensor::<T, (D, D)>::from_elem(vt_shape, T::default());

        for i in 0..min_mn {
            s_mda[[0, i]] = T::from_real(singular_values[i]);
        }

        let u_cols = u.ncols();
        for i in 0..m {
            for j in 0..u_cols {
                u_mda[[i, j]] = u[(i, j)];
            }
            for j in u_cols..m {
                u_mda[[i, j]] = T::zero();
            }
        }

        for i in 0..n {
            for j in 0..n {
                vt_mda[[i, j]] = v_t[(i, j)];
            }
        }

        Ok(SVDDecomp {
            s: s_mda,
            u: u_mda,
            vt: vt_mda,
        })
    }

    /// Compute only singular values with new allocated matrix
    fn svd_s(&self, a: &mut Slice<T, (D, D), L>) -> Result<Tensor<T, (D, D)>, SVDError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);
        let a_nalgebra = nalgebra::DMatrix::<T>::from_fn(m, n, |i, j| a[[i, j]]);
        let svd_result = a_nalgebra.svd(false, false);
        let singular_values = svd_result.singular_values;
        let s_shape = <(D, D) as Shape>::from_dims(&[min_mn, min_mn]);
        let mut s_mda = Tensor::<T, (D, D)>::from_elem(s_shape, T::default());

        for i in 0..min_mn {
            s_mda[[0, i]] = T::from_real(singular_values[i]);
        }

        Ok(s_mda)
    }

    /// Compute full SVD, overwriting existing matrices
    fn svd_write<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s_mda: &mut Slice<T, (D, D), Ls>,
        u_mda: &mut Slice<T, (D, D), Lu>,
        vt_mda: &mut Slice<T, (D, D), Lvt>,
    ) -> Result<(), SVDError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);

        let a_nalgebra = nalgebra::DMatrix::<T>::from_fn(m, n, |i, j| a[[i, j]]);

        let svd_result = a_nalgebra.svd(true, true);

        let singular_values = svd_result.singular_values;
        let u = svd_result.u.ok_or(SVDError::BackendError(-1))?;
        let v_t = svd_result.v_t.ok_or(SVDError::BackendError(-1))?;

        for i in 0..min_mn {
            s_mda[[0, i]] = T::from_real(singular_values[i]);
        }

        let u_cols = u.ncols();
        for i in 0..m {
            for j in 0..u_cols {
                u_mda[[i, j]] = u[(i, j)];
            }
            for j in u_cols..m {
                u_mda[[i, j]] = T::zero();
            }
        }

        for i in 0..n {
            for j in 0..n {
                vt_mda[[i, j]] = v_t[(i, j)];
            }
        }

        Ok(())
    }

    /// Compute only singular values, overwriting existing matrix
    fn svd_write_s<Ls: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s_mda: &mut Slice<T, (D, D), Ls>,
    ) -> Result<(), SVDError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);

        let a_nalgebra = nalgebra::DMatrix::<T>::from_fn(m, n, |i, j| a[[i, j]]);

        let svd_result = a_nalgebra.svd(false, false);

        let singular_values = svd_result.singular_values;
        for i in 0..min_mn {
            s_mda[[0, i]] = T::from_real(singular_values[i]);
        }

        Ok(())
    }
}
