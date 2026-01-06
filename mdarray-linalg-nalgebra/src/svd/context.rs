// Singular Value Decomposition (SVD):
//     A = U * Σ * V^T
// where:
//     - A is m × n         (input matrix)
//     - U is m × m        (left singular vectors, orthogonal)
//     - Σ is m × n         (diagonal matrix with singular values on the diagonal)
//     - V^T is n × n      (transpose of right singular vectors, orthogonal)
//     - s (Σ) contains min(m, n) singular values (non-negative, sorted in descending order)
use std::fmt::Debug;

use mdarray::{Dim, Layout, Shape, Slice, Tensor};
use mdarray_linalg::svd::{SVD, SVDDecomp, SVDError, SVDResult};
use num_complex::ComplexFloat;

use matamorph::ref_::MataConvertRef;

use crate::Nalgebra;

impl<T, D, L> SVD<T, D, L> for Nalgebra
where
    T: ComplexFloat
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + Debug
        + simba::scalar::ComplexField<RealField = T>
        + 'static,
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
        // let a_nalgebra = a.view(.., ..).to_nalgebra();

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
            s_mda[[0, i]] = singular_values[i];
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
        todo!()
    }

    /// Compute full SVD, overwriting existing matrices
    fn svd_write<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<T, (D, D), Ls>,
        u: &mut Slice<T, (D, D), Lu>,
        vt: &mut Slice<T, (D, D), Lvt>,
    ) -> Result<(), SVDError> {
        todo!()
    }

    /// Compute only singular values, overwriting existing matrix
    fn svd_write_s<Ls: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<T, (D, D), Ls>,
    ) -> Result<(), SVDError> {
        todo!()
    }
}
