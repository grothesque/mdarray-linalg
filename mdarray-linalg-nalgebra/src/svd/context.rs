// Singular Value Decomposition (SVD):
//     A = U * Σ * V^T
// where:
//     - A is m × n         (input matrix)
//     - U is m × m        (left singular vectors, orthogonal)
//     - Σ is m × n         (diagonal matrix with singular values on the diagonal)
//     - V^T is n × n      (transpose of right singular vectors, orthogonal)
//     - s (Σ) contains min(m, n) singular values (non-negative, sorted in descending order)
use std::fmt::Debug;

use mdarray::{DSlice, DTensor, Dense, Dim, Layout, Shape, Slice, tensor};
use mdarray_linalg::svd::{SVD, SVDDecomp, SVDError};
use num_complex::ComplexFloat;

use matamorph::ref_::MataConvertRef;

use crate::Nalgebra;

impl<T, D0, D1, L> SVD<T, D0, D1, L> for Nalgebra
where
    T: ComplexFloat
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + Debug
        + simba::scalar::ComplexField<RealField = T>
        + 'static,
    D0: Dim,
    D1: Dim,
    L: Layout,
    for<'a> mdarray::View<'a, T, (D0, D1), L>: MataConvertRef<'a, T>,
{
    /// Compute full SVD with new allocated matrices
    fn svd(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<SVDDecomp<T>, SVDError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);
        let max_mn = m.max(n);

        let a_nalgebra = nalgebra::DMatrix::<T>::from_fn(m, n, |i, j| a[[i, j]]);
        // let a_nalgebra = a.view(.., ..).to_nalgebra();

        let svd_result = a_nalgebra.svd(true, true);

        let singular_values = svd_result.singular_values;
        dbg!(&min_mn);
        // dbg!(&singular_values);
        let u = svd_result.u.ok_or(SVDError::BackendError(-1))?;
        let v_t = svd_result.v_t.ok_or(SVDError::BackendError(-1))?;

        let mut s_mda = tensor![[T::default(); m]; n];
        let mut u_mda = tensor![[T::default(); m]; m];
        let mut vt_mda = tensor![[T::default(); n]; n];

        for i in 0..min_mn {
            s_mda[[0, i]] = singular_values[i];
        }

        dbg!(&m);
        dbg!(&min_mn);
        dbg!(&u_mda);
        dbg!(&u);

        for i in 0..m {
            for j in 0..min_mn {
                u_mda[[i, j]] = u[(i, j)];
            }
        }

        dbg!("ici");

        for i in 0..min_mn {
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
    fn svd_s(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<DTensor<T, 2>, SVDError> {
        todo!()
    }

    /// Compute full SVD, overwriting existing matrices
    fn svd_write<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        s: &mut DSlice<T, 2, Ls>,
        u: &mut DSlice<T, 2, Lu>,
        vt: &mut DSlice<T, 2, Lvt>,
    ) -> Result<(), SVDError> {
        todo!()
    }

    /// Compute only singular values, overwriting existing matrix
    fn svd_write_s<Ls: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        s: &mut DSlice<T, 2, Ls>,
    ) -> Result<(), SVDError> {
        todo!()
    }
}
