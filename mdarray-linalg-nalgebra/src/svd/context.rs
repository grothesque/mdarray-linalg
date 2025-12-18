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

use matamorph::mut_::MataConvertMut;
use matamorph::own::MataConvertOwn;
use matamorph::ref_;
use matamorph::ref_::MataConvertRef;

use mdarray::View;

// use super::simple::svd_nalgebra;
use crate::Nalgebra;

impl<'a, T, D0, D1, L> SVD<T, D0, D1, L> for Nalgebra
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
    ref_::MataRef<T>: From<View<'a, T, (D0, D1), L>>,
{
    /// Compute full SVD with new allocated matrices
    /// Compute full SVD with new allocated matrices
    fn svd(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<SVDDecomp<T>, SVDError>
    where
        mdarray::View<'a, T, (D0, D1), L>: MataConvertRef<'a, T>,
    {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);

        let a_nalgebra = a.view(.., ..).to_nalgebra();
        let svd_result = a_nalgebra.svd(true, true);

        let singular_values = svd_result.singular_values;
        let u = svd_result.u.ok_or(SVDError::BackendError(-1))?;
        let v_t = svd_result.v_t.ok_or(SVDError::BackendError(-1))?;

        let mut s_mda = tensor![[T::default(); min_mn]; min_mn];
        let mut u_mda = tensor![[T::default(); m]; m];
        let mut vt_mda = tensor![[T::default(); n]; n];

        for i in 0..min_mn {
            s_mda[[i, i]] = singular_values[i];
        }

        for i in 0..m {
            for j in 0..m {
                u_mda[[i, j]] = u[(i, j)];
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
    fn svd_s(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<DTensor<T, 2>, SVDError> {
        todo!()
        // let ash = *a.shape();
        // let (m, n) = (ash.dim(0), ash.dim(1));

        // let min_mn = m.min(n);
        // let mut s_mda = tensor![[T::default(); min_mn]; min_mn];

        // match svd_nalgebra::<T, D0, D1, L, Dense, Dense, Dense>(a, &mut s_mda, None, None) {
        //     Err(_) => Err(SVDError::BackendDidNotConverge {
        //         superdiagonals: (0),
        //     }),
        //     Ok(_) => Ok(s_mda),
        // }
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
        // svd_nalgebra::<T, D0, D1, L, Ls, Lu, Lvt>(a, s, Some(u), Some(vt))
    }

    /// Compute only singular values, overwriting existing matrix
    fn svd_write_s<Ls: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        s: &mut DSlice<T, 2, Ls>,
    ) -> Result<(), SVDError> {
        todo!()
        // svd_nalgebra::<T, D0, D1, L, Ls, Dense, Dense>(a, s, None, None)
    }
}
