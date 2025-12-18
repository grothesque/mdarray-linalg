// Singular Value Decomposition (SVD):
//     A = U * Σ * V^T
// where:
//     - A is m × n         (input matrix)
//     - U is m × m        (left singular vectors, orthogonal)
//     - Σ is m × n         (diagonal matrix with singular values on the diagonal)
//     - V^T is n × n      (transpose of right singular vectors, orthogonal)
//     - s (Σ) contains min(m, n) singular values (non-negative, sorted in descending order)

use mdarray::{DSlice, DTensor, Dense, Dim, Layout, Shape, Slice, tensor};
use mdarray_linalg::svd::{SVD, SVDDecomp, SVDError};
use num_complex::ComplexFloat;

use matamorph::mut_::MataConvertMut;
use matamorph::own::MataConvertOwn;
use matamorph::ref_::MataConvertRef;

// use super::simple::svd_nalgebra;
use crate::Nalgebra;

impl<T, D0, D1> SVD<T, D0, D1> for Nalgebra
where
    T: ComplexFloat
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
    D0: Dim,
    D1: Dim,
{
    /// Compute full SVD with new allocated matrices
    fn svd<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<SVDDecomp<T>, SVDError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);

        let a_nalgebra = a.view(.., ..).to_nalgebra();

        let mut s_mda = tensor![[T::default(); min_mn]; min_mn];
        let mut u_mda = tensor![[T::default(); m]; m];
        let mut vt_mda = tensor![[T::default(); n]; n];

        // match svd_nalgebra(a, &mut s_mda, Some(&mut u_mda), Some(&mut vt_mda)) {
        //     Err(_) => Err(SVDError::BackendDidNotConverge {
        //         superdiagonals: (0),
        //     }),
        //     Ok(_) => Ok(SVDDecomp {
        //         s: s_mda,
        //         u: u_mda,
        //         vt: vt_mda,
        //     }),
        // }
    }

    /// Compute only singular values with new allocated matrix
    fn svd_s<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<DTensor<T, 2>, SVDError> {
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
    fn svd_write<L: Layout, Ls: Layout, Lu: Layout, Lvt: Layout>(
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
    fn svd_write_s<L: Layout, Ls: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        s: &mut DSlice<T, 2, Ls>,
    ) -> Result<(), SVDError> {
        todo!()
        // svd_nalgebra::<T, D0, D1, L, Ls, Dense, Dense>(a, s, None, None)
    }
}
