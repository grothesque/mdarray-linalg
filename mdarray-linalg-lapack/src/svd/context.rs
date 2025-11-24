//! Singular Value Decomposition (SVD):
//!     A = U * Σ * V^T
//! where:
//!     - A is m × n         (input matrix)
//!     - U is m × m         (left singular vectors, orthogonal)
//!     - Σ is m × n         (diagonal matrix with singular values on the diagonal)
//!     - V^T is n × n       (transpose of right singular vectors, orthogonal)
//!     - s (Σ) contains min(m, n) singular values (non-negative, sorted in descending order) in the first row

use mdarray::{DSlice, DTensor, Dense, Layout, tensor};
use mdarray_linalg::{
    get_dims, into_i32,
    svd::{SVD, SVDDecomp, SVDError},
};
use num_complex::ComplexFloat;

use super::{
    scalar::{LapackScalar, NeedsRwork},
    simple::gsvd,
};
use crate::Lapack;

impl<T> SVD<T> for Lapack
where
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
    T::Real: Into<T>,
{
    // Computes full SVD with new allocated matrices
    fn svd<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> Result<SVDDecomp<T>, SVDError> {
        let (m, n) = get_dims!(a);
        let min_mn = m.min(n);

        let mut s = tensor![[T::default(); min_mn as usize]; min_mn as usize];
        let mut u = tensor![[T::default(); m as usize]; m as usize];
        let mut vt = tensor![[T::default(); n as usize]; n as usize];

        match gsvd(a, &mut s, Some(&mut u), Some(&mut vt), self.svd_config) {
            Ok(_) => Ok(SVDDecomp { s, u, vt }),
            Err(e) => Err(e),
        }
    }

    // Computes only singular values with new allocated matrix
    fn svd_s<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> Result<DTensor<T, 2>, SVDError> {
        let (m, n) = get_dims!(a);
        let min_mn = m.min(n);

        // Only allocate space for singular values
        let mut s = tensor![[T::default(); min_mn as usize]; min_mn as usize];

        match gsvd::<L, Dense, Dense, Dense, T>(a, &mut s, None, None, self.svd_config) {
            Ok(_) => Ok(s),
            Err(err) => Err(err),
        }
    }

    // Computes full SVD, overwriting existing matrices
    fn svd_write<L: Layout, Ls: Layout, Lu: Layout, Lvt: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        s: &mut DSlice<T, 2, Ls>,
        u: &mut DSlice<T, 2, Lu>,
        vt: &mut DSlice<T, 2, Lvt>,
    ) -> Result<(), SVDError> {
        gsvd(a, s, Some(u), Some(vt), self.svd_config)
    }

    // Computes only singular values, overwriting existing matrix
    fn svd_write_s<L: Layout, Ls: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        s: &mut DSlice<T, 2, Ls>,
    ) -> Result<(), SVDError> {
        gsvd::<L, Ls, Dense, Dense, T>(a, s, None, None, self.svd_config)
    }
}
