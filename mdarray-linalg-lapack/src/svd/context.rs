//! Singular Value Decomposition (SVD):
//!     A = U * Σ * V^T
//! where:
//!     - A is m × n         (input matrix)
//!     - U is m × m         (left singular vectors, orthogonal)
//!     - Σ is µ × µ         (diagonal matrix with singular values on the diagonal, µ = min(m,n))
//!     - V^T is n × n       (transpose of right singular vectors, orthogonal)
//!     - s (Σ) contains min(m, n) singular values (non-negative, sorted in descending order) in the first row

use mdarray::{Dense, Dim, Layout, Shape, Slice, Tensor};
use mdarray_linalg::svd::{SVD, SVDDecomp, SVDError};
use num_complex::ComplexFloat;

use super::{
    scalar::{LapackScalar, NeedsRwork},
    simple::gsvd,
};
use crate::Lapack;

impl<T, D, L> SVD<T, D, L> for Lapack
where
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
    T::Real: Into<T>,
    D: Dim,
    L: Layout,
{
    // Computes full SVD with new allocated matrices
    fn svd(&self, a: &mut Slice<T, (D, D), L>) -> Result<SVDDecomp<T, D>, SVDError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));
        let min_mn = m.min(n);

        let s_shape = <(D, D) as Shape>::from_dims(&[min_mn, min_mn]);
        let u_shape = <(D, D) as Shape>::from_dims(&[m, m]);
        let vt_shape = <(D, D) as Shape>::from_dims(&[n, n]);

        let mut s = Tensor::from_elem(s_shape, T::default());
        let mut u = Tensor::from_elem(u_shape, T::default());
        let mut vt = Tensor::from_elem(vt_shape, T::default());

        match gsvd(a, &mut s, Some(&mut u), Some(&mut vt), self.svd_config) {
            Ok(_) => Ok(SVDDecomp { s, u, vt }),
            Err(e) => Err(e),
        }
    }

    // Computes only singular values with new allocated matrix
    fn svd_s(&self, a: &mut Slice<T, (D, D), L>) -> Result<Tensor<T, (D, D)>, SVDError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);

        // Only allocate space for singular values
        let s_shape = <(D, D) as Shape>::from_dims(&[min_mn, min_mn]);
        let mut s = Tensor::from_elem(s_shape, T::default());

        match gsvd::<T, D, L, Dense, Dense, Dense>(a, &mut s, None, None, self.svd_config) {
            Ok(_) => Ok(s),
            Err(err) => Err(err),
        }
    }

    // Computes full SVD, overwriting existing matrices
    fn svd_write<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<T, (D, D), Ls>,
        u: &mut Slice<T, (D, D), Lu>,
        vt: &mut Slice<T, (D, D), Lvt>,
    ) -> Result<(), SVDError> {
        gsvd(a, s, Some(u), Some(vt), self.svd_config)
    }

    // Computes only singular values, overwriting existing matrix
    fn svd_write_s<Ls: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<T, (D, D), Ls>,
    ) -> Result<(), SVDError> {
        gsvd::<T, D, L, Ls, Dense, Dense>(a, s, None, None, self.svd_config)
    }
}
