//! Singular Value Decomposition (SVD):
//!     A = U * Σ * Vᵀ
//! where:
//!     - A is m × n         (input matrix)
//!     - U is m × m         (left singular vectors, orthogonal)
//!     - Σ is m × n         (diagonal matrix with singular values on the diagonal)
//!     - Vᵀ is n × n        (transpose of right singular vectors, orthogonal)
//!     - s (Σ) contains min(m, n) singular values (non-negative, sorted in descending order) in the first row

use super::simple::{dgesdd, dgesdd_uninit};
use crate::get_dims;

use super::simple::into_i32;
use mdarray::{DSlice, DTensor, Dense, Layout, tensor};
use mdarray_linalg::{SVD, SVDBuilder, SVDError};
use num_complex::ComplexFloat;
use std::mem::MaybeUninit;
pub struct Lapack;
use super::scalar::{LapackScalar, NeedsRwork};

struct LapackSVDBuilder<'a, T, L>
where
    L: Layout,
{
    a: &'a mut DSlice<T, 2, L>,
}

impl<'a, T, L> SVDBuilder<'a, T, L> for LapackSVDBuilder<'a, T, L>
where
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
    T::Real: Into<T>,
    L: Layout,
{
    fn overwrite_suvt<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &mut self,
        s: &mut DSlice<T, 2, Ls>,
        u: &mut DSlice<T, 2, Lu>,
        vt: &mut DSlice<T, 2, Lvt>,
    ) -> Result<(), SVDError> {
        dgesdd(self.a, s, Some(u), Some(vt))
    }

    fn overwrite_s<Ls: Layout>(&mut self, s: &mut DSlice<T, 2, Ls>) -> Result<(), SVDError> {
        dgesdd::<L, Ls, Dense, Dense, T>(self.a, s, None, None)
    }

    fn eval<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &mut self,
    ) -> Result<(DTensor<T, 2>, DTensor<T, 2>, DTensor<T, 2>), SVDError> {
        let (m, n) = get_dims!(self.a);
        let s = tensor![[MaybeUninit::<T>::uninit(); n as usize]; m as usize];
        let u = tensor![[MaybeUninit::<T>::uninit(); m as usize]; m as usize];
        let vt = tensor![[MaybeUninit::<T>::uninit(); n as usize]; n as usize];
        dgesdd_uninit::<_, Lu, Ls, Lvt, T>(self.a, s, Some(u), Some(vt))
    }

    fn eval_s<Ls: Layout, Lu: Layout, Lvt: Layout>(&mut self) -> Result<DTensor<T, 2>, SVDError> {
        let (m, n) = get_dims!(self.a);
        let s = tensor![[MaybeUninit::<T>::uninit(); n as usize]; m as usize];
        match dgesdd_uninit::<_, Lu, Ls, Lvt, T>(self.a, s, None, None) {
            Ok((s, _, _)) => Ok(s),
            Err(err) => Err(err),
        }
    }
}

impl<T> SVD<T> for Lapack
where
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
    T::Real: Into<T>,
{
    fn svd<'a, L: Layout>(&self, a: &'a mut DSlice<T, 2, L>) -> impl SVDBuilder<'a, T, L> {
        LapackSVDBuilder { a }
    }
}
