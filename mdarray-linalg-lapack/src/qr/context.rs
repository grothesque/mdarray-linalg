//! QR Decomposition:
//!     A = Q * R
//! where:
//!     - A is m × n (input matrix)
//!     - Q is m × m (orthogonal matrix)
//!     - R is m × n (upper triangular matrix)
//! This decomposition is used to solve linear equations, least squares problems, and eigenvalue problems.
//! The function `geqrf` (LAPACK) computes the QR factorization of a general m-by-n matrix A using a blocking algorithm.
//! The matrix Q is orthogonal, and R is upper triangular.

use super::simple::{geqrf, geqrf_uninit};
use mdarray_linalg::get_dims;

use super::scalar::LapackScalar;
use mdarray::{DSlice, DTensor, Layout, tensor};
use mdarray_linalg::QR;
use mdarray_linalg::into_i32;
use num_complex::ComplexFloat;
use std::mem::MaybeUninit;

use crate::Lapack;

impl<T> QR<T> for Lapack
where
    T: ComplexFloat + Default + LapackScalar,
    T::Real: Into<T>,
{
    fn qr_overwrite<L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        q: &mut DSlice<T, 2, Lq>,
        r: &mut DSlice<T, 2, Lr>,
    ) {
        geqrf(a, q, r)
    }

    fn qr<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> (DTensor<T, 2>, DTensor<T, 2>) {
        let (m, n) = get_dims!(a);
        let q = tensor![[MaybeUninit::<T>::uninit(); m as usize]; m as usize];
        let r = tensor![[MaybeUninit::<T>::uninit(); n as usize]; m as usize];
        geqrf_uninit::<L, T>(a, q, r)
    }
}
