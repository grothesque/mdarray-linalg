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

use mdarray::{DSlice, DTensor, Layout, tensor};
use mdarray_linalg::into_i32;
use mdarray_linalg::{QR, QRBuilder};
use num_complex::ComplexFloat;
use std::mem::MaybeUninit;
pub struct Lapack;
use super::scalar::LapackScalar;

struct LapackQRBuilder<'a, T, L>
where
    L: Layout,
{
    a: &'a mut DSlice<T, 2, L>,
}

impl<'a, T, L> QRBuilder<'a, T, L> for LapackQRBuilder<'a, T, L>
where
    T: ComplexFloat + Default + LapackScalar,
    T::Real: Into<T>,
    L: Layout,
{
    fn overwrite<Lq: Layout, Lr: Layout>(
        &mut self,
        q: &mut DSlice<T, 2, Lq>,
        r: &mut DSlice<T, 2, Lr>,
    ) {
        geqrf(self.a, q, r)
    }

    fn eval<Lq: Layout, Lr: Layout>(&mut self) -> (DTensor<T, 2>, DTensor<T, 2>) {
        let (m, n) = get_dims!(self.a);
        let q = tensor![[MaybeUninit::<T>::uninit(); m as usize]; m as usize];
        let r = tensor![[MaybeUninit::<T>::uninit(); n as usize]; m as usize];
        geqrf_uninit::<L, T>(self.a, q, r)
    }
}

impl<T> QR<T> for Lapack
where
    T: ComplexFloat + Default + LapackScalar,
    T::Real: Into<T>,
{
    fn qr<'a, L: Layout>(&self, a: &'a mut DSlice<T, 2, L>) -> impl QRBuilder<'a, T, L> {
        LapackQRBuilder { a }
    }
}
