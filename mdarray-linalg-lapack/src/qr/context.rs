//! QR Decomposition:
//!     A = Q * R
//! where:
//!     - A is m × n (input matrix)
//!     - Q is m × m (orthogonal matrix)
//!     - R is m × n (upper triangular matrix)
//! This decomposition is used to solve linear equations, least squares problems, and eigenvalue problems.
//! The function `geqrf` (LAPACK) computes the QR factorization of a general m-by-n matrix A using a blocking algorithm.
//! The matrix Q is orthogonal, and R is upper triangular.

use super::scalar::NeedsRwork;
use super::simple::geqrf;
use mdarray_linalg::get_dims;

use super::scalar::LapackScalar;
use mdarray::{DSlice, DTensor, Layout, tensor};
use mdarray_linalg::into_i32;
use mdarray_linalg::qr::QR;
use num_complex::ComplexFloat;

use crate::Lapack;

impl<T> QR<T> for Lapack
where
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
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
        let mut q = tensor![[T::default(); m as usize]; m as usize];
        let mut r = tensor![[T::default(); n as usize]; m as usize];

        geqrf(a, &mut q, &mut r);

        (q, r)
    }
}
