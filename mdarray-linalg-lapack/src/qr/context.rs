//! QR Decomposition:
//!     A = Q * R
//! where:
//!     - A is m × n (input matrix)
//!     - Q is m × m (orthogonal matrix)
//!     - R is m × n (upper triangular matrix)
//! This decomposition is used to solve linear equations, least squares problems, and eigenvalue problems.
//! The function `geqrf` (LAPACK) computes the QR factorization of a general m-by-n matrix A using a blocking algorithm.
//! The matrix Q is orthogonal, and R is upper triangular.

use mdarray::{DSlice, DTensor, Dim, Layout, Shape, Slice, tensor};
use mdarray_linalg::{into_i32, qr::QR};
use num_complex::ComplexFloat;

use super::{
    scalar::{LapackScalar, NeedsRwork},
    simple::geqrf,
};
use crate::Lapack;

impl<T, D0: Dim, D1: Dim> QR<T, D0, D1> for Lapack
where
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
    T::Real: Into<T>,
{
    fn qr_write<L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        q: &mut DSlice<T, 2, Lq>,
        r: &mut DSlice<T, 2, Lr>,
    ) {
        geqrf(a, q, r)
    }

    fn qr<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> (DTensor<T, 2>, DTensor<T, 2>) {
        let ash = *a.shape();
        let (m, n) = (into_i32(ash.dim(0)), into_i32(ash.dim(1)));

        let mut q = tensor![[T::default(); m as usize]; m as usize];
        let mut r = tensor![[T::default(); n as usize]; m as usize];

        geqrf(a, &mut q, &mut r);

        (q, r)
    }
}
