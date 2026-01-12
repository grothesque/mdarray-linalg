//! QR Decomposition:
//!     A = Q * R
//! where:
//!     - A is m × n (input matrix)
//!     - Q is m × m (orthogonal matrix)
//!     - R is m × n (upper triangular matrix)
//! This decomposition is used to solve linear equations, least squares problems, and eigenvalue problems.
//! The function `geqrf` (LAPACK) computes the QR factorization of a general m-by-n matrix A using a blocking algorithm.
//! The matrix Q is orthogonal, and R is upper triangular.

use mdarray::{Dim, Layout, Slice, Tensor};
use mdarray_linalg::qr::QR;
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
        q: &mut Slice<T, (D0, D1), Lq>,
        r: &mut Slice<T, (D0, D1), Lr>,
    ) {
        geqrf(a, q, r)
    }

    fn qr<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Tensor<T, (D0, D1)>, Tensor<T, (D0, D1)>) {
        let ash = *a.shape();

        let mut q = Tensor::from_elem(ash, T::default());
        let mut r = Tensor::from_elem(ash, T::default());

        geqrf(a, &mut q, &mut r);

        (q, r)
    }
}
