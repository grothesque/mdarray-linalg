use mdarray::{Dim, Layout, Shape, Slice};
use num_complex::ComplexFloat;
use num_traits::Zero;

use crate::to_dmatrix;

/// Compute the reduced QR decomposition.
pub(super) fn qr_reduced<T, D0, D1, L>(
    a: &Slice<T, (D0, D1), L>,
) -> (nalgebra::DMatrix<T>, nalgebra::DMatrix<T>)
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let qr = to_dmatrix(a).qr();
    (qr.q(), qr.r())
}

/// Compute the complete QR decomposition.
pub(super) fn qr_complete<T, D0, D1, L>(
    a: &Slice<T, (D0, D1), L>,
) -> (nalgebra::DMatrix<T>, nalgebra::DMatrix<T>)
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let m = a.shape().dim(0);
    let n = a.shape().dim(1);
    let k = m.min(n);

    let qr = to_dmatrix(a).qr();
    let r_thin = qr.r();

    let mut qh = nalgebra::DMatrix::identity(m, m);
    qr.q_tr_mul(&mut qh);
    let q = qh.adjoint();

    let mut r = nalgebra::DMatrix::from_element(m, n, T::zero());
    for i in 0..k {
        for j in 0..n {
            r[(i, j)] = r_thin[(i, j)];
        }
    }

    (q, r)
}
