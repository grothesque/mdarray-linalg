use mdarray::{Dim, Layout, Shape, Slice};
use num_complex::ComplexFloat;
use num_traits::{One, Zero};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign};

use crate::{to_dmatrix, to_dvector, write_dmatrix, write_dvector};

pub(super) fn gemv<T, D0, D1, La, Lx, Ly>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    x: &Slice<T, (D1,), Lx>,
    beta: T,
    y: &mut Slice<T, (D0,), Ly>,
) where
    T: nalgebra::Scalar + ComplexFloat + Zero + One + ClosedAddAssign + ClosedMulAssign + Copy,
    D0: Dim,
    D1: Dim,
    La: Layout,
    Lx: Layout,
    Ly: Layout,
{
    let a_nalgebra = to_dmatrix(a);
    let x_nalgebra = to_dvector(x);
    let mut y_nalgebra = if beta.is_zero() {
        nalgebra::DVector::from_element(y.len(), T::zero())
    } else {
        to_dvector(y)
    };

    y_nalgebra.gemv(alpha, &a_nalgebra, &x_nalgebra, beta);
    write_dvector(&y_nalgebra, y);
}

pub(super) fn ger<T, Dx, Dy, Lx, Ly, La>(
    alpha: T,
    x: &Slice<T, (Dx,), Lx>,
    y: &Slice<T, (Dy,), Ly>,
    beta: T,
    a: &mut Slice<T, (Dx, Dy), La>,
) where
    T: nalgebra::Scalar + ComplexFloat + Zero + One + ClosedAddAssign + ClosedMulAssign + Copy,
    Dx: Dim,
    Dy: Dim,
    Lx: Layout,
    Ly: Layout,
    La: Layout,
{
    let x_nalgebra = to_dvector(x);
    let y_nalgebra = to_dvector(y);
    let mut a_nalgebra = if beta.is_zero() {
        nalgebra::DMatrix::from_element(a.shape().dim(0), a.shape().dim(1), T::zero())
    } else {
        to_dmatrix(a)
    };

    a_nalgebra.ger(alpha, &x_nalgebra, &y_nalgebra, beta);
    write_dmatrix(&a_nalgebra, a);
}

pub(super) fn axpy<T, D1, Lx, Ly>(alpha: T, x: &Slice<T, (D1,), Lx>, y: &mut Slice<T, (D1,), Ly>)
where
    T: nalgebra::Scalar + ComplexFloat + Zero + One + ClosedAddAssign + ClosedMulAssign + Copy,
    D1: Dim,
    Lx: Layout,
    Ly: Layout,
{
    let x_nalgebra = to_dvector(x);
    let mut y_nalgebra = to_dvector(y);
    y_nalgebra.axpy(alpha, &x_nalgebra, T::one());
    write_dvector(&y_nalgebra, y);
}
