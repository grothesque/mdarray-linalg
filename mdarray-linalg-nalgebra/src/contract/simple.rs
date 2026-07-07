use mdarray::{Dim, Layout, Shape, Slice};
use num_complex::ComplexFloat;
use num_traits::{One, Zero};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign};

use crate::{to_dmatrix, write_dmatrix};

pub(super) fn gemm<T, La, Lb, Lc, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    c: &mut Slice<T, (D0, D2), Lc>,
) where
    T: nalgebra::Scalar + ComplexFloat + Zero + One + ClosedAddAssign + ClosedMulAssign + Copy,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let a_nalgebra = to_dmatrix(a);
    let b_nalgebra = to_dmatrix(b);
    let mut c_nalgebra = if beta.is_zero() {
        nalgebra::DMatrix::from_element(c.shape().dim(0), c.shape().dim(1), T::zero())
    } else {
        to_dmatrix(c)
    };

    c_nalgebra.gemm(alpha, &a_nalgebra, &b_nalgebra, beta);
    write_dmatrix(&c_nalgebra, c);
}
