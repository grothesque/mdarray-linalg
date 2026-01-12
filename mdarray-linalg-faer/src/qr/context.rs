// QR Decomposition:
//     A = Q * R
// where:
//     - A is m × n         (input matrix)
//     - Q is m × m        (orthogonal matrix)
//     - R is m × n         (upper triangular/trapezoidal matrix)
//     - For thin QR: Q is m × min(m,n) and R is min(m,n) × n

use faer_traits::ComplexField;
use mdarray::{Dim, Layout, Shape, Slice, Tensor};
use mdarray_linalg::{identity, into_i32, qr::QR};
use num_complex::ComplexFloat;

use super::simple::qr_faer;
use crate::Faer;

impl<T, D0: Dim, D1: Dim> QR<T, D0, D1> for Faer
where
    T: ComplexFloat
        + ComplexField
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
{
    /// Compute full QR decomposition with new allocated matrices
    fn qr<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Tensor<T, (D0, D1)>, Tensor<T, (D0, D1)>) {
        let ash = *a.shape();
        let (m, _) = (into_i32(ash.dim(0)), into_i32(ash.dim(1)));

        let mut q_mda = identity(m as usize);

        let mut r_mda = Tensor::from_elem(ash, T::default());

        qr_faer(a, Some(&mut q_mda), &mut r_mda);
        (q_mda, r_mda)
    }

    /// Compute full QR decomposition, overwriting existing matrices
    fn qr_write<L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        q: &mut Slice<T, (D0, D1), Lq>,
        r: &mut Slice<T, (D0, D1), Lr>,
    ) {
        qr_faer::<T, D0, D1, L, Lq, Lr>(a, Some(q), r)
    }
}
