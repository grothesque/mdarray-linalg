// QR Decomposition:
//     A = Q * R
// where:
//     - A is m × n         (input matrix)
//     - Q is m × m        (orthogonal matrix)
//     - R is m × n         (upper triangular/trapezoidal matrix)
//     - For thin QR: Q is m × min(m,n) and R is min(m,n) × n

use super::simple::qr_faer;
use faer_traits::ComplexField;
use mdarray::{DSlice, DTensor, Layout, tensor};
use mdarray_linalg::{QR, identity};
use num_complex::ComplexFloat;

use crate::Faer;

impl<T> QR<T> for Faer
where
    T: ComplexFloat
        + ComplexField
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
{
    /// Compute full QR decomposition with new allocated matrices
    fn qr<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> (DTensor<T, 2>, DTensor<T, 2>) {
        let (m, n) = *a.shape();
        let mut q_mda = identity(m);
        let mut r_mda = tensor![[T::default(); n]; m];

        qr_faer(a, Some(&mut q_mda), &mut r_mda);
        (q_mda, r_mda)
    }

    /// Compute full QR decomposition, overwriting existing matrices
    fn qr_overwrite<L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        q: &mut DSlice<T, 2, Lq>,
        r: &mut DSlice<T, 2, Lr>,
    ) {
        qr_faer::<T, L, Lq, Lr>(a, Some(q), r)
    }
}
