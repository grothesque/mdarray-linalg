// QR Decomposition:
//     A = Q * R
// where:
//     - A is m × n         (input matrix)
//     - Q is m × m        (orthogonal matrix)
//     - R is m × n         (upper triangular/trapezoidal matrix)
//     - For thin QR: Q is m × min(m,n) and R is min(m,n) × n

use faer_traits::ComplexField;
use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::qr::QR;
use num_complex::ComplexFloat;

use super::simple::qr_faer;
use crate::Faer;

impl<T, D0: Dim, D1: Dim> QR<T, D0, D1> for Faer
where
    T: ComplexFloat + ComplexField + Default,
{
    fn qr<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Array<T, (D0, usize)>, Array<T, (usize, D1)>) {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));
        let k = m.min(n);

        let q_shape = <(D0, usize) as Shape>::from_dims(&[m, k]);
        let r_shape = <(usize, D1) as Shape>::from_dims(&[k, n]);

        let mut q_mda = Array::from_elem(q_shape, T::default());
        let mut r_mda = Array::from_elem(r_shape, T::default());

        qr_faer(a, Some(&mut q_mda), &mut r_mda);
        (q_mda, r_mda)
    }

    fn qr_write<D2: Dim, L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        q: &mut Slice<T, (D0, D2), Lq>,
        r: &mut Slice<T, (D2, D1), Lr>,
    ) {
        qr_faer(a, Some(q), r)
    }
}
