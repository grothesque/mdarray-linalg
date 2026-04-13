use mdarray::{Array, Dim, Layout, Shape, Slice};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};

use super::simple::naive_qr;
use crate::Naive;
use crate::qr::QR;

impl<T, D0: Dim, D1: Dim> QR<T, D0, D1> for Naive
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T>,
{
    fn qr_write<D2: Dim, L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        q: &mut Slice<T, (D0, D2), Lq>,
        r: &mut Slice<T, (D2, D1), Lr>,
    ) {
        naive_qr(a, q, r);
    }

    fn qr<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Array<T, (D0, usize)>, Array<T, (usize, D1)>) {
        let ash = *a.shape();
        let m = ash.dim(0);
        let n = ash.dim(1);

        // let mut q = Array::<T, (D0, usize)>::from_elem([m, m], T::zero());
        // let mut r = Array::<T, (usize, D1)>::from_elem([m, m], T::zero());

        let q_shape = <(D0, usize) as Shape>::from_dims(&[m, m]);
        let r_shape = <(usize, D1) as Shape>::from_dims(&[m, n]);

        let mut q = Array::<T, (D0, usize)>::from_elem(q_shape, T::zero());
        let mut r = Array::<T, (usize, D1)>::from_elem(r_shape, T::zero());

        naive_qr(a, &mut q, &mut r);

        (q, r)
    }
}
