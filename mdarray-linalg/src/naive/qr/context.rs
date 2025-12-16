use mdarray::{DSlice, DTensor, Dim, Layout, Slice};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};

use super::simple::naive_qr;
use crate::Naive;
use crate::qr::QR;

impl<T, D0: Dim, D1: Dim> QR<T, D0, D1> for Naive
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T>,
{
    fn qr_write<L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        q: &mut DSlice<T, 2, Lq>,
        r: &mut DSlice<T, 2, Lr>,
    ) {
        naive_qr(a, q, r);
    }

    fn qr<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> (DTensor<T, 2>, DTensor<T, 2>) {
        let (m, n) = *a.shape();

        let mut q = DTensor::<T, 2>::from_elem([m.size(), n.size()], T::zero());
        let mut r = DTensor::<T, 2>::from_elem([n.size(), n.size()], T::zero());

        naive_qr(a, &mut q, &mut r);

        (q, r)
    }
}
