use mdarray::{DSlice, DTensor, Dim, Layout};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};

use super::simple::naive_qr;
use crate::Naive;
use crate::qr::QR;

impl<T> QR<T> for Naive
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T>,
{
    fn qr_write<L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        q: &mut DSlice<T, 2, Lq>,
        r: &mut DSlice<T, 2, Lr>,
    ) {
        naive_qr(a, q, r);
    }

    fn qr<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> (DTensor<T, 2>, DTensor<T, 2>) {
        let (m, n) = *a.shape();

        let mut q = DTensor::<T, 2>::from_elem([m, n], T::zero());
        let mut r = DTensor::<T, 2>::from_elem([n, n], T::zero());

        naive_qr(a, &mut q, &mut r);

        (q, r)
    }
}
