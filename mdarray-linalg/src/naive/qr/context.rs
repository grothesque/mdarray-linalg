use mdarray::{Dim, Layout, Slice, Tensor};
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
        q: &mut Slice<T, (D0, D1), Lq>,
        r: &mut Slice<T, (D0, D1), Lr>,
    ) {
        naive_qr(a, q, r);
    }

    fn qr<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Tensor<T, (D0, D1)>, Tensor<T, (D0, D1)>) {
        let mut q = Tensor::<T, (D0, D1)>::from_elem(*a.shape(), T::zero());
        let mut r = Tensor::<T, (D0, D1)>::from_elem(*a.shape(), T::zero());

        naive_qr(a, &mut q, &mut r);

        (q, r)
    }
}
