use mdarray::{Slice, Tensor};
use num_complex::ComplexFloat;
use num_traits::Zero;

use mdarray_linalg::tensordot::{Tensordot, TensordotBuilder};

use super::simple::{Axes, tensordot};
use crate::Naive;

struct NaiveTensordotBuilder<'a, T> {
    a: &'a Slice<T>,
    b: &'a Slice<T>,
    axes: Axes,
}

impl<'a, T> TensordotBuilder<'a, T> for NaiveTensordotBuilder<'a, T>
where
    T: Zero + ComplexFloat + std::fmt::Debug,
{
    fn contract_k(mut self, k: isize) -> Self {
        self.axes = Axes::LastFirst { k };
        self
    }

    fn specific(mut self, axes_a: &[isize], axes_b: &[isize]) -> Self {
        self.axes = Axes::Specific(
            axes_a.to_vec().into_boxed_slice(),
            axes_b.to_vec().into_boxed_slice(),
        );
        self
    }

    fn eval(self) -> Tensor<T> {
        tensordot(self.a, self.b, self.axes)
    }

    fn overwrite(self, c: &mut Slice<T>) {
        let result = tensordot(self.a, self.b, self.axes);
        c.assign(&result);
    }
}

impl<T> Tensordot<T> for Naive
where
    T: Zero + ComplexFloat + std::fmt::Debug,
{
    fn tensordot<'a>(&self, a: &'a Slice<T>, b: &'a Slice<T>) -> impl TensordotBuilder<'a, T> {
        NaiveTensordotBuilder {
            a,
            b,
            axes: Axes::All,
        }
    }
}
