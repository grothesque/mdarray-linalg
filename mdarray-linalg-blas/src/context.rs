use std::mem::MaybeUninit;

use mdarray::{DSlice, DTensor, Layout, tensor};
use num_complex::ComplexFloat;

use mdarray_linalg::{MatMul, MatMulBuilder};

use super::scalar::BlasScalar;
use super::simple::{gemm, gemm_uninit};

pub struct Blas;

struct BlasMatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
{
    alpha: T,
    a: &'a DSlice<T, 2, La>,
    b: &'a DSlice<T, 2, Lb>,
}

impl<'a, T, La, Lb> MatMulBuilder<'a, T, La, Lb> for BlasMatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> DTensor<T, 2> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let c = tensor![[MaybeUninit::<T>::uninit(); n]; m];
        gemm_uninit(self.alpha, self.a, self.b, c)
    }

    fn overwrite<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>) {
        gemm(self.alpha, self.a, self.b, 0.into().into(), c);
    }

    fn add_to<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>) {
        gemm(self.alpha, self.a, self.b, 1.into().into(), c);
    }

    fn add_to_scaled<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>, beta: T) {
        gemm(self.alpha, self.a, self.b, beta, c);
    }
}

impl<T> MatMul<T> for Blas
where
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    fn matmul<'a, La, Lb>(
        &self,
        a: &'a DSlice<T, 2, La>,
        b: &'a DSlice<T, 2, Lb>,
    ) -> impl MatMulBuilder<'a, T, La, Lb>
    where
        La: Layout,
        Lb: Layout,
    {
        BlasMatMulBuilder { alpha: 1.into().into(), a, b }
    }
}
