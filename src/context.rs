use std::mem::MaybeUninit;

use mdarray::{DSlice, DTensor, Layout, tensor};
use num_complex::ComplexFloat;

use crate::BlasScalar;
use crate::simple::{gemm, gemm_uninit};
use crate::traits::MatMul;

pub struct Blas;

pub struct MatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
{
    alpha: T,
    a: &'a DSlice<T, 2, La>,
    b: &'a DSlice<T, 2, Lb>,
}

impl<T, La, Lb> MatMulBuilder<'_, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    pub fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    pub fn to_owned(self) -> DTensor<T, 2> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let c = tensor![[MaybeUninit::<T>::uninit(); n]; m];
        gemm_uninit(self.alpha, self.a, self.b, c)
    }

    pub fn overwrite<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>) {
        gemm(self.alpha, self.a, self.b, 0.into().into(), c);
    }

    pub fn add_to<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>) {
        gemm(self.alpha, self.a, self.b, 1.into().into(), c);
    }

    pub fn add_to_scaled<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>, beta: T) {
        gemm(self.alpha, self.a, self.b, beta, c);
    }
}

impl MatMul for Blas {
    type MatMulBuilder<'a, T, La, Lb>
        = MatMulBuilder<'a, T, La, Lb>
    where
        La: Layout,
        Lb: Layout,
        T: 'a,
        La: 'a,
        Lb: 'a;

    fn matmul<'a, T, La, Lb>(
        &self,
        a: &'a DSlice<T, 2, La>,
        b: &'a DSlice<T, 2, Lb>,
    ) -> Self::MatMulBuilder<'a, T, La, Lb>
    where
        La: Layout,
        Lb: Layout,
        T: ComplexFloat,
        i8: Into<T::Real>,
        T::Real: Into<T>,
    {
        MatMulBuilder { alpha: 1.into().into(), a, b }
    }
}
