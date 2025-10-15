use std::mem::MaybeUninit;

use cblas_sys::{CBLAS_SIDE, CBLAS_UPLO};
use mdarray::{DSlice, DTensor, Dense, Layout, tensor};
use num_complex::ComplexFloat;

use mdarray_linalg::matmul::{MatMul, MatMulBuilder, Side, Triangle, Type};

use super::scalar::BlasScalar;
use super::simple::{gemm, gemm_uninit, hemm_uninit, symm_uninit, trmm};

use crate::Blas;

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
    fn parallelize(self) -> Self {
        self
    }

    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> DTensor<T, 2> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let c = tensor![[MaybeUninit::<T>::uninit(); n]; m];
        gemm_uninit::<T, La, Lb, Dense>(self.alpha, self.a, self.b, 0.into().into(), c)
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

    fn special(self, lr: Side, type_of_matrix: Type, tr: Triangle) -> DTensor<T, 2> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let c = tensor![[MaybeUninit::<T>::uninit(); n]; m];
        let cblas_side = match lr {
            Side::Left => CBLAS_SIDE::CblasLeft,
            Side::Right => CBLAS_SIDE::CblasRight,
        };
        let cblas_triangle = match tr {
            Triangle::Lower => CBLAS_UPLO::CblasLower,
            Triangle::Upper => CBLAS_UPLO::CblasUpper,
        };
        match type_of_matrix {
            Type::Her => hemm_uninit::<T, La, Lb, Dense>(
                self.alpha,
                self.a,
                self.b,
                0.into().into(),
                c,
                cblas_side,
                cblas_triangle,
            ),
            Type::Sym => symm_uninit::<T, La, Lb, Dense>(
                self.alpha,
                self.a,
                self.b,
                0.into().into(),
                c,
                cblas_side,
                cblas_triangle,
            ),
            Type::Tri => {
                let mut b_copy = DTensor::<T, 2>::from_elem(*self.b.shape(), 0.into().into());
                b_copy.assign(self.b);
                trmm(self.alpha, self.a, &mut b_copy, cblas_side, cblas_triangle);
                b_copy
            }
        }
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
        BlasMatMulBuilder {
            alpha: 1.into().into(),
            a,
            b,
        }
    }
}
