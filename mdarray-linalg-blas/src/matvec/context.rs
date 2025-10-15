use cblas_sys::CBLAS_UPLO;
use mdarray::{DSlice, DTensor, Layout, Shape, Slice};
use num_complex::ComplexFloat;

use mdarray_linalg::matmul::{Triangle, Type};
use mdarray_linalg::matvec::{MatVec, MatVecBuilder, VecOps};

use crate::Blas;

use super::scalar::BlasScalar;
use super::simple::{asum, axpy, dotc, dotu, gemv, ger, her, nrm2, syr};

struct BlasMatVecBuilder<'a, T, La, Lx>
where
    La: Layout,
    Lx: Layout,
{
    alpha: T,
    a: &'a DSlice<T, 2, La>,
    x: &'a DSlice<T, 1, Lx>,
}

impl<'a, T, La, Lx> MatVecBuilder<'a, T, La, Lx> for BlasMatVecBuilder<'a, T, La, Lx>
where
    La: Layout,
    Lx: Layout,
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    fn parallelize(self) -> Self {
        self
    }

    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    fn eval(self) -> DTensor<T, 1> {
        let mut y = DTensor::<T, 1>::from_elem(self.x.len(), 0.into().into());
        gemv(self.alpha, self.a, self.x, 0.into().into(), &mut y);
        y
    }

    fn overwrite<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>) {
        gemv(self.alpha, self.a, self.x, 0.into().into(), y);
    }

    fn add_to<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>) {
        gemv(self.alpha, self.a, self.x, 1.into().into(), y);
    }

    fn add_to_scaled<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>, beta: T) {
        gemv(self.alpha, self.a, self.x, beta, y);
    }

    fn add_outer<Ly: Layout>(self, y: &DSlice<T, 1, Ly>, beta: T) -> DTensor<T, 2> {
        let mut a_copy = DTensor::<T, 2>::from_elem(*self.a.shape(), 0.into().into());
        a_copy.assign(self.a);

        // Apply scale factor to preserve builder pattern logic: the alpha parameter
        // may have been modified before this call, so we must scale the matrix
        // before applying the rank-1 update. Unlike gemm operations, this requires
        // a separate pass since BLAS lacks a direct matrix-scalar multiplication.

        if self.alpha != 1.into().into() {
            a_copy = a_copy.map(|x| x * self.alpha);
        }

        ger(beta, self.x, y, &mut a_copy);
        a_copy
    }

    fn add_outer_special(self, beta: T, ty: Type, tr: Triangle) -> DTensor<T, 2> {
        let mut a_copy = DTensor::<T, 2>::from_elem(*self.a.shape(), 0.into().into());
        a_copy.assign(self.a);

        if self.alpha != 1.into().into() {
            a_copy = a_copy.map(|x| x * self.alpha);
        }

        let cblas_uplo = match tr {
            Triangle::Lower => CBLAS_UPLO::CblasLower,
            Triangle::Upper => CBLAS_UPLO::CblasUpper,
        };

        match ty {
            Type::Her => her(cblas_uplo, beta.re(), self.x, &mut a_copy),
            Type::Sym => syr(cblas_uplo, beta, self.x, &mut a_copy),
            Type::Tri => {
                ger(beta, self.x, self.x, &mut a_copy);
            }
        }

        a_copy
    }
}

impl<T> MatVec<T> for Blas
where
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    fn matvec<'a, La, Lx>(
        &self,
        a: &'a DSlice<T, 2, La>,
        x: &'a DSlice<T, 1, Lx>,
    ) -> impl MatVecBuilder<'a, T, La, Lx>
    where
        La: Layout,
        Lx: Layout,
    {
        BlasMatVecBuilder {
            alpha: 1.into().into(),
            a,
            x,
        }
    }
}

impl<T: ComplexFloat + BlasScalar + 'static> VecOps<T> for Blas {
    fn add_to_scaled<Lx: Layout, Ly: Layout>(
        &self,
        alpha: T,
        x: &DSlice<T, 1, Lx>,
        y: &mut DSlice<T, 1, Ly>,
    ) {
        axpy(alpha, x, y);
    }

    fn dot<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T {
        dotu(x, y)
    }

    fn dotc<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T {
        dotc(x, y)
    }

    fn norm2<Lx: Layout>(&self, x: &DSlice<T, 1, Lx>) -> T::Real {
        nrm2(x)
    }

    fn norm1<Lx: Layout>(&self, x: &DSlice<T, 1, Lx>) -> T::Real
    where
        T: ComplexFloat,
    {
        asum(x)
    }

    fn argmax_overwrite<Lx: Layout, S: Shape>(
        &self,
        x: &Slice<T, S, Lx>,
        output: &mut Vec<usize>,
    ) -> bool {
        todo!()
    }

    fn argmax<Lx: Layout, S: Shape>(&self, _x: &Slice<T, S, Lx>) -> Option<Vec<usize>> {
        todo!()
    }
    fn copy<Lx: Layout, Ly: Layout>(&self, _x: &DSlice<T, 1, Lx>, _y: &mut DSlice<T, 1, Ly>) {
        todo!()
    }
    fn scal<Lx: Layout>(&self, _alpha: T, _x: &mut DSlice<T, 1, Lx>) {
        todo!()
    }
    fn swap<Lx: Layout, Ly: Layout>(&self, _x: &mut DSlice<T, 1, Lx>, _y: &mut DSlice<T, 1, Ly>) {
        todo!()
    }
    fn rot<Lx: Layout, Ly: Layout>(
        &self,
        _x: &mut DSlice<T, 1, Lx>,
        _y: &mut DSlice<T, 1, Ly>,
        _c: T::Real,
        _s: T,
    ) where
        T: ComplexFloat,
    {
        todo!()
    }
}
