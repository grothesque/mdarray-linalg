use std::mem::MaybeUninit;

use cblas_sys::{CBLAS_SIDE, CBLAS_UPLO};
use mdarray::{DSlice, DTensor, Dense, Layout, tensor};
use num_complex::ComplexFloat;

use mdarray_linalg::{MatVec, MatVecBuilder, Side, Triangle, Type};

use crate::Blas;

use super::scalar::BlasScalar;
use super::simple::{gemv, ger, scal};

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

    /// For symmetric, Hermitian, or triangular matrices
    /// BLAS: SSYMV, DSYMV, CHEMV, ZHEMV, STRMV, DTRMV, CTRMV, ZTRMV
    fn symmetric(self, tr: Triangle) -> Self {
        unimplemented!()
    }
    fn hermitian(self, tr: Triangle) -> Self {
        unimplemented!()
    }
    fn triangular(self, tr: Triangle) -> Self {
        unimplemented!()
    }

    fn add_outer<Ly: Layout>(self, y: &DSlice<T, 1, Ly>, beta: T) -> DTensor<T, 2> {
        // if self.alpha != 1.into().into() {
        //     let mut x_copy = DTensor::<T, 1>::from_elem(*self.x.shape(), 0.into().into());
        //     x_copy.assign(self.x);
        //     scal(self.alpha, &mut x_copy);
        //     self.x.assign(x_copy);
        // }
        let mut a_copy = DTensor::<T, 2>::from_elem(*self.a.shape(), 0.into().into());
        a_copy.assign(self.a);
        ger(beta, self.x, y, &mut a_copy);
        a_copy
    }

    fn add_outer_special<Ly: Layout>(
        self,
        y: &DSlice<T, 1, Ly>,
        beta: T,
        ty: Type,
    ) -> DTensor<T, 2> {
        // if self.alpha != 1.into().into() {
        //     scal(self.alpha, &mut self.x);
        // }
        let mut a_copy = DTensor::<T, 2>::from_elem(*self.a.shape(), 0.into().into());
        a_copy.assign(self.a);
        ger(beta, self.x, y, &mut a_copy);
        a_copy
    }

    /// Symmetric rank-1 update: A := alpha * x * x^T + A
    /// BLAS: SSYR, DSYR, CHER, ZHER
    fn syr(self, tr: Triangle) {
        unimplemented!()
    }
    fn her(self, tr: Triangle) {
        unimplemented!()
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
