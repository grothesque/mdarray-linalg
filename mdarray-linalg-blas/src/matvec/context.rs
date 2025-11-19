use cblas_sys::{
    CBLAS_DIAG, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO, cblas_dcopy, cblas_drotg, cblas_dscal,
    cblas_dswap,
};
use mdarray::{DSlice, DTensor, Layout, Shape, Slice};
use num_complex::ComplexFloat;
use num_traits::Zero;
use std::ops::{Add, Mul};

use mdarray_linalg::matmul::{Triangle, Type};
use mdarray_linalg::matvec::{Argmax, MatVec, MatVecBuilder, Outer, OuterBuilder, VecOps};
use mdarray_linalg::utils::unravel_index;

use super::scalar::BlasScalar;
use super::simple::{amax, asum, axpy, dotc, dotu, gemv, ger, her, nrm2, syr};
use crate::Blas;

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
        self // BLAS gère souvent le parallélisme internement
    }

    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    fn eval(self) -> DTensor<T, 1> {
        let mut y = DTensor::<T, 1>::from_elem([self.a.shape().0], 0.into().into());
        gemv(self.alpha, self.a, self.x, T::zero(), &mut y);
        y
    }

    fn write<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>) {
        gemv(self.alpha, self.a, self.x, T::zero(), y);
    }

    fn add_to_vec<Ly: Layout>(self, y: &DSlice<T, 1, Ly>) -> DTensor<T, 1> {
        let mut result = DTensor::<T, 1>::from_elem([self.a.shape().0], 0.into().into());
        result.assign(y);
        gemv(self.alpha, self.a, self.x, T::one(), &mut result);
        result
    }

    fn add_to_scaled_vec<Ly: Layout>(self, y: &DSlice<T, 1, Ly>, beta: T) -> DTensor<T, 1> {
        let mut result = DTensor::<T, 1>::from_elem([self.a.shape().0], 0.into().into());
        result.assign(y);
        gemv(self.alpha, self.a, self.x, beta, &mut result);
        result
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
            alpha: T::one(),
            a,
            x,
        }
    }
}

impl<T: ComplexFloat + BlasScalar + 'static + Add<Output = T> + Mul<Output = T> + Zero + Copy>
    VecOps<T> for Blas
{
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

    fn copy<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &mut DSlice<T, 1, Ly>) {
        // Wrapper BLAS pour copy (exemple pour f64 ; adapte pour Complex via cblas_zcopy)
        unsafe {
            cblas_dcopy(
                x.len() as i32,
                x.as_ptr() as *const f64,
                1,
                y.as_mut_ptr() as *mut f64,
                1,
            );
        }
        // TODO: Gérer Complex<T> avec cblas_zcopy si T::Real == f64
    }

    fn scal<Lx: Layout>(&self, alpha: T, x: &mut DSlice<T, 1, Lx>) {
        todo!()
    }

    fn swap<Lx: Layout, Ly: Layout>(&self, x: &mut DSlice<T, 1, Lx>, y: &mut DSlice<T, 1, Ly>) {
        todo!()
    }

    fn rot<Lx: Layout, Ly: Layout>(
        &self,
        x: &mut DSlice<T, 1, Lx>,
        y: &mut DSlice<T, 1, Ly>,
        c: T::Real,
        s: T,
    ) where
        T: ComplexFloat,
    {
        todo!()
    }
}

impl<
    T: ComplexFloat
        + std::cmp::PartialOrd
        + BlasScalar
        + 'static
        + Add<Output = T>
        + Mul<Output = T>
        + Zero
        + Copy,
> Argmax<T> for Blas
where
    T::Real: PartialOrd,
{
    fn argmax_write<Lx: Layout, S: Shape>(
        &self,
        x: &Slice<T, S, Lx>,
        output: &mut Vec<usize>,
    ) -> bool {
        output.clear();
        if x.is_empty() {
            return false;
        }
        if x.rank() == 0 {
            return true;
        }
        let mut max_flat_idx = 0;
        let mut max_val = *x.iter().next().unwrap();
        for (flat_idx, val) in x.iter().enumerate().skip(1) {
            if *val > max_val {
                max_val = *val;
                max_flat_idx = flat_idx;
            }
        }
        let indices = unravel_index(x, max_flat_idx);
        output.extend_from_slice(&indices);
        true
    }

    fn argmax<Lx: Layout, S: Shape>(&self, x: &Slice<T, S, Lx>) -> Option<Vec<usize>> {
        let mut result = Vec::new();
        if self.argmax_write(x, &mut result) {
            Some(result)
        } else {
            None
        }
    }

    fn argmax_abs_write<Lx: Layout, S: Shape>(
        &self,
        x: &Slice<T, S, Lx>,
        output: &mut Vec<usize>,
    ) -> bool {
        output.clear();
        if x.is_empty() {
            return false;
        }
        if x.rank() == 0 {
            return true;
        }
        let max_flat_idx = amax(x);
        let indices = unravel_index(x, max_flat_idx);
        output.extend_from_slice(&indices);
        true
    }

    fn argmax_abs<Lx: Layout, S: Shape>(&self, x: &Slice<T, S, Lx>) -> Option<Vec<usize>> {
        let mut result = Vec::new();
        if self.argmax_abs_write(x, &mut result) {
            Some(result)
        } else {
            None
        }
    }
}

// Ajout pour Outer
struct BlasOuterBuilder<'a, T, Lx, Ly>
where
    Lx: Layout,
    Ly: Layout,
{
    alpha: T,
    x: &'a DSlice<T, 1, Lx>,
    y: &'a DSlice<T, 1, Ly>,
}

impl<'a, T, Lx, Ly> OuterBuilder<'a, T, Lx, Ly> for BlasOuterBuilder<'a, T, Lx, Ly>
where
    Lx: Layout,
    Ly: Layout,
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    fn eval(self) -> DTensor<T, 2> {
        let shape = [self.x.len(), self.y.len()];
        let mut a = DTensor::<T, 2>::from_elem(shape, 0.into().into());
        ger(self.alpha, self.x, self.y, &mut a);
        a
    }

    fn write<La: Layout>(self, a: &mut DSlice<T, 2, La>) {
        // Écrase avec α * x * y^T
        let zero = T::zero();
        a.fill(zero);
        ger(self.alpha, self.x, self.y, a);
    }

    fn add_to<La: Layout>(self, a: &DSlice<T, 2, La>) -> DTensor<T, 2> {
        let mut result = DTensor::<T, 2>::from_elem(*a.shape(), 0.into().into());
        result.assign(a);
        ger(self.alpha, self.x, self.y, &mut result);
        result
    }

    fn add_to_write<La: Layout>(self, a: &mut DSlice<T, 2, La>) {
        ger(self.alpha, self.x, self.y, a);
    }

    fn add_to_special(self, a: &DSlice<T, 2>, ty: Type, tr: Triangle) -> DTensor<T, 2> {
        let mut result = DTensor::<T, 2>::from_elem(*a.shape(), 0.into().into());

        result.assign(a);
        let cblas_uplo = match tr {
            Triangle::Lower => CBLAS_UPLO::CblasLower,
            Triangle::Upper => CBLAS_UPLO::CblasUpper,
        };
        match ty {
            Type::Sym => syr(cblas_uplo, self.alpha, self.x, &mut result), // Assume x == y pour special
            Type::Her => her(cblas_uplo, self.alpha.re(), self.x, &mut result),
            Type::Tri => ger(self.alpha, self.x, self.x, &mut result), // Fallback à ger pour tri
        }
        result
    }

    fn add_to_special_write(self, a: &mut DSlice<T, 2>, ty: Type, tr: Triangle) {
        let cblas_uplo = match tr {
            Triangle::Lower => CBLAS_UPLO::CblasLower,
            Triangle::Upper => CBLAS_UPLO::CblasUpper,
        };
        match ty {
            Type::Sym => syr(cblas_uplo, self.alpha, self.x, a),
            Type::Her => her(cblas_uplo, self.alpha.re(), self.x, a),
            Type::Tri => ger(self.alpha, self.x, self.x, a),
        }
    }
}

impl<T> Outer<T> for Blas
where
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    fn outer<'a, Lx, Ly>(
        &self,
        x: &'a DSlice<T, 1, Lx>,
        y: &'a DSlice<T, 1, Ly>,
    ) -> impl OuterBuilder<'a, T, Lx, Ly>
    where
        Lx: Layout,
        Ly: Layout,
    {
        // Note: Le trait dit MatVecBuilder, mais c'est OuterBuilder (corrige dans matvec.rs si besoin)
        BlasOuterBuilder {
            alpha: T::one(),
            x,
            y,
        }
    }
}
