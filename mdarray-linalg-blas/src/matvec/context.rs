use std::ops::{Add, Mul};

use cblas_sys::CBLAS_UPLO;
use mdarray::{Dim, Layout, Shape, Slice, Tensor};
use mdarray_linalg::{
    matmul::{Triangle, Type},
    matvec::{Argmax, MatVec, MatVecBuilder, Outer, OuterBuilder, VecOps},
    utils::unravel_index,
};
use num_complex::ComplexFloat;
use num_traits::Zero;

use super::{
    scalar::BlasScalar,
    simple::{amax, asum, axpy, dotc, dotu, gemv, ger, her, nrm2, syr},
};
use crate::Blas;

struct BlasMatVecBuilder<'a, T, D0, D1, La, Lx>
where
    D0: Dim,
    D1: Dim,
    La: Layout,
    Lx: Layout,
{
    alpha: T,
    a: &'a Slice<T, (D0, D1), La>,
    x: &'a Slice<T, (D1,), Lx>,
}

impl<'a, T, La, Lx, D0: Dim, D1: Dim> MatVecBuilder<'a, T, La, Lx, D0, D1>
    for BlasMatVecBuilder<'a, T, D0, D1, La, Lx>
where
    La: Layout,
    Lx: Layout,
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
    D0: Dim,
    D1: Dim,
{
    fn parallelize(self) -> Self {
        self
    }

    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    fn eval(self) -> Tensor<T, (D1,)> {
        let mut y = Tensor::<T, (D1,)>::from_elem(
            <(D1,) as Shape>::from_dims(&[self.a.shape().dim(0)]),
            0.into().into(),
        );
        gemv(self.alpha, self.a, self.x, T::zero(), &mut y);
        y
    }

    fn write<Ly: Layout>(self, y: &mut Slice<T, (D1,), Ly>) {
        gemv(self.alpha, self.a, self.x, T::zero(), y);
    }

    fn add_to_vec<Ly: Layout>(self, y: &mut Slice<T, (D1,), Ly>) {
        gemv(self.alpha, self.a, self.x, T::one(), y);
    }

    fn add_to_scaled_vec<Ly: Layout>(self, y: &mut Slice<T, (D1,), Ly>, beta: T) {
        gemv(self.alpha, self.a, self.x, beta, y);
    }
}

impl<T, D0: Dim, D1: Dim> MatVec<T, D0, D1> for Blas
where
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    fn matvec<'a, La, Lx>(
        &self,
        a: &'a Slice<T, (D0, D1), La>,
        x: &'a Slice<T, (D1,), Lx>,
    ) -> impl MatVecBuilder<'a, T, La, Lx, D0, D1>
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

impl<
    T: ComplexFloat + BlasScalar + 'static + Add<Output = T> + Mul<Output = T> + Zero + Copy,
    D1: Dim,
> VecOps<T, D1> for Blas
{
    fn add_to_scaled<Lx: Layout, Ly: Layout>(
        &self,
        alpha: T,
        x: &Slice<T, (D1,), Lx>,
        y: &mut Slice<T, (D1,), Ly>,
    ) {
        axpy(alpha, x, y);
    }

    fn dot<Lx: Layout, Ly: Layout>(&self, x: &Slice<T, (D1,), Lx>, y: &Slice<T, (D1,), Ly>) -> T {
        dotu(x, y)
    }

    fn dotc<Lx: Layout, Ly: Layout>(&self, x: &Slice<T, (D1,), Lx>, y: &Slice<T, (D1,), Ly>) -> T {
        dotc(x, y)
    }

    fn norm2<Lx: Layout>(&self, x: &Slice<T, (D1,), Lx>) -> T::Real {
        nrm2(x)
    }

    fn norm1<Lx: Layout>(&self, x: &Slice<T, (D1,), Lx>) -> T::Real
    where
        T: ComplexFloat,
    {
        asum(x)
    }

    fn rot<Lx: Layout, Ly: Layout>(
        &self,
        _x: &mut Slice<T, (D1,), Lx>,
        _y: &mut Slice<T, (D1,), Ly>,
        _c: T::Real,
        _s: T,
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

struct BlasOuterBuilder<'a, T, Dx, Dy, Lx, Ly>
where
    Lx: Layout,
    Ly: Layout,
    Dx: Dim,
    Dy: Dim,
{
    alpha: T,
    x: &'a Slice<T, (Dx,), Lx>,
    y: &'a Slice<T, (Dy,), Ly>,
}

impl<'a, T, Dx, Dy, Lx, Ly> OuterBuilder<'a, T, Lx, Ly, Dx, Dy>
    for BlasOuterBuilder<'a, T, Dx, Dy, Lx, Ly>
where
    Lx: Layout,
    Ly: Layout,
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
    Dx: Dim,
    Dy: Dim,
{
    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    fn eval(self) -> Tensor<T, (Dx, Dy)> {
        let shape = <(Dx, Dy) as Shape>::from_dims(&[self.x.len(), self.y.len()]);
        let mut a = Tensor::<T, (Dx, Dy)>::from_elem(shape, 0.into().into());
        ger(self.alpha, self.x, self.y, &mut a);
        a
    }

    fn write<La: Layout>(self, a: &mut Slice<T, (Dx, Dy), La>) {
        let zero = T::zero();
        a.fill(zero);
        ger(self.alpha, self.x, self.y, a);
    }

    fn add_to<La: Layout>(self, a: &mut Slice<T, (Dx, Dy), La>) {
        ger(self.alpha, self.x, self.y, a);
    }

    fn add_to_special(self, a: &mut Slice<T, (Dx, Dy)>, ty: Type, tr: Triangle) {
        let cblas_uplo = match tr {
            Triangle::Lower => CBLAS_UPLO::CblasLower,
            Triangle::Upper => CBLAS_UPLO::CblasUpper,
        };
        match ty {
            Type::Sym => syr(cblas_uplo, self.alpha, self.x, a), // Assume x == y
            Type::Her => her(cblas_uplo, self.alpha.re(), self.x, a),
            Type::Tri => ger(self.alpha, self.x, self.y, a),
        }
    }
}

impl<T, Dx, Dy> Outer<T, Dx, Dy> for Blas
where
    T: BlasScalar + ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
    Dx: Dim,
    Dy: Dim,
{
    fn outer<'a, Lx, Ly>(
        &self,
        x: &'a Slice<T, (Dx,), Lx>,
        y: &'a Slice<T, (Dy,), Ly>,
    ) -> impl OuterBuilder<'a, T, Lx, Ly, Dx, Dy>
    where
        Lx: Layout,
        Ly: Layout,
    {
        BlasOuterBuilder {
            alpha: T::one(),
            x,
            y,
        }
    }
}
