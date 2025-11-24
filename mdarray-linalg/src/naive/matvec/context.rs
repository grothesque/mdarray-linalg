use std::ops::{Add, Mul};

use mdarray::{DSlice, DTensor, Layout, Shape, Slice};
use num_complex::ComplexFloat;
use num_traits::Zero;

use super::simple::naive_outer;
use crate::{
    Naive,
    matmul::{Triangle, Type},
    matvec::{Argmax, MatVec, MatVecBuilder, Outer, OuterBuilder, VecOps},
    utils::unravel_index,
};

struct NaiveMatVecBuilder<'a, T, La, Lx>
where
    La: Layout,
    Lx: Layout,
{
    alpha: T,
    a: &'a DSlice<T, 2, La>,
    x: &'a DSlice<T, 1, Lx>,
}

impl<'a, T, La, Lx> MatVecBuilder<'a, T, La, Lx> for NaiveMatVecBuilder<'a, T, La, Lx>
where
    La: Layout,
    Lx: Layout,
    T: ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    fn parallelize(self) -> Self {
        self
    }

    /// `α := α·α'`
    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    fn eval(self) -> DTensor<T, 1> {
        let (m, n) = *self.a.shape();
        let x_len = self.x.shape().0;

        assert!(n == x_len, "Matrix columns must match vector length");

        let mut result = DTensor::<T, 1>::from_elem([m], 0.into().into());

        for i in 0..m {
            let mut sum = 0.into().into();
            for j in 0..n {
                sum = sum + self.a[[i, j]] * self.x[[j]];
            }
            result[[i]] = self.alpha * sum;
        }
        result
    }

    fn write<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>) {
        let (m, n) = *self.a.shape();
        let x_len = self.x.shape().0;

        assert!(n == x_len, "Matrix columns must match vector length");

        for i in 0..m {
            let mut sum = 0.into().into();
            for j in 0..n {
                sum = sum + self.a[[i, j]] * self.x[[j]];
            }
            y[[i]] = self.alpha * sum;
        }
    }

    fn add_to_vec<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>) {
        let (m, n) = *self.a.shape();
        let x_len = self.x.shape().0;
        let y_len = y.shape().0;

        assert!(n == x_len, "Matrix columns must match x vector length");
        assert!(m == y_len, "Matrix rows must match y vector length");

        for i in 0..m {
            for j in 0..n {
                y[[i]] = y[[i]] + self.a[[i, j]] * self.x[[j]];
            }
        }
    }

    fn add_to_scaled_vec<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>, beta: T) {
        let (m, n) = *self.a.shape();
        let x_len = self.x.shape().0;
        let y_len = y.shape().0;

        assert!(n == x_len, "Matrix columns must match x vector length");
        assert!(m == y_len, "Matrix rows must match y vector length");

        for i in 0..m {
            y[[i]] = beta * y[[i]];
        }

        for i in 0..m {
            for j in 0..n {
                y[[i]] = y[[i]] + self.a[[i, j]] * self.x[[j]];
            }
        }
    }
}

impl<T> MatVec<T> for Naive
where
    T: ComplexFloat,
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
        NaiveMatVecBuilder {
            alpha: 1.into().into(),
            a,
            x,
        }
    }
}

impl<T: ComplexFloat + 'static + Add<Output = T> + Mul<Output = T> + Zero + Copy> VecOps<T>
    for Naive
{
    fn add_to_scaled<Lx: Layout, Ly: Layout>(
        &self,
        alpha: T,
        x: &DSlice<T, 1, Lx>,
        y: &mut DSlice<T, 1, Ly>,
    ) {
        for (elem_x, elem_y) in std::iter::zip(x.into_iter(), y.into_iter()) {
            *elem_y = alpha * (*elem_x) + *elem_y;
        }
    }

    fn dot<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T {
        let mut result = T::zero();
        for (elem_x, elem_y) in std::iter::zip(x.into_iter(), y.into_iter()) {
            result = result + *elem_x * (*elem_y);
        }
        result
    }

    fn dotc<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T {
        let mut result = T::zero();
        for (elem_x, elem_y) in std::iter::zip(x.into_iter(), y.into_iter()) {
            result = result + elem_x.conj() * (*elem_y);
        }
        result
    }

    fn norm2<Lx: Layout>(&self, x: &DSlice<T, 1, Lx>) -> T::Real {
        let mut sum_sq = T::Real::zero();
        for elem in x.into_iter() {
            sum_sq = sum_sq + elem.abs().powi(2);
        }
        sum_sq.sqrt()
    }

    fn norm1<Lx: Layout>(&self, x: &DSlice<T, 1, Lx>) -> T::Real
    where
        T: ComplexFloat,
    {
        let mut sum = T::Real::zero();
        for elem in x.into_iter() {
            sum = sum + elem.re().abs() + elem.im().abs();
        }
        sum
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

impl<
    T: ComplexFloat<Real = T> + 'static + PartialOrd + Add<Output = T> + Mul<Output = T> + Zero + Copy,
> Argmax<T> for Naive
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
        let mut max_val = x.iter().next().unwrap();

        for (flat_idx, val) in x.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
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

        let mut max_flat_idx = 0;
        let mut max_val = x.iter().next().unwrap().abs();

        for (flat_idx, val) in x.iter().enumerate().skip(1) {
            if val.abs() > max_val {
                max_val = val.abs();
                max_flat_idx = flat_idx;
            }
        }

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

impl<T> Outer<T> for Naive
where
    T: ComplexFloat,
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
        NaiveOuterBuilder {
            alpha: 1.into().into(),
            x,
            y,
        }
    }
}

struct NaiveOuterBuilder<'a, T, Lx, Ly>
where
    Lx: Layout,
    Ly: Layout,
{
    alpha: T,
    x: &'a DSlice<T, 1, Lx>,
    y: &'a DSlice<T, 1, Ly>,
}

impl<'a, T, Lx, Ly> OuterBuilder<'a, T, Lx, Ly> for NaiveOuterBuilder<'a, T, Lx, Ly>
where
    Lx: Layout,
    Ly: Layout,
    T: ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    /// `α := α·α'`
    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    /// Returns `α·xy`
    fn eval(self) -> DTensor<T, 2> {
        let m = self.x.shape().0;
        let n = self.y.shape().0;
        let mut a = DTensor::<T, 2>::from_elem([m, n], 0.into().into());

        naive_outer(&mut a, self.x, self.y, self.alpha, None, None);

        a
    }

    /// `a := α·xy`
    fn write<La: Layout>(self, a: &mut DSlice<T, 2, La>) {
        let m = self.x.shape().0;
        let n = self.y.shape().0;

        let (ma, na) = *a.shape();

        assert!(ma == m, "Output shape must match input vector length");
        assert!(na == n, "Output shape must match input vector length");

        naive_outer(a, self.x, self.y, self.alpha, None, None);
    }

    /// Rank-1 update: `A := α·x·yᵀ + A`
    fn add_to<La: Layout>(self, a: &mut DSlice<T, 2, La>) {
        let m = self.x.shape().0;
        let n = self.y.shape().0;

        let (ma, na) = *a.shape();

        assert!(ma == m, "Output shape must match input vector length");
        assert!(na == n, "Output shape must match input vector length");

        naive_outer(a, self.x, self.y, self.alpha, None, None);
    }

    /// Rank-1 update: `A := α·x·xᵀ (or x·x†) + A` on special matrix
    fn add_to_special(self, a: &mut DSlice<T, 2>, ty: Type, tr: Triangle) {
        let n = self.x.shape().0;
        let (ma, na) = *a.shape();

        assert!(ma == na, "Input matrix must be square");
        assert!(na == n, "Output shape must match input vector length");

        naive_outer(a, self.x, self.y, self.alpha, Some(ty), Some(tr));
    }
}
