use std::ops::{Add, Mul};

use mdarray::{Dim, Layout, Shape, Slice, Tensor};
use num_complex::ComplexFloat;
use num_traits::Zero;

use super::simple::naive_outer;
use crate::{
    Naive,
    matmul::{Triangle, Type},
    matvec::{Argmax, MatVec, MatVecBuilder, Outer, OuterBuilder, VecOps},
    utils::unravel_index,
};

struct NaiveMatVecBuilder<'a, T, La, Lx, D0, D1>
where
    La: Layout,
    Lx: Layout,
    D0: Dim,
    D1: Dim,
{
    alpha: T,
    a: &'a Slice<T, (D0, D1), La>,
    x: &'a Slice<T, (D1,), Lx>,
}

impl<'a, T, La, Lx, D0, D1> MatVecBuilder<'a, T, La, Lx, D0, D1>
    for NaiveMatVecBuilder<'a, T, La, Lx, D0, D1>
where
    La: Layout,
    Lx: Layout,
    T: ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
    D0: Dim,
    D1: Dim,
{
    fn parallelize(self) -> Self {
        self
    }

    /// `α := α·α'`
    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    fn eval(self) -> Tensor<T, (D1,)> {
        let ash = *self.a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));
        let x_len = self.x.shape().dim(0);

        assert!(n == x_len, "Matrix columns must match vector length");

        let result_shape = <(D1,) as Shape>::from_dims(&[m]);
        let mut result = Tensor::<T, (D1,)>::from_elem(result_shape, 0.into().into());

        for i in 0..m {
            let mut sum = 0.into().into();
            for j in 0..n {
                sum = sum + self.a[[i, j]] * self.x[[j]];
            }
            result[[i]] = self.alpha * sum;
        }
        result
    }

    fn write<Ly: Layout>(self, y: &mut Slice<T, (D1,), Ly>) {
        let ash = *self.a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));
        let x_len = self.x.shape().dim(0);

        assert!(n == x_len, "Matrix columns must match vector length");

        for i in 0..m {
            let mut sum = 0.into().into();
            for j in 0..n {
                sum = sum + self.a[[i, j]] * self.x[[j]];
            }
            y[[i]] = self.alpha * sum;
        }
    }

    fn add_to_vec<Ly: Layout>(self, y: &mut Slice<T, (D1,), Ly>) {
        let ash = *self.a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));
        let x_len = self.x.shape().dim(0);
        let y_len = y.shape().dim(0);

        assert!(n == x_len, "Matrix columns must match x vector length");
        assert!(m == y_len, "Matrix rows must match y vector length");

        for i in 0..m {
            for j in 0..n {
                y[[i]] = y[[i]] + self.a[[i, j]] * self.x[[j]];
            }
        }
    }

    fn add_to_scaled_vec<Ly: Layout>(self, y: &mut Slice<T, (D1,), Ly>, beta: T) {
        let ash = *self.a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));
        let x_len = self.x.shape().dim(0);
        let y_len = y.shape().dim(0);

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

impl<T, D0: Dim, D1: Dim> MatVec<T, D0, D1> for Naive
where
    T: ComplexFloat,
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
        NaiveMatVecBuilder {
            alpha: 1.into().into(),
            a,
            x,
        }
    }
}

impl<T: ComplexFloat + 'static + Add<Output = T> + Mul<Output = T> + Zero + Copy, D: Dim>
    VecOps<T, D> for Naive
{
    fn add_to_scaled<Lx: Layout, Ly: Layout>(
        &self,
        alpha: T,
        x: &Slice<T, (D,), Lx>,
        y: &mut Slice<T, (D,), Ly>,
    ) {
        for (elem_x, elem_y) in std::iter::zip(x.into_iter(), y.into_iter()) {
            *elem_y = alpha * (*elem_x) + *elem_y;
        }
    }

    fn dot<Lx: Layout, Ly: Layout>(&self, x: &Slice<T, (D,), Lx>, y: &Slice<T, (D,), Ly>) -> T {
        let mut result = T::zero();
        for (elem_x, elem_y) in std::iter::zip(x.into_iter(), y.into_iter()) {
            result = result + *elem_x * (*elem_y);
        }
        result
    }

    fn dotc<Lx: Layout, Ly: Layout>(&self, x: &Slice<T, (D,), Lx>, y: &Slice<T, (D,), Ly>) -> T {
        let mut result = T::zero();
        for (elem_x, elem_y) in std::iter::zip(x.into_iter(), y.into_iter()) {
            result = result + elem_x.conj() * (*elem_y);
        }
        result
    }

    fn norm2<Lx: Layout>(&self, x: &Slice<T, (D,), Lx>) -> T::Real {
        let mut sum_sq = T::Real::zero();
        for elem in x.into_iter() {
            sum_sq = sum_sq + elem.abs().powi(2);
        }
        sum_sq.sqrt()
    }

    fn norm1<Lx: Layout>(&self, x: &Slice<T, (D,), Lx>) -> T::Real
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
        _x: &mut Slice<T, (D,), Lx>,
        _y: &mut Slice<T, (D,), Ly>,
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

impl<T, Dx, Dy> Outer<T, Dx, Dy> for Naive
where
    T: ComplexFloat,
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
        NaiveOuterBuilder {
            alpha: 1.into().into(),
            x,
            y,
        }
    }
}

struct NaiveOuterBuilder<'a, T, Lx, Ly, Dx, Dy>
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

impl<'a, T, Lx, Ly, Dx, Dy> OuterBuilder<'a, T, Lx, Ly, Dx, Dy>
    for NaiveOuterBuilder<'a, T, Lx, Ly, Dx, Dy>
where
    Lx: Layout,
    Ly: Layout,
    Dx: Dim,
    Dy: Dim,
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
    fn eval(self) -> Tensor<T, (Dx, Dy)> {
        let m = self.x.shape().dim(0);
        let n = self.y.shape().dim(0);

        let a_shape = <(Dx, Dy) as Shape>::from_dims(&[m, n]);
        let mut a = Tensor::<T, (Dx, Dy)>::from_elem(a_shape, 0.into().into());

        naive_outer(&mut a, self.x, self.y, self.alpha, None, None);

        a
    }

    /// `a := α·xy`
    fn write<La: Layout>(self, a: &mut Slice<T, (Dx, Dy), La>) {
        let m = self.x.shape().dim(0);
        let n = self.y.shape().dim(0);

        let ash = *a.shape();
        let (ma, na) = (ash.dim(0), ash.dim(1));

        assert!(ma == m, "Output shape must match input vector length");
        assert!(na == n, "Output shape must match input vector length");

        naive_outer(a, self.x, self.y, self.alpha, None, None);
    }

    /// Rank-1 update: `A := α·x·yᵀ + A`
    fn add_to<La: Layout>(self, a: &mut Slice<T, (Dx, Dy), La>) {
        let m = self.x.shape().dim(0);
        let n = self.y.shape().dim(0);

        let ash = *a.shape();
        let (ma, na) = (ash.dim(0), ash.dim(1));

        assert!(ma == m, "Output shape must match input vector length");
        assert!(na == n, "Output shape must match input vector length");

        naive_outer(a, self.x, self.y, self.alpha, None, None);
    }

    /// Rank-1 update: `A := α·x·xᵀ (or x·x† ) + A` on special matrix
    fn add_to_special(self, a: &mut Slice<T, (Dx, Dy)>, ty: Type, tr: Triangle) {
        let n = self.x.shape().dim(0);
        let ash = *a.shape();
        let (ma, na) = (ash.dim(0), ash.dim(1));

        assert!(ma == na, "Input matrix must be square");
        assert!(na == n, "Output shape must match input vector length");

        naive_outer(a, self.x, self.y, self.alpha, Some(ty), Some(tr));
    }
}
