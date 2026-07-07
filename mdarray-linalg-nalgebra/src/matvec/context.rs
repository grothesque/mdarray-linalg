use std::ops::{Add, Mul};

use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::{
    contract::{Triangle, Type},
    matvec::{Argmax, MatVec, MatVecBuilder, Outer, OuterBuilder, VecOps},
    utils::unravel_index,
};
use num_complex::ComplexFloat;
use num_traits::{One, Signed, Zero};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign};

use super::simple::{axpy, gemv, ger};
use crate::{Nalgebra, to_dvector_view};

struct NalgebraMatVecBuilder<'a, T, D0, D1, La, Lx>
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

impl<'a, T, La, Lx, D0, D1> MatVecBuilder<'a, T, La, Lx, D0, D1>
    for NalgebraMatVecBuilder<'a, T, D0, D1, La, Lx>
where
    T: nalgebra::Scalar + ComplexFloat + Zero + One + ClosedAddAssign + ClosedMulAssign + Copy,
    D0: Dim,
    D1: Dim,
    La: Layout,
    Lx: Layout,
{
    fn parallelize(self) -> Self {
        self
    }

    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    fn eval(self) -> Array<T, (D0,)> {
        let mut y = Array::<T, (D0,)>::from_elem(
            <(D0,) as Shape>::from_dims(&[self.a.shape().dim(0)]),
            T::zero(),
        );
        gemv(self.alpha, self.a, self.x, T::zero(), &mut y);
        y
    }

    fn write<Ly: Layout>(self, y: &mut Slice<T, (D0,), Ly>) {
        gemv(self.alpha, self.a, self.x, T::zero(), y);
    }

    fn add_to_vec<Ly: Layout>(self, y: &mut Slice<T, (D0,), Ly>) {
        gemv(self.alpha, self.a, self.x, T::one(), y);
    }

    fn add_to_scaled_vec<Ly: Layout>(self, y: &mut Slice<T, (D0,), Ly>, beta: T) {
        gemv(self.alpha, self.a, self.x, beta, y);
    }
}

impl<T, D0, D1> MatVec<T, D0, D1> for Nalgebra
where
    T: nalgebra::Scalar + ComplexFloat + Zero + One + ClosedAddAssign + ClosedMulAssign + Copy,
    D0: Dim,
    D1: Dim,
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
        NalgebraMatVecBuilder {
            alpha: T::one(),
            a,
            x,
        }
    }
}

impl<T, D1> VecOps<T, D1> for Nalgebra
where
    T: nalgebra::Scalar
        + ComplexFloat
        + Zero
        + One
        + ClosedAddAssign
        + ClosedMulAssign
        + Copy
        + Add<Output = T>
        + Mul<Output = T>,
    D1: Dim,
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
        let x_nalgebra = to_dvector_view(x);
        let y_nalgebra = to_dvector_view(y);
        x_nalgebra.dot(&y_nalgebra)
    }

    fn dotc<Lx: Layout, Ly: Layout>(&self, x: &Slice<T, (D1,), Lx>, y: &Slice<T, (D1,), Ly>) -> T {
        assert_eq!(x.len(), y.len(), "Vector lengths must match");
        x.iter()
            .zip(y.iter())
            .fold(T::zero(), |acc, (xi, yi)| acc + xi.conj() * *yi)
    }

    fn norm2<Lx: Layout>(&self, x: &Slice<T, (D1,), Lx>) -> T::Real {
        x.iter()
            .fold(T::Real::zero(), |acc, xi| {
                let abs = xi.abs();
                acc + abs * abs
            })
            .sqrt()
    }

    fn norm1<Lx: Layout>(&self, x: &Slice<T, (D1,), Lx>) -> T::Real
    where
        T: ComplexFloat,
    {
        x.iter().fold(T::Real::zero(), |acc, xi| acc + xi.l1_norm())
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

impl<T> Argmax<T> for Nalgebra
where
    T: nalgebra::Scalar + ComplexFloat + PartialOrd + Signed + Copy,
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
            output.push(0);
            return true;
        }

        let flat_idx = to_dvector_view(x).argmax().0;
        output.extend_from_slice(&unravel_index(x, flat_idx));
        true
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
            output.push(0);
            return true;
        }

        let flat_idx = to_dvector_view(x).iamax();
        output.extend_from_slice(&unravel_index(x, flat_idx));
        true
    }

    fn argmax<Lx: Layout, S: Shape>(&self, x: &Slice<T, S, Lx>) -> Option<Vec<usize>> {
        let mut output = Vec::new();
        self.argmax_write(x, &mut output).then_some(output)
    }

    fn argmax_abs<Lx: Layout, S: Shape>(&self, x: &Slice<T, S, Lx>) -> Option<Vec<usize>> {
        let mut output = Vec::new();
        self.argmax_abs_write(x, &mut output).then_some(output)
    }
}

struct NalgebraOuterBuilder<'a, T, Dx, Dy, Lx, Ly>
where
    Dx: Dim,
    Dy: Dim,
    Lx: Layout,
    Ly: Layout,
{
    alpha: T,
    x: &'a Slice<T, (Dx,), Lx>,
    y: &'a Slice<T, (Dy,), Ly>,
}

impl<'a, T, Dx, Dy, Lx, Ly> OuterBuilder<'a, T, Lx, Ly, Dx, Dy>
    for NalgebraOuterBuilder<'a, T, Dx, Dy, Lx, Ly>
where
    T: nalgebra::Scalar + ComplexFloat + Zero + One + ClosedAddAssign + ClosedMulAssign + Copy,
    Dx: Dim,
    Dy: Dim,
    Lx: Layout,
    Ly: Layout,
{
    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    fn eval(self) -> Array<T, (Dx, Dy)> {
        let shape = <(Dx, Dy) as Shape>::from_dims(&[self.x.len(), self.y.len()]);
        let mut a = Array::<T, (Dx, Dy)>::from_elem(shape, T::zero());
        ger(self.alpha, self.x, self.y, T::zero(), &mut a);
        a
    }

    fn write<La: Layout>(self, a: &mut Slice<T, (Dx, Dy), La>) {
        ger(self.alpha, self.x, self.y, T::zero(), a);
    }

    fn add_to<La: Layout>(self, a: &mut Slice<T, (Dx, Dy), La>) {
        ger(self.alpha, self.x, self.y, T::one(), a);
    }

    fn add_to_special<La: Layout>(self, a: &mut Slice<T, (Dx, Dy), La>, ty: Type, tr: Triangle) {
        match ty {
            Type::Tri => self.add_to(a),
            Type::Sym | Type::Her => {
                assert_eq!(a.shape().dim(0), a.shape().dim(1));
                assert_eq!(a.shape().dim(0), self.x.len());
                assert_eq!(a.shape().dim(1), self.y.len());

                for i in 0..a.shape().dim(0) {
                    for j in 0..a.shape().dim(1) {
                        let stored = match tr {
                            Triangle::Upper => i <= j,
                            Triangle::Lower => i >= j,
                        };
                        if stored {
                            let rhs = match ty {
                                Type::Sym => self.alpha * self.x[[i]] * self.y[[j]],
                                Type::Her => self.alpha * self.x[[i]] * self.y[[j]].conj(),
                                Type::Tri => unreachable!(),
                            };
                            a[[i, j]] += rhs;
                        }
                    }
                }
            }
        }
    }
}

impl<T, Dx, Dy> Outer<T, Dx, Dy> for Nalgebra
where
    T: nalgebra::Scalar + ComplexFloat + Zero + One + ClosedAddAssign + ClosedMulAssign + Copy,
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
        NalgebraOuterBuilder {
            alpha: T::one(),
            x,
            y,
        }
    }
}
