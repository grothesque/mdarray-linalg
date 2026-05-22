use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::matmul::{
    _contract, Axes, ContractBuilder, MatMul, MatMulBuilder, Side, Triangle, Type,
};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign};

use super::simple::gemm;
use crate::{Nalgebra, to_dmatrix, write_dmatrix};

/// Rebuild the missing half of a structured matrix before dispatching to nalgebra.
fn copy_special_matrix<T, D0, D1, L>(
    a: &Slice<T, (D0, D1), L>,
    ty: &Type,
    tr: &Triangle,
) -> nalgebra::DMatrix<T>
where
    T: nalgebra::Scalar + ComplexFloat + Zero + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let rows = a.shape().dim(0);
    let cols = a.shape().dim(1);
    assert_eq!(
        rows, cols,
        "special matrix operations require a square matrix"
    );

    let mut out = nalgebra::DMatrix::from_element(rows, cols, T::zero());

    for i in 0..rows {
        for j in 0..cols {
            let stored = match tr {
                Triangle::Upper => i <= j,
                Triangle::Lower => i >= j,
            };

            out[(i, j)] = if stored {
                a[[i, j]]
            } else {
                match ty {
                    Type::Sym => a[[j, i]],
                    Type::Her => a[[j, i]].conj(),
                    Type::Tri => T::zero(),
                }
            };
        }
    }

    out
}

struct NalgebraMatMulBuilder<'a, T, La, Lb, D0, D1, D2>
where
    La: Layout,
    Lb: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    alpha: T,
    a: &'a Slice<T, (D0, D1), La>,
    b: &'a Slice<T, (D1, D2), Lb>,
}

struct NalgebraContractBuilder<'a, T, La, Lb, S>
where
    La: Layout,
    Lb: Layout,
    S: Shape,
{
    alpha: T,
    a: &'a Slice<T, S, La>,
    b: &'a Slice<T, S, Lb>,
    axes: Axes<'a>,
}

impl<'a, T, La, Lb, D0, D1, D2> MatMulBuilder<'a, T, La, Lb, D0, D1, D2>
    for NalgebraMatMulBuilder<'a, T, La, Lb, D0, D1, D2>
where
    T: nalgebra::Scalar + ComplexFloat + Zero + One + ClosedAddAssign + ClosedMulAssign + Copy,
    La: Layout,
    Lb: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha *= factor;
        self
    }

    fn eval(self) -> Array<T, (D0, D2)> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let mut c = Array::<T, (D0, D2)>::from_elem((m, n), T::zero());
        gemm(self.alpha, self.a, self.b, T::zero(), &mut c);
        c
    }

    fn write<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        gemm(self.alpha, self.a, self.b, T::zero(), c);
    }

    fn add_to<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        gemm(self.alpha, self.a, self.b, T::one(), c);
    }

    fn add_to_scaled<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>, beta: T) {
        gemm(self.alpha, self.a, self.b, beta, c);
    }

    fn special(self, lr: Side, ty: Type, tr: Triangle) -> Array<T, (D0, D2)> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let mut c_nalgebra = nalgebra::DMatrix::from_element(m.size(), n.size(), T::zero());

        let a_nalgebra = match lr {
            Side::Left => copy_special_matrix(self.a, &ty, &tr),
            Side::Right => to_dmatrix(self.a),
        };
        let b_nalgebra = match lr {
            Side::Left => to_dmatrix(self.b),
            Side::Right => copy_special_matrix(self.b, &ty, &tr),
        };

        c_nalgebra.gemm(self.alpha, &a_nalgebra, &b_nalgebra, T::zero());

        let mut c = Array::<T, (D0, D2)>::from_elem((m, n), T::zero());
        write_dmatrix(&c_nalgebra, &mut c);
        c
    }
}

impl<'a, T, La, Lb, S> ContractBuilder<'a, T, La, Lb> for NalgebraContractBuilder<'a, T, La, Lb, S>
where
    T: nalgebra::Scalar
        + ComplexFloat
        + Zero
        + One
        + ClosedAddAssign
        + ClosedMulAssign
        + Copy
        + MulAdd<Output = T>,
    La: Layout,
    Lb: Layout,
    S: Shape,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha *= factor;
        self
    }

    fn eval(self) -> Array<T> {
        _contract(Nalgebra, self.a, self.b, self.axes, self.alpha)
    }

    fn write(self, _c: &mut Slice<T>) {
        todo!()
    }
}

impl<T> MatMul<T> for Nalgebra
where
    T: nalgebra::Scalar
        + ComplexFloat
        + Zero
        + One
        + ClosedAddAssign
        + ClosedMulAssign
        + Copy
        + MulAdd<Output = T>,
{
    fn matmul<'a, La, Lb, D0, D1, D2>(
        &self,
        a: &'a Slice<T, (D0, D1), La>,
        b: &'a Slice<T, (D1, D2), Lb>,
    ) -> impl MatMulBuilder<'a, T, La, Lb, D0, D1, D2>
    where
        La: Layout,
        Lb: Layout,
        D0: Dim,
        D1: Dim,
        D2: Dim,
    {
        NalgebraMatMulBuilder {
            alpha: T::one(),
            a,
            b,
        }
    }

    fn contract_all<'a, La, Lb, S>(
        &self,
        a: &'a Slice<T, S, La>,
        b: &'a Slice<T, S, Lb>,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout,
        S: Shape,
    {
        NalgebraContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::All,
        }
    }

    fn contract_n<'a, La, Lb, S>(
        &self,
        a: &'a Slice<T, S, La>,
        b: &'a Slice<T, S, Lb>,
        n: usize,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout,
        S: Shape,
    {
        NalgebraContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::LastFirst { k: n },
        }
    }

    fn contract<'a, La, Lb, S>(
        &self,
        a: &'a Slice<T, S, La>,
        b: &'a Slice<T, S, Lb>,
        axes_a: &'a [usize],
        axes_b: &'a [usize],
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout,
        S: Shape,
    {
        NalgebraContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(axes_a, axes_b),
        }
    }
}
