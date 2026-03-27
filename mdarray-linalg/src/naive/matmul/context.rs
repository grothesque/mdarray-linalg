use mdarray::{Array, Dim, Layout, Shape, Slice};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};

use super::simple::naive_matmul;
use crate::{
    Naive,
    matmul::{_contract, Axes, Contract, ContractBuilder, MatMulBuilder},
};

struct NaiveMatMulBuilder<'a, T, La, Lb, D0, D1, D2>
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

struct NaiveContractBuilder<'a, T, La, Lb, S>
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

impl<'a, T, La, Lb, D0, D1, D2> MatMulBuilder<'a, T, D0, D1, D2, La, Lb>
    for NaiveMatMulBuilder<'a, T, D0, D1, D2, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + Zero + One + MulAdd<Output = T>,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    /// Multiplies the result by a scalar factor.
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    /// Returns a new owned tensor containing the result.
    fn eval(self) -> Array<T, (D0, D2)> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let mut c = Array::from_elem((m, n), T::zero());
        naive_matmul(self.alpha, self.a, self.b, T::zero(), &mut c);
        c
    }

    /// Overwrites the provided slice with the result.
    fn write<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        naive_matmul(self.alpha, self.a, self.b, T::zero(), c);
    }

    /// Adds the result to the provided slice.
    fn add_to<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        naive_matmul(self.alpha, self.a, self.b, T::one(), c);
    }

    /// Adds the result to the provided slice after scaling the slice by `beta`
    /// (i.e. C := beta * C + result).
    fn add_to_scaled<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>, beta: T) {
        naive_matmul(self.alpha, self.a, self.b, beta, c);
    }
}

impl<'a, T, La, Lb, S> ContractBuilder<'a, T, Sa, Sb, La, Lb>
    for NaiveContractBuilder<'a, T, Sa, Sb, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + Zero + One + MulAdd<Output = T>,
    S: Shape,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Array<T> {
        _contract(Naive, self.a, self.b, self.axes, self.alpha)
    }

    fn write(self, _c: &mut Slice<T>) {
        todo!()
    }

    /// Adds the result to the provided slice.
    fn add_to<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>) {
        todo!()
    }

    /// Adds the result to the provided slice after scaling the slice by `beta`
    /// (i.e. C := beta * C + result).
    fn add_to_scaled<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>, beta: T) {
        todo!()
    }
}

impl<T> Contract<T> for Naive
where
    T: ComplexFloat + MulAdd<Output = T>,
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
        NaiveMatMulBuilder {
            alpha: T::one(),
            a,
            b,
        }
    }

    /// Contracts all axes of the first tensor with all axes of the second tensor.
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
        NaiveContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::All,
        }
    }

    /// Contracts the last `n` axes of the first tensor with the first `n` axes of the second tensor.
    /// # Example
    /// For two matrices (2D tensors), `contract_n(1)` performs standard matrix multiplication.
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
        NaiveContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::LastFirst { k: (n) },
        }
    }

    /// Specifies exactly which axes to contract_all.
    /// # Example
    /// `specific([1, 2], [3, 4])` contracts axis 1 and 2 of `a`
    /// with axes 3 and 4 of `b`.
    fn contract<'a, La, Lb, S>(
        &self,
        a: &'a Slice<T, Sa, La>,
        b: &'a Slice<T, Sb, Lb>,
        axes_a: &'a [usize],
        axes_b: &'a [usize],
    ) -> impl ContractBuilder<'a, T, Sa, Sb, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout,
        Sa: Shape,
        Sb: Shape,
    {
        NaiveContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(axes_a, axes_b),
        }
    }

    fn contract_pairs<'a, Sa, Sb, La, Lb>(
        &self,
        a: &'a Slice<T, Sa, La>,
        b: &'a Slice<T, Sb, Lb>,
        axes_a: &'a [usize],
        axes_b: &'a [usize],
    ) -> impl ContractBuilder<'a, T, Sa, Sb, La, Lb>
    where
        T: 'a,
        Sa: Shape,
        Sb: Shape,
        La: Layout,
        Lb: Layout,
    {
        {
            NaiveContractBuilder {
                alpha: T::one(),
                a,
                b,
                axes: Axes::Specific(axes_a, axes_b),
            }
        }
    }
}
