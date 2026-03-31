use std::mem::MaybeUninit;

use mdarray::{Array, Dense, Dim, DynRank, Layout, Shape, Slice};
use mdarray_linalg::matmul::{_contract, Axes, Contract, ContractBuilder, MatMulBuilder};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};

use super::{
    scalar::BlasScalar,
    simple::{gemm, gemm_uninit},
};
use crate::Blas;

struct BlasMatMulBuilder<'a, T, D0, D1, D2, La, Lb>
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

struct BlasContractBuilder<'a, T, Sa, Sb, La, Lb>
where
    La: Layout,
    Lb: Layout,
    Sa: Shape,
    Sb: Shape,
{
    alpha: T,
    a: &'a Slice<T, Sa, La>,
    b: &'a Slice<T, Sb, Lb>,
    axes: Axes<'a>,
}

impl<'a, T, D0, D1, D2, La, Lb> MatMulBuilder<'a, T, D0, D1, D2, La, Lb>
    for BlasMatMulBuilder<'a, T, D0, D1, D2, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: BlasScalar + ComplexFloat + Zero + One,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Array<T, (D0, D2)> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let c = Array::from_elem((m, n), MaybeUninit::<T>::uninit());
        gemm_uninit::<T, La, Lb, Dense, D0, D1, D2>(self.alpha, self.a, self.b, T::zero(), c)
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
}

impl<'a, T, Sa, Sb, La, Lb> ContractBuilder<'a, T, Sa, Sb, La, Lb>
    for BlasContractBuilder<'a, T, Sa, Sb, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: BlasScalar + ComplexFloat + Zero + One + MulAdd<Output = T>,
    Sa: Shape,
    Sb: Shape,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Array<T, DynRank> {
        _contract(Blas, self.a, self.b, self.axes, self.alpha)
    }

    fn write<Sc: Shape, Lc: Layout>(self, _c: &mut Slice<T, Sc, Lc>) {
        todo!()
    }

    fn add_to<Sc: Shape, Lc: Layout>(self, _c: &mut Slice<T, Sc, Lc>) {
        todo!()
    }

    fn add_to_scaled<Sc: Shape, Lc: Layout>(self, _c: &mut Slice<T, Sc, Lc>, _beta: T) {
        todo!()
    }
}

impl<T> Contract<T> for Blas
where
    T: BlasScalar + ComplexFloat + Zero + One + MulAdd<Output = T>,
{
    fn matmul<'a, D0, D1, D2, La, Lb>(
        &self,
        a: &'a Slice<T, (D0, D1), La>,
        b: &'a Slice<T, (D1, D2), Lb>,
    ) -> impl MatMulBuilder<'a, T, D0, D1, D2, La, Lb>
    where
        La: Layout,
        Lb: Layout,
        D0: Dim,
        D1: Dim,
        D2: Dim,
    {
        BlasMatMulBuilder {
            alpha: T::one(),
            a,
            b,
        }
    }

    fn contract_all<'a, Sa, Sb, La, Lb>(
        &self,
        a: &'a Slice<T, Sa, La>,
        b: &'a Slice<T, Sb, Lb>,
    ) -> T
    where
        T: 'a,
        Sa: Shape,
        Sb: Shape,
        La: Layout,
        Lb: Layout,
    {
        _contract(Blas, a, b, Axes::All, T::one()).into_scalar()
    }

    fn contract_n<'a, Sa, Sb, La, Lb>(
        &self,
        a: &'a Slice<T, Sa, La>,
        b: &'a Slice<T, Sb, Lb>,
        n: usize,
    ) -> impl ContractBuilder<'a, T, Sa, Sb, La, Lb>
    where
        T: 'a,
        Sa: Shape,
        Sb: Shape,
        La: Layout,
        Lb: Layout,
    {
        BlasContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::LastFirst { k: n },
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
        BlasContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(axes_a, axes_b),
        }
    }

    fn contract<'a, Sa, Sb, La, Lb>(
        &self,
        a: &'a Slice<T, Sa, La>,
        b: &'a Slice<T, Sb, Lb>,
        indices_a: &'a [u8],
        indices_b: &'a [u8],
        indices_c: &'a [u8],
    ) -> impl ContractBuilder<'a, T, Sa, Sb, La, Lb>
    where
        T: 'a,
        Sa: Shape,
        Sb: Shape,
        La: Layout,
        Lb: Layout,
    {
        // TODO: _hypercontract once implemented
        let _ = indices_c;
        let axes_a: Vec<usize> = indices_a.iter().map(|&i| i as usize).collect();
        let axes_b: Vec<usize> = indices_b.iter().map(|&i| i as usize).collect();
        BlasContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(
                Box::leak(axes_a.into_boxed_slice()),
                Box::leak(axes_b.into_boxed_slice()),
            ),
        }
    }
}
