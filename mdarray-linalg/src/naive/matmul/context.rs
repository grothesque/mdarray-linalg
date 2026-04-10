use std::iter::Sum;
use std::ops::AddAssign;

use mdarray::{Array, Dim, DynRank, Layout, Shape, Slice, View};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};

use super::simple::naive_matmul;
use crate::{
    Naive,
    matmul::{_contract, _hypercontract2, Axes, Contract, ContractBuilder, MatMulBuilder},
};

struct NaiveMatMulBuilder<'a, T, D0, D1, D2, La, Lb>
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

struct NaiveContractBuilder<'a, T, Sa, Sb, La, Lb>
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
    einsum: bool,
}

impl<'a, T, D0, D1, D2, La, Lb> MatMulBuilder<'a, T, D0, D1, D2, La, Lb>
    for NaiveMatMulBuilder<'a, T, D0, D1, D2, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + Zero + One + MulAdd<Output = T>,
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
        let mut c = Array::from_elem((m, n), T::zero());
        naive_matmul(self.alpha, self.a, self.b, T::zero(), &mut c);
        c
    }

    fn write<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        naive_matmul(self.alpha, self.a, self.b, T::zero(), c);
    }

    fn add_to<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        naive_matmul(self.alpha, self.a, self.b, T::one(), c);
    }

    fn add_to_scaled<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>, beta: T) {
        naive_matmul(self.alpha, self.a, self.b, beta, c);
    }
}

impl<'a, T, Sa, Sb, La, Lb> ContractBuilder<'a, T, Sa, Sb, La, Lb>
    for NaiveContractBuilder<'a, T, Sa, Sb, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + AddAssign + Sum,
    Sa: Shape,
    Sb: Shape,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Array<T, DynRank> {
        if self.einsum {
            let axes_a_storage: Option<Vec<usize>>;
            let axes_b_storage: Option<Vec<usize>>;

            let (ax_a, ax_b) = match self.axes {
                Axes::SpecificOwned(ax_a, ax_b) => {
                    axes_a_storage = Some(ax_a);
                    axes_b_storage = Some(ax_b);
                    (
                        axes_a_storage.as_deref().unwrap(),
                        axes_b_storage.as_deref().unwrap(),
                    )
                }
                _ => todo!(),
            };

            let a = self.a.to_array().into_dyn();
            let b = self.b.to_array().into_dyn();

            // TODO: it is very likely that this copy is useless. It
            // should be removed in a near future (9th april
            // 2026). However due to tricky typing error I did not
            // manage to give the view to _hypercontract without
            // removing the genericity toward Layout and shape :(

            _hypercontract2(Naive, a.expr(), b.expr(), ax_a, ax_b)
        } else {
            _contract(Naive, self.a, self.b, self.axes, self.alpha)
        }
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

impl<T> Contract<T> for Naive
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + AddAssign + Sum,
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
        NaiveMatMulBuilder {
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
        _contract(Naive, a, b, Axes::All, T::one()).into_scalar()
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
        NaiveContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::LastFirst { k: n },
            einsum: false,
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
        NaiveContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(axes_a, axes_b),
            einsum: false,
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
        let free: std::collections::HashSet<u8> = indices_c.iter().copied().collect();

        // let axes_a: Vec<usize> = indices_a
        //     .iter()
        //     .enumerate()
        //     .filter(|(_, idx)| !free.contains(*idx))
        //     .map(|(pos, _)| pos)
        //     .collect();

        let axes_a: Vec<usize> = indices_a
            .iter()
            .filter(|idx| !free.contains(*idx))
            .map(|&idx| idx as usize) // <-- valeur logique
            .collect();

        // let axes_b: Vec<usize> = indices_b
        //     .iter()
        //     .enumerate()
        //     .filter(|(_, idx)| !free.contains(*idx))
        //     .map(|(pos, _)| pos)
        //     .collect();

        let axes_b: Vec<usize> = indices_b
            .iter()
            .filter(|idx| !free.contains(*idx))
            .map(|&idx| idx as usize) // <-- valeur logique
            .collect();

        NaiveContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::SpecificOwned(axes_a, axes_b),
            einsum: true,
        }
    }
}
