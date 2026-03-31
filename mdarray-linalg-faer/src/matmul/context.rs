use std::num::NonZero;

use faer::{Accum, Par, linalg::matmul::matmul};
use faer_traits::ComplexField;
use mdarray::{Array, Dim, DynRank, Layout, Shape, Slice};
use mdarray_linalg::matmul::{
    _contract, Axes, Contract, ContractAxes, ContractBuilder, MatMulBuilder, extract_axes,
};
use mdarray_linalg::{finish_contraction, prepare_contraction};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};

use crate::{Faer, into_faer, into_faer_mut};

struct FaerMatMulBuilder<'a, T, D0, D1, D2, La, Lb>
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
    par: Par,
}

struct FaerContractBuilder<'a, T, Sa, Sb, La, Lb>
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
    par: Par,
}

impl<'a, T, D0, D1, D2, La, Lb> MatMulBuilder<'a, T, D0, D1, D2, La, Lb>
    for FaerMatMulBuilder<'a, T, D0, D1, D2, La, Lb>
where
    La: Layout,
    Lb: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
    T: ComplexFloat + ComplexField + One + Zero + 'static,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Array<T, (D0, D2)> {
        let (ma, _) = *self.a.shape();
        let (_, nb) = *self.b.shape();

        let a_faer = into_faer(self.a);
        let b_faer = into_faer(self.b);

        let mut c = Array::<T, (D0, D2)>::from_elem((ma, nb), T::zero());
        let mut c_faer = into_faer_mut(&mut c);

        matmul(
            &mut c_faer,
            Accum::Replace,
            a_faer,
            b_faer,
            self.alpha,
            self.par,
        );

        c
    }

    fn write<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        let mut c_faer = into_faer_mut(c);
        matmul(
            &mut c_faer,
            Accum::Replace,
            into_faer(self.a),
            into_faer(self.b),
            self.alpha,
            self.par,
        );
    }

    fn add_to<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        let mut c_faer = into_faer_mut(c);
        matmul(
            &mut c_faer,
            Accum::Add,
            into_faer(self.a),
            into_faer(self.b),
            self.alpha,
            self.par,
        );
    }

    fn add_to_scaled<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>, _beta: T) {
        let mut c_faer = into_faer_mut(c);
        matmul(
            &mut c_faer,
            Accum::Add,
            into_faer(self.a),
            into_faer(self.b),
            self.alpha,
            self.par,
        );
        todo!(); // multiplication by beta not implemented in faer ?
    }
}

impl<'a, T, Sa, Sb, La, Lb> ContractBuilder<'a, T, Sa, Sb, La, Lb>
    for FaerContractBuilder<'a, T, Sa, Sb, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + ComplexField + Zero + One + 'static + MulAdd<Output = T>,
    Sa: Shape,
    Sb: Shape,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Array<T, DynRank> {
        let (a_2d, b_2d, keep_shape_a, keep_shape_b) =
            prepare_contraction!(self.axes, self.a, self.b);
        let a_faer = into_faer(&a_2d);
        let b_faer = into_faer(&b_2d);

        let (m, _) = *a_2d.shape();
        let (_, n) = *b_2d.shape();
        let mut c = Array::<T, (usize, usize)>::from_elem((m, n), T::zero());
        let mut c_faer = into_faer_mut(&mut c);

        matmul(
            &mut c_faer,
            Accum::Replace,
            a_faer,
            b_faer,
            self.alpha,
            self.par,
        );

        finish_contraction!(c, keep_shape_a, keep_shape_b)
    }

    fn write<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>) {
        // TODO WRITE test because I'm not sure it works
        let (a_2d, b_2d, keep_shape_a, keep_shape_b) =
            prepare_contraction!(self.axes, self.a, self.b);

        let a_faer = into_faer(&a_2d);
        let b_faer = into_faer(&b_2d);

        let (m, _) = *a_2d.shape();
        let (_, n) = *b_2d.shape();

        let mut c_reshaped = c.reshape_mut([m, n]);
        let mut c_faer = into_faer_mut(&mut c_reshaped);

        matmul(
            &mut c_faer,
            Accum::Replace,
            a_faer,
            b_faer,
            self.alpha,
            self.par,
        );
    }

    fn add_to<Sc: Shape, Lc: Layout>(self, _c: &mut Slice<T, Sc, Lc>) {
        todo!()
    }

    fn add_to_scaled<Sc: Shape, Lc: Layout>(self, _c: &mut Slice<T, Sc, Lc>, _beta: T) {
        todo!()
    }
}

impl<T> Contract<T> for Faer
where
    T: ComplexFloat + ComplexField + Zero + One + 'static + MulAdd<Output = T>,
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
        FaerMatMulBuilder {
            alpha: T::one(),
            a,
            b,
            par: if self.parallelize {
                Par::Rayon(NonZero::new(num_cpus::get()).unwrap())
            } else {
                Par::Seq
            },
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
        _contract(
            Faer {
                parallelize: self.parallelize,
            },
            a,
            b,
            Axes::All,
            T::one(),
        )
        .into_scalar()
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
        FaerContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::LastFirst { k: n },
            par: if self.parallelize {
                Par::Rayon(NonZero::new(num_cpus::get()).unwrap())
            } else {
                Par::Seq
            },
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
        FaerContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(axes_a, axes_b),
            par: if self.parallelize {
                Par::Rayon(NonZero::new(num_cpus::get()).unwrap())
            } else {
                Par::Seq
            },
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
        FaerContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(
                Box::leak(axes_a.into_boxed_slice()),
                Box::leak(axes_b.into_boxed_slice()),
            ),
            par: if self.parallelize {
                Par::Rayon(NonZero::new(num_cpus::get()).unwrap())
            } else {
                Par::Seq
            },
        }
    }
}
