use std::num::NonZero;

use faer::{Accum, Mat, Par, linalg::matmul::matmul};
use faer_traits::ComplexField;
use mdarray::{DSlice, DTensor, DynRank, Layout, Slice, Tensor};
use mdarray_linalg::matmul::{
    _contract, Axes, ContractBuilder, MatMul, MatMulBuilder, Side, Triangle, Type,
};
use num_complex::ComplexFloat;
use num_traits::{One, Zero};

use crate::{Faer, into_faer, into_faer_mut, into_mdarray};

struct FaerMatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
{
    alpha: T,
    a: &'a DSlice<T, 2, La>,
    b: &'a DSlice<T, 2, Lb>,
    par: Par,
}

struct FaerContractBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
{
    alpha: T,
    a: &'a Slice<T, DynRank, La>,
    b: &'a Slice<T, DynRank, Lb>,
    axes: Axes,
}

impl<'a, T, La, Lb> FaerMatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + ComplexField + One + 'static,
{
    #[allow(dead_code)]
    pub fn parallelize(mut self) -> Self {
        // Alternative ??? : use faer::get_global_parallelism()
        self.par = Par::Rayon(NonZero::new(num_cpus::get()).unwrap());
        self
    }
}

impl<'a, T, La, Lb> MatMulBuilder<'a, T, La, Lb> for FaerMatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + ComplexField + One + 'static,
{
    fn parallelize(mut self) -> Self {
        // Alternative ?????
        self.par = Par::Rayon(NonZero::new(num_cpus::get()).unwrap());
        self
    }

    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> DTensor<T, 2> {
        let (ma, _) = *self.a.shape();
        let (_, nb) = *self.b.shape();

        let a_faer = into_faer(self.a);
        let b_faer = into_faer(self.b);

        let mut c_faer = Mat::<T>::zeros(ma, nb);

        matmul(
            &mut c_faer,
            Accum::Replace,
            a_faer,
            b_faer,
            self.alpha,
            self.par,
        );

        into_mdarray::<T>(c_faer)
    }

    fn write<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>) {
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

    fn add_to<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>) {
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

    fn add_to_scaled<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>, _beta: T) {
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

    fn special(self, _lr: Side, _type_of_matrix: Type, _tr: Triangle) -> DTensor<T, 2> {
        self.eval()
    }
}

impl<'a, T, La, Lb> ContractBuilder<'a, T, La, Lb> for FaerContractBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + Zero + One + ComplexField + 'static,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Tensor<T, DynRank> {
        _contract(Faer, self.a, self.b, self.axes, self.alpha)
    }

    fn write(self, _c: &mut Slice<T>) {
        todo!()
    }
}

impl<T> MatMul<T> for Faer
where
    T: ComplexFloat + ComplexField + One + 'static,
{
    fn matmul<'a, La, Lb>(
        &self,
        a: &'a DSlice<T, 2, La>,
        b: &'a DSlice<T, 2, Lb>,
    ) -> impl MatMulBuilder<'a, T, La, Lb>
    where
        La: Layout,
        Lb: Layout,
    {
        FaerMatMulBuilder {
            alpha: T::one(),
            a,
            b,
            par: Par::Seq,
        }
    }

    /// Contracts all axes of the first tensor with all axes of the second tensor.
    fn contract_all<'a, La, Lb>(
        &self,
        a: &'a Slice<T, DynRank, La>,
        b: &'a Slice<T, DynRank, Lb>,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout,
    {
        FaerContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::All,
        }
    }

    /// Contracts the last `n` axes of the first tensor with the first `n` axes of the second tensor.
    /// # Example
    /// For two matrices (2D tensors), `contract_n(1)` performs standard matrix multiplication.
    fn contract_n<'a, La, Lb>(
        &self,
        a: &'a Slice<T, DynRank, La>,
        b: &'a Slice<T, DynRank, Lb>,
        n: usize,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout,
    {
        FaerContractBuilder {
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
    fn contract<'a, La, Lb>(
        &self,
        a: &'a Slice<T, DynRank, La>,
        b: &'a Slice<T, DynRank, Lb>,
        axes_a: impl Into<Box<[usize]>>,
        axes_b: impl Into<Box<[usize]>>,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout,
    {
        FaerContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(axes_a.into(), axes_b.into()),
        }
    }
}
