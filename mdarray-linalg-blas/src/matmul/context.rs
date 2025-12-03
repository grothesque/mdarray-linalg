use std::mem::MaybeUninit;

use cblas_sys::{CBLAS_SIDE, CBLAS_UPLO};
use mdarray::{DSlice, DTensor, Dense, Dim, DynRank, Layout, Slice, Tensor, tensor};
use mdarray_linalg::matmul::{
    _contract, Axes, ContractBuilder, MatMul, MatMulBuilder, Side, Triangle, Type,
};
use num_complex::ComplexFloat;
use num_traits::{One, Zero};

use super::{
    scalar::BlasScalar,
    simple::{gemm, gemm_uninit, hemm_uninit, symm_uninit, trmm},
};
use crate::Blas;

struct BlasMatMulBuilder<'a, T, La, Lb, D0, D1, D2>
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

struct BlasContractBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
{
    alpha: T,
    a: &'a Slice<T, DynRank, La>,
    b: &'a Slice<T, DynRank, Lb>,
    axes: Axes,
}

impl<'a, T, La, Lb, D0, D1, D2> MatMulBuilder<'a, T, La, Lb, D0, D1, D2>
    for BlasMatMulBuilder<'a, T, La, Lb, D0, D1, D2>
where
    La: Layout,
    Lb: Layout,
    T: BlasScalar + ComplexFloat + Zero + One,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    fn parallelize(self) -> Self {
        self
    }

    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Tensor<T, (D0, D2)> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let c = Tensor::from_elem((m, n), MaybeUninit::<T>::uninit());
        gemm_uninit::<T, La, Lb, Dense, D0, D1, D2>(self.alpha, self.a, self.b, T::zero(), c)
        // formerly 0.into().into() instead of T::zero() but
        // propagating the associated bounds was causing a lot of
        // trouble
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

    fn special(self, lr: Side, type_of_matrix: Type, tr: Triangle) -> Tensor<T, (D0, D2)> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let c = Tensor::from_elem((m, n), MaybeUninit::<T>::uninit());
        let cblas_side = match lr {
            Side::Left => CBLAS_SIDE::CblasLeft,
            Side::Right => CBLAS_SIDE::CblasRight,
        };
        let cblas_triangle = match tr {
            Triangle::Lower => CBLAS_UPLO::CblasLower,
            Triangle::Upper => CBLAS_UPLO::CblasUpper,
        };
        match type_of_matrix {
            Type::Her => hemm_uninit::<T, La, Lb, Dense, D0, D1, D2>(
                self.alpha,
                self.a,
                self.b,
                T::zero(),
                c,
                cblas_side,
                cblas_triangle,
            ),
            Type::Sym => symm_uninit::<T, La, Lb, Dense, D0, D1, D2>(
                self.alpha,
                self.a,
                self.b,
                T::zero(),
                c,
                cblas_side,
                cblas_triangle,
            ),
            Type::Tri => {
                let mut b_copy = Tensor::<T, (D1, D2)>::from_elem(*self.b.shape(), T::zero());
                b_copy.assign(self.b);
                trmm(self.alpha, self.a, &mut b_copy, cblas_side, cblas_triangle);

                let mut c_copy = Tensor::<T, (D0, D2)>::from_elem(*c.shape(), T::zero());
                c_copy.assign(b_copy);
                c_copy
            }
        }
    }
}

impl<'a, T, La, Lb> ContractBuilder<'a, T, La, Lb> for BlasContractBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: BlasScalar + ComplexFloat + Zero + One,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Tensor<T> {
        _contract(Blas, self.a, self.b, self.axes, self.alpha)
    }

    fn write(self, _c: &mut Slice<T>) {
        todo!()
    }
}

impl<T> MatMul<T> for Blas
where
    T: BlasScalar + ComplexFloat,
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
        BlasMatMulBuilder {
            alpha: T::one(),
            a,
            b,
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
        BlasContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::All,
        }
    }

    /// Contracts the last `n` axes of the first tensor with the first `n` axes of the second tensor.
    /// # Example
    /// For two matrices (2D tensors), `contract_n(1)` performs standard matrix multiplication.
    fn contract_n<'a, La: Layout, Lb: Layout>(
        &self,
        a: &'a Slice<T, DynRank, La>,
        b: &'a Slice<T, DynRank, Lb>,
        n: usize,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
    {
        BlasContractBuilder {
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
    fn contract<'a, La: Layout, Lb: Layout>(
        &self,
        a: &'a Slice<T, DynRank, La>,
        b: &'a Slice<T, DynRank, Lb>,
        axes_a: impl Into<Box<[usize]>>,
        axes_b: impl Into<Box<[usize]>>,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
    {
        BlasContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(axes_a.into(), axes_b.into()),
        }
    }
}
