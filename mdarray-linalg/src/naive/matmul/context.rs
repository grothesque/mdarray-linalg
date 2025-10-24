use num_complex::ComplexFloat;
use num_traits::{One, Zero};

use mdarray::{DSlice, DTensor, DynRank, Layout, Slice, Tensor, tensor};

use crate::matmul::{Axes, Side, Triangle, Type, _contract};
use crate::prelude::*;

use crate::Naive;

use super::simple::naive_matmul;

struct NaiveMatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
{
    alpha: T,
    a: &'a DSlice<T, 2, La>,
    b: &'a DSlice<T, 2, Lb>,
}

struct NaiveContractBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
{
    alpha: T,
    a: &'a Slice<T, DynRank, La>,
    b: &'a Slice<T, DynRank, Lb>,
    axes: Axes,
}

impl<'a, T, La, Lb> MatMulBuilder<'a, T, La, Lb> for NaiveMatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + Zero + One,
    // i8: Into<T::Real>,
    // T::Real: Into<T>,
{
    /// Enable parallelization.
    fn parallelize(self) -> Self {
        self
    }

    /// Multiplies the result by a scalar factor.
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    /// Returns a new owned tensor containing the result.
    fn eval(self) -> DTensor<T, 2> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let mut c = tensor![[T::zero(); n]; m];
        naive_matmul(self.alpha, self.a, self.b, T::zero(), &mut c);
        c
    }

    /// Overwrites the provided slice with the result.
    fn overwrite<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>) {
        naive_matmul(self.alpha, self.a, self.b, T::zero(), c);
    }

    /// Adds the result to the provided slice.
    fn add_to<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>) {
        naive_matmul(self.alpha, self.a, self.b, T::one(), c);
    }

    /// Adds the result to the provided slice after scaling the slice by `beta`
    /// (i.e. C := beta * C + result).
    fn add_to_scaled<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>, beta: T) {
        naive_matmul(self.alpha, self.a, self.b, beta, c);
    }

    /// Computes a matrix product where the first operand is a special
    /// matrix (symmetric, Hermitian, or triangular) and the other is
    /// general.
    ///
    /// The special matrix is always treated as `A`. `lr` determines the multiplication order:
    /// - `Side::Left`  : C := alpha * A * B
    /// - `Side::Right` : C := alpha * B * A
    ///
    /// # Parameters
    /// * `lr` - side of multiplication (left or right)
    /// * `type_of_matrix` - special matrix type: `Sym`, `Her`, or `Tri`
    /// * `tr` - triangle containing stored data: `Upper` or `Lower`
    ///
    /// Only the specified triangle needs to be stored for symmetric/Hermitian matrices;
    /// for triangular matrices it specifies which half is used.
    ///
    /// # Returns
    /// A new tensor with the result.
    fn special(self, _lr: Side, _type_of_matrix: Type, _tr: Triangle) -> DTensor<T, 2> {
        todo!()
    }
}

impl<'a, T, La, Lb> ContractBuilder<'a, T, La, Lb> for NaiveContractBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + Zero + One,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Tensor<T> {
        _contract(Naive, self.a, self.b, self.axes, self.alpha)
    }

    fn overwrite(self, _c: &mut Slice<T>) {
        todo!()
    }
}

impl<T> MatMul<T> for Naive
where
    T: ComplexFloat,
    // i8: Into<T::Real>,
    // T::Real: Into<T>,
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
        NaiveMatMulBuilder {
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
        NaiveContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(axes_a.into(), axes_b.into()),
        }
    }
}
