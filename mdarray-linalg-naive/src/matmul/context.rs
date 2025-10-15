use num_complex::ComplexFloat;
use num_traits::{One, Zero};

use mdarray::{DSlice, DTensor, Layout, tensor};

use mdarray_linalg::matmul::{Side, Triangle, Type};
use mdarray_linalg::prelude::*;

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
    fn special(self, lr: Side, type_of_matrix: Type, tr: Triangle) -> DTensor<T, 2> {
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
}
