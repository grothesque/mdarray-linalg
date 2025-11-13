//! Basic vector and matrix-vector operations, including Ax, Ax + βy, Givens rotations, argmax, and rank-1 updates
use mdarray::{DSlice, DTensor, Layout, Shape, Slice};

use crate::matmul::{Triangle, Type};

use num_complex::ComplexFloat;

/// Matrix-vector multiplication and transformations
pub trait MatVec<T> {
    fn matvec<'a, La, Lx>(
        &self,
        a: &'a DSlice<T, 2, La>,
        x: &'a DSlice<T, 1, Lx>,
    ) -> impl MatVecBuilder<'a, T, La, Lx>
    where
        La: Layout,
        Lx: Layout;
}

/// Builder interface for configuring matrix-vector operations
pub trait MatVecBuilder<'a, T, La, Lx>
where
    La: Layout,
    Lx: Layout,
    T: 'a,
    La: 'a,
    Lx: 'a,
{
    fn parallelize(self) -> Self;

    /// `α := α·α'`
    fn scale(self, alpha: T) -> Self;

    /// Returns `α·A·x`
    fn eval(self) -> DTensor<T, 1>;

    /// `y := α·A·x`
    fn overwrite<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>);

    /// `Returns α·A·x + y`
    fn add_to_vec<Ly: Layout>(self, y: &DSlice<T, 1, Ly>) -> DTensor<T, 1>;

    /// `Returns α·A·x + β·y`
    fn add_to_scaled_vec<Ly: Layout>(self, y: &DSlice<T, 1, Ly>, beta: T) -> DTensor<T, 1>;
}

/// Vector operations and basic linear algebra utilities
pub trait VecOps<T: ComplexFloat> {
    /// Accumulate a scaled vector: `y := α·x + y`
    fn add_to_scaled<Lx: Layout, Ly: Layout>(
        &self,
        alpha: T,
        x: &DSlice<T, 1, Lx>,
        y: &mut DSlice<T, 1, Ly>,
    );

    /// Dot product: `∑xᵢyᵢ`
    fn dot<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T;

    /// Conjugated dot product: `∑(xᵢ * conj(yᵢ))`
    fn dotc<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T;

    /// L2 norm: `√(∑|xᵢ|²)`
    fn norm2<Lx: Layout>(&self, x: &DSlice<T, 1, Lx>) -> T::Real;

    /// L1 norm: `∑|xᵢ|`
    fn norm1<Lx: Layout>(&self, x: &DSlice<T, 1, Lx>) -> T::Real
    where
        T: ComplexFloat;

    /// Copy vector: `y := x` (**TODO**)
    fn copy<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &mut DSlice<T, 1, Ly>);

    /// Scale vector: `x := α·xᵢ` (**TODO**)
    fn scal<Lx: Layout>(&self, alpha: T, x: &mut DSlice<T, 1, Lx>);

    /// Swap vectors: `x ↔ y` (**TODO**)
    fn swap<Lx: Layout, Ly: Layout>(&self, x: &mut DSlice<T, 1, Lx>, y: &mut DSlice<T, 1, Ly>);

    /// Givens rotation (**TODO**)
    fn rot<Lx: Layout, Ly: Layout>(
        &self,
        x: &mut DSlice<T, 1, Lx>,
        y: &mut DSlice<T, 1, Ly>,
        c: T::Real,
        s: T,
    ) where
        T: ComplexFloat;
}

/// Argmax for tensors, unlike other traits: it requires `T: PartialOrd` and works on tensor of any rank.
pub trait Argmax<T: ComplexFloat + std::cmp::PartialOrd> {
    fn argmax_overwrite<Lx: Layout, S: Shape>(
        &self,
        x: &Slice<T, S, Lx>,
        output: &mut Vec<usize>,
    ) -> bool;

    fn argmax_abs_overwrite<Lx: Layout, S: Shape>(
        &self,
        x: &Slice<T, S, Lx>,
        output: &mut Vec<usize>,
    ) -> bool;

    /// Index of max xᵢ (argmaxᵢ xᵢ)
    fn argmax<Lx: Layout, S: Shape>(&self, x: &Slice<T, S, Lx>) -> Option<Vec<usize>>;

    /// Index of max |xᵢ| (argmaxᵢ |xᵢ|)
    fn argmax_abs<Lx: Layout, S: Shape>(&self, x: &Slice<T, S, Lx>) -> Option<Vec<usize>>;
}

/// Outer product and rank-1 update
pub trait Outer<T> {
    fn outer<'a, Lx, Ly>(
        &self,
        x: &'a DSlice<T, 1, Lx>,
        y: &'a DSlice<T, 1, Ly>,
    ) -> impl OuterBuilder<'a, T, Lx, Ly>
    where
        Lx: Layout,
        Ly: Layout;
}

/// Builder interface for configuring outer product and rank-1 update
pub trait OuterBuilder<'a, T, Lx, Ly>
where
    Lx: Layout,
    Ly: Layout,
    T: 'a,
    Lx: 'a,
    Ly: 'a,
{
    /// `α := α·α'`
    fn scale(self, alpha: T) -> Self;

    /// Returns `α·xy`
    fn eval(self) -> DTensor<T, 2>;

    /// `a := α·xy`
    fn overwrite<La: Layout>(self, a: &mut DSlice<T, 2, La>);

    /// Rank-1 update, returns `α·x·yᵀ + A`
    fn add_to<La: Layout>(self, a: &DSlice<T, 2, La>) -> DTensor<T, 2>;

    /// Rank-1 update: `A := α·x·yᵀ + A`
    fn add_to_overwrite<La: Layout>(self, a: &mut DSlice<T, 2, La>);

    /// Rank-1 update: returns `α·x·xᵀ (or x·x†) + A` on special matrix
    fn add_to_special(self, a: &DSlice<T, 2>, ty: Type, tr: Triangle) -> DTensor<T, 2>;

    /// Rank-1 update: `A := α·x·xᵀ (or x·x†) + A` on special matrix
    fn add_to_special_overwrite(self, a: &mut DSlice<T, 2>, ty: Type, tr: Triangle);
}
