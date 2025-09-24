use mdarray::{DSlice, DTensor, Layout, Shape, View};

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

    /// `A := α·A`
    fn scale(self, alpha: T) -> Self;

    /// Returns `α·A·x`
    fn eval(self) -> DTensor<T, 1>;

    /// `A := α·A·x`
    fn overwrite<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>);

    /// `A := α·A·x + y`
    fn add_to<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>);

    /// `A := α·A·x + β·y`
    fn add_to_scaled<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>, beta: T);

    /// Rank-1 update: `β·x·yᵀ + α·A`
    fn add_outer<Ly: Layout>(self, y: &DSlice<T, 1, Ly>, beta: T) -> DTensor<T, 2>;

    /// Rank-1 update: `β·x·xᵀ (or x·x†) + α·A`
    fn add_outer_special(self, beta: T, ty: Type, tr: Triangle) -> DTensor<T, 2>;

    // Special rank-2 update: beta * (x * y^T + y * x^T) + alpha * A
    // syr2 her2

    // Special rank-k update: beta * AA^T + alpha * C
    // syrk herk
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

    /// Index of max |xᵢ| (argmaxᵢ |xᵢ|) (**TODO**)
    fn argmax<Lx: Layout, S: Shape>(&self, x: &View<'_, T, S, Lx>) -> Vec<usize>;

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
