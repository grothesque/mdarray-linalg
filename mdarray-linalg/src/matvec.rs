use mdarray::{DSlice, DTensor, DynRank, Layout, Shape};

use crate::matmul::{Triangle, Type};

use num_complex::ComplexFloat;

/// Matrix-vector operations (GEMV, SYMV, TRMV, etc.)
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

/// Builder for matrix-vector operations
pub trait MatVecBuilder<'a, T, La, Lx>
where
    La: Layout,
    Lx: Layout,
    T: 'a,
    La: 'a,
    Lx: 'a,
{
    fn parallelize(self) -> Self;
    fn scale(self, alpha: T) -> Self;
    fn eval(self) -> DTensor<T, 1>;
    fn overwrite<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>);
    fn add_to<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>);
    fn add_to_scaled<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>, beta: T);

    /// Rank-1 update: beta * x * y^T + alpha * A
    fn add_outer<Ly: Layout>(self, y: &DSlice<T, 1, Ly>, beta: T) -> DTensor<T, 2>;
    fn add_outer_special(self, beta: T, ty: Type, tr: Triangle) -> DTensor<T, 2>;

    // Special rank-2 update: beta * (x * y^T + y * x^T) + alpha * A
    // syr2 her2

    // Special rank-k update: beta * AA^T + alpha * C
    // syrk herk
}

/// Vector operations and utilities
pub trait VecOps<T: ComplexFloat> {
    /// AXPY: y := alpha * x + y
    /// BLAS: SAXPY, DAXPY, CAXPY, ZAXPY
    fn add_to_scaled<Lx: Layout, Ly: Layout>(
        &self,
        alpha: T,
        x: &DSlice<T, 1, Lx>,
        y: &mut DSlice<T, 1, Ly>,
    );

    /// Dot product
    /// BLAS: SDOT, DDOT, CDOTU, ZDOTU
    fn dot<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T;

    /// Conjugated dot product
    /// BLAS: CDOTC, ZDOTC
    fn dotc<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T;

    /// Euclidean norm
    /// BLAS: SNRM2, DNRM2, SCNRM2, DZNRM2
    fn norm2<Lx: Layout>(&self, x: &DSlice<T, 1, Lx>) -> T::Real;

    /// Sum of absolute values
    /// BLAS: SASUM, DASUM, SCASUM, DZASUM
    fn norm1<Lx: Layout>(&self, x: &DSlice<T, 1, Lx>) -> T::Real
    where
        T: ComplexFloat;

    /// Index of maximum absolute value
    /// BLAS: ISAMAX, IDAMAX, ICAMAX, IZAMAX
    fn argmax<Lx: Layout>(&self, x: &DSlice<T, 2, Lx>) -> Vec<usize>;

    /// Copy vector: y := x
    /// BLAS: SCOPY, DCOPY, CCOPY, ZCOPY
    fn copy<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &mut DSlice<T, 1, Ly>);

    /// Scale vector: x := alpha * x
    /// BLAS: SSCAL, DSCAL, CSCAL, ZSCAL
    fn scal<Lx: Layout>(&self, alpha: T, x: &mut DSlice<T, 1, Lx>);

    /// Swap vectors: x <-> y
    /// BLAS: SSWAP, DSWAP, CSWAP, ZSWAP
    fn swap<Lx: Layout, Ly: Layout>(&self, x: &mut DSlice<T, 1, Lx>, y: &mut DSlice<T, 1, Ly>);

    /// Givens rotation
    fn rot<Lx: Layout, Ly: Layout>(
        &self,
        x: &mut DSlice<T, 1, Lx>,
        y: &mut DSlice<T, 1, Ly>,
        c: T::Real,
        s: T,
    ) where
        T: ComplexFloat;
}

/// Rank-2 updates
pub trait Rank2Update<T> {
    fn rank2_update<'a, Lx, Ly>(
        &self,
        x: &'a DSlice<T, 1, Lx>,
        y: &'a DSlice<T, 1, Ly>,
    ) -> impl Rank2Builder<'a, T, Lx, Ly>
    where
        Lx: Layout,
        Ly: Layout;
}

/// Builder for rank-2 operations, similar to special() in matmul
/// BLAS: SSYR2, DSYR2, CHER2, ZHER2
pub trait Rank2Builder<'a, T, Lx, Ly>
where
    Lx: Layout,
    Ly: Layout,
    T: 'a,
    Lx: 'a,
    Ly: 'a,
{
    fn scale(self, alpha: T) -> Self;

    /// Symmetric rank-2 update: A := alpha * x * y^T + alpha * y * x^T + A
    /// BLAS: SSYR2, DSYR2
    fn syr2<La: Layout>(self, tr: Triangle, a: &mut DSlice<T, 2, La>);

    /// Hermitian rank-2 update: A := alpha * x * y^H + conj(alpha) * y * x^H + A
    /// BLAS: CHER2, ZHER2
    fn her2<La: Layout>(self, tr: Triangle, a: &mut DSlice<T, 2, La>);
}
