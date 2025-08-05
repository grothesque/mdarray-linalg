use mdarray::{DSlice, DTensor, Layout};

pub trait MatMul<T> {
    fn matmul<'a, La, Lb>(
        &self,
        a: &'a DSlice<T, 2, La>,
        b: &'a DSlice<T, 2, Lb>,
    ) -> impl MatMulBuilder<'a, T, La, Lb>
    where
        La: Layout,
        Lb: Layout;
}

pub trait MatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: 'a,
    La: 'a,
    Lb: 'a,
{
    /// Enable parallelization.
    fn parallelize(self) -> Self;

    /// Multiplies the result by a scalar factor.
    fn scale(self, factor: T) -> Self;

    /// Returns a new owned tensor containing the result.
    fn eval(self) -> DTensor<T, 2>;

    /// Overwrites the provided slice with the result.
    fn overwrite<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>);

    /// Adds the result to the provided slice.
    fn add_to<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>);

    /// Adds the result to the provided slice after scaling the slice by `beta`
    /// (i.e. C := beta * C + result).
    fn add_to_scaled<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>, beta: T);
}
