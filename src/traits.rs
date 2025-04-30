use mdarray::{DSlice, Layout};
use num_complex::ComplexFloat;

pub trait MatMul {
    type MatMulBuilder<'a, T, La, Lb>
    where
        La: Layout,
        Lb: Layout,
        T: 'a,
        La: 'a,
        Lb: 'a;

    fn matmul<'a, T, La, Lb>(
        &self,
        a: &'a DSlice<T, 2, La>,
        b: &'a DSlice<T, 2, Lb>,
    ) -> Self::MatMulBuilder<'a, T, La, Lb>
    where
        La: Layout,
        Lb: Layout,
        T: ComplexFloat,
        i8: Into<T::Real>,
        T::Real: Into<T>;
}
