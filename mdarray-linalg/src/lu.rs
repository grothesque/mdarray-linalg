use mdarray::{DSlice, DTensor, Layout};

///  LU decomposition for solving linear systems and matrix inversion
pub trait LU<T> {
    /// Compute LU decomposition overwriting existing matrices
    fn lu_overwrite<L: Layout, Ll: Layout, Lu: Layout, Lp: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        l: &mut DSlice<T, 2, Ll>,
        u: &mut DSlice<T, 2, Lu>,
        p: &mut DSlice<T, 2, Lp>,
    );

    /// Compute LU decomposition with new allocated matrices
    fn lu<L: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
    ) -> (DTensor<T, 2>, DTensor<T, 2>, DTensor<T, 2>);
}
