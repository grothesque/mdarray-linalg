use mdarray::{DSlice, DTensor, Layout};

/// Linear system solver using LU decomposition
pub trait Solve<T> {
    /// Solves linear system AX = b overwriting existing matrices
    /// A is overwritten with its LU decomposition
    /// B is overwritten with the solution X
    /// P is filled with the permutation matrix such that A = P*L*U
    fn solve_overwrite<La: Layout, Lb: Layout, Lp: Layout>(
        &self,
        a: &mut DSlice<T, 2, La>,
        b: &mut DSlice<T, 2, Lb>,
        p: &mut DSlice<T, 2, Lp>,
    );

    /// Solves linear system AX = B with new allocated solution matrix
    /// A is modified (overwritten with LU decomposition)
    /// Returns the solution X and P the permutation matrix
    fn solve<La: Layout, Lb: Layout>(
        &self,
        a: &mut DSlice<T, 2, La>,
        b: &DSlice<T, 2, Lb>,
    ) -> (DTensor<T, 2>, DTensor<T, 2>);
}
