use mdarray::{DSlice, DTensor, Layout};

pub struct PRRLUDecomp<T> {
    pub p: DTensor<T, 2>,
    pub l: DTensor<T, 2>,
    pub d: DTensor<T, 2>,
    pub u: DTensor<T, 2>,
    pub q: DTensor<T, 2>,
    pub rank: usize,
}

pub trait PRRLU<T> {
    /// Compute full Partial Rank-Revealing LU decomposition
    ///
    /// Decomposes matrix A into the form: A = P * L * D * U * Q
    /// where P, Q are permutation matrices, L is unit lower triangular,
    /// D is diagonal, and U is unit upper triangular.
    ///
    /// Algorithm: iteratively selects the maximum element as pivot,
    /// permutes rows/columns to bring it to diagonal position,
    /// computes Schur complement for the (n-1)×(n-1) subblock,
    /// and repeats until completion or pivot falls below numerical precision.
    fn prrlu<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> PRRLUDecomp<T>;

    /// Compute PRRLU with specified target rank
    fn prrlu_rank<L: Layout>(&self, a: &mut DSlice<T, 2, L>, target_rank: usize) -> PRRLUDecomp<T>;

    /// Compute PRRLU decomposition, overwriting existing matrices
    fn prrlu_overwrite<L: Layout, Lp: Layout, Ll: Layout, Ld: Layout, Lu: Layout, Lq: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        p: &mut DSlice<T, 2, Lp>,
        l: &mut DSlice<T, 2, Ll>,
        d: &mut DSlice<T, 2, Ld>,
        u: &mut DSlice<T, 2, Lu>,
        q: &mut DSlice<T, 2, Lq>,
    ) -> usize;

    /// Compute only the rank of the matrix using PRRLU
    fn rank<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> usize;

    /// Compute only the diagonal elements from PRRLU
    fn pivots<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> DTensor<T, 2>;
}
