//! Partial rank-revealing LU decomposition utilities.
//!```rust
//!use mdarray::tensor;
//!use mdarray_linalg::Naive;
//!use mdarray_linalg::prrlu::PRRLUDecomp;
//!use crate::mdarray_linalg::prelude::PRRLU;
//!// ----- Naive backend -----
//!let a = tensor![[1., 2.], [3., 4.]];
//!let PRRLUDecomp { p, l, u, q, rank } = Naive.prrlu(&mut a.clone());
//!println!("PRRLU decomposition done (Naive backend)");
//!println!(
//!    "p: {:?}, l: {:?}, u: {:?}, q: {:?}, rank: {:?}",
//!    p, l, u, q, rank
//!);
//!```

use mdarray::{DSlice, DTensor, Layout};

/// Holds the results of a pivoted, rank-revealing LU decomposition,
/// including permutation matrices and the computed rank
pub struct PRRLUDecomp<T> {
    pub p: DTensor<T, 2>,
    pub l: DTensor<T, 2>,
    pub u: DTensor<T, 2>,
    pub q: DTensor<T, 2>,
    pub rank: usize,
}

/// Pivoted LU decomposition with rank-revealing
pub trait PRRLU<T> {
    /// Compute full Partial Rank-Revealing LU decomposition
    ///
    /// Decomposes matrix `A` into the form: `A = P * L * U * Q` where
    /// `P`, `Q` are permutation matrices, `L` is unit lower
    /// triangular and `U` is unit upper triangular.
    ///
    /// Algorithm: iteratively selects the maximum element as pivot,
    /// permutes rows/columns to bring it to diagonal position,
    /// computes Schur complement for the (n-1)×(n-1) subblock,
    /// and repeats until completion or pivot falls below numerical precision.
    fn prrlu<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> PRRLUDecomp<T>;

    /// Compute PRRLU with specified target rank
    fn prrlu_rank<L: Layout>(&self, a: &mut DSlice<T, 2, L>, target_rank: usize) -> PRRLUDecomp<T>;

    /// Compute PRRLU decomposition, overwriting existing matrices, `u`  is stored in `a`.
    fn prrlu_overwrite<L: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        p: &mut DSlice<T, 2>,
        l: &mut DSlice<T, 2>,
        q: &mut DSlice<T, 2>,
    ) -> usize;

    /// Compute only the rank of the matrix using PRRLU
    fn rank<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> usize;
}
