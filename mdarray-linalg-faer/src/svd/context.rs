// Singular Value Decomposition (SVD):
//     A = U * Σ * V^T
// where:
//     - A is m × n         (input matrix)
//     - U is m × m        (left singular vectors, orthogonal)
//     - Σ is m × n         (diagonal matrix with singular values on the diagonal)
//     - V^T is n × n      (transpose of right singular vectors, orthogonal)
//     - s (Σ) contains min(m, n) singular values (non-negative, sorted in descending order)

use super::scalar::BlasScalar;
use super::simple::svd_faer;
use faer_traits::ComplexField;
use mdarray::{DSlice, DTensor, Dense, Layout, tensor};
use mdarray_linalg::{SVD, SVDDecomp, SVDError};
use num_complex::ComplexFloat;

use crate::Faer;

impl<T> SVD<T> for Faer
where
    T: ComplexFloat
        + ComplexField
        + Default
        + BlasScalar
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
{
    // Compute full SVD with new allocated matrices
    fn svd<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> Result<SVDDecomp<T>, SVDError> {
        let (m, n) = *a.shape();
        let min_mn = m.min(n);
        let mut s_mda = tensor![[T::default(); min_mn]; min_mn];
        let mut u_mda = tensor![[T::default(); m]; m];
        let mut vt_mda = tensor![[T::default(); n]; n];

        // NOTE:
        // These tensors were previously created with `MaybeUninit` to avoid default-initialization.
        // However, after benchmarking, we observed **no measurable performance benefit**,
        // so for the sake of simplicity and safety, `T::default()` is now used instead.
        //
        // LLVM aggressively optimizes trivial memory initialization (like zeroing floats or ints),
        // either lowering them to a single `memset` or eliminating them entirely if they're unused.
        //
        // See:
        // - LLVM memset/memcpy optimizer: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/Scalar/MemCpyOptimizer.cpp
        //
        // In this context, using `MaybeUninit` adds complexity and potential for undefined behavior
        // with no real performance gain, so we stick to `T::default()`.

        match svd_faer(a, &mut s_mda, Some(&mut u_mda), Some(&mut vt_mda)) {
            Err(_) => Err(SVDError::BackendDidNotConverge {
                superdiagonals: (0),
            }),
            Ok(_) => Ok(SVDDecomp {
                s: s_mda,
                u: u_mda,
                vt: vt_mda,
            }),
        }
    }

    // Compute only singular values with new allocated matrix
    fn svd_s<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> Result<DTensor<T, 2>, SVDError> {
        let (m, n) = *a.shape();
        let min_mn = m.min(n);
        let mut s_mda = tensor![[T::default(); min_mn]; min_mn];

        // NOTE:
        // Same rationale as in `svd`: `T::default()` is used instead of `MaybeUninit`,
        // because LLVM already optimizes default initializations effectively.

        match svd_faer::<T, L, Dense, Dense, Dense>(a, &mut s_mda, None, None) {
            Err(_) => Err(SVDError::BackendDidNotConverge {
                superdiagonals: (0),
            }),
            Ok(_) => Ok(s_mda),
        }
    }

    // Compute full SVD, overwriting existing matrices
    fn svd_overwrite<L: Layout, Ls: Layout, Lu: Layout, Lvt: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        s: &mut DSlice<T, 2, Ls>,
        u: &mut DSlice<T, 2, Lu>,
        vt: &mut DSlice<T, 2, Lvt>,
    ) -> Result<(), SVDError> {
        svd_faer::<T, L, Ls, Lu, Lvt>(a, s, Some(u), Some(vt))
    }

    // Compute only singular values, overwriting existing matrix
    fn svd_overwrite_s<L: Layout, Ls: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        s: &mut DSlice<T, 2, Ls>,
    ) -> Result<(), SVDError> {
        svd_faer::<T, L, Ls, Dense, Dense>(a, s, None, None)
    }
}
