// Singular Value Decomposition (SVD):
//     A = U * Σ * V^T
// where:
//     - A is m × n         (input matrix)
//     - U is m × m        (left singular vectors, orthogonal)
//     - Σ is µ × µ         (diagonal matrix with singular values on the diagonal, µ = min(m,n))
//     - V^T is n × n      (transpose of right singular vectors, orthogonal)
//     - s (Σ) contains min(m, n) singular values (non-negative, sorted in descending order)

use faer_traits::ComplexField;
use mdarray::{Dense, Dim, Layout, Shape, Slice, Tensor};
use mdarray_linalg::svd::{SVD, SVDDecomp, SVDError};
use num_complex::ComplexFloat;

use super::simple::svd_faer;
use crate::Faer;

impl<T, D, L> SVD<T, D, L> for Faer
where
    T: ComplexFloat
        + ComplexField
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
    D: Dim,
    L: Layout,
{
    /// Compute full SVD with new allocated matrices
    fn svd(&self, a: &mut Slice<T, (D, D), L>) -> Result<SVDDecomp<T, D>, SVDError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);

        let s_shape = <(D, D) as Shape>::from_dims(&[min_mn, min_mn]);
        let u_shape = <(D, D) as Shape>::from_dims(&[m, m]);
        let vt_shape = <(D, D) as Shape>::from_dims(&[n, n]);

        let mut s_mda = Tensor::from_elem(s_shape, T::default());
        let mut u_mda = Tensor::from_elem(u_shape, T::default());
        let mut vt_mda = Tensor::from_elem(vt_shape, T::default());

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

    /// Compute only singular values with new allocated matrix
    fn svd_s(&self, a: &mut Slice<T, (D, D), L>) -> Result<Tensor<T, (D, D)>, SVDError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);

        let s_shape = <(D, D) as Shape>::from_dims(&[min_mn, min_mn]);
        let mut s_mda = Tensor::from_elem(s_shape, T::default());

        // NOTE:
        // Same rationale as in `svd`: `T::default()` is used instead of `MaybeUninit`,
        // because LLVM already optimizes default initializations effectively.

        match svd_faer::<T, D, L, Dense, Dense, Dense>(a, &mut s_mda, None, None) {
            Err(_) => Err(SVDError::BackendDidNotConverge {
                superdiagonals: (0),
            }),
            Ok(_) => Ok(s_mda),
        }
    }

    /// Compute full SVD, overwriting existing matrices
    fn svd_write<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<T, (D, D), Ls>,
        u: &mut Slice<T, (D, D), Lu>,
        vt: &mut Slice<T, (D, D), Lvt>,
    ) -> Result<(), SVDError> {
        svd_faer::<T, D, L, Ls, Lu, Lvt>(a, s, Some(u), Some(vt))
    }

    /// Compute only singular values, overwriting existing matrix
    fn svd_write_s<Ls: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<T, (D, D), Ls>,
    ) -> Result<(), SVDError> {
        svd_faer::<T, D, L, Ls, Dense, Dense>(a, s, None, None)
    }
}
