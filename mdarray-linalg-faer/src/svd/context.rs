// Singular Value Decomposition (SVD):
//     A = U * Σ * Vᵀ
// where:
//     - A is m × n         (input matrix)Layout
//     - U is m × m        tors, orthogonal)
//     - Σ is m × n         (diagonal matrix with singular values on the diagonal)
//     - Vᵀ is n × n        (transpose of right singular vectors, orthogonal)
//     - s (Σ) contains min(m, n) singular values (non-negative, sorted in descending order)

use super::scalar::BlasScalar;
use super::simple::svd_faer;
use faer_traits::ComplexField;
use mdarray::{DSlice, DTensor, Dense, Layout, tensor};
use mdarray_linalg::{SVD, SVDBuilder, SVDError};
use num_complex::ComplexFloat;

#[derive(Debug)]
pub struct Faer;

struct FaerSVDBuilder<'a, T, L>
where
    L: Layout,
{
    a: &'a mut DSlice<T, 2, L>,
}

impl<'a, T, L> SVDBuilder<'a, T, L> for FaerSVDBuilder<'a, T, L>
where
    T: ComplexFloat + Default + BlasScalar + ComplexField + 'static,
    // T::Real: Into<T>,
    L: Layout,
{
    fn overwrite_suvt<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &mut self,
        s: &mut DSlice<T, 2, Ls>,
        u: &mut DSlice<T, 2, Lu>,
        vt: &mut DSlice<T, 2, Lvt>,
    ) -> Result<(), SVDError> {
        svd_faer::<T, L, Ls, Lu, Lvt>(self.a, s, Some(u), Some(vt))
    }

    fn overwrite_s<Ls: Layout>(&mut self, s: &mut DSlice<T, 2, Ls>) -> Result<(), SVDError> {
        svd_faer::<T, L, Ls, Dense, Dense>(self.a, s, None, None)
    }

    fn eval<Ls: Layout, Lu: Layout, Lvt: Layout>(
        &mut self,
    ) -> Result<(DTensor<T, 2>, DTensor<T, 2>, DTensor<T, 2>), SVDError> {
        let (m, n) = *self.a.shape();
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
        // either lowering them to a single `memset` or eliminating them entirely if they’re unused.
        //
        // See:
        // - LLVM memset/memcpy optimizer: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/Scalar/MemCpyOptimizer.cpp
        //
        // In this context, using `MaybeUninit` adds complexity and potential for undefined behavior
        // with no real performance gain, so we stick to `T::default()`.

        match svd_faer(self.a, &mut s_mda, Some(&mut u_mda), Some(&mut vt_mda)) {
            Err(_) => Err(SVDError::BackendDidNotConverge {
                superdiagonals: (0),
            }),
            Ok(_) => Ok((s_mda, u_mda, vt_mda)),
        }
    }

    fn eval_s<Ls: Layout, Lu: Layout, Lvt: Layout>(&mut self) -> Result<DTensor<T, 2>, SVDError> {
        let (m, n) = *self.a.shape();
        let min_mn = m.min(n);
        let mut s_mda = tensor![[T::default(); min_mn]; min_mn];

        // NOTE:
        // Same rationale as in `eval`: `T::default()` is used instead of `MaybeUninit`,
        // because LLVM already optimizes default initializations effectively.

        match svd_faer::<T, L, Dense, Dense, Dense>(self.a, &mut s_mda, None, None) {
            Err(_) => Err(SVDError::BackendDidNotConverge {
                superdiagonals: (0),
            }),
            Ok(_) => Ok(s_mda),
        }
    }
}

impl<T> SVD<T> for Faer
where
    T: ComplexFloat
        + ComplexField
        + Default
        + BlasScalar
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
{
    fn print_name(&self) {
        println!("Backend: Faer");
    }

    fn svd<'a, L: Layout>(&self, a: &'a mut DSlice<T, 2, L>) -> impl SVDBuilder<'a, T, L> {
        FaerSVDBuilder { a }
    }
}
