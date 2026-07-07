use mdarray::{Dim, Layout, Shape, Slice};
use num_complex::ComplexFloat;

/// Performs naively A += α·x·yᵀ
pub fn naive_outer<T: ComplexFloat, La: Layout, Lx: Layout, Ly: Layout, D0, D1>(
    a: &mut Slice<T, (D0, D1), La>,
    x: &Slice<T, (D0,), Lx>,
    y: &Slice<T, (D1,), Ly>,
    alpha: T,
) where
    D0: Dim,
    D1: Dim,
{
    let m = x.shape().dim(0);
    let n = y.shape().dim(0);

    let ash = *a.shape();
    assert!(
        ash.dim(0) == m,
        "Output shape must match input vector length"
    );
    assert!(
        ash.dim(1) == n,
        "Output shape must match input vector length"
    );

    for i in 0..m {
        for j in 0..n {
            a[[i, j]] = a[[i, j]] + alpha * x[[i]] * y[[j]];
        }
    }
}
