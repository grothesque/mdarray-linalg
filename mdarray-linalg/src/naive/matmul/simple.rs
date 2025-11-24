use mdarray::{DSlice, Layout};
use num_complex::ComplexFloat;

/// Textbook implementation of matrix multiplication, useful for
/// debugging and simple tests without relying on a external backend
pub fn naive_matmul<T: ComplexFloat, La: Layout, Lb: Layout, Lc: Layout>(
    alpha: T,
    a: &DSlice<T, 2, La>,
    b: &DSlice<T, 2, Lb>,
    beta: T,
    c: &mut DSlice<T, 2, Lc>,
) {
    for mut ci in c.rows_mut() {
        for cij in ci.expr_mut() {
            *cij = beta * *cij;
        }
    }

    for (mut ci, ai) in c.rows_mut().into_iter().zip(a.rows()) {
        for (aik, bk) in ai.expr().into_iter().zip(b.rows()) {
            for (cij, bkj) in ci.expr_mut().into_iter().zip(bk) {
                *cij = alpha * (*aik) * (*bkj) + *cij;
            }
        }
    }
}
