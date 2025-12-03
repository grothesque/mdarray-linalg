use mdarray::{Dim, Layout, Slice};
use num_complex::ComplexFloat;

/// Textbook implementation of matrix multiplication, useful for
/// debugging and simple tests without relying on a external backend
pub fn naive_matmul<
    T: ComplexFloat,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    c: &mut Slice<T, (D0, D2), Lc>,
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
