// Helper module with common code for integration tests.
// See https://doc.rust-lang.org/rust-by-example/testing/integration_testing.html

use mdarray::{DSlice, expr, expr::Expression as _};

pub fn example_matrix(
    shape: [usize; 2],
) -> expr::FromFn<(usize, usize), impl FnMut(&[usize]) -> f64> {
    expr::from_fn(shape, move |i| (shape[1] * i[0] + i[1] + 1) as f64)
}

pub fn naive_matmul(a: &DSlice<f64, 2>, b: &DSlice<f64, 2>, c: &mut DSlice<f64, 2>) {
    for (mut ci, ai) in c.rows_mut().zip(a.rows()) {
        for (aik, bk) in ai.expr().zip(b.rows()) {
            for (cij, bkj) in ci.expr_mut().zip(bk) {
                *cij = aik.mul_add(*bkj, *cij);
            }
        }
    }
}
