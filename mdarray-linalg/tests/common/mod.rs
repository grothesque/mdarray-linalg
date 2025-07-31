// Helper module with common code for integration tests.
// See https://doc.rust-lang.org/rust-by-example/testing/integration_testing.html

use mdarray::expr;

pub fn example_matrix(
    shape: [usize; 2],
) -> expr::FromFn<(usize, usize), impl FnMut(&[usize]) -> f64> {
    expr::from_fn(shape, move |i| (shape[1] * i[0] + i[1] + 1) as f64)
}
