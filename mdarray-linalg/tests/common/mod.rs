// Helper module with common code for integration tests.
// See https://doc.rust-lang.org/rust-by-example/testing/integration_testing.html

use mdarray::{DTensor, expr};
use mdarray_linalg::naive_matmul;

use rand::Rng;

pub fn example_matrix(
    shape: [usize; 2],
) -> expr::FromFn<(usize, usize), impl FnMut(&[usize]) -> f64> {
    expr::from_fn(shape, move |i| (shape[1] * i[0] + i[1] + 1) as f64)
}

#[macro_export]
macro_rules! assert_matrix_eq {
    ($a:expr, $b:expr) => {
        assert_matrix_eq!($a, $b, 1e-8f64)
    };
    ($a:expr, $b:expr, $epsilon:expr) => {
        assert_eq!($a.shape(), $b.shape(), "Matrix shapes don't match");
        let shape = $a.shape();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                assert_relative_eq!($a[[i, j]], $b[[i, j]], epsilon = $epsilon);
            }
        }
    };
}

#[macro_export]
macro_rules! assert_complex_matrix_eq {
    ($a:expr, $b:expr) => {
        assert_complex_matrix_eq!($a, $b, 1e-8)
    };
    ($a:expr, $b:expr, $epsilon:expr) => {
        assert_eq!($a.shape(), $b.shape(), "Matrix shapes don't match");
        let shape = $a.shape();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                assert_relative_eq!(
                    Complex::norm($a[[i, j]]),
                    Complex::norm($b[[i, j]]),
                    epsilon = $epsilon
                );
            }
        }
    };
}

/// Generate a random matrix of size m x n
pub fn random_matrix(m: usize, n: usize) -> DTensor<f64, 2> {
    let mut rng = rand::rng();
    DTensor::<f64, 2>::from_fn([m, n], |_| rng.random_range(0.0..1.0))
}

/// Generate a rank-k matrix by multiplying m×k and k×n matrices
pub fn rank_k_matrix(m: usize, n: usize, k: usize) -> DTensor<f64, 2> {
    assert!(k <= n.min(m));

    let a = random_matrix(m, k);
    let b = random_matrix(k, n);
    let mut result = DTensor::<f64, 2>::zeros([m, n]);

    naive_matmul(&a, &b, &mut result);
    result
}
