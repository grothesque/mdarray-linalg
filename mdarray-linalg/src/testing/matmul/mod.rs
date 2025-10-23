use mdarray::{DTensor, Tensor, expr, expr::Expression as _};
use num_complex::Complex64;

use super::common::*;
use crate::prelude::*;

pub fn create_test_matrix_f64(
    shape: [usize; 2],
) -> expr::FromFn<(usize, usize), impl FnMut(&[usize]) -> f64> {
    expr::from_fn(shape, move |i| (shape[1] * i[0] + i[1] + 1) as f64)
}

pub fn create_test_matrix_complex(
    shape: [usize; 2],
) -> expr::FromFn<(usize, usize), impl FnMut(&[usize]) -> Complex64> {
    expr::from_fn(shape, move |i| {
        let val = (shape[1] * i[0] + i[1] + 1) as f64;
        Complex64::new(val, val * 0.5)
    })
}

pub fn test_matmul_complex_with_scaling_impl(backend: &impl MatMul<Complex64>) {
    let a = create_test_matrix_complex([2, 3]).eval();
    let b = create_test_matrix_complex([3, 2]).eval();
    let scale_factor = Complex64::new(2.0, 1.5);

    let result = backend.matmul(&a, &b).scale(scale_factor).eval();

    let expected = naive_matmul(&a, &b);
    let expected = (expr::fill(scale_factor) * &expected).eval();

    assert_eq!(result, expected);
}

pub fn create_symmetric_matrix_f64(size: usize) -> DTensor<f64, 2> {
    let mut matrix = Tensor::from_elem([size, size], 0.0);
    for i in 0..size {
        for j in 0..size {
            let value = ((i + 1) * (j + 1)) as f64;
            matrix[[i, j]] = value;
            matrix[[j, i]] = value; // Assurer la symétrie
        }
    }
    matrix
}

pub fn create_upper_triangular_f64(size: usize) -> DTensor<f64, 2> {
    let mut matrix = Tensor::from_elem([size, size], 0.0);
    for i in 0..size {
        for j in i..size {
            matrix[[i, j]] = ((i + 1) * (j + 1)) as f64;
        }
    }
    matrix
}

pub fn create_lower_triangular_f64(size: usize) -> DTensor<f64, 2> {
    let mut matrix = Tensor::from_elem([size, size], 0.0);
    for i in 0..size {
        for j in 0..=i {
            matrix[[i, j]] = ((i + 1) * (j + 1)) as f64;
        }
    }
    matrix
}

pub fn create_hermitian_matrix_complex(size: usize) -> DTensor<Complex64, 2> {
    let mut matrix = Tensor::from_elem([size, size], Complex64::new(0.0, 0.0));
    for i in 0..size {
        for j in 0..size {
            if i == j {
                matrix[[i, j]] = Complex64::new((i + 1) as f64, 0.0);
            } else if i < j {
                let real = ((i + 1) * (j + 1)) as f64;
                let imag = (i + j + 1) as f64;
                matrix[[i, j]] = Complex64::new(real, imag);
                matrix[[j, i]] = Complex64::new(real, -imag);
            }
        }
    }
    matrix
}
