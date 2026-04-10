use mdarray::{Array, DArray, array, expr, expr::Expression as _};
use num_complex::Complex64;

use super::common::*;
use crate::{matmul::Contract, prelude::*};

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

pub fn test_matmul_complex_with_scaling_impl(backend: &impl Contract<Complex64>) {
    let a = create_test_matrix_complex([2, 3]).eval();
    let b = create_test_matrix_complex([3, 2]).eval();
    let scale_factor = Complex64::new(2.0, 1.5);

    let result = backend.matmul(&a, &b).scale(scale_factor).eval();

    let expected = naive_matmul(&a, &b);
    let expected = (expr::fill(scale_factor) * &expected).eval();

    assert_eq!(result, expected);
}

pub fn create_symmetric_matrix_f64(size: usize) -> DArray<f64, 2> {
    let mut matrix = Array::from_elem([size, size], 0.0);
    for i in 0..size {
        for j in 0..size {
            let value = ((i + 1) * (j + 1)) as f64;
            matrix[[i, j]] = value;
            matrix[[j, i]] = value; // Assurer la symétrie
        }
    }
    matrix
}

pub fn create_upper_triangular_f64(size: usize) -> DArray<f64, 2> {
    let mut matrix = Array::from_elem([size, size], 0.0);
    for i in 0..size {
        for j in i..size {
            matrix[[i, j]] = ((i + 1) * (j + 1)) as f64;
        }
    }
    matrix
}

pub fn create_lower_triangular_f64(size: usize) -> DArray<f64, 2> {
    let mut matrix = Array::from_elem([size, size], 0.0);
    for i in 0..size {
        for j in 0..=i {
            matrix[[i, j]] = ((i + 1) * (j + 1)) as f64;
        }
    }
    matrix
}

pub fn create_hermitian_matrix_complex(size: usize) -> DArray<Complex64, 2> {
    let mut matrix = Array::from_elem([size, size], Complex64::new(0.0, 0.0));
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

pub fn contract_einsum_matrix_multiplication_impl(backend: &impl Contract<f64>) {
    // ij,jk->ik
    let a = array![[1., 2.], [3., 4.]].into_dyn();
    let b = array![[5., 6.], [7., 8.]].into_dyn();
    let expected = array![[19., 22.], [43., 50.]].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1], &[1, 2], &[0, 2]).eval();
    assert_eq!(result, expected);
}

pub fn contract_einsum_full_contraction_impl(backend: &impl Contract<f64>) {
    // ij,ij->
    let a = array![[1., 2.], [3., 4.]].into_dyn();
    let b = array![[5., 6.], [7., 8.]].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1], &[0, 1], &[]).eval();
    assert_eq!(result.into_scalar(), 70.);
}

pub fn contract_einsum_outer_product_impl(backend: &impl Contract<f64>) {
    // i,j->ij
    let a = array![1., 2.].into_dyn();
    let b = array![3., 4.].into_dyn();
    let expected = array![[3., 4.], [6., 8.]].into_dyn();
    let result = backend.contract(&a, &b, &[0], &[1], &[0, 1]).eval();
    assert_eq!(result, expected);
}

pub fn contract_einsum_trace_diagonal_impl(backend: &impl Contract<f64>) {
    // ijj,ij-> = 32
    let a = array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn();
    let b = array![[0., 1.], [2., 3.]].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1, 1], &[0, 1], &[]).eval();
    assert_eq!(result.into_scalar(), 32.);
}

pub fn contract_einsum_index_relabelling_impl(backend: &impl Contract<f64>) {
    // Same as above with permuted label assignments: result must be identical.
    let a = array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn();
    let b = array![[0., 1.], [2., 3.]].into_dyn();
    let result = backend.contract(&a, &b, &[1, 0, 0], &[1, 0], &[]).eval();
    assert_eq!(result.into_scalar(), 32.);
}

pub fn contract_einsum_partial_trace_then_contract_impl(backend: &impl Contract<f64>) {
    // ijj,ik->k  expected = [22, 36]
    let a = array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn();
    let b = array![[0., 1.], [2., 3.]].into_dyn();
    let expected = array![22., 36.].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1, 1], &[0, 2], &[2]).eval();
    assert_eq!(result, expected);
}

pub fn contract_einsum_cross_diagonal_impl(backend: &impl Contract<f64>) {
    // ijj,iij-> = 76
    let a = array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn();
    let b = array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1, 1], &[0, 0, 1], &[]).eval();
    assert_eq!(result.into_scalar(), 76.);
}

pub fn contract_einsum_vector_result_impl(backend: &impl Contract<f64>) {
    // ijjj,ijkl->l
    let a = mdarray::DArray::<f64, 4>::from_fn([2, 2, 2, 2], |i| {
        (i[0] * 8 + i[1] * 4 + i[2] * 2 + i[3]) as f64
    })
    .into_dyn();
    let b = mdarray::DArray::<f64, 4>::from_fn([2, 2, 2, 2], |i| {
        (i[0] * 8 + i[1] * 4 + i[2] * 2 + i[3]) as f64
    })
    .into_dyn();

    let mut expected = [0f64; 2];
    for l in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    expected[l] +=
                        (i * 8 + j * 4 + j * 2 + j) as f64 * (i * 8 + j * 4 + k * 2 + l) as f64;
                }
            }
        }
    }

    let result = backend
        .contract(&a, &b, &[0, 1, 1, 1], &[0, 1, 2, 3], &[3])
        .eval();
    assert_eq!(result, array![expected[0], expected[1]].into_dyn());
}
