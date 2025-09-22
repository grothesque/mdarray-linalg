use mdarray::{DTensor, Tensor, expr, expr::Expression as _, tensor};
use num_complex::Complex64;
use openblas_src as _;

use mdarray_linalg::naive_matmul;
use mdarray_linalg::{MatMul, Side, Triangle, Type, prelude::*};
use mdarray_linalg_blas::Blas;
use mdarray_linalg_faer::Faer;

// Helper functions to create test matrices with known values using mdarray expressions
fn create_test_matrix_f64(
    shape: [usize; 2],
) -> expr::FromFn<(usize, usize), impl FnMut(&[usize]) -> f64> {
    expr::from_fn(shape, move |i| (shape[1] * i[0] + i[1] + 1) as f64)
}

fn create_test_matrix_complex(
    shape: [usize; 2],
) -> expr::FromFn<(usize, usize), impl FnMut(&[usize]) -> Complex64> {
    expr::from_fn(shape, move |i| {
        let val = (shape[1] * i[0] + i[1] + 1) as f64;
        Complex64::new(val, val * 0.5)
    })
}

fn test_matmul_complex_with_scaling_impl(backend: &impl MatMul<Complex64>) {
    let a = create_test_matrix_complex([2, 3]).eval();
    let b = create_test_matrix_complex([3, 2]).eval();
    let scale_factor = Complex64::new(2.0, 1.5);

    let result = backend.matmul(&a, &b).scale(scale_factor).eval();

    let mut expected = Tensor::from_elem([2, 2], Complex64::new(0.0, 0.0));
    naive_matmul(&a, &b, &mut expected);
    let expected = (expr::fill(scale_factor) * &expected).eval();

    assert_eq!(result, expected);
}

fn create_symmetric_matrix_f64(size: usize) -> DTensor<f64, 2> {
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

fn create_upper_triangular_f64(size: usize) -> DTensor<f64, 2> {
    let mut matrix = Tensor::from_elem([size, size], 0.0);
    for i in 0..size {
        for j in i..size {
            matrix[[i, j]] = ((i + 1) * (j + 1)) as f64;
        }
    }
    matrix
}

fn create_lower_triangular_f64(size: usize) -> DTensor<f64, 2> {
    let mut matrix = Tensor::from_elem([size, size], 0.0);
    for i in 0..size {
        for j in 0..=i {
            matrix[[i, j]] = ((i + 1) * (j + 1)) as f64;
        }
    }
    matrix
}

fn create_hermitian_matrix_complex(size: usize) -> DTensor<Complex64, 2> {
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

#[test]
fn matmul_complex_with_scaling() {
    test_matmul_complex_with_scaling_impl(&Blas);
    test_matmul_complex_with_scaling_impl(&Faer);
}

#[test]
#[should_panic]
fn dimension_mismatch_panic() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([4, 2]).eval(); // Wrong inner dimension

    let _result = Blas.matmul(&a, &b).eval();
    let _result = Faer.matmul(&a, &b).eval();
}

#[test]
fn empty_matrix_multiplication() {
    let a = Tensor::from_elem([0, 3], 0.0f64);
    let b = Tensor::from_elem([3, 0], 0.0f64);

    let blas_result = Blas.matmul(&a, &b).eval();
    let faer_result = Faer.matmul(&a, &b).eval();

    assert_eq!(blas_result, faer_result);
}

#[test]
fn single_element_matrices() {
    let a = tensor![[3.]];
    let b = tensor![[4.]];

    let blas_result = Blas.matmul(&a, &b).eval();
    let faer_result = Faer.matmul(&a, &b).eval();

    assert_eq!(blas_result, faer_result);
}

#[test]
fn rectangular_matrices() {
    let a = create_test_matrix_f64([3, 5]).eval();
    let b = create_test_matrix_f64([5, 4]).eval();

    let blas_result = Blas.matmul(&a, &b).eval();
    let faer_result = Faer.matmul(&a, &b).eval();

    assert_eq!(*blas_result.shape(), (3, 4));
    assert_eq!(*faer_result.shape(), (3, 4));

    let mut expected = Tensor::from_elem([3, 4], 0.0);
    naive_matmul(&a, &b, &mut expected);

    assert_eq!(blas_result, expected);
    assert_eq!(faer_result, expected);
    assert_eq!(blas_result, faer_result);
}

#[test]
fn zero_matrices() {
    let a = Tensor::from_elem([2, 3], 0.0f64);
    let b = Tensor::from_elem([3, 2], 5.0f64);

    let blas_result = Blas.matmul(&a, &b).eval();
    let faer_result = Faer.matmul(&a, &b).eval();

    assert_eq!(*blas_result.shape(), (2, 2));
    assert_eq!(*faer_result.shape(), (2, 2));

    assert!(blas_result.iter().all(|&x| x == 0.0));
    assert!(faer_result.iter().all(|&x| x == 0.0));
    assert_eq!(blas_result, faer_result);
}

#[test]
fn chained_operations() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();

    // Test scale then overwrite
    let scale_factor = 2.0;
    let mut c_blas = create_test_matrix_f64([2, 2]).eval();
    let mut c_faer = c_blas.clone();

    Blas.matmul(&a, &b)
        .scale(scale_factor)
        .overwrite(&mut c_blas);
    Faer.matmul(&a, &b)
        .scale(scale_factor)
        .overwrite(&mut c_faer);

    let mut expected = Tensor::from_elem([2, 2], 0.0);
    naive_matmul(&a, &b, &mut expected);
    let expected = (expr::fill(scale_factor) * &expected).eval();

    assert_eq!(c_blas, expected);
    assert_eq!(c_faer, expected);
    assert_eq!(c_blas, c_faer);
}

#[test]
fn backend_defaults() {
    let _blas = Blas::default();
    let _faer = Faer::default();
}

#[test]
fn special_symmetric_left_lower() {
    let a_sym = create_symmetric_matrix_f64(3);
    let b = create_test_matrix_f64([3, 4]).eval();

    let result = Blas
        .matmul(&a_sym, &b)
        .special(Side::Left, Type::Sym, Triangle::Lower);

    assert_eq!(*result.shape(), (3, 4));

    let result_upper = Blas
        .matmul(&a_sym, &b)
        .special(Side::Left, Type::Sym, Triangle::Upper);

    assert_eq!(result, result_upper);
}

#[test]
fn special_triangular_upper_left() {
    let a_tri = create_upper_triangular_f64(3);
    let b = create_test_matrix_f64([3, 4]).eval();

    let result = Blas
        .matmul(&a_tri, &b)
        .special(Side::Left, Type::Tri, Triangle::Upper);

    let result_std = Blas.matmul(&a_tri, &b).eval();

    assert_eq!(result, result_std);
}

#[test]
fn special_triangular_lower_left() {
    let a_tri = create_lower_triangular_f64(3);
    let b = create_test_matrix_f64([3, 4]).eval();

    let result = Blas
        .matmul(&a_tri, &b)
        .special(Side::Left, Type::Tri, Triangle::Lower);

    let result_std = Blas.matmul(&a_tri, &b).eval();
    assert_eq!(result, result_std);
}

#[test]
fn special_triangular_upper_right() {
    let a = create_test_matrix_f64([3, 4]).eval();
    let b_tri = create_upper_triangular_f64(3);

    let result = Blas
        .matmul(&b_tri, &a)
        .special(Side::Left, Type::Tri, Triangle::Upper);

    let result_std = Blas.matmul(&b_tri, &a).eval();
    assert_eq!(result, result_std);
}

#[test]
fn special_hermitian_left_lower() {
    let a_her = create_hermitian_matrix_complex(3);
    let b = create_test_matrix_complex([3, 4]).eval();

    let result = Blas
        .matmul(&a_her, &b)
        .special(Side::Left, Type::Her, Triangle::Lower);

    assert_eq!(*result.shape(), (3, 4));

    let result_upper = Blas.matmul(&a_her, &b).eval();

    for (a, b) in result.iter().zip(result_upper.iter()) {
        assert!((a - b).norm() < 1e-10);
    }
}

#[test]
fn special_with_scaling() {
    let a_sym = create_symmetric_matrix_f64(3);
    let b = create_test_matrix_f64([3, 4]).eval();
    let scale_factor = 2.5;

    let result =
        Blas.matmul(&a_sym, &b)
            .scale(scale_factor)
            .special(Side::Left, Type::Sym, Triangle::Upper);

    let result_std =
        Blas.matmul(&a_sym, &b)
            .scale(scale_factor)
            .special(Side::Left, Type::Sym, Triangle::Upper);

    assert_eq!(result, result_std);
}

#[test]
fn special_single_element() {
    let a = tensor![[5.0]];
    let b = tensor![[2.0]];

    let result = Blas
        .matmul(&a, &b)
        .special(Side::Left, Type::Sym, Triangle::Upper);

    assert_eq!(*result.shape(), (1, 1));
    assert_eq!(result[[0, 0]], 10.0);
}
