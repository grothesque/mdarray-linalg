use mdarray::{Tensor, expr, expr::Expression as _, tensor};
use num_complex::Complex64;
use openblas_src as _;

use mdarray_linalg::naive_matmul;
use mdarray_linalg::{MatMul, prelude::*};
use mdarray_linalg_blas::Blas;
use mdarray_linalg_faer::matmul::Faer;

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

fn test_basic_matmul_f64_impl(backend: &impl MatMul<f64>) {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();

    let result = backend.matmul(&a, &b).eval();

    assert_eq!(*result.shape(), (2, 2));

    let mut expected = Tensor::from_elem([2, 2], 0.0);
    naive_matmul(&a, &b, &mut expected);
    assert_eq!(result, expected);
}

fn test_matmul_with_scaling_f64_impl(backend: &impl MatMul<f64>) {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();
    let scale_factor = 2.5;

    let result = backend.matmul(&a, &b).scale(scale_factor).eval();

    let mut expected = Tensor::from_elem([2, 2], 0.0);
    naive_matmul(&a, &b, &mut expected);
    let expected = (expr::fill(scale_factor) * &expected).eval();

    assert_eq!(result, expected);
}

fn test_add_to_f64_impl(backend: &impl MatMul<f64>) {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();
    let mut c = create_test_matrix_f64([2, 2]).eval();
    let c_original = c.clone();

    backend.matmul(&a, &b).add_to(&mut c);

    let mut ab = Tensor::from_elem([2, 2], 0.0);
    naive_matmul(&a, &b, &mut ab);
    let expected = ab + c_original;

    assert_eq!(c, expected);
}

fn test_add_to_scaled_f64_impl(backend: &impl MatMul<f64>) -> Result<(), &'static str> {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();
    let mut c = create_test_matrix_f64([2, 2]).eval();
    let c_original = c.clone();
    let scale_factor = 3.0;

    // This might panic for some backends like Faer
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.matmul(&a, &b).add_to_scaled(&mut c, scale_factor);
    }))
    .map_err(|_| "Backend doesn't support add_to_scaled")?;

    let mut ab = Tensor::from_elem([2, 2], 0.0);
    naive_matmul(&a, &b, &mut ab);
    let expected = ab + (expr::fill(scale_factor) * &c_original).eval();

    assert_eq!(c, expected);
    Ok(())
}

fn test_overwrite_f64_impl(backend: &impl MatMul<f64>) {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();
    let mut c = create_test_matrix_f64([2, 2]).eval();

    backend.matmul(&a, &b).overwrite(&mut c);

    let mut expected = Tensor::from_elem([2, 2], 0.0);
    naive_matmul(&a, &b, &mut expected);

    assert_eq!(c, expected);
}

fn test_basic_matmul_complex_impl(backend: &impl MatMul<Complex64>) {
    let a = create_test_matrix_complex([2, 3]).eval();
    let b = create_test_matrix_complex([3, 2]).eval();

    let result = backend.matmul(&a, &b).eval();

    assert_eq!(*result.shape(), (2, 2));

    let mut expected = Tensor::from_elem([2, 2], Complex64::new(0.0, 0.0));
    naive_matmul(&a, &b, &mut expected);
    assert_eq!(result, expected);
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

// Atomic Tests - Each tests all backends
#[test]
fn test_basic_matmul_f64() {
    test_basic_matmul_f64_impl(&Blas);
    test_basic_matmul_f64_impl(&Faer);
}

#[test]
fn test_matmul_with_scaling_f64() {
    test_matmul_with_scaling_f64_impl(&Blas);
    test_matmul_with_scaling_f64_impl(&Faer);
}

#[test]
fn test_add_to_f64() {
    test_add_to_f64_impl(&Blas);
    test_add_to_f64_impl(&Faer);
}

#[test]
fn test_add_to_scaled_f64() {
    // BLAS should work
    assert!(test_add_to_scaled_f64_impl(&Blas).is_ok());

    // Faer is expected to fail
    assert!(test_add_to_scaled_f64_impl(&Faer).is_err());
}

#[test]
fn test_overwrite_f64() {
    test_overwrite_f64_impl(&Blas);
    test_overwrite_f64_impl(&Faer);
}

#[test]
fn test_basic_matmul_complex() {
    test_basic_matmul_complex_impl(&Blas);
    test_basic_matmul_complex_impl(&Faer);
}

#[test]
fn test_matmul_complex_with_scaling() {
    test_matmul_complex_with_scaling_impl(&Blas);
    test_matmul_complex_with_scaling_impl(&Faer);
}

#[test]
#[should_panic]
fn test_dimension_mismatch_panic() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([4, 2]).eval(); // Wrong inner dimension

    let _result = Blas.matmul(&a, &b).eval();
    let _result = Faer.matmul(&a, &b).eval();
}

// Backend consistency tests

#[test]
fn test_empty_matrix_multiplication() {
    let a = Tensor::from_elem([0, 3], 0.0f64);
    let b = Tensor::from_elem([3, 0], 0.0f64);

    let blas_result = Blas.matmul(&a, &b).eval();
    let faer_result = Faer.matmul(&a, &b).eval();

    assert_eq!(*blas_result.shape(), (0, 0));
    assert_eq!(*faer_result.shape(), (0, 0));
    assert_eq!(blas_result, faer_result);
}

#[test]
fn test_single_element_matrices() {
    let a = tensor![[3.]];
    let b = tensor![[4.]];

    let blas_result = Blas.matmul(&a, &b).eval();
    let faer_result = Faer.matmul(&a, &b).eval();

    assert_eq!(*blas_result.shape(), (1, 1));
    assert_eq!(*faer_result.shape(), (1, 1));
    assert_eq!(blas_result[[0, 0]], 12.0);
    assert_eq!(faer_result[[0, 0]], 12.0);
    assert_eq!(blas_result, faer_result);
}

#[test]
fn test_large_matrix_multiplication() {
    let size = 100;
    let a = Tensor::from_elem([size, size], 1.0f64);
    let b = Tensor::from_elem([size, size], 2.0f64);

    let blas_result = Blas.matmul(&a, &b).eval();
    let faer_result = Faer.matmul(&a, &b).eval();

    assert_eq!(*blas_result.shape(), (size, size));
    assert_eq!(*faer_result.shape(), (size, size));

    // Each element should be size * 1.0 * 2.0 = size * 2.0
    assert_eq!(blas_result[[0, 0]], (size as f64) * 2.0);
    assert_eq!(faer_result[[0, 0]], (size as f64) * 2.0);
    assert_eq!(blas_result, faer_result);
}

#[test]
fn test_rectangular_matrices() {
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
fn test_zero_matrices() {
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
fn test_backends_consistency_comprehensive() {
    let test_cases = vec![
        ([2, 3], [3, 4]),
        ([1, 5], [5, 1]),
        ([4, 4], [4, 4]),
        ([10, 20], [20, 15]),
    ];

    for (a_shape, b_shape) in test_cases {
        let a = create_test_matrix_f64(a_shape).eval();
        let b = create_test_matrix_f64(b_shape).eval();

        let blas_result = Blas.matmul(&a, &b).eval();
        let faer_result = Faer.matmul(&a, &b).eval();

        assert_eq!(
            blas_result, faer_result,
            "Backend results differ for shapes {:?} x {:?}",
            a_shape, b_shape
        );
    }
}

#[test]
fn test_chained_operations() {
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
fn test_backend_defaults() {
    let _blas = Blas::default();
    let _faer = Faer::default();
}
