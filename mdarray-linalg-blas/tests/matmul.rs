use mdarray::expr::Expression;
use mdarray::{Tensor, tensor};

use mdarray_linalg::prelude::*;

use mdarray_linalg_blas::Blas;
use mdarray_linalg_blas::gemm;

use mdarray_linalg_tests::common::*;
use mdarray_linalg_tests::matmul::*;

use mdarray_linalg::matmul::{Side, Triangle, Type};

#[test]
fn matmul_complex_with_scaling() {
    test_matmul_complex_with_scaling_impl(&Blas);
}

#[test]
#[should_panic]
fn dimension_mismatch_panic() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([4, 2]).eval(); // Wrong inner dimension

    let _result = Blas.matmul(&a, &b).eval();
}

#[test]
fn empty_matrix_multiplication() {
    let a = Tensor::from_elem([0, 3], 0.0f64);
    let b = Tensor::from_elem([3, 0], 0.0f64);

    let result = Blas.matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn single_element_matrices() {
    let a = tensor![[3.]];
    let b = tensor![[4.]];

    let result = Blas.matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn rectangular_matrices() {
    let a = create_test_matrix_f64([3, 5]).eval();
    let b = create_test_matrix_f64([5, 4]).eval();

    let result = Blas.matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn zero_matrices() {
    let a = Tensor::from_elem([2, 3], 0.0f64);
    let b = Tensor::from_elem([3, 2], 5.0f64);

    let result = Blas.matmul(&a, &b).eval();

    assert_eq!(*result.shape(), (2, 2));

    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn chained_operations() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();

    // Test scale then overwrite
    let scale_factor = 2.0;
    let mut c = create_test_matrix_f64([2, 2]).eval();

    Blas.matmul(&a, &b).scale(scale_factor).overwrite(&mut c);

    let expected = naive_matmul(&a, &b);

    for (cij, eij) in std::iter::zip(c, expected) {
        assert_eq!(cij, 2. * eij);
    }
}

#[test]
fn backend_defaults() {
    let _bd = Blas::default();
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

#[test]
fn test_gemm() {
    let a = example_matrix([2, 3]).eval();
    let b = example_matrix([3, 4]).eval();
    let c_expr = || example_matrix([2, 4]);
    let mut c = c_expr().eval();
    let ab_plus_c = {
        let mut ab = naive_matmul(&a, &b);
        ab + &c
    };

    // Test vanilla gemm with all matrices in column major order and Dense mapping.
    gemm(1.0, &a, &b, 1.0, &mut c);
    assert!(c == ab_plus_c);

    ////////////////
    // Test all combinations of row- and column major for the three matrices A, B, and C.  The
    // layout is always ‘Strided’ here, so we never test calling gemm with mixed layout, but we
    // know anyway statically that this must work.

    let a_cmajor = a.transpose().to_tensor();
    let a_cmajor = a_cmajor.transpose();
    let b_cmajor = b.transpose().to_tensor();
    let b_cmajor = b_cmajor.transpose();

    // Convert to a ‘Strided’ layout (still row major) so that ‘a’ has the same type as ‘a_cmajor’.
    let a = a.remap();
    let b = b.remap();
    let mut c = c.remap_mut();

    let mut c_cmajor = c.transpose().to_tensor();
    let mut c_cmajor = c_cmajor.transpose_mut();

    for a in [&a, &a_cmajor] {
        for b in [&b, &b_cmajor] {
            for c in [&mut c, &mut c_cmajor] {
                c.assign(c_expr());
                gemm(1.0, a, &b, 1.0, c);
                assert!(*c == ab_plus_c);
            }
        }
    }
}
