use mdarray::{Array, Shape, Strided, StridedMapping, View, array, expr::Expression, tensor};
use mdarray_linalg::{
    matmul::{Side, Triangle, Type},
    prelude::*,
    testing::{common::*, matmul::*},
};
use mdarray_linalg_nalgebra::Nalgebra;

#[test]
fn matmul_complex_with_scaling() {
    test_matmul_complex_with_scaling_impl(&Nalgebra::default());
}

#[test]
#[should_panic]
fn dimension_mismatch_panic() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([4, 2]).eval();

    let _ = Nalgebra::default().matmul(&a, &b).eval();
}

#[test]
fn empty_matrix_multiplication() {
    let a = Array::from_elem([0, 3], 0.0f64);
    let b = Array::from_elem([3, 0], 0.0f64);

    let result = Nalgebra::default().matmul(&a, &b).eval();
    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn single_element_matrices() {
    let a = tensor![[3.0]];
    let b = tensor![[4.0]];

    let result = Nalgebra::default().matmul(&a, &b).eval();
    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn rectangular_matrices() {
    let a = create_test_matrix_f64([3, 5]).eval();
    let b = create_test_matrix_f64([5, 4]).eval();

    let result = Nalgebra::default().matmul(&a, &b).eval();
    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn zero_matrices() {
    let a = Array::from_elem([2, 3], 0.0f64);
    let b = Array::from_elem([3, 2], 5.0f64);

    let result = Nalgebra::default().matmul(&a, &b).eval();

    assert_eq!(*result.shape(), (2, 2));
    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn chained_operations() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();
    let mut c = create_test_matrix_f64([2, 2]).eval();

    Nalgebra::default().matmul(&a, &b).scale(2.0).write(&mut c);

    let expected = naive_matmul(&a, &b);
    for (cij, eij) in std::iter::zip(c, expected) {
        assert_eq!(cij, 2.0 * eij);
    }
}

#[test]
fn backend_defaults() {
    let _ = Nalgebra::default();
}

#[test]
fn special_symmetric_left_lower() {
    let a_sym = create_symmetric_matrix_f64(3);
    let b = create_test_matrix_f64([3, 4]).eval();

    let result = Nalgebra::default()
        .matmul(&a_sym, &b)
        .special(Side::Left, Type::Sym, Triangle::Lower);
    let result_upper = Nalgebra::default()
        .matmul(&a_sym, &b)
        .special(Side::Left, Type::Sym, Triangle::Upper);

    assert_eq!(*result.shape(), (3, 4));
    assert_eq!(result, result_upper);
}

#[test]
fn special_triangular_upper_left() {
    let a_tri = create_upper_triangular_f64(3);
    let b = create_test_matrix_f64([3, 4]).eval();

    let result = Nalgebra::default()
        .matmul(&a_tri, &b)
        .special(Side::Left, Type::Tri, Triangle::Upper);
    let result_std = Nalgebra::default().matmul(&a_tri, &b).eval();

    assert_eq!(result, result_std);
}

#[test]
fn special_triangular_lower_left() {
    let a_tri = create_lower_triangular_f64(3);
    let b = create_test_matrix_f64([3, 4]).eval();

    let result = Nalgebra::default()
        .matmul(&a_tri, &b)
        .special(Side::Left, Type::Tri, Triangle::Lower);
    let result_std = Nalgebra::default().matmul(&a_tri, &b).eval();

    assert_eq!(result, result_std);
}

#[test]
fn special_hermitian_left_lower() {
    let a_her = create_hermitian_matrix_complex(3);
    let b = create_test_matrix_complex([3, 4]).eval();

    let result = Nalgebra::default()
        .matmul(&a_her, &b)
        .special(Side::Left, Type::Her, Triangle::Lower);
    let result_std = Nalgebra::default().matmul(&a_her, &b).eval();

    assert_eq!(*result.shape(), (3, 4));
    for (lhs, rhs) in result.iter().zip(result_std.iter()) {
        assert!((lhs - rhs).norm() < 1e-10);
    }
}

#[test]
fn special_with_scaling() {
    let a_sym = create_symmetric_matrix_f64(3);
    let b = create_test_matrix_f64([3, 4]).eval();

    let result = Nalgebra::default()
        .matmul(&a_sym, &b)
        .scale(2.5)
        .special(Side::Left, Type::Sym, Triangle::Upper);
    let result_std = Nalgebra::default()
        .matmul(&a_sym, &b)
        .scale(2.5)
        .special(Side::Left, Type::Sym, Triangle::Upper);

    assert_eq!(result, result_std);
}

#[test]
fn special_single_element() {
    let a = tensor![[5.0]];
    let b = tensor![[2.0]];

    let result = Nalgebra::default()
        .matmul(&a, &b)
        .special(Side::Left, Type::Sym, Triangle::Upper);

    assert_eq!(*result.shape(), (1, 1));
    assert_eq!(result[[0, 0]], 10.0);
}

#[test]
pub fn non_contiguous_along_both_axis() {
    let bufa: Vec<f64> = vec![1., 0., 3., 0., 2., 0., 4.];
    let av: View<'_, _, (usize, usize), Strided> = unsafe {
        let sh = <(usize, usize) as Shape>::from_dims(&[2, 2]);
        let mapping = StridedMapping::new(sh, &[2, 4]);
        View::new_unchecked(bufa.as_ptr(), mapping)
    };

    let bufb: Vec<f64> = vec![5., 0., 7., 0., 6., 0., 8.];
    let bv: View<'_, _, (usize, usize), Strided> = unsafe {
        let sh = <(usize, usize) as Shape>::from_dims(&[2, 2]);
        let mapping = StridedMapping::new(sh, &[2, 4]);
        View::new_unchecked(bufb.as_ptr(), mapping)
    };

    let c = Nalgebra::default().matmul(&av, &bv).eval();
    assert_eq!(c, array![[19., 22.], [43., 50.]]);
}

#[test]
fn macro_matmul() {
    let a = create_test_matrix_f64([3, 5]).eval();
    let b = create_test_matrix_f64([5, 4]).eval();
    let c = create_test_matrix_f64([4, 2]).eval();
    let d = create_test_matrix_f64([2, 6]).eval();

    let cd = Nalgebra::default().matmul(&c, &d).eval();
    let bcd = Nalgebra::default().matmul(&b, &cd).eval();
    let result = Nalgebra::default().matmul(&a, &bcd).eval();
    let expected = naive_matmul(&a, &naive_matmul(&b, &naive_matmul(&c, &d)));

    assert_eq!(result, expected);
}
