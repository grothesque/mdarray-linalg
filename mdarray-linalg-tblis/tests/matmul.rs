use mdarray::{Array, Shape, Strided, StridedMapping, View, array, expr::Expression, tensor};
use mdarray_linalg::{
    prelude::*,
    testing::{common::*, matmul::*},
};
use mdarray_linalg_tblis::{Tblis, matmul};

#[test]
fn matmul_complex_with_scaling() {
    test_matmul_complex_with_scaling_impl(&Tblis);
}

#[test]
#[should_panic]
fn dimension_mismatch_panic() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([4, 2]).eval();

    let _result = Tblis.matmul(&a, &b).eval();
}

#[test]
fn empty_matrix_multiplication() {
    let a = Array::from_elem([0, 3], 0.0f64);
    let b = Array::from_elem([3, 0], 0.0f64);

    let result = Tblis.matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn single_element_matrices() {
    let a = tensor![[3.]];
    let b = tensor![[4.]];

    let result = Tblis.matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn rectangular_matrices() {
    let a = create_test_matrix_f64([3, 5]).eval();
    let b = create_test_matrix_f64([5, 4]).eval();

    let result = Tblis.matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn zero_matrices() {
    let a = Array::from_elem([2, 3], 0.0f64);
    let b = Array::from_elem([3, 2], 5.0f64);

    let result = Tblis.matmul(&a, &b).eval();

    assert_eq!(*result.shape(), (2, 2));
    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn chained_operations() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();
    let scale_factor = 2.0;
    let mut c = create_test_matrix_f64([2, 2]).eval();

    Tblis.matmul(&a, &b).scale(scale_factor).write(&mut c);

    let expected = naive_matmul(&a, &b);

    for (cij, eij) in std::iter::zip(c, expected) {
        assert_eq!(cij, 2. * eij);
    }
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

    let c = Tblis.matmul(&av, &bv).eval();

    assert_eq!(c, array![[19., 22.], [43., 50.]]);
}

#[test]
fn macro_matmul() {
    let a = create_test_matrix_f64([3, 5]).eval();
    let b = create_test_matrix_f64([5, 4]).eval();
    let c = create_test_matrix_f64([4, 2]).eval();
    let d = create_test_matrix_f64([2, 6]).eval();

    let result = matmul!(&a, &b, &c, &d);

    let expected = naive_matmul(&a, &naive_matmul(&b, &naive_matmul(&c, &d)));

    assert_eq!(result, expected);
}
