use mdarray::{Array, Shape, Strided, StridedMapping, View, array, expr::Expression, tensor};
use mdarray_linalg::{
    prelude::*,
    testing::{common::*, contract::*},
};
use mdarray_linalg_faer::Faer;

#[test]
fn matmul_complex_with_scaling() {
    matmul_complex_with_scaling_impl(&Faer::default());
}

#[test]
#[should_panic]
fn dimension_mismatch_panic() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([4, 2]).eval(); // Wrong inner dimension

    let _result = Faer::default().matmul(&a, &b).eval();
}

#[test]
fn empty_matrix_multiplication() {
    let a = Array::from_elem([0, 3], 0.0f64);
    let b = Array::from_elem([3, 0], 0.0f64);

    let result = Faer::default().matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn single_element_matrices() {
    let a = tensor![[3.]];
    let b = tensor![[4.]];

    let result = Faer::default().matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn rectangular_matrices() {
    let a = create_test_matrix_f64([3, 5]).eval();
    let b = create_test_matrix_f64([5, 4]).eval();

    let result = Faer::default().matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn zero_matrices() {
    let a = Array::from_elem([2, 3], 0.0f64);
    let b = Array::from_elem([3, 2], 5.0f64);

    let result = Faer::default().matmul(&a, &b).eval();

    assert_eq!(*result.shape(), (2, 2));

    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn chained_operations() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();

    // Test scale then write
    let scale_factor = 2.0;
    let mut c = create_test_matrix_f64([2, 2]).eval();

    Faer::default()
        .matmul(&a, &b)
        .scale(scale_factor)
        .write(&mut c);

    let expected = naive_matmul(&a, &b);

    for (cij, eij) in std::iter::zip(c, expected) {
        assert_eq!(cij, 2. * eij);
    }
}

#[test]
fn backend_defaults() {
    let _bd = Faer::default();
}

#[test]
fn matmul_builder_methods() {
    matmul_builder_methods_impl(&Faer::default());
}

#[test]
fn contract_builder_methods() {
    contract_builder_methods_impl(&Faer::default());
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

    let c = Faer::default().matmul(&av, &bv).eval();

    assert_eq!(c, array![[19., 22.], [43., 50.]]);
}

#[test]
fn macro_matmul() {
    let a = create_test_matrix_f64([3, 5]).eval();
    let b = create_test_matrix_f64([5, 4]).eval();
    let c = create_test_matrix_f64([4, 2]).eval();
    let d = create_test_matrix_f64([2, 6]).eval();

    let cd = Faer::default().matmul(&c, &d).eval();
    let bcd = Faer::default().matmul(&b, &cd).eval();
    let result = Faer::default().matmul(&a, &bcd).eval();

    let expected = naive_matmul(&a, &naive_matmul(&b, &naive_matmul(&c, &d)));

    assert_eq!(result, expected);
}

// --- Structured contractions ---

#[test]
fn contract_all() {
    contract_all_impl(&Faer::default());
}

#[test]
fn contract_n_2_should_match_all_axes() {
    contract_n_2_should_match_all_axes_impl(&Faer::default());
}

#[test]
fn contract_pairs_matrix_multiplication() {
    contract_pairs_matrix_multiplication_impl(&Faer::default());
}

#[test]
fn contract_n_0_should_outer_product() {
    contract_n_0_should_outer_product_impl(&Faer::default());
}

#[test]
fn contract_scalar_inputs_should_multiply() {
    contract_scalar_inputs_should_multiply_impl(&Faer::default());
}

#[test]
fn contract_increase_deep() {
    contract_increase_deep_impl(&Faer::default());
}

#[test]
fn contract_vector_dot_product() {
    contract_vector_dot_product_impl(&Faer::default());
}

#[test]
fn contract_mismatched_dimensions_should_panic() {
    contract_mismatched_dimensions_should_panic_impl(&Faer::default());
}

#[test]
fn contract_outer_should_match_manual_kronecker() {
    contract_outer_should_match_manual_kronecker_impl(&Faer::default());
}

// --- Einsum-style contractions ---

#[test]
fn contract_einsum_matrix_multiplication() {
    contract_einsum_matrix_multiplication_impl(&Faer::default())
}

#[test]
fn contract_einsum_full_contraction() {
    contract_einsum_full_contraction_impl(&Faer::default())
}

#[test]
fn contract_einsum_output_permutation() {
    contract_einsum_output_permutation_impl(&Faer::default())
}

#[test]
fn contract_einsum_outer_product() {
    contract_einsum_outer_product_impl(&Faer::default())
}

#[test]
fn contract_einsum_trace_diagonal() {
    contract_einsum_trace_diagonal_impl(&Faer::default())
}

#[test]
fn contract_einsum_index_relabelling() {
    contract_einsum_index_relabelling_impl(&Faer::default())
}

#[test]
fn contract_einsum_partial_trace_then_contract() {
    contract_einsum_partial_trace_then_contract_impl(&Faer::default())
}

#[test]
fn contract_einsum_cross_diagonal() {
    contract_einsum_cross_diagonal_impl(&Faer::default())
}

#[test]
fn contract_einsum_vector_result() {
    contract_einsum_vector_result_impl(&Faer::default())
}
