use mdarray::{Array, expr::Expression as _};
use mdarray_linalg::{Contract, Naive, prelude::*, testing::contract::*};

// --- Structured contractions ---

#[test]
fn contract_all() {
    contract_all_impl(&Naive);
}

#[test]
fn contract_n_2_should_match_all_axes() {
    contract_n_2_should_match_all_axes_impl(&Naive);
}

#[test]
fn contract_pairs_matrix_multiplication() {
    contract_pairs_matrix_multiplication_impl(&Naive);
}

#[test]
fn contract_n_0_should_outer_product() {
    contract_n_0_should_outer_product_impl(&Naive);
}

#[test]
fn contract_scalar_inputs_should_multiply() {
    contract_scalar_inputs_should_multiply_impl(&Naive);
}

#[test]
fn contract_increase_deep() {
    contract_increase_deep_impl(&Naive);
}

#[test]
fn contract_vector_dot_product() {
    contract_vector_dot_product_impl(&Naive);
}

#[test]
fn contract_all_invalid_shapes_should_panic() {
    contract_all_invalid_shapes_should_panic_impl(&Naive);
}

#[test]
fn contract_outer_should_match_manual_kronecker() {
    contract_outer_should_match_manual_kronecker_impl(&Naive);
}

// --- Builder APIs ---

#[test]
fn matmul_builder_methods() {
    matmul_builder_methods_impl(&Naive);
}

#[test]
fn contract_builder_methods() {
    contract_builder_methods_impl(&Naive);
}

// --- Einsum-style contractions ---

#[test]
fn contract_einsum_zero_length_free_axis() {
    // Regression from tensoratu/examples/dmrg_tree.rs: safe Rust must not SIGSEGV.
    // C[i, j] = sum_k A[i, k] * B[j, k], with an empty free axis i.
    // Avoid from_fn().eval(), which crashes for this empty shape in mdarray 0.8.0.
    let a = Array::from_elem([0, 2], 0.0_f64).into_dyn();
    let b = create_test_matrix_f64([2, 2]).eval().into_dyn();
    assert_eq!(a.dim(0), 0);
    assert_eq!(a.dim(1), 2);

    let result = Naive.contract(&a, &b, &[0, 2], &[1, 2], &[0, 1]).eval();

    // The output is empty but must retain its shape [0, 2].
    assert_eq!(result, a);
}

#[test]
fn contract_einsum_matrix_multiplication() {
    contract_einsum_matrix_multiplication_impl(&Naive)
}

#[test]
fn contract_einsum_full_contraction() {
    contract_einsum_full_contraction_impl(&Naive)
}

#[test]
fn contract_einsum_output_permutation() {
    contract_einsum_output_permutation_impl(&Naive)
}

#[test]
fn contract_einsum_outer_product() {
    contract_einsum_outer_product_impl(&Naive)
}

#[test]
fn contract_einsum_trace_diagonal() {
    contract_einsum_trace_diagonal_impl(&Naive)
}

#[test]
fn contract_einsum_index_relabelling() {
    contract_einsum_index_relabelling_impl(&Naive)
}

#[test]
fn contract_einsum_partial_trace_then_contract() {
    contract_einsum_partial_trace_then_contract_impl(&Naive)
}

#[test]
fn contract_einsum_cross_diagonal() {
    contract_einsum_cross_diagonal_impl(&Naive)
}

#[test]
fn contract_einsum_vector_result() {
    contract_einsum_vector_result_impl(&Naive)
}

#[test]
fn hypercontract_empty_inputs() {
    use mdarray_linalg::contract::{_hypercontract, FREE_AXIS};
    for (shape_a, shape_b) in [([0, 2], [2, 2]), ([2, 0], [3, 0])] {
        let a = Array::from_elem(shape_a, 1.0_f64).into_dyn();
        let b = Array::from_elem(shape_b, 1.0_f64).into_dyn();
        let result = _hypercontract(Naive, a.expr(), b.expr(), &[FREE_AXIS, 0], &[FREE_AXIS, 0]);
        assert_eq!(
            result,
            Array::from_elem([shape_a[0], shape_b[0]], 0.0).into_dyn()
        );
    }
}

#[test]
fn contract_empty_einsum() {
    contract_empty_einsum_impl(&Naive);
}

#[test]
fn contract_empty_structured() {
    contract_empty_structured_impl(&Naive);
}

#[test]
fn contract_empty_builders() {
    contract_empty_builders_impl(&Naive);
}

#[test]
fn contract_empty_invalid_shapes() {
    contract_empty_invalid_shapes_impl(&Naive);
}
