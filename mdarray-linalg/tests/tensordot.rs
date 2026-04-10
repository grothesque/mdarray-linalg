use mdarray_linalg::{Naive, testing::matmul::*, testing::tensordot::*};

// --- Basic functionality ---

#[test]
fn tensordot_all_axes() {
    tensordot_all_axes_impl(&Naive);
}

#[test]
fn tensordot_contract_k_2_should_match_all_axes() {
    tensordot_contract_k_2_should_match_all_axes_impl(&Naive);
}

#[test]
fn tensordot_specific_axes_matrix_multiplication() {
    tensordot_specific_axes_matrix_multiplication_impl(&Naive);
}

#[test]
fn tensordot_specific_empty_axes_should_outer_product() {
    tensordot_specific_empty_axes_should_outer_product_impl(&Naive);
}

// --- Edge cases ---

#[test]
fn tensordot_scalar_inputs_should_multiply() {
    tensordot_scalar_inputs_should_multiply_impl(&Naive);
}

#[test]
fn tensordot_increase_deep() {
    tensordot_increase_deep_impl(&Naive);
}

#[test]
fn tensordot_vector_dot_product() {
    tensordot_vector_dot_product_impl(&Naive);
}

#[test]
fn tensordot_mismatched_dimensions_should_panic() {
    tensordot_mismatched_dimensions_should_panic_impl(&Naive);
}

// --- Structural and mathematical properties ---

#[test]
fn tensordot_outer_should_match_manual_kronecker() {
    tensordot_outer_should_match_manual_kronecker_impl(&Naive);
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
