use mdarray_linalg::{Naive, testing::contract::*};

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
fn contract_mismatched_dimensions_should_panic() {
    contract_mismatched_dimensions_should_panic_impl(&Naive);
}

#[test]
fn contract_outer_should_match_manual_kronecker() {
    contract_outer_should_match_manual_kronecker_impl(&Naive);
}

// --- Einsum-style contractions ---

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
