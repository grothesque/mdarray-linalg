extern crate tblis_src as _;
use mdarray_linalg::{testing::matmul::*, testing::tensordot::*};
use mdarray_linalg_tblis::Tblis;

#[test]
fn tensordot_all_axes() {
    tensordot_all_axes_impl(&Tblis);
}

#[test]
fn tensordot_contract_k_2_should_match_all_axes() {
    tensordot_contract_k_2_should_match_all_axes_impl(&Tblis);
}

#[test]
fn tensordot_specific_axes_matrix_multiplication() {
    tensordot_specific_axes_matrix_multiplication_impl(&Tblis);
}

#[test]
fn tensordot_specific_empty_axes_should_outer_product() {
    tensordot_specific_empty_axes_should_outer_product_impl(&Tblis);
}

#[test]
fn tensordot_scalar_inputs_should_multiply() {
    tensordot_scalar_inputs_should_multiply_impl(&Tblis);
}

#[test]
fn tensordot_increase_deep() {
    tensordot_increase_deep_impl(&Tblis);
}

#[test]
fn tensordot_vector_dot_product() {
    tensordot_vector_dot_product_impl(&Tblis);
}

#[test]
fn tensordot_mismatched_dimensions_should_panic() {
    tensordot_mismatched_dimensions_should_panic_impl(&Tblis);
}

#[test]
fn tensordot_outer_should_match_manual_kronecker() {
    tensordot_outer_should_match_manual_kronecker_impl(&Tblis);
}

#[test]
fn contract_einsum_matrix_multiplication() {
    contract_einsum_matrix_multiplication_impl(&Tblis)
}

#[test]
fn contract_einsum_full_contraction() {
    contract_einsum_full_contraction_impl(&Tblis)
}

#[test]
fn contract_einsum_output_permutation() {
    contract_einsum_output_permutation_impl(&Tblis)
}

#[test]
fn contract_einsum_outer_product() {
    contract_einsum_outer_product_impl(&Tblis)
}

#[test]
fn contract_einsum_trace_diagonal() {
    contract_einsum_trace_diagonal_impl(&Tblis)
}

#[test]
fn contract_einsum_index_relabelling() {
    contract_einsum_index_relabelling_impl(&Tblis)
}

#[test]
fn contract_einsum_partial_trace_then_contract() {
    contract_einsum_partial_trace_then_contract_impl(&Tblis)
}

#[test]
fn contract_einsum_cross_diagonal() {
    contract_einsum_cross_diagonal_impl(&Tblis)
}

#[test]
fn contract_einsum_vector_result() {
    contract_einsum_vector_result_impl(&Tblis)
}
