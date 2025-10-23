use mdarray_linalg_naive::Naive;
use mdarray_linalg::testing::prrlu::*;

#[test]
fn rank_deficient() {
    test_rank_deficient(Naive)
}

#[test]
fn full_rank() {
    test_full_rank(Naive)
}

#[test]
fn rectangular() {
    test_rectangular(Naive)
}

#[test]
fn hilbert_matrix() {
    test_hilbert_matrix(Naive)
}
