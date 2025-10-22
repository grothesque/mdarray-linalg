use mdarray_linalg_lapack::Lapack;
use mdarray_linalg_tests::solve::*;

#[test]
fn solve_single_rhs() {
    test_solve_single_rhs(&Lapack::default());
}

#[test]
fn solve_multiple_rhs() {
    test_solve_multiple_rhs(&Lapack::default());
}

#[test]
fn solve_overwrite() {
    test_solve_overwrite(&Lapack::default());
}

#[test]
fn solve_identity_matrix() {
    test_solve_identity_matrix(&Lapack::default());
}

#[test]
fn solve_complex() {
    test_solve_complex(&Lapack::default());
}
