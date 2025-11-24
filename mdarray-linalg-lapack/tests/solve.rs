use mdarray_linalg::testing::solve::*;
use mdarray_linalg_lapack::Lapack;

#[test]
fn solve_single_rhs() {
    test_solve_single_rhs(&Lapack::default());
}

#[test]
fn solve_multiple_rhs() {
    test_solve_multiple_rhs(&Lapack::default());
}

#[test]
fn solve_write() {
    test_solve_write(&Lapack::default());
}

#[test]
fn solve_identity_matrix() {
    test_solve_identity_matrix(&Lapack::default());
}

#[test]
fn solve_complex() {
    test_solve_complex(&Lapack::default());
}
