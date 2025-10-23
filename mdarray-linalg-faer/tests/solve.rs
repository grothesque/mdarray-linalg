use mdarray_linalg_faer::Faer;
use mdarray_linalg::testing::solve::*;

#[test]
fn solve_single_rhs() {
    test_solve_single_rhs(&Faer);
}

#[test]
fn solve_multiple_rhs() {
    test_solve_multiple_rhs(&Faer);
}

#[test]
fn solve_overwrite() {
    test_solve_overwrite(&Faer);
}

#[test]
fn solve_identity_matrix() {
    test_solve_identity_matrix(&Faer);
}

#[test]
fn solve_complex() {
    test_solve_complex(&Faer);
}
