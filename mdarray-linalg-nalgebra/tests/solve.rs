use mdarray_linalg::testing::solve::*;
use mdarray_linalg_nalgebra::Nalgebra;

#[test]
fn solve_single_rhs() {
    test_solve_single_rhs(&Nalgebra::default());
}

#[test]
fn solve_static_rhs_shape() {
    test_solve_static_rhs_shape(&Nalgebra::default());
}

#[test]
fn solve_multiple_rhs() {
    test_solve_multiple_rhs(&Nalgebra::default());
}

#[test]
fn solve_write() {
    test_solve_write(&Nalgebra::default());
}

#[test]
fn solve_identity_matrix() {
    test_solve_identity_matrix(&Nalgebra::default());
}

#[test]
fn solve_complex() {
    test_solve_complex(&Nalgebra::default());
}
