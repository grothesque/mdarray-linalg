use mdarray_linalg::testing::lu::*;
use mdarray_linalg_nalgebra::Nalgebra;

#[test]
fn lu_decomposition() {
    test_lu_decomposition(&Nalgebra::default());
}

#[test]
fn lu_decomposition_rectangular() {
    test_lu_decomposition_rectangular(&Nalgebra::default());
}

#[test]
fn lu_write() {
    test_lu_write(&Nalgebra::default());
}

#[test]
fn lu_write_rectangular() {
    test_lu_write_rectangular(&Nalgebra::default());
}

#[test]
fn inverse() {
    test_inverse(&Nalgebra::default());
}

#[test]
fn inverse_write() {
    test_inverse_write(&Nalgebra::default());
}

#[test]
fn determinant() {
    test_determinant(&Nalgebra::default());
}

#[test]
fn determinant_dummy() {
    test_determinant_dummy(&Nalgebra::default());
}

#[test]
fn cholesky_decomposition() {
    test_cholesky_decomposition(&Nalgebra::default());
}

#[test]
fn cholesky_write() {
    test_cholesky_write(&Nalgebra::default());
}

#[test]
fn cholesky_not_positive_definite() {
    test_cholesky_not_positive_definite(&Nalgebra::default());
}

#[test]
fn cholesky_identity_matrix() {
    test_cholesky_identity_matrix(&Nalgebra::default());
}
