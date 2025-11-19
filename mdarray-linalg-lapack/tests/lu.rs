use mdarray_linalg_lapack::Lapack;
use mdarray_linalg::testing::lu::*;

#[test]
fn lu_decomposition() {
    test_lu_decomposition(&Lapack::default());
}

#[test]
fn lu_decomposition_rectangular() {
    test_lu_decomposition_rectangular(&Lapack::default());
}

#[test]
fn lu_write() {
    test_lu_write(&Lapack::default());
}

#[test]
fn lu_write_rectangular() {
    test_lu_write_rectangular(&Lapack::default());
}

#[test]
fn inverse() {
    test_inverse(&Lapack::default());
}

#[test]
fn inverse_write() {
    test_inverse_write(&Lapack::default());
}

#[test]
#[should_panic]
fn inverse_singular_should_panic() {
    test_inverse_singular_should_panic(&Lapack::default());
}

#[test]
fn determinant() {
    test_determinant(&Lapack::default());
}

#[test]
fn determinant_dummy() {
    test_determinant_dummy(&Lapack::default());
}

#[test]
fn cholesky_decomposition() {
    test_cholesky_decomposition(&Lapack::default());
}

#[test]
fn cholesky_write() {
    test_cholesky_write(&Lapack::default());
}

#[test]
fn cholesky_not_positive_definite() {
    test_cholesky_not_positive_definite(&Lapack::default());
}

#[test]
fn cholesky_identity_matrix() {
    test_cholesky_identity_matrix(&Lapack::default());
}
