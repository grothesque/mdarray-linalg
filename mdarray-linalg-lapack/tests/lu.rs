use mdarray_linalg_lapack::Lapack;
use mdarray_linalg_tests::lu::*;

#[test]
fn lu_decomposition() {
    test_lu_decomposition(&Lapack::default());
}

#[test]
fn lu_decomposition_rectangular() {
    test_lu_decomposition_rectangular(&Lapack::default());
}

#[test]
fn lu_overwrite() {
    test_lu_overwrite(&Lapack::default());
}

#[test]
fn lu_overwrite_rectangular() {
    test_lu_overwrite_rectangular(&Lapack::default());
}

#[test]
fn inverse() {
    test_inverse(&Lapack::default());
}

#[test]
fn inverse_overwrite() {
    test_inverse_overwrite(&Lapack::default());
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
fn cholesky_overwrite() {
    test_cholesky_overwrite(&Lapack::default());
}

#[test]
fn cholesky_not_positive_definite() {
    test_cholesky_not_positive_definite(&Lapack::default());
}

#[test]
fn cholesky_identity_matrix() {
    test_cholesky_identity_matrix(&Lapack::default());
}
