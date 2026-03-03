use mdarray_linalg::testing::lu::*;
use mdarray_linalg_faer::Faer;

#[test]
fn lu_decomposition() {
    test_lu_decomposition(&Faer::default());
}

#[test]
fn lu_decomposition_rectangular() {
    test_lu_decomposition_rectangular(&Faer::default());
}

#[test]
fn lu_write() {
    test_lu_write(&Faer::default());
}

#[test]
fn lu_write_rectangular() {
    test_lu_write_rectangular(&Faer::default());
}

#[test]
fn inverse() {
    test_inverse(&Faer::default());
}

#[test]
fn inverse_write() {
    test_inverse_write(&Faer::default());
}

// #[test]
// #[should_panic]
// fn inverse_singular_should_panic() {
//     test_inverse_singular_should_panic(&Faer::default());
// }

#[test]
fn determinant() {
    test_determinant(&Faer::default());
}

#[test]
fn determinant_dummy() {
    test_determinant_dummy(&Faer::default());
}

// #[test]
// fn cholesky_decomposition() {
//     test_cholesky_decomposition(&Faer::default());
// }

// #[test]
// fn cholesky_write() {
//     test_cholesky_write(&Faer::default());
// }

// #[test]
// fn cholesky_not_positive_definite() {
//     test_cholesky_not_positive_definite(&Faer::default());
// }

// #[test]
// fn cholesky_identity_matrix() {
//     test_cholesky_identity_matrix(&Faer::default());
// }
