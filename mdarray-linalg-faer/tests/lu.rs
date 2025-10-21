use mdarray_linalg_faer::Faer;
use mdarray_linalg_tests::lu::*;

#[test]
fn lu_decomposition() {
    test_lu_decomposition(&Faer);
}

#[test]
fn lu_decomposition_rectangular() {
    test_lu_decomposition_rectangular(&Faer);
}

#[test]
fn lu_overwrite() {
    test_lu_overwrite(&Faer);
}

#[test]
fn lu_overwrite_rectangular() {
    test_lu_overwrite_rectangular(&Faer);
}

#[test]
fn inverse() {
    test_inverse(&Faer);
}

#[test]
fn inverse_overwrite() {
    test_inverse_overwrite(&Faer);
}

// #[test]
// #[should_panic]
// fn inverse_singular_should_panic() {
//     test_inverse_singular_should_panic(&Faer);
// }

#[test]
fn determinant() {
    test_determinant(&Faer);
}

#[test]
fn determinant_dummy() {
    test_determinant_dummy(&Faer);
}

// #[test]
// fn cholesky_decomposition() {
//     test_cholesky_decomposition(&Faer);
// }

// #[test]
// fn cholesky_overwrite() {
//     test_cholesky_overwrite(&Faer);
// }

// #[test]
// fn cholesky_not_positive_definite() {
//     test_cholesky_not_positive_definite(&Faer);
// }

// #[test]
// fn cholesky_identity_matrix() {
//     test_cholesky_identity_matrix(&Faer);
// }
