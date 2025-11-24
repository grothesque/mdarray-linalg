use mdarray_linalg::testing::qr::*;
use mdarray_linalg_lapack::Lapack;

#[test]
fn qr_random_matrix() {
    test_qr_random_matrix(&Lapack::default());
}

#[test]
fn qr_structured_matrix() {
    test_qr_structured_matrix(&Lapack::default());
}

#[test]
fn qr_complex_matrix() {
    test_qr_complex_matrix(&Lapack::default());
}
