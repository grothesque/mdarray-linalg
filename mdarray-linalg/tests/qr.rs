use mdarray_linalg::Naive;
use mdarray_linalg::testing::qr::*;

#[test]
fn qr_random_matrix() {
    test_qr_random_matrix(&Naive);
}

#[test]
fn qr_structured_matrix() {
    test_qr_structured_matrix(&Naive);
}

#[test]
fn qr_complex_matrix() {
    test_qr_complex_matrix(&Naive);
}
