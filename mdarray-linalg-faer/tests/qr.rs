use mdarray_linalg::testing::qr::*;
use mdarray_linalg_faer::Faer;

#[test]
fn qr_random_matrix() {
    test_qr_random_matrix(&Faer);
}

#[test]
fn qr_structured_matrix() {
    test_qr_structured_matrix(&Faer);
}

#[test]
fn qr_complex_matrix() {
    test_qr_complex_matrix(&Faer);
}
