use mdarray_linalg::qr::QR;
use mdarray_linalg::testing::qr::*;
use mdarray_linalg_nalgebra::Nalgebra;

#[test]
fn qr_random_matrix() {
    test_qr_random_matrix(&Nalgebra::default());
}

#[test]
fn qr_structured_matrix() {
    test_qr_structured_matrix(&Nalgebra::default());
}

#[test]
fn qr_complex_matrix() {
    test_qr_complex_matrix(&Nalgebra::default());
}

#[test]
fn qr_rectangular() {
    let bd = Nalgebra::default();
    test_qr_rectangular_matrix(&bd);

    let a = mdarray::darray![[1., 2.], [3., 4.], [5., 6.]];
    let (q, _) = bd.qr(&mut a.clone());
    let (m, n) = q.shape();
    assert!(*m == 3);
    assert!(*n == 2);
}
