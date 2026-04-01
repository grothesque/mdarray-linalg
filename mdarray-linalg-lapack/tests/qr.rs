use mdarray_linalg::qr::QR;
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

#[test]
fn qr_rectangular() {
    let bd = Lapack::default().config_qr(mdarray_linalg_lapack::QRConfig::Complete);
    test_qr_rectangular_matrix(&bd);
    let bd2 = Lapack::default();
    test_qr_rectangular_matrix(&bd2);
    let a = mdarray::darray![[1., 2.], [3., 4.], [5., 6.]];
    let (q, _) = bd.qr(&mut a.clone());
    let (m, n) = q.shape();
    assert!(m == n);
    let (q2, _) = bd2.qr(&mut a.clone());
    let (m2, n2) = q2.shape();
    assert!(*m2 == 3);
    assert!(*n2 == 2);
}
