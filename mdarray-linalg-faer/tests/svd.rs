use mdarray_linalg::testing::svd::*;
use mdarray_linalg_faer::Faer;

#[test]
fn test_backend_svd_square_matrix() {
    test_svd_square_matrix(&Faer::default());
}

#[test]
fn test_backend_svd_rectangular_m_gt_n() {
    test_svd_rectangular_m_gt_n(&Faer::default());
}

#[test]
fn test_backend_svd_rectangular_n_gt_m() {
    test_svd_rectangular_n_gt_m(&Faer::default());
}

#[test]
fn test_backend_big_square_matrix() {
    test_svd_big_square_matrix(&Faer::default());
}

#[test]
fn test_backend_svd_random_matrix() {
    test_svd_random_matrix(&Faer::default());
}

#[test]
fn test_backend_svd_cplx_square_matrix() {
    test_svd_cplx_square_matrix(&Faer::default());
}

#[test]
fn test_backend_svd_cplx_random_matrix() {
    test_svd_cplx_random_matrix(&Faer::default());
}
