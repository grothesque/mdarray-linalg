use mdarray_linalg_tests::svd::*;

use mdarray_linalg_lapack::Lapack;
use mdarray_linalg_lapack::SVDConfig;

#[test]
fn test_backend_svd_square_matrix() {
    test_svd_square_matrix(&Lapack::default().config_svd(SVDConfig::DivideConquer));
}

#[test]
fn test_backend_svd_rectangular_m_gt_n() {
    test_svd_rectangular_m_gt_n(&Lapack::default().config_svd(SVDConfig::Auto));
}

#[test]
fn test_backend_big_square_matrix() {
    test_svd_big_square_matrix(&Lapack::default().config_svd(SVDConfig::Jacobi));
}

#[test]
fn test_backend_svd_random_matrix() {
    test_svd_random_matrix(&Lapack::default());
}

#[test]
fn test_backend_svd_cplx_square_matrix() {
    test_svd_cplx_square_matrix(&Lapack::default());
}
