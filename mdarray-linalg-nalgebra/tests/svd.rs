use mdarray_linalg::testing::svd::*;
use mdarray_linalg_nalgebra::Nalgebra;

#[test]
fn test_backend_svd_square_matrix() {
    test_svd_square_matrix(&Nalgebra);
}

#[test]
fn test_backend_svd_rectangular_m_gt_n() {
    test_svd_rectangular_m_gt_n(&Nalgebra);
}

#[test]
fn test_backend_big_square_matrix() {
    test_svd_big_square_matrix(&Nalgebra);
}

#[test]
fn test_backend_svd_random_matrix() {
    test_svd_random_matrix(&Nalgebra);
}

// #[test]
// fn test_backend_svd_cplx_square_matrix() {
//     test_svd_cplx_square_matrix(&Nalgebra);
// }
