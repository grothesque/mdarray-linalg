use mdarray_linalg::testing::eig::*;
use mdarray_linalg_lapack::Lapack;

#[test]
#[should_panic]
fn non_square_matrix() {
    test_non_square_matrix(&Lapack::default());
}

#[test]
fn square_matrix() {
    test_square_matrix(&Lapack::default());
}

#[test]
fn cplx_square_matrix() {
    test_eig_cplx_square_matrix(&Lapack::default());
}

#[test]
fn eig_values_only() {
    test_eig_values_only(&Lapack::default());
}

#[test]
fn eigh_symmetric() {
    test_eigh_symmetric(&Lapack::default());
}

#[test]
fn eigh_complex_hermitian() {
    test_eigh_complex_hermitian(&Lapack::default());
}

#[test]
#[should_panic]
fn eig_full_non_square() {
    test_eig_full_non_square(&Lapack::default());
}

#[test]
#[should_panic]
fn eig_values_non_square() {
    test_eig_values_non_square(&Lapack::default());
}

#[test]
fn schur_decomp() {
    test_schur(&Lapack::default());
}

#[test]
fn schur_decomp_cplx() {
    test_schur_cplx(&Lapack::default());
}
