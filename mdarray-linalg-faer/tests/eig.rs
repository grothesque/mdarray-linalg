use mdarray_linalg::testing::eig::*;
use mdarray_linalg_faer::Faer;

#[test]
#[should_panic]
fn non_square_matrix() {
    test_non_square_matrix(&Faer::default());
}

#[test]
fn square_matrix() {
    test_square_matrix(&Faer::default());
}

#[test]
fn cplx_square_matrix() {
    test_eig_cplx_square_matrix(&Faer::default());
}

#[test]
fn eig_values_only() {
    test_eig_values_only(&Faer::default());
}

#[test]
fn eigh_symmetric() {
    test_eigh_symmetric(&Faer::default());
}

#[test]
fn eigh_complex_hermitian() {
    test_eigh_complex_hermitian(&Faer::default());
}

#[test]
#[should_panic]
fn eig_full_non_square() {
    test_eig_full_non_square(&Faer::default());
}

#[test]
#[should_panic]
fn eig_values_non_square() {
    test_eig_values_non_square(&Faer::default());
}

// #[test]
// fn schur_decomp() {
//     test_schur(&Faer::default());
// }

// #[test]
// fn schur_decomp_cplx() {
//     test_schur_cplx(&Faer::default());
// }
