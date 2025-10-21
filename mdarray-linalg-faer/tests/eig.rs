use mdarray_linalg_faer::Faer;
use mdarray_linalg_tests::eig::*;

#[test]
#[should_panic]
fn non_square_matrix() {
    test_non_square_matrix(&Faer);
}

#[test]
fn square_matrix() {
    test_square_matrix(&Faer);
}

#[test]
fn cplx_square_matrix() {
    test_eig_cplx_square_matrix(&Faer);
}

#[test]
fn eig_values_only() {
    test_eig_values_only(&Faer);
}

#[test]
fn eigh_symmetric() {
    test_eigh_symmetric(&Faer);
}

#[test]
fn eigh_complex_hermitian() {
    test_eigh_complex_hermitian(&Faer);
}

#[test]
#[should_panic]
fn eig_full_non_square() {
    test_eig_full_non_square(&Faer);
}

#[test]
#[should_panic]
fn eig_values_non_square() {
    test_eig_values_non_square(&Faer);
}

// #[test]
// fn schur_decomp() {
//     test_schur(&Faer);
// }

// #[test]
// fn schur_decomp_cplx() {
//     test_schur_cplx(&Faer);
// }
