use mdarray_linalg::testing::eig::*;
use mdarray_linalg_nalgebra::Nalgebra;

#[test]
#[should_panic]
fn non_square_matrix() {
    test_non_square_matrix(&Nalgebra::default());
}

#[test]
fn square_matrix() {
    test_square_matrix(&Nalgebra::default());
}

#[test]
fn cplx_square_matrix() {
    test_eig_cplx_square_matrix(&Nalgebra::default());
}

#[test]
fn eig_values_only() {
    test_eig_values_only(&Nalgebra::default());
}

#[test]
fn eigh_symmetric() {
    test_eigh_symmetric(&Nalgebra::default());
}

#[test]
fn eigh_complex_hermitian() {
    test_eigh_complex_hermitian(&Nalgebra::default());
}

#[test]
fn eig_full() {
    test_eig_full(&Nalgebra::default());
}

#[test]
fn eig_full_complex() {
    test_eig_full_complex(&Nalgebra::default());
}

#[test]
fn eig_full_real_complex_pair() {
    test_eig_full_real_complex_pair(&Nalgebra::default());
}

#[test]
fn eig_full_complex_singleton() {
    test_eig_full_complex_singleton(&Nalgebra::default());
}

#[test]
#[should_panic]
fn eig_full_non_square() {
    test_eig_full_non_square(&Nalgebra::default());
}

#[test]
#[should_panic]
fn eig_values_non_square() {
    test_eig_values_non_square(&Nalgebra::default());
}

#[test]
fn schur_decomp() {
    test_schur(&Nalgebra::default());
}

#[test]
fn schur_decomp_cplx() {
    test_schur_cplx(&Nalgebra::default());
}
