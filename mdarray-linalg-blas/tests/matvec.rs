use mdarray_linalg::testing::matvec::*;
use mdarray_linalg_blas::Blas;

#[test]
fn eval_and_overwrite() {
    test_eval_and_overwrite(Blas)
}

#[test]
fn add_to_scaled() {
    test_add_to_scaled(Blas)
}

#[test]
fn add_to() {
    test_add_to(Blas)
}

#[test]
fn add_outer_basic() {
    test_add_outer_basic(Blas)
}

#[test]
fn add_outer_sym() {
    test_add_outer_sym(Blas)
}

#[test]
fn add_outer_her() {
    test_add_outer_her(Blas)
}

#[test]
fn add_to_scaled_vecvec() {
    test_add_to_scaled_vecvec(Blas)
}

#[test]
fn dot_real() {
    test_dot_real(Blas)
}

#[test]
fn dot_complex() {
    test_dot_complex(Blas)
}

#[test]
fn dotc_complex() {
    test_dotc_complex(Blas)
}

#[test]
fn norm1_complex() {
    test_norm1_complex(Blas)
}

#[test]
fn norm2_complex() {
    test_norm2_complex(Blas)
}

#[test]
fn argmax_real() {
    test_argmax_real(Blas);
}

#[test]
fn argmax_abs() {
    test_argmax_abs(Blas)
}

#[test]
fn argmax_overwrite_real() {
    test_argmax_overwrite_real(Blas)
}
