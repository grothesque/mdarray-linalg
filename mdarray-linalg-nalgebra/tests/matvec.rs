use mdarray_linalg::testing::matvec::*;
use mdarray_linalg_nalgebra::Nalgebra;

#[test]
fn eval_and_write() {
    test_eval_and_write(Nalgebra)
}

#[test]
fn eval_and_write_rectangular() {
    test_eval_and_write_rectangular(Nalgebra)
}

#[test]
fn add_to_scaled() {
    test_add_to_scaled(Nalgebra)
}

#[test]
fn add_to() {
    test_add_to(Nalgebra)
}

#[test]
fn add_outer_basic() {
    test_add_outer_basic(Nalgebra)
}

#[test]
fn add_outer_subview() {
    test_add_outer_subview(Nalgebra)
}

#[test]
fn add_outer_cplx() {
    test_add_outer_cplx(Nalgebra)
}

#[test]
fn add_outer_sym() {
    test_add_outer_sym(Nalgebra)
}

#[test]
fn add_outer_her() {
    test_add_outer_her(Nalgebra)
}

#[test]
fn add_to_scaled_vecvec() {
    test_add_to_scaled_vecvec(Nalgebra)
}

#[test]
fn dot_real() {
    test_dot_real(Nalgebra)
}

#[test]
fn dot_complex() {
    test_dot_complex(Nalgebra)
}

#[test]
fn dotc_complex() {
    test_dotc_complex(Nalgebra)
}

#[test]
fn norm1_complex() {
    test_norm1_complex(Nalgebra)
}

#[test]
fn norm2_complex() {
    test_norm2_complex(Nalgebra)
}

#[test]
fn argmax_real() {
    test_argmax_real(Nalgebra)
}

#[test]
fn argmax_abs() {
    test_argmax_abs(Nalgebra)
}

#[test]
fn argmax_write_real() {
    test_argmax_write_real(Nalgebra)
}
