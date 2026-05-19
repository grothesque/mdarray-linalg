use mdarray_linalg::{matvec::VecOps, testing::matvec::*};
use mdarray_linalg_faer::Faer;
use num_complex::Complex;

#[test]
fn eval_and_write() {
    test_eval_and_write(Faer::default())
}

#[test]
fn eval_and_write_rectangular() {
    test_eval_and_write_rectangular(Faer::default())
}

#[test]
fn add_to_scaled() {
    test_add_to_scaled(Faer::default())
}

#[test]
fn add_to() {
    test_add_to(Faer::default())
}

#[test]
fn add_outer_basic() {
    test_add_outer_basic(Faer::default())
}

#[test]
fn add_outer_cplx() {
    test_add_outer_cplx(Faer::default())
}

#[test]
fn add_outer_subview() {
    test_add_outer_subview(Faer::default())
}

#[test]
fn dot_real() {
    test_dot_real(Faer::default())
}

#[test]
fn dot_complex() {
    test_dot_complex(Faer::default())
}

#[test]
fn dotc_complex() {
    test_dotc_complex(Faer::default())
}

#[test]
fn norm1_complex() {
    test_norm1_complex(Faer::default())
}

#[test]
fn norm2_complex() {
    test_norm2_complex(Faer::default())
}

#[test]
fn vector_ops_complex_smoke() {
    let bd = Faer::default();
    let x = mdarray::tensor![Complex::new(1.0, 2.0), Complex::new(3.0, -1.0)];
    let y = mdarray::tensor![Complex::new(2.0, -1.0), Complex::new(-4.0, 0.5)];
    let _ = bd.dot(&x, &y);
    let _ = bd.dotc(&x, &y);
    let _ = bd.norm1(&x);
    let _ = bd.norm2(&x);
}
