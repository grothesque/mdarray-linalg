use mdarray::DTensor;

use mdarray_linalg::prelude::*;
use mdarray_linalg::qr::QR;
use mdarray_linalg_faer::Faer;
use mdarray_linalg_lapack::Lapack;
use mdarray_linalg_naive::Naive;

use approx::assert_relative_eq;
use num_complex::Complex;
use rand::prelude::*;

use crate::{assert_complex_matrix_eq, assert_matrix_eq};
use mdarray_linalg::pretty_print;

#[test]
fn test_backend_qr_random_matrix() {
    test_qr_random_matrix(&Lapack::default());
    test_qr_random_matrix(&Faer);
}

fn test_qr_random_matrix(bd: &impl QR<f64>) {
    let (m, n) = (5, 5);
    let mut rng = rand::rng();

    let a = DTensor::<f64, 2>::from_fn([m, n], |_| rng.random::<f64>());
    test_qr_reconstruction(bd, &a);
}

#[test]
fn test_backend_qr_structured_matrix() {
    test_qr_structured_matrix(&Lapack::default());
    test_qr_structured_matrix(&Faer);
}

fn test_qr_structured_matrix(bd: &impl QR<f64>) {
    let (m, n) = (3, 3);

    let a = DTensor::<f64, 2>::from_fn([m, n], |i| (i[0] * i[1] + 1) as f64);
    test_qr_reconstruction(bd, &a);
}

#[test]
fn test_backend_qr_complex_matrix() {
    test_qr_complex_matrix(&Lapack::default());
    test_qr_complex_matrix(&Faer);
}

fn test_qr_complex_matrix(bd: &impl QR<Complex<f64>>) {
    let (m, n) = (3, 3);

    let mut a = DTensor::<Complex<f64>, 2>::from_fn([m, n], |i| {
        Complex::new((i[0] + 1) as f64, (i[1] + 1) as f64)
    });

    a[[1, 2]] = Complex::new(1., 5.); // destroy symmetry

    let mut q = DTensor::<Complex<f64>, 2>::zeros([m, m]);
    let mut r = DTensor::<Complex<f64>, 2>::zeros([m, n]);

    let mut reconstructed = DTensor::<Complex<f64>, 2>::zeros([m, n]);
    bd.qr_overwrite(&mut a.clone(), &mut q, &mut r);
    Naive.matmul(&q, &r).overwrite(&mut reconstructed);
    assert_complex_matrix_eq!(a, reconstructed);

    let mut reconstructed = DTensor::<Complex<f64>, 2>::zeros([m, n]);
    let (q, r) = bd.qr(&mut a.clone());
    Naive.matmul(&q, &r).overwrite(&mut reconstructed);
    assert_complex_matrix_eq!(a, reconstructed);

    pretty_print(&a);
    pretty_print(&reconstructed);
}

fn test_qr_reconstruction<T>(bd: &impl QR<T>, a: &DTensor<T, 2>)
where
    T: num_traits::float::FloatConst
        + Default
        + Copy
        + std::fmt::Debug
        + approx::AbsDiffEq<Epsilon = f64>
        + std::fmt::Display
        + approx::RelativeEq
        + num_traits::Float
        + std::convert::From<i8>,
{
    let (m, n) = *a.shape();
    let mut q = DTensor::<T, 2>::zeros([m, m]);
    let mut r = DTensor::<T, 2>::zeros([m, n]);

    let mut reconstructed = DTensor::<T, 2>::zeros([m, n]);
    bd.qr_overwrite(&mut a.clone(), &mut q, &mut r);
    Naive.matmul(&q, &r).overwrite(&mut reconstructed);

    pretty_print(&q);
    pretty_print(&r);

    pretty_print(a);
    pretty_print(&reconstructed);

    assert_matrix_eq!(a, reconstructed);

    let mut reconstructed = DTensor::<T, 2>::zeros([m, n]);
    let (q, r) = bd.qr(&mut a.clone());
    Naive.matmul(&q, &r).overwrite(&mut reconstructed);

    pretty_print(a);
    pretty_print(&reconstructed);

    assert_matrix_eq!(a, reconstructed);
}
