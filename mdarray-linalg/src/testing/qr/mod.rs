use approx::assert_relative_eq;
use mdarray::DTensor;
use num_complex::Complex;
use rand::prelude::*;

use super::common::naive_matmul;
use crate::{assert_complex_matrix_eq, assert_matrix_eq, pretty_print, qr::QR};

pub fn test_qr_random_matrix(bd: &impl QR<f64>) {
    let (m, n) = (5, 5);
    let mut rng = rand::rng();

    let a = DTensor::<f64, 2>::from_fn([m, n], |_| rng.random::<f64>());
    test_qr_reconstruction(bd, &a);
}

pub fn test_qr_structured_matrix(bd: &impl QR<f64>) {
    let (m, n) = (3, 3);

    let a = DTensor::<f64, 2>::from_fn([m, n], |i| (i[0] * i[1] + 1) as f64);
    test_qr_reconstruction(bd, &a);
}

pub fn test_qr_complex_matrix(bd: &impl QR<Complex<f64>>) {
    let (m, n) = (3, 3);

    let mut a = DTensor::<Complex<f64>, 2>::from_fn([m, n], |i| {
        Complex::new((i[0] + 1) as f64, (i[1] + 1) as f64)
    });

    a[[1, 2]] = Complex::new(1., 5.); // destroy symmetry

    let mut q = DTensor::<Complex<f64>, 2>::zeros([m, m]);
    let mut r = DTensor::<Complex<f64>, 2>::zeros([m, n]);

    bd.qr_write(&mut a.clone(), &mut q, &mut r);
    let reconstructed = naive_matmul(&q, &r);
    assert_complex_matrix_eq!(a, reconstructed);

    let (q, r) = bd.qr(&mut a.clone());
    let reconstructed = naive_matmul(&q, &r);
    assert_complex_matrix_eq!(a, reconstructed);

    pretty_print(&a);
    pretty_print(&reconstructed);
}

pub fn test_qr_reconstruction<T>(bd: &impl QR<T>, a: &DTensor<T, 2>)
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

    bd.qr_write(&mut a.clone(), &mut q, &mut r);
    let reconstructed = naive_matmul(&q, &r);

    pretty_print(&q);
    pretty_print(&r);

    pretty_print(a);
    pretty_print(&reconstructed);

    assert_matrix_eq!(a, reconstructed);

    let (q, r) = bd.qr(&mut a.clone());
    let reconstructed = naive_matmul(&q, &r);

    pretty_print(a);
    pretty_print(&reconstructed);

    assert_matrix_eq!(a, reconstructed);
}
