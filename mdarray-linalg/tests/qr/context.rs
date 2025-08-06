use mdarray::{DTensor, Strided};

use mdarray_linalg::{QR, QRBuilder};
// use qr::faer::Faer;
use mdarray_linalg_lapack::qr::Lapack;

use approx::assert_relative_eq;
use num_complex::Complex;
use rand::prelude::*;

use crate::{assert_complex_matrix_eq, assert_matrix_eq};
use mdarray_linalg::{naive_matmul, pretty_print};

#[test]
fn test_backend_qr_random_matrix() {
    test_qr_random_matrix(&Lapack);
    // test_qr_random_matrix(&Faer);
}

fn test_qr_random_matrix(bd: &impl QR<f64>) {
    let (m, n) = (5, 5);
    let mut rng = rand::rng();

    let a = DTensor::<f64, 2>::from_fn([m, n], |_| rng.random::<f64>());
    test_qr_reconstruction(bd, &a);
}

#[test]
fn test_backend_qr_structured_matrix() {
    test_qr_structured_matrix(&Lapack);
    // test_qr_structured_matrix(&Faer);
}

fn test_qr_structured_matrix(bd: &impl QR<f64>) {
    let (m, n) = (3, 3);

    let a = DTensor::<f64, 2>::from_fn([m, n], |i| (i[0] * i[1] + 1) as f64);
    test_qr_reconstruction(bd, &a);
}

#[test]
fn test_backend_qr_complex_matrix() {
    test_qr_complex_matrix(&Lapack);
    // test_qr_complex_matrix(&Faer);
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
    bd.qr(&mut a.clone()).overwrite(&mut q, &mut r);
    naive_matmul(&q, &r, &mut reconstructed);
    assert_complex_matrix_eq!(a, reconstructed);

    pretty_print(&a);
    pretty_print(&reconstructed);

    let mut reconstructed = DTensor::<Complex<f64>, 2>::zeros([m, n]);
    let (q, r) = bd.qr(&mut a.clone()).eval::<Strided, Strided>();
    naive_matmul(&q, &r, &mut reconstructed);
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
        + num_traits::Float,
{
    let (m, n) = *a.shape();
    let mut q = DTensor::<T, 2>::zeros([m, m]);
    let mut r = DTensor::<T, 2>::zeros([m, n]);

    let mut reconstructed = DTensor::<T, 2>::zeros([m, n]);
    bd.qr(&mut a.clone()).overwrite(&mut q, &mut r);
    naive_matmul(&q, &r, &mut reconstructed);
    assert_matrix_eq!(a, reconstructed);

    let mut reconstructed = DTensor::<T, 2>::zeros([m, n]);
    let (q, r) = bd.qr(&mut a.clone()).eval::<Strided, Strided>();
    naive_matmul(&q, &r, &mut reconstructed);
    assert_matrix_eq!(a, reconstructed);
}
