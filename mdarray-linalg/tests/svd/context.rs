use approx::assert_relative_eq;
use num_complex::Complex;

use crate::{assert_complex_matrix_eq, assert_matrix_eq};
use mdarray::{DTensor, Dense, tensor};
use mdarray_linalg::{SVD, SVDBuilder, naive_matmul, pretty_print};
use mdarray_linalg_faer::svd::Faer;
use mdarray_linalg_lapack::svd::Lapack;

use num_complex::ComplexFloat;
use rand::Rng;

fn test_svd_reconstruction<T>(bd: &impl SVD<T>, a: &DTensor<T, 2>, debug_print: bool)
where
    T: ComplexFloat<Real = f64>
        + Default
        + Copy
        + std::fmt::Debug
        + approx::AbsDiffEq<Epsilon = T::Real>
        + std::fmt::Display
        + approx::RelativeEq,
    T::Real: std::fmt::Display,
{
    let (m, n) = (a.shape().0, a.shape().1);
    let min_dim = m.min(n);

    let Ok((s, u, vt)) = bd.svd(&mut a.clone()).eval::<Dense, Dense, Dense>() else {
        panic!("SVD failed");
    };

    assert_eq!(*s.shape(), (n, n));
    assert_eq!(*u.shape(), (m, m));
    assert_eq!(*vt.shape(), (n, n));

    let mut sigma = DTensor::<T, 2>::zeros([m, n]);
    for i in 0..min_dim {
        sigma[[i, i]] = s[[0, i]];
    }

    let mut us = DTensor::<T, 2>::zeros([m, n]);
    let mut usvt = DTensor::<T, 2>::zeros([m, n]);

    if debug_print {
        bd.print_name();
        println!("=== Σ (Sigma) ===");
        pretty_print(&sigma);
        println!("=== U ===");
        pretty_print(&u);
        println!("=== Vᵀ ===");
        pretty_print(&vt);
    }

    naive_matmul(&u, &sigma, &mut us);
    if debug_print {
        println!("=== U × Σ ===");
        pretty_print(&us);
    }

    naive_matmul(&us, &vt, &mut usvt);
    if debug_print {
        println!("=== U × Σ × Vᵀ  ===");
        pretty_print(&usvt);
        println!("=== A original ===");
        pretty_print(&a);
    }

    assert_matrix_eq!(*a, usvt);
}

#[test]
fn test_backend_svd_square_matrix() {
    test_svd_square_matrix(&Lapack);
    test_svd_square_matrix(&Faer);
}

fn test_svd_square_matrix(bd: &impl SVD<f64>) {
    let n = 3;
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * i[1]) as f64);
    test_svd_reconstruction(bd, &a, true);
}

#[test]
fn test_backend_svd_rectangular_m_gt_n() {
    test_svd_rectangular_m_gt_n(&Lapack);
    test_svd_rectangular_m_gt_n(&Faer);
}

fn test_svd_rectangular_m_gt_n(bd: &impl SVD<f64>) {
    let (m, n) = (4, 3);
    let a = DTensor::<f64, 2>::from_fn([m, n], |i| (i[0] * i[1]) as f64);
    test_svd_reconstruction(bd, &a, true);
}

#[test]
fn test_backend_big_square_matrix() {
    test_svd_big_square_matrix(&Lapack);
    test_svd_big_square_matrix(&Faer);
}

fn test_svd_big_square_matrix(bd: &impl SVD<f64>) {
    let n = 200;
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * i[1]) as f64);
    test_svd_reconstruction(bd, &a, false);
}

#[test]
fn test_backend_svd_random_matrix() {
    test_svd_random_matrix(&Lapack);
    test_svd_random_matrix(&Faer);
}

fn test_svd_random_matrix(bd: &impl SVD<f64>) {
    let mut rng = rand::rng();
    let n = 10;
    let a = DTensor::<f64, 2>::from_fn([n, n], |_| rng.random::<f64>());
    test_svd_reconstruction(bd, &a, false);
}

#[test]
fn test_backend_svd_cplx_square_matrix() {
    test_svd_cplx_square_matrix(&Lapack);
}

fn test_svd_cplx_square_matrix(bd: &impl SVD<Complex<f64>>) {
    let n = 3;
    let a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |i| {
        Complex::new((i[0] * i[1]) as f64, i[1] as f64)
    });

    let Ok((s, u, vt)) = bd.svd(&mut a.clone()).eval::<Dense, Dense, Dense>() else {
        panic!("SVD failed");
    };

    assert_eq!(*s.shape(), (n, n));
    assert_eq!(*u.shape(), (n, n));
    assert_eq!(*vt.shape(), (n, n));

    let mut sigma = DTensor::<Complex<f64>, 2>::zeros([n, n]);
    for i in 0..n {
        sigma[[i, i]] = s[[0, i]];
    }

    let mut us = tensor![[Complex::new(0.,0.);n];n];
    let mut usvt = tensor![[Complex::new(0.,0.);n];n];

    println!("=== Σ (Sigma) ===");
    pretty_print(&sigma);
    println!("=== U ===");
    pretty_print(&u);
    println!("=== Vᵀ ===");
    pretty_print(&vt);

    naive_matmul(&u, &sigma, &mut us);
    println!("=== U × Σ ===");
    pretty_print(&us);
    naive_matmul(&us, &vt, &mut usvt);
    println!("=== U × Σ × Vᵀ  ===");
    pretty_print(&usvt);
    println!("=== A original ===");
    pretty_print(&a);

    assert_complex_matrix_eq!(a, usvt);
}
