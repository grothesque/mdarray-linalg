use approx::assert_relative_eq;
use mdarray::DTensor;
use num_complex::{Complex, ComplexFloat};
use rand::Rng;

use super::common::naive_matmul;
use crate::{
    assert_complex_matrix_eq, assert_matrix_eq, pretty_print,
    svd::{SVD, SVDDecomp},
};

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

    let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");

    assert_eq!(*s.shape(), (n, n));
    assert_eq!(*u.shape(), (m, m));
    assert_eq!(*vt.shape(), (n, n));

    let mut sigma = DTensor::<T, 2>::zeros([m, n]);
    for i in 0..min_dim {
        sigma[[i, i]] = s[[0, i]];
    }

    if debug_print {
        println!("=== Σ (Sigma) ===");
        pretty_print(&sigma);
        println!("=== U ===");
        pretty_print(&u);
        println!("=== Vᵀ ===");
        pretty_print(&vt);
    }

    let us = naive_matmul(&u, &sigma);
    if debug_print {
        println!("=== U × Σ ===");
        pretty_print(&us);
    }

    let usvt = naive_matmul(&us, &vt);
    if debug_print {
        println!("=== U × Σ × Vᵀ  ===");
        pretty_print(&usvt);
        println!("=== A original ===");
        pretty_print(a);
    }

    assert_matrix_eq!(*a, usvt);
}

pub fn test_svd_square_matrix(bd: &impl SVD<f64>) {
    let n = 3;
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * i[1]) as f64);
    test_svd_reconstruction(bd, &a, true);
}

pub fn test_svd_rectangular_m_gt_n(bd: &impl SVD<f64>) {
    let (m, n) = (4, 3);
    let a = DTensor::<f64, 2>::from_fn([m, n], |i| (i[0] * i[1]) as f64);
    test_svd_reconstruction(bd, &a, true);
}

pub fn test_svd_big_square_matrix(bd: &impl SVD<f64>) {
    let n = 200;
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * i[1]) as f64);
    test_svd_reconstruction(bd, &a, false);
}

pub fn test_svd_random_matrix(bd: &impl SVD<f64>) {
    let mut rng = rand::rng();
    let n = 10;
    let a = DTensor::<f64, 2>::from_fn([n, n], |_| rng.random::<f64>());
    test_svd_reconstruction(bd, &a, false);
}

pub fn test_svd_cplx_square_matrix(bd: &impl SVD<Complex<f64>>) {
    let n = 3;
    let a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |i| {
        Complex::new((i[0] * i[1]) as f64, i[1] as f64)
    });

    let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");

    assert_eq!(*s.shape(), (n, n));
    assert_eq!(*u.shape(), (n, n));
    assert_eq!(*vt.shape(), (n, n));

    let mut sigma = DTensor::<Complex<f64>, 2>::zeros([n, n]);
    for i in 0..n {
        sigma[[i, i]] = s[[0, i]];
    }

    println!("=== Σ (Sigma) ===");
    pretty_print(&sigma);
    println!("=== U ===");
    pretty_print(&u);
    println!("=== Vᵀ ===");
    pretty_print(&vt);

    let us = naive_matmul(&u, &sigma);
    println!("=== U × Σ ===");
    pretty_print(&us);
    let usvt = naive_matmul(&us, &vt);
    println!("=== U × Σ × Vᵀ  ===");
    pretty_print(&usvt);
    println!("=== A original ===");
    pretty_print(&a);

    assert_complex_matrix_eq!(a, usvt);
}
