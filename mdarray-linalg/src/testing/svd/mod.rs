use approx::assert_relative_eq;
use mdarray::{DTensor, Dyn};
use num_complex::{Complex, ComplexFloat};
use rand::Rng;

use super::common::naive_matmul;
use crate::{
    assert_complex_matrix_eq, assert_matrix_eq, pretty_print,
    svd::{SVD, SVDDecomp},
};

fn test_svd_reconstruction<T>(bd: &impl SVD<T, Dyn, Dyn>, a: &DTensor<T, 2>, debug_print: bool)
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

pub fn test_svd_square_matrix(bd: &impl SVD<f64, Dyn, Dyn>) {
    let n = 3;
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * i[1]) as f64);
    test_svd_reconstruction(bd, &a, true);
}

pub fn test_svd_rectangular_m_gt_n(bd: &impl SVD<f64, Dyn, Dyn>) {
    let (m, n) = (4, 3);
    let a = DTensor::<f64, 2>::from_fn([m, n], |i| (i[0] * i[1]) as f64);
    test_svd_reconstruction(bd, &a, true);
}

pub fn test_svd_big_square_matrix(bd: &impl SVD<f64, Dyn, Dyn>) {
    let n = 200;
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * i[1]) as f64);
    test_svd_reconstruction(bd, &a, false);
}

pub fn test_svd_random_matrix(bd: &impl SVD<f64, Dyn, Dyn>) {
    let mut rng = rand::rng();
    let n = 10;
    let a = DTensor::<f64, 2>::from_fn([n, n], |_| rng.random::<f64>());
    test_svd_reconstruction(bd, &a, false);
}

pub fn test_svd_cplx_square_matrix(bd: &impl SVD<Complex<f64>, Dyn, Dyn>) {
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

/// Test complex SVD with random matrix having significant imaginary parts.
/// This test is specifically designed to catch the V^T vs V^H bug.
pub fn test_svd_cplx_random_matrix(bd: &impl SVD<Complex<f64>, Dyn, Dyn>) {
    let mut rng = rand::rng();
    let n = 5;

    // Create random complex matrix with significant imaginary parts
    let a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |_| {
        Complex::new(rng.random::<f64>() * 2.0 - 1.0, rng.random::<f64>() * 2.0 - 1.0)
    });

    let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");

    // Build sigma matrix
    let mut sigma = DTensor::<Complex<f64>, 2>::zeros([n, n]);
    for i in 0..n {
        sigma[[i, i]] = s[[0, i]];
    }

    // Reconstruct: A = U * Σ * V^H (vt should be V^H)
    let us = naive_matmul(&u, &sigma);
    let usvt = naive_matmul(&us, &vt);

    assert_complex_matrix_eq!(a, usvt);
}

/// Test complex SVD with rectangular matrix (m > n).
/// This catches potential issues with non-square complex matrices.
pub fn test_svd_cplx_rectangular_m_gt_n(bd: &impl SVD<Complex<f64>, Dyn, Dyn>) {
    let mut rng = rand::rng();
    let (m, n) = (5, 3);

    let a = DTensor::<Complex<f64>, 2>::from_fn([m, n], |_| {
        Complex::new(rng.random::<f64>() * 2.0 - 1.0, rng.random::<f64>() * 2.0 - 1.0)
    });

    let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");

    assert_eq!(*u.shape(), (m, m));
    assert_eq!(*vt.shape(), (n, n));

    // Build sigma matrix (m x n)
    let min_dim = m.min(n);
    let mut sigma = DTensor::<Complex<f64>, 2>::zeros([m, n]);
    for i in 0..min_dim {
        sigma[[i, i]] = s[[0, i]];
    }

    // Reconstruct: A = U * Σ * V^H
    let us = naive_matmul(&u, &sigma);
    let usvt = naive_matmul(&us, &vt);

    assert_complex_matrix_eq!(a, usvt);
}

/// Test complex SVD with rectangular matrix (m < n).
pub fn test_svd_cplx_rectangular_m_lt_n(bd: &impl SVD<Complex<f64>, Dyn, Dyn>) {
    let mut rng = rand::rng();
    let (m, n) = (3, 5);

    let a = DTensor::<Complex<f64>, 2>::from_fn([m, n], |_| {
        Complex::new(rng.random::<f64>() * 2.0 - 1.0, rng.random::<f64>() * 2.0 - 1.0)
    });

    let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");

    assert_eq!(*u.shape(), (m, m));
    assert_eq!(*vt.shape(), (n, n));

    // Build sigma matrix (m x n)
    let min_dim = m.min(n);
    let mut sigma = DTensor::<Complex<f64>, 2>::zeros([m, n]);
    for i in 0..min_dim {
        sigma[[i, i]] = s[[0, i]];
    }

    // Reconstruct: A = U * Σ * V^H
    let us = naive_matmul(&u, &sigma);
    let usvt = naive_matmul(&us, &vt);

    assert_complex_matrix_eq!(a, usvt);
}

/// Helper to compute V^H (Hermitian conjugate) from vt
fn hermitian_conjugate(vt: &DTensor<Complex<f64>, 2>) -> DTensor<Complex<f64>, 2> {
    let (rows, cols) = (vt.shape().0, vt.shape().1);
    DTensor::<Complex<f64>, 2>::from_fn([cols, rows], |idx| vt[[idx[1], idx[0]]].conj())
}

/// Test that V^H * V = I (unitary property).
/// This directly tests that vt is V^H, not V^T.
pub fn test_svd_cplx_unitary_property(bd: &impl SVD<Complex<f64>, Dyn, Dyn>) {
    let mut rng = rand::rng();
    let n = 4;

    let a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |_| {
        Complex::new(rng.random::<f64>() * 2.0 - 1.0, rng.random::<f64>() * 2.0 - 1.0)
    });

    let SVDDecomp { s: _, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");

    // For SVD, U and V should be unitary: U^H * U = I, V^H * V = I
    // Since vt = V^H, we have: vt * vt^H = V^H * V = I

    // Compute V from vt (V = vt^H)
    let v = hermitian_conjugate(&vt);

    // Check V^H * V = vt * V = I
    let vhv = naive_matmul(&vt, &v);

    // Check it's approximately identity
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
            let diff = (vhv[[i, j]] - expected).norm();
            assert!(
                diff < 1e-10,
                "V^H * V not identity at [{}, {}]: got {:?}, expected {:?}",
                i,
                j,
                vhv[[i, j]],
                expected
            );
        }
    }

    // Also check U^H * U = I
    let uh = hermitian_conjugate(&u);
    let uhu = naive_matmul(&uh, &u);

    for i in 0..n {
        for j in 0..n {
            let expected = if i == j {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
            let diff = (uhu[[i, j]] - expected).norm();
            assert!(
                diff < 1e-10,
                "U^H * U not identity at [{}, {}]: got {:?}, expected {:?}",
                i,
                j,
                uhu[[i, j]],
                expected
            );
        }
    }
}
