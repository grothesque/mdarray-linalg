use approx::assert_relative_eq;
use num_complex::ComplexFloat;

use crate::common::random_matrix;
use mdarray::DTensor;
use mdarray_linalg::{LU, naive_matmul};
use mdarray_linalg_lapack::Lapack;

fn test_lu_reconstruction<T>(
    a: &DTensor<T, 2>,
    l: &DTensor<T, 2>,
    u: &DTensor<T, 2>,
    p: &DTensor<T, 2>,
) where
    T: Default
        + ComplexFloat
        + std::fmt::Debug
        + Copy
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>,
    f64: From<T>,
{
    let (n, m) = *a.shape();

    let mut pa = DTensor::<T, 2>::zeros([n, m]);
    naive_matmul(p, a, &mut pa);

    let mut lu = DTensor::<T, 2>::zeros([n, m]);
    naive_matmul(l, u, &mut lu);

    // Verify that P * A = L * U
    for i in 0..n {
        for j in 0..m {
            let diff = f64::from(pa[[i, j]]) - f64::from(lu[[i, j]]);
            assert_relative_eq!(diff, 0.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn lu_decomposition() {
    test_lu_decomposition(&Lapack::default());
}

fn test_lu_decomposition(bd: &impl LU<f64>) {
    let n = 4;
    let mut a = random_matrix(n, n);
    let original_a = a.clone();

    let (l, u, p) = bd.lu(&mut a);

    test_lu_reconstruction(&original_a, &l, &u, &p);
}

#[test]
fn lu_decomposition_rectangular() {
    test_lu_decomposition_rectangular(&Lapack::default());
}

fn test_lu_decomposition_rectangular(bd: &impl LU<f64>) {
    let n = 5;
    let m = 3;
    let mut a = random_matrix(n, m);
    let original_a = a.clone();

    let (l, u, p) = bd.lu(&mut a);

    test_lu_reconstruction(&original_a, &l, &u, &p);
}

#[test]
fn lu_overwrite() {
    test_lu_overwrite(&Lapack::default());
}

fn test_lu_overwrite(bd: &impl LU<f64>) {
    let n = 4;
    let mut a = random_matrix(n, n);
    let original_a = a.clone();

    let mut l = DTensor::<f64, 2>::zeros([n, n]);
    let mut u = DTensor::<f64, 2>::zeros([n, n]);
    let mut p = DTensor::<f64, 2>::zeros([n, n]);

    bd.lu_overwrite(&mut a, &mut l, &mut u, &mut p);

    test_lu_reconstruction(&original_a, &l, &u, &p);
}

#[test]
fn lu_overwrite_rectangular() {
    test_lu_overwrite_rectangular(&Lapack::default());
}

fn test_lu_overwrite_rectangular(bd: &impl LU<f64>) {
    let n = 5;
    let m = 3;
    let mut a = random_matrix(n, m);
    let original_a = a.clone();

    let mut l = DTensor::<f64, 2>::zeros([n, std::cmp::min(n, m)]);
    let mut u = DTensor::<f64, 2>::zeros([std::cmp::min(n, m), m]);
    let mut p = DTensor::<f64, 2>::zeros([n, n]);

    bd.lu_overwrite(&mut a, &mut l, &mut u, &mut p);

    test_lu_reconstruction(&original_a, &l, &u, &p);
}

#[test]
fn inverse() {
    test_inverse(&Lapack::default());
}

fn test_inverse(bd: &impl LU<f64>) {
    let n = 4;
    let a = random_matrix(n, n);
    let a_inv = bd.inv(&mut a.clone()).unwrap();

    let mut product = DTensor::<f64, 2>::zeros([n, n]);
    naive_matmul(&a, &a_inv, &mut product);

    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(product[[i, j]], expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn inverse_overwrite() {
    test_inverse_overwrite(&Lapack::default());
}

fn test_inverse_overwrite(bd: &impl LU<f64>) {
    let n = 4;
    let mut a = random_matrix(n, n);
    let a_clone = a.clone();
    let _ = bd.inv_overwrite(&mut a);

    let mut product = DTensor::<f64, 2>::zeros([n, n]);
    naive_matmul(&a, &a_clone, &mut product);

    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(product[[i, j]], expected, epsilon = 1e-10);
        }
    }
}

#[test]
#[should_panic]
fn inverse_singular_should_panic() {
    test_inverse_singular_should_panic(&Lapack::default());
}

fn test_inverse_singular_should_panic(bd: &impl LU<f64>) {
    let n = 4;
    let mut a = DTensor::<f64, 2>::from_elem([n, n], 1.);
    let _ = bd.inv(&mut a);
}
