use approx::assert_relative_eq;

use crate::common::random_matrix;
use mdarray::DTensor;
use mdarray_linalg::LU;
use mdarray_linalg_lapack::Lapack;

fn test_lu_reconstruction<T>(
    original_a: &DTensor<T, 2>,
    l: &DTensor<T, 2>,
    u: &DTensor<T, 2>,
    p: &DTensor<T, 2>,
) where
    T: Default
        + std::fmt::Debug
        + Copy
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>,
    f64: From<T>,
{
    let (n, m) = *original_a.shape();

    // Compute P * A
    let mut pa = DTensor::<T, 2>::zeros([n, m]);
    for i in 0..n {
        for j in 0..m {
            let mut sum = T::default();
            for k in 0..n {
                sum = sum + p[[i, k]] * original_a[[k, j]];
            }
            pa[[i, j]] = sum;
        }
    }

    // Compute L * U
    let mut lu = DTensor::<T, 2>::zeros([n, m]);
    for i in 0..n {
        for j in 0..m {
            let mut sum = T::default();
            for k in 0..std::cmp::min(i + 1, j + 1) {
                sum = sum + l[[i, k]] * u[[k, j]];
            }
            lu[[i, j]] = sum;
        }
    }

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
