use approx::assert_relative_eq;

use crate::common::random_matrix;
use mdarray::DTensor;
use mdarray_linalg::Solve;
use mdarray_linalg_lapack::Lapack;

fn test_solve_verification<T>(original_a: &DTensor<T, 2>, x: &DTensor<T, 2>, b: &DTensor<T, 2>)
where
    T: Default
        + std::fmt::Debug
        + Copy
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>,
    f64: From<T>,
{
    let (n, nrhs) = *b.shape();

    let mut ax = DTensor::<T, 2>::zeros([n, nrhs]);
    for i in 0..n {
        for j in 0..nrhs {
            let mut sum = T::default();
            for k in 0..n {
                sum = sum + original_a[[i, k]] * x[[k, j]];
            }
            ax[[i, j]] = sum;
        }
    }

    for i in 0..n {
        for j in 0..nrhs {
            let diff = f64::from(ax[[i, j]]) - f64::from(b[[i, j]]);
            assert_relative_eq!(diff, 0.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn solve_single_rhs() {
    test_solve_single_rhs(&Lapack::default());
}

fn test_solve_single_rhs(bd: &impl Solve<f64>) {
    let n = 4;
    let mut a = random_matrix(n, n);
    let original_a = a.clone();
    let b = random_matrix(n, 1);

    let (x, _p) = bd.solve(&mut a, &b);

    test_solve_verification(&original_a, &x, &b);
}

#[test]
fn solve_multiple_rhs() {
    test_solve_multiple_rhs(&Lapack::default());
}

fn test_solve_multiple_rhs(bd: &impl Solve<f64>) {
    let n = 5;
    let nrhs = 3;
    let mut a = random_matrix(n, n);
    let original_a = a.clone();
    let b = random_matrix(n, nrhs);

    let (x, _p) = bd.solve(&mut a, &b);

    test_solve_verification(&original_a, &x, &b);
}

#[test]
fn solve_overwrite() {
    test_solve_overwrite(&Lapack::default());
}

fn test_solve_overwrite(bd: &impl Solve<f64>) {
    let n = 4;
    let nrhs = 2;
    let mut a = random_matrix(n, n);
    let original_a = a.clone();
    let mut b = random_matrix(n, nrhs);
    let original_b = b.clone();
    let mut p = DTensor::<f64, 2>::zeros([n, n]);

    bd.solve_overwrite(&mut a, &mut b, &mut p);

    // b now contains the solution x
    test_solve_verification(&original_a, &b, &original_b);
}

#[test]
fn solve_identity_matrix() {
    test_solve_identity_matrix(&Lapack::default());
}

fn test_solve_identity_matrix(bd: &impl Solve<f64>) {
    let n = 3;
    let nrhs = 2;

    let mut a = DTensor::<f64, 2>::zeros([n, n]);
    for i in 0..n {
        a[[i, i]] = 1.0;
    }
    let original_a = a.clone();

    let b = random_matrix(n, nrhs);

    let (x, _p) = bd.solve(&mut a, &b);

    for i in 0..n {
        for j in 0..nrhs {
            let diff = x[[i, j]] - b[[i, j]];
            assert_relative_eq!(diff, 0.0, epsilon = 1e-14);
        }
    }

    test_solve_verification(&original_a, &x, &b);
}
