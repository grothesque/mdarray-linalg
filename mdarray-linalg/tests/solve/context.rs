use approx::assert_relative_eq;

use crate::common::random_matrix;
use mdarray::DTensor;
use mdarray_linalg::{Solve, SolveResult};
use mdarray_linalg_faer::Faer;
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
    test_solve_single_rhs(&Faer);
}

fn test_solve_single_rhs(bd: &impl Solve<f64>) {
    let n = 4;
    let a = random_matrix(n, n);
    let original_a = a.clone();
    let b = random_matrix(n, 1);

    let SolveResult { x, .. } = bd.solve(&mut a.clone(), &b).expect("");

    test_solve_verification(&original_a, &x, &b);
}

#[test]
fn solve_multiple_rhs() {
    test_solve_multiple_rhs(&Lapack::default());
    test_solve_multiple_rhs(&Faer);
}

fn test_solve_multiple_rhs(bd: &impl Solve<f64>) {
    let n = 5;
    let nrhs = 3;
    let mut a = random_matrix(n, n);
    let original_a = a.clone();
    let b = random_matrix(n, nrhs);

    let SolveResult { x, .. } = bd.solve(&mut a, &b).expect("");

    test_solve_verification(&original_a, &x, &b);
}

#[test]
fn solve_overwrite() {
    test_solve_overwrite(&Lapack::default());
    test_solve_overwrite(&Faer);
}

fn test_solve_overwrite(bd: &impl Solve<f64>) {
    let n = 4;
    let nrhs = 2;
    let mut a = random_matrix(n, n);
    let original_a = a.clone();
    let mut b = random_matrix(n, nrhs);
    let original_b = b.clone();
    let mut p = DTensor::<f64, 2>::zeros([n, n]);

    let _ = bd.solve_overwrite(&mut a, &mut b, &mut p);

    // b now contains the solution x
    test_solve_verification(&original_a, &b, &original_b);
}

#[test]
fn solve_identity_matrix() {
    test_solve_identity_matrix(&Lapack::default());
    test_solve_identity_matrix(&Faer);
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

    let SolveResult { x, .. } = bd.solve(&mut a, &b).expect("");

    for i in 0..n {
        for j in 0..nrhs {
            let diff = x[[i, j]] - b[[i, j]];
            assert_relative_eq!(diff, 0.0, epsilon = 1e-14);
        }
    }

    test_solve_verification(&original_a, &x, &b);
}

#[test]
fn solve_complex() {
    test_solve_complex(&Lapack::default());
    test_solve_complex(&Faer);
}

fn test_solve_complex(bd: &impl Solve<num_complex::Complex<f64>>) {
    use num_complex::Complex;

    let n = 4;
    let nrhs = 2;

    let re = random_matrix(n, n);
    let im = random_matrix(n, n);

    let mut a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |i| {
        Complex::new(re[[i[0], i[1]]], im[[i[0], i[1]]])
    });
    println!("a={:?}", a);
    let original_a = a.clone();

    // Create random complex right-hand side
    let b = DTensor::<Complex<f64>, 2>::from_fn([n, nrhs], |i| {
        Complex::new((i[0] + 2 * i[1] + 1) as f64, (2 * i[0] + i[1] + 1) as f64)
    });
    println!("b={:?}", b);

    let SolveResult { x, .. } = bd.solve(&mut a, &b).expect("");

    println!("{:?}", x);

    // Verify A * X = B for complex matrices
    for i in 0..n {
        for j in 0..nrhs {
            let mut sum = Complex::new(0.0, 0.0);
            for k in 0..n {
                sum += original_a[[i, k]] * x[[k, j]];
            }
            let diff_real = sum.re - b[[i, j]].re;
            let diff_imag = sum.im - b[[i, j]].im;
            assert_relative_eq!(diff_real, 0.0, epsilon = 1e-10);
            assert_relative_eq!(diff_imag, 0.0, epsilon = 1e-10);
        }
    }
}
