use approx::assert_relative_eq;
use num_complex::ComplexFloat;

use super::common::{naive_matmul, random_matrix};
use crate::prelude::*;
use crate::{identity, pretty_print, transpose_in_place};
use mdarray::{DSlice, DTensor, Dense, tensor};

pub fn test_lu_reconstruction<T>(
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

    let pa = naive_matmul(p, a);
    let lu = naive_matmul(l, u);

    // Verify that P * A = L * U
    for i in 0..n {
        for j in 0..m {
            let diff = f64::from(pa[[i, j]]) - f64::from(lu[[i, j]]);
            assert_relative_eq!(diff, 0.0, epsilon = 1e-10);
        }
    }
}

pub fn test_lu_decomposition(bd: &impl LU<f64>) {
    let mut a = tensor![
        [0.16931568150114162, 0.5524301997803323],
        [0.10477204466703971, 0.33895423448188766]
    ];

    let original_a = a.clone();

    let (l, u, p) = bd.lu(&mut a);

    println!("{a:?}");
    pretty_print(&a);
    pretty_print(&l);
    pretty_print(&u);

    test_lu_reconstruction(&original_a, &l, &u, &p);
}

pub fn test_lu_decomposition_rectangular(bd: &impl LU<f64>) {
    let n = 5;
    let m = 3;
    let mut a = random_matrix(n, m);
    let original_a = a.clone();

    let (l, u, p) = bd.lu(&mut a);

    test_lu_reconstruction(&original_a, &l, &u, &p);
}

pub fn test_lu_write(bd: &impl LU<f64>) {
    let n = 4;
    let mut a = random_matrix(n, n);
    let original_a = a.clone();

    let mut l = DTensor::<f64, 2>::zeros([n, n]);
    let mut u = DTensor::<f64, 2>::zeros([n, n]);
    let mut p = DTensor::<f64, 2>::zeros([n, n]);

    bd.lu_write(&mut a, &mut l, &mut u, &mut p);

    test_lu_reconstruction(&original_a, &l, &u, &p);
}

pub fn test_lu_write_rectangular(bd: &impl LU<f64>) {
    let n = 5;
    let m = 3;
    let mut a = random_matrix(n, m);
    let original_a = a.clone();

    let mut l = DTensor::<f64, 2>::zeros([n, std::cmp::min(n, m)]);
    let mut u = DTensor::<f64, 2>::zeros([std::cmp::min(n, m), m]);
    let mut p = DTensor::<f64, 2>::zeros([n, n]);

    bd.lu_write(&mut a, &mut l, &mut u, &mut p);

    test_lu_reconstruction(&original_a, &l, &u, &p);
}

pub fn test_inverse(bd: &impl LU<f64>) {
    let n = 4;
    let a = random_matrix(n, n);
    let a_inv = bd.inv(&mut a.clone()).unwrap();

    let product = naive_matmul(&a, &a_inv);

    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(product[[i, j]], expected, epsilon = 1e-10);
        }
    }
}

pub fn test_inverse_write(bd: &impl LU<f64>) {
    let n = 4;
    let mut a = random_matrix(n, n);
    let a_clone = a.clone();
    let _ = bd.inv_write(&mut a);

    let product = naive_matmul(&a, &a_clone);

    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(product[[i, j]], expected, epsilon = 1e-10);
        }
    }
}

pub fn test_inverse_singular_should_panic(bd: &impl LU<f64>) {
    let n = 4;
    let mut a = DTensor::<f64, 2>::from_elem([n, n], 1.);
    let _ = bd.inv(&mut a);
}

pub fn test_determinant(bd: &impl LU<f64>) {
    let n = 4;
    let a = random_matrix(n, n);

    let d = bd.det(&mut a.clone());

    assert_relative_eq!(det_permutations(&a), d, epsilon = 1e-6);
}

pub fn test_determinant_dummy(bd: &impl LU<f64>) {
    let a = identity(3);
    let d = bd.det(&mut a.clone());
    println!("{d}");
    assert_relative_eq!(1., d, epsilon = 1e-6);
}

use itertools::Itertools;

/// Computes the determinant of an n×n matrix using the Leibniz formula.
/// Very slow (O(n!)), only for testing / small matrices.
pub fn det_permutations<T>(a: &DSlice<T, 2, Dense>) -> T
where
    T: ComplexFloat
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + Copy
        + Default,
{
    let (n, m) = *a.shape();
    assert_eq!(n, m, "Matrix must be square");

    let mut det = T::default();

    for perm in (0..n).permutations(n) {
        let mut sign = 1;
        for i in 0..n {
            for j in i + 1..n {
                if perm[i] > perm[j] {
                    sign = -sign;
                }
            }
        }

        let mut prod = T::one();
        for i in 0..n {
            prod = prod * a[[i, perm[i]]];
        }

        if sign == 1 {
            det = det + prod;
        } else {
            det = det - prod;
        }
    }
    det
}

pub fn random_positive_definite_matrix(n: usize) -> DTensor<f64, 2> {
    // A^T + A + nI is positive definite
    let a = random_matrix(n, n);
    let mut a_t = a.clone();
    transpose_in_place(&mut a_t);
    let mut b = a + a_t;
    for i in 0..n {
        b[[i, i]] += n as f64;
    }
    b
}

pub fn test_cholesky_reconstruction<T>(a: &DTensor<T, 2>, l: &DTensor<T, 2>)
where
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
    assert_eq!(n, m, "Matrix must be square");
    let mut ln = DTensor::<T, 2>::zeros([n, n]);
    for i in 0..n {
        for j in 0..n {
            if i >= j {
                ln[[i, j]] = l[[i, j]];
            }
        }
    }

    // Compute L^T
    let mut lt = DTensor::<T, 2>::zeros([n, m]);
    for i in 0..n {
        for j in 0..m {
            if i <= j {
                lt[[i, j]] = l[[j, i]];
            }
        }
    }

    // Compute L * L^T
    let llt = naive_matmul(&ln, &lt);

    // Verify that A = L * L^T
    for i in 0..n {
        for j in 0..m {
            let diff = f64::from(a[[i, j]]) - f64::from(llt[[i, j]]);
            assert_relative_eq!(diff, 0.0, epsilon = 1e-10);
        }
    }
}

pub fn test_cholesky_decomposition(bd: &impl LU<f64>) {
    let n = 4;
    let mut a = random_positive_definite_matrix(n);
    let original_a = a.clone();

    let l = bd.choleski(&mut a).unwrap();

    println!("{l:?}");

    test_cholesky_reconstruction(&original_a, &l);
}

pub fn test_cholesky_write(bd: &impl LU<f64>) {
    let n = 4;
    let a = random_positive_definite_matrix(n);
    let original_a = a.clone();
    let mut a_copy = a.clone();

    let l = bd.choleski(&mut a_copy.clone()).unwrap();
    bd.choleski_write(&mut a_copy).unwrap();

    println!("{l:?}");
    println!("{a_copy:?}");

    test_cholesky_reconstruction(&original_a, &a_copy);
}

pub fn test_cholesky_not_positive_definite(bd: &impl LU<f64>) {
    let n = 4;
    // Create a matrix that is not positive definite (has negative eigenvalues)
    let mut a = DTensor::<f64, 2>::zeros([n, n]);

    // Fill with a pattern that creates a non-positive definite matrix
    for i in 0..n {
        for j in 0..n {
            if i == j {
                a[[i, j]] = -1.0; // Negative diagonal elements
            } else {
                a[[i, j]] = 0.1;
            }
        }
    }

    // This should fail
    let result = bd.choleski(&mut a);
    assert!(result.is_err());
}

pub fn test_cholesky_identity_matrix(bd: &impl LU<f64>) {
    let n = 4;
    let mut a = DTensor::<f64, 2>::zeros([n, n]);

    // Create identity matrix (positive definite)
    for i in 0..n {
        a[[i, i]] = 1.0;
    }

    let l = bd.choleski(&mut a).unwrap();

    // Cholesky of identity should be identity
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(l[[i, j]], expected, epsilon = 1e-14);
        }
    }
}
