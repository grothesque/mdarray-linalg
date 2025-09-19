use approx::assert_relative_eq;

use crate::common::random_matrix;
use mdarray::{DTensor, Dense};
use mdarray_linalg::{Eig, EigDecomp};
// use mdarray_linalg_faer::eig::Faer;
use mdarray_linalg_lapack::Lapack;
use num_complex::{Complex, ComplexFloat};

fn test_eigen_reconstruction<T>(
    a: &DTensor<T, 2>,
    eigenvalues: &DTensor<Complex<T::Real>, 2>,
    right_eigenvectors: &DTensor<Complex<T::Real>, 2>,
) where
    T: Default + std::fmt::Debug + ComplexFloat<Real = f64>,
{
    let (n, _) = *a.shape();

    let x = T::default();

    for i in 0..n {
        let λ = eigenvalues[[0, i]];
        let v = right_eigenvectors.view(.., i).to_owned();

        let mut av = DTensor::<_, 1>::from_elem([n], Complex::new(x.re(), x.re()));
        let mut λv = DTensor::<_, 1>::from_elem([n], Complex::new(x.re(), x.re()));

        for row in 0..n {
            let mut sum = Complex::new(x.re(), x.re());
            for col in 0..n {
                sum += Complex::new(a[[row, col]].re(), x.re()) * v[[col]];
            }
            av[[row]] = sum;
            λv[[row]] = λ * v[[row]];
        }

        for row in 0..n {
            let diff = av[[row]] - λv[[row]];
            println!("{:?}", diff);
            assert_relative_eq!(diff.re(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(diff.im(), 0.0, epsilon = 1e-10);
        }
    }
}

#[test]
#[should_panic]
fn non_square_matrix() {
    test_non_square_matrix(&Lapack::default());
    // test_eig_square_matrix(&Faer);
}

fn test_non_square_matrix(bd: &impl Eig<f64>) {
    let n = 3;
    let m = 5;
    let a = random_matrix(m, n);

    let EigDecomp { .. } = bd
        .eig(&mut a.clone())
        .expect("Eigenvalue decomposition failed");
}

#[test]
fn square_matrix() {
    test_square_matrix(&Lapack::default());
    // test_eig_square_matrix(&Faer);
}

fn test_square_matrix(bd: &impl Eig<f64>) {
    let n = 2;
    let a = random_matrix(n, n);

    let EigDecomp {
        eigenvalues,
        right_eigenvectors,
        ..
    } = bd
        .eig(&mut a.clone())
        .expect("Eigenvalue decomposition failed");

    test_eigen_reconstruction(&a, &eigenvalues, &right_eigenvectors.unwrap());
}

#[test]
fn cplx_square_matrix() {
    test_eig_cplx_square_matrix(&Lapack::default());
}

fn test_eig_cplx_square_matrix(bd: &impl Eig<Complex<f64>>) {
    let n = 3;
    let a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |i| {
        Complex::new((i[0] + i[1]) as f64, (i[0] + i[1]) as f64)
    });
    let EigDecomp {
        eigenvalues,
        right_eigenvectors,
        ..
    } = bd
        .eig(&mut a.clone())
        .expect("Eigenvalue decomposition failed");

    test_eigen_reconstruction(&a, &eigenvalues, &right_eigenvectors.unwrap());
}

// Add these tests to your existing context.rs file

#[test]
fn eig_full() {
    test_eig_full(&Lapack::default());
}

fn test_eig_full(bd: &impl Eig<f64>) {
    let n = 3;
    let a = random_matrix(n, n);

    let EigDecomp {
        eigenvalues,
        left_eigenvectors,
        right_eigenvectors,
    } = bd
        .eig_full(&mut a.clone())
        .expect("Full eigenvalue decomposition failed");

    // Test right eigenvectors
    test_eigen_reconstruction(&a, &eigenvalues, &right_eigenvectors.unwrap());

    // Verify left eigenvectors are computed
    assert!(left_eigenvectors.is_some());
}

#[test]
fn eig_values_only() {
    test_eig_values_only(&Lapack::default());
}

fn test_eig_values_only(bd: &impl Eig<f64>) {
    let n = 3;
    let a = random_matrix(n, n);

    let EigDecomp {
        eigenvalues,
        left_eigenvectors,
        right_eigenvectors,
    } = bd
        .eig_values(&mut a.clone())
        .expect("Eigenvalues computation failed");

    // Check that eigenvalues are computed
    assert_eq!(*eigenvalues.shape(), (1, n));

    // Check that eigenvectors are not computed
    assert!(left_eigenvectors.is_none());
    assert!(right_eigenvectors.is_none());
}

#[test]
fn eig_overwrite() {
    test_eig_overwrite(&Lapack::default());
}

fn test_eig_overwrite(bd: &impl Eig<f64>) {
    let n = 3;
    let mut a = random_matrix(n, n);
    let original_a = a.clone();

    let mut eigenvalues_real = DTensor::<f64, 2>::zeros([1, n]);
    let mut eigenvalues_imag = DTensor::<f64, 2>::zeros([1, n]);
    let mut right_eigenvectors_raw = DTensor::<f64, 2>::zeros([n, n]);

    bd.eig_overwrite::<Dense, Dense, Dense, Dense>(
        &mut a,
        &mut eigenvalues_real,
        &mut eigenvalues_imag,
        &mut right_eigenvectors_raw,
    )
    .expect("Overwrite eigenvalue decomposition failed");

    // Reconstruct complex eigenvalues and eigenvectors from LAPACK format
    let mut eigenvalues = DTensor::<Complex<f64>, 2>::zeros([1, n]);
    let mut complex_eigenvectors = DTensor::<Complex<f64>, 2>::zeros([n, n]);

    for i in 0..n {
        eigenvalues[[0, i]] = Complex::new(eigenvalues_real[[0, i]], eigenvalues_imag[[0, i]]);
    }

    let mut j = 0_usize;
    while j < n {
        let imag = eigenvalues_imag[[0, j]];
        if imag == 0.0 {
            // Real eigenvalue: copy the real eigenvector
            for i in 0..n {
                let re = right_eigenvectors_raw[[i, j]];
                complex_eigenvectors[[i, j]] = Complex::new(re, 0.0);
            }
            j += 1;
        } else {
            // Complex conjugate pair: reconstruct both eigenvectors
            for i in 0..n {
                let re = right_eigenvectors_raw[[i, j]];
                let im = right_eigenvectors_raw[[i, j + 1]];
                complex_eigenvectors[[i, j]] = Complex::new(re, im); // v = Re + i*Im
                complex_eigenvectors[[i, j + 1]] = Complex::new(re, -im); // v̄ = Re - i*Im
            }
            j += 2;
        }
    }

    test_eigen_reconstruction(&original_a, &eigenvalues, &complex_eigenvectors);
}

#[test]
fn eigh_symmetric() {
    test_eigh_symmetric(&Lapack::default());
}

fn test_eigh_symmetric(bd: &impl Eig<f64>) {
    let n = 3;
    let mut a = random_matrix(n, n);

    // Make matrix symmetric
    for i in 0..n {
        for j in 0..n {
            let val = (a[[i, j]] + a[[j, i]]) / 2.0;
            a[[i, j]] = val;
            a[[j, i]] = val;
        }
    }

    let EigDecomp {
        eigenvalues,
        right_eigenvectors,
        ..
    } = bd
        .eigh(&mut a.clone())
        .expect("Hermitian eigenvalue decomposition failed");

    // For symmetric real matrices, eigenvalues should be real
    for i in 0..n {
        assert_relative_eq!(eigenvalues[[0, i]].im(), 0.0, epsilon = 1e-10);
    }

    test_eigen_reconstruction(&a, &eigenvalues, &right_eigenvectors.unwrap());
}

#[test]
fn eigh_complex_hermitian() {
    test_eigh_complex_hermitian(&Lapack::default());
}

fn test_eigh_complex_hermitian(bd: &impl Eig<Complex<f64>>) {
    let n = 3;
    let mut a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |i| {
        Complex::new((i[0] + i[1]) as f64, (i[0] * i[1]) as f64)
    });

    // Make matrix Hermitian
    for i in 0..n {
        for j in 0..n {
            if i == j {
                a[[i, j]] = Complex::new(a[[i, j]].re(), 0.0); // Diagonal must be real
            } else {
                a[[j, i]] = a[[i, j]].conj();
            }
        }
    }

    let EigDecomp {
        eigenvalues,
        right_eigenvectors,
        ..
    } = bd
        .eigh(&mut a.clone())
        .expect("Complex Hermitian eigenvalue decomposition failed");

    // For Hermitian matrices, eigenvalues should be real
    for i in 0..n {
        assert_relative_eq!(eigenvalues[[0, i]].im(), 0.0, epsilon = 1e-10);
    }

    test_eigen_reconstruction(&a, &eigenvalues, &right_eigenvectors.unwrap());
}

#[test]
#[should_panic]
fn eig_full_non_square() {
    test_eig_full_non_square(&Lapack::default());
}

fn test_eig_full_non_square(bd: &impl Eig<f64>) {
    let n = 3;
    let m = 5;
    let a = random_matrix(m, n);

    let EigDecomp { .. } = bd
        .eig_full(&mut a.clone())
        .expect("Full eigenvalue decomposition failed");
}

#[test]
#[should_panic]
fn eig_values_non_square() {
    test_eig_values_non_square(&Lapack::default());
}

fn test_eig_values_non_square(bd: &impl Eig<f64>) {
    let n = 3;
    let m = 5;
    let a = random_matrix(m, n);

    let EigDecomp { .. } = bd
        .eig_values(&mut a.clone())
        .expect("Eigenvalues computation failed");
}
