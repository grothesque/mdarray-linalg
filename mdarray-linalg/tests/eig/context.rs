use approx::assert_relative_eq;

use crate::common::random_matrix;
use mdarray::DTensor;
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
