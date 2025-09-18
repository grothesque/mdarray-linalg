use approx::assert_relative_eq;

use crate::common::random_matrix;
use crate::{assert_complex_matrix_eq, assert_matrix_eq};
use mdarray::{DTensor, tensor};
use mdarray_linalg::{Eig, EigDecomp, naive_matmul, pretty_print};
// use mdarray_linalg_faer::eig::Faer;
use mdarray_linalg_lapack::Lapack;

use num_complex::{Complex, ComplexFloat};
use rand::Rng;

fn test_eigen_reconstruction<T>(bd: &impl Eig<T>, a: &DTensor<T, 2>, debug_print: bool)
where
    T: ComplexFloat<Real = f64>
        + Default
        + Copy
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::AddAssign,
    T::Real: std::fmt::Display,
{
    let (n, m) = (a.shape().0, a.shape().1);
    assert_eq!(n, m, "Eigen decomposition requires square matrix");

    println!("{:?}", a);

    let EigDecomp {
        eigenvalues_real,
        eigenvalues_imag,
        right_eigenvectors,
        ..
    } = bd
        .eig(&mut a.clone())
        .expect("Eigenvalue decomposition failed");

    let eigenvalues_imag = eigenvalues_imag.unwrap();
    let right_eigenvectors = right_eigenvectors.unwrap();

    assert_eq!(*eigenvalues_real.shape(), (1, n));
    assert_eq!(*right_eigenvectors.shape(), (n, n));

    if debug_print {
        println!("=== Eigenvalues (λ) ===");
        println!("{:?}", eigenvalues_real);
        // pretty_print(&);
        println!("=== Eigenvectors (V) ===");
        println!("{:?}", right_eigenvectors);
        // pretty_print(&right_eigenvectors);
    }

    println!("{:?}", *eigenvalues_real.shape());
    println!("{:?}", *right_eigenvectors.shape());

    for i in 0..n {
        // let λ = Complex::new(eigenvalues_real[[0, i]], eigenvalues_imag[[0, i]]);
        let λ = eigenvalues_real[[0, i]];
        let v = right_eigenvectors.view(.., i).to_owned();
        let mut av = DTensor::<T, 1>::zeros([n]);
        let mut λv = DTensor::<T, 1>::zeros([n]);

        // A × v
        for row in 0..n {
            let mut sum = T::default();
            for col in 0..n {
                sum = sum + a[[row, col]] * v[[col]];
            }
            av[[row]] = sum;
            λv[[row]] = λ * v[[row]];
        }

        if debug_print {
            println!("v{}:", i);
            // pretty_print(&v);
            println!("A v{}:", i);
            // pretty_print(&av);
            println!("λ v{}:", i);
            // pretty_print(&λv);
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
    test_eigen_reconstruction(bd, &a, true);
}

#[test]
fn square_matrix() {
    test_square_matrix(&Lapack::default());
    // test_eig_square_matrix(&Faer);
}

fn test_square_matrix(bd: &impl Eig<f64>) {
    let n = 2;
    let a = random_matrix(n, n);
    let a = tensor![
        [-1.3687125300798364, -5.83702909406171],
        [5.030167836010184, 5.50660879498793]
    ];
    // let a = DTensor::<f64, 2>::from_fn([n, n], |i| {
    //     (1 / (10 * i[0] + i[1] + 1) * (i[0] + 1) * (i[1] + 1)) as f64
    // });

    test_eigen_reconstruction(bd, &a, true);
}

// #[test]
// fn cplx_square_matrix() {
//     test_eig_cplx_square_matrix(&Lapack::default());
// }

// fn test_eig_cplx_square_matrix(bd: &impl Eig<Complex<f64>>) {
//     let n = 3;
//     let a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |i| {
//         Complex::new((i[0] + i[1]) as f64, (i[0] + i[1]) as f64)
//     });
//     test_eigen_reconstruction(bd, &a, true);
// }
