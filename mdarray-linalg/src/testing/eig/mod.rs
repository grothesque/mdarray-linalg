use approx::assert_relative_eq;
use mdarray::DTensor;
use num_complex::{Complex, ComplexFloat};

use super::common::{naive_matmul, random_matrix};
use crate::{
    assert_complex_matrix_eq, assert_matrix_eq,
    eig::{Eig, EigDecomp, SchurDecomp},
    pretty_print,
};

fn test_eigen_reconstruction<T>(
    a: &DTensor<T, 2>,
    eigenvalues: &DTensor<Complex<T::Real>, 1>,
    right_eigenvectors: &DTensor<Complex<T::Real>, 2>,
) where
    T: Default + std::fmt::Debug + ComplexFloat<Real = f64>,
{
    let (n, _) = *a.shape();

    let x = T::default();

    for i in 0..n {
        let λ = eigenvalues[i];
        let v = right_eigenvectors.view(.., i).to_owned();

        let mut av = DTensor::<_, 1>::from_elem([n], Complex::new(x.re(), x.re()));
        let mut λv = DTensor::<_, 1>::from_elem([n], Complex::new(x.re(), x.re()));

        let norm: f64 = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        assert!(norm > 1e-12, "Null vector found");

        for row in 0..n {
            let mut sum = Complex::new(x.re(), x.re());
            for col in 0..n {
                sum += Complex::new(a[[row, col]].re(), a[[row, col]].im()) * v[[col]];
            }
            av[[row]] = sum;
            λv[[row]] = λ * v[[row]];
        }
        for row in 0..n {
            let diff = av[[row]] - λv[[row]];
            assert_relative_eq!(diff.re(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(diff.im(), 0.0, epsilon = 1e-10);
        }
    }
}

pub fn test_non_square_matrix(bd: &impl Eig<f64, usize, usize>) {
    let n = 3;
    let m = 5;
    let a = random_matrix(m, n);

    let EigDecomp { .. } = bd
        .eig(&mut a.clone())
        .expect("Eigenvalue decomposition failed");
}

pub fn test_square_matrix(bd: &impl Eig<f64, usize, usize>) {
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

pub fn test_eig_cplx_square_matrix(bd: &impl Eig<Complex<f64>, usize, usize>) {
    let n = 4;
    let a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |i| {
        Complex::new((i[0] + i[1]) as f64, (i[0] * i[1]) as f64)
    });
    println!("{a:?}");
    let EigDecomp {
        eigenvalues,
        right_eigenvectors,
        ..
    } = bd
        .eig(&mut a.clone())
        .expect("Eigenvalue decomposition failed");
    println!("{eigenvalues:?}");
    println!("{right_eigenvectors:?}");

    test_eigen_reconstruction(&a, &eigenvalues, &right_eigenvectors.unwrap());
}

// #[test]
// fn eig_full() {
//     test_eig_full(&Lapack::default());
//     test_eig_full(&Faer);
// }

// fn test_eig_full(bd: &impl Eig<f64>) {
//     let n = 3;
//     let a = random_matrix(n, n);

//     let EigDecomp {
//         eigenvalues,
//         left_eigenvectors,
//         right_eigenvectors,
//     } = bd
//         .eig_full(&mut a.clone())
//         .expect("Full eigenvalue decomposition failed");

//     // Test right eigenvectors
//     test_eigen_reconstruction(&a, &eigenvalues, &right_eigenvectors.unwrap());

//     // Verify left eigenvectors are computed
//     assert!(left_eigenvectors.is_some());
// }

// #[test]
// TODO
// fn test_eig_full_reconstruction() {
//     test_eig_full_reconstruction_impl(&Lapack::default());
//     // test_eig_full_reconstruction_impl(&Faer);
// }

// fn test_eig_full_reconstruction_impl(bd: &impl Eig<Complex<f64>>) {
//     let n = 4;
//     // let mut a = random_matrix(n, n);
//     // let mut b = random_matrix(n, n);
//     let mut a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |i| {
//         Complex::new((i[0] + i[1]) as f64, (i[0] * i[1]) as f64)
//     });

//     let EigDecomp {
//         eigenvalues,
//         left_eigenvectors,
//         right_eigenvectors,
//     } = bd
//         .eig_full(&mut a.clone())
//         .expect("Full eigen decomposition failed");

//     pretty_print(&right_eigenvectors.clone().unwrap());
//     pretty_print(&left_eigenvectors.clone().unwrap());

//     test_eigen_reconstruction_full(
//         &a,
//         &eigenvalues,
//         &left_eigenvectors.unwrap(),
//         &right_eigenvectors.unwrap(),
//     );
// }

pub fn test_eigen_reconstruction_full<T>(
    a: &DTensor<T, 2>,
    eigenvalues: &DTensor<Complex<T::Real>, 2>,
    left_eigenvectors: &DTensor<Complex<T::Real>, 2>,
    right_eigenvectors: &DTensor<Complex<T::Real>, 2>,
) where
    T: Default + std::fmt::Debug + ComplexFloat<Real = f64>,
{
    let (n, _) = *a.shape();
    let x = T::default();

    for i in 0..n {
        let λ = eigenvalues[[0, i]];
        let vr = right_eigenvectors.view(.., i).to_owned();
        let vl = left_eigenvectors.view(.., i).to_owned();

        let norm_r: f64 = vr.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let norm_l: f64 = vl.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();

        assert!(norm_r > 1e-12, "Null right eigenvector found");
        assert!(norm_l > 1e-12, "Null left eigenvector found");

        // A * vr = λ * vr
        for row in 0..n {
            let mut sum = Complex::new(x.re(), x.re());
            for col in 0..n {
                sum += Complex::new(a[[row, col]].re(), a[[row, col]].im()) * vr[[col]];
            }
            let diff = sum - λ * vr[[row]];
            assert_relative_eq!(diff.re(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(diff.im(), 0.0, epsilon = 1e-10);
        }

        //  vl^H * A = λ * vl^H
        for col in 0..n {
            let mut sum = Complex::new(x.re(), x.re());
            for row in 0..n {
                sum += vl[[row]].conj() * Complex::new(a[[row, col]].re(), a[[row, col]].im());
            }
            let diff = sum - λ * vl[[col]].conj();
            assert_relative_eq!(diff.re(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(diff.im(), 0.0, epsilon = 1e-10);
        }
    }
}

pub fn test_eig_values_only(bd: &impl Eig<f64, usize, usize>) {
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
    assert_eq!(*eigenvalues.shape(), (n,));

    // Check that eigenvectors are not computed
    assert!(left_eigenvectors.is_none());
    assert!(right_eigenvectors.is_none());
}

// test on write removed as the function has been temporary removed.

// #[test]
// fn eig_write() {
//     test_eig_write(&Lapack::default());
// }

// fn test_eig_write(bd: &impl Eig<f64>) {
//     let n = 3;
//     let mut a = random_matrix(n, n);
//     let original_a = a.clone();

//     let mut eigenvalues = DTensor::<Complex<f64>, 2>::zeros([1, n]);
//     let mut right_eigenvectors_raw = DTensor::<f64, 2>::zeros([n, n]);

//     bd.eig_write::<Dense, Dense, Dense, Dense>(
//         &mut a,
//         &mut eigenvalues,
//         &mut right_eigenvectors_raw,
//     )
//     .expect("Overwrite eigenvalue decomposition failed");

//     // Reconstruct complex eigenvalues and eigenvectors from LAPACK format
//     // let mut eigenvalues = DTensor::<Complex<f64>, 2>::zeros([1, n]);
//     let mut complex_eigenvectors = DTensor::<Complex<f64>, 2>::zeros([n, n]);

//     let mut j = 0_usize;
//     while j < n {
//         let imag = eigenvalues[[0, j]].im();
//         if imag == 0.0 {
//             // Real eigenvalue: copy the real eigenvector
//             for i in 0..n {
//                 let re = right_eigenvectors_raw[[i, j]];
//                 complex_eigenvectors[[i, j]] = Complex::new(re, 0.0);
//             }
//             j += 1;
//         } else {
//             // Complex conjugate pair: reconstruct both eigenvectors
//             for i in 0..n {
//                 let re = right_eigenvectors_raw[[i, j]];
//                 let im = right_eigenvectors_raw[[i, j + 1]];
//                 complex_eigenvectors[[i, j]] = Complex::new(re, im); // v = Re + i*Im
//                 complex_eigenvectors[[i, j + 1]] = Complex::new(re, -im); // v̄ = Re - i*Im
//             }
//             j += 2;
//         }
//     }

//     test_eigen_reconstruction(&original_a, &eigenvalues, &complex_eigenvectors);
// }

pub fn test_eigh_symmetric(bd: &impl Eig<f64, usize, usize>) {
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

    let mut a_clone = a.clone();

    let EigDecomp {
        eigenvalues,
        right_eigenvectors,
        ..
    } = bd
        .eigs(&mut a_clone)
        .expect("Hermitian eigenvalue decomposition failed");

    println!("{a_clone:?}");
    println!("{right_eigenvectors:?}");
    println!("{eigenvalues:?}");

    // For symmetric real matrices, eigenvalues should be real
    for i in 0..n {
        assert_relative_eq!(eigenvalues[i].im(), 0.0, epsilon = 1e-10);
    }

    test_eigen_reconstruction(&a, &eigenvalues, &right_eigenvectors.unwrap());
}

pub fn test_eigh_complex_hermitian(bd: &impl Eig<Complex<f64>, usize, usize>) {
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
    a[[0, 0]] = Complex::new(1., 0.);

    // println!("{:?}", a);
    pretty_print(&a);

    let EigDecomp {
        eigenvalues,
        right_eigenvectors,
        ..
    } = bd
        .eigh(&mut a.clone())
        .expect("Complex Hermitian eigenvalue decomposition failed");

    pretty_print(&right_eigenvectors.clone().unwrap());
    println!("{eigenvalues:?}");

    // For Hermitian matrices, eigenvalues should be real
    for i in 0..n {
        assert_relative_eq!(eigenvalues[i].im(), 0.0, epsilon = 1e-10);
    }

    test_eigen_reconstruction(&a, &eigenvalues, &right_eigenvectors.unwrap());
}

pub fn test_eig_full_non_square(bd: &impl Eig<f64, usize, usize>) {
    let n = 3;
    let m = 5;
    let a = random_matrix(m, n);

    let EigDecomp { .. } = bd
        .eig_full(&mut a.clone())
        .expect("Full eigenvalue decomposition failed");
}

pub fn test_eig_values_non_square(bd: &impl Eig<f64, usize, usize>) {
    let n = 3;
    let m = 5;
    let a = random_matrix(m, n);

    let EigDecomp { .. } = bd
        .eig_values(&mut a.clone())
        .expect("Eigenvalues computation failed");
}

pub fn test_schur(bd: &impl Eig<f64, usize, usize>) {
    let n = 4;
    let a = random_matrix(n, n);

    let SchurDecomp { t, z } = bd
        .schur(&mut a.clone())
        .expect("Schur decomposition failed");

    assert_eq!(t.shape(), &(n, n));
    assert_eq!(z.shape(), &(n, n));

    let zt = z.transpose().to_tensor();

    println!("{a:?}");
    println!("{t:?}");
    println!("{z:?}");

    let reconstructed_tmp = naive_matmul(&z, &t);
    let a_reconstructed = naive_matmul(&reconstructed_tmp, &zt);

    assert_matrix_eq!(&a, &a_reconstructed);
}

pub fn test_schur_cplx(bd: &impl Eig<Complex<f64>, usize, usize>) {
    let n = 4;
    let a = random_matrix(n, n);
    let b = random_matrix(n, n);

    let c = DTensor::<Complex<f64>, 2>::from_fn([n, n], |i| {
        Complex::new(a[[i[0], i[1]]], b[[i[0], i[1]]])
    });

    let SchurDecomp { t, z } = bd
        .schur_complex(&mut c.clone())
        .expect("Schur decomposition failed");

    assert_eq!(t.shape(), &(n, n));
    assert_eq!(z.shape(), &(n, n));

    let mut zt = z.transpose().to_tensor();

    for i in 0..n {
        for j in 0..n {
            zt[[i, j]] = zt[[i, j]].conj();
        }
    }

    println!("{c:?}");
    println!("{t:?}");
    println!("{z:?}");

    let c_reconstructed_tmp = naive_matmul(&z, &t);
    let c_reconstructed = naive_matmul(&c_reconstructed_tmp, &zt);

    println!("---------------------------------------------");
    println!("{c:?}");
    println!("{c_reconstructed:?}");

    assert_complex_matrix_eq!(&c, &c_reconstructed);
}
