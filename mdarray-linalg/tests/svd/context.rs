use approx::assert_relative_eq;
use num_complex::Complex;

use mdarray::{DSlice, DTensor, Dense, tensor};
use mdarray_linalg::{SVD, SVDBuilder};
use mdarray_linalg_lapack::Lapack;

use num_complex::ComplexFloat;

use rand::Rng;

// fn main() {
//     let mut rng = rand::rng();
//     let (m, n) = (1000, 1000);
//     // let a = DTensor::<f64, 2>::from_fn([m, n], |i| (i[0] + i[1]) as f64);
//     let a = DTensor::<f64, 2>::from_fn([m, n], |_| rng.random::<f64>() as f64);
//     // let (m, n) = (2, 2);
//     // let a = tensor![[1., 1e-6], [0., 1.]];

//     let mut s = tensor![[0.; m.min(n)];m.min(n)];
//     let _ = Faer.svd(&mut a.clone()).overwrite_s(&mut s);
//     // pretty_print(&s);

//     let mut s = tensor![[0.; m.min(n)];m.min(n)];
//     let mut u = tensor![[0.; m];m];
//     let mut vt = tensor![[0.; n];n];

//     let start = Instant::now();
//     let _ = Lapack
//         .svd(&mut a.clone())
//         .overwrite_suvt(&mut s, &mut u, &mut vt);
//     println!("{:?}", start.elapsed());
//     panic!();

// pretty_print(&a);
// let n_iter = 10;
// let mut times = Vec::with_capacity(n_iter);
// for _ in 0..n_iter {
//     let a = DTensor::<f64, 2>::from_fn([m, n], |_| rng.random::<f64>() as f64);
//     let start = Instant::now();
//     let _ = Faer
//         .svd(&mut a.clone())
//         .overwrite_suvt(&mut s, &mut u, &mut vt);
//     times.push(start.elapsed().as_secs_f64());
// }

// println!("{:?}", times.into_iter().sum::<f64>() / (n_iter as f64));

// pretty_print(&s);
// pretty_print(&u);
// pretty_print(&vt);

//     let start = Instant::now();
//     let Ok((s, u, vt)) = Faer.svd(&mut a.clone()).eval::<Dense, Dense, Dense>() else {
//         panic!("")
//     };
//     println!("{:?}", start.elapsed());
//     pretty_print(&s);
//     pretty_print(&u);
//     pretty_print(&vt);

//     // let Ok(s) = Faer.svd(&mut a.clone()).eval_s::<Dense, Dense, Dense>() else {
//     //     panic!("")
//     // };
//     // pretty_print(&s);
// }

fn pretty_print<T: ComplexFloat + std::fmt::Display>(mat: &DTensor<T, 2>)
where
    <T as num_complex::ComplexFloat>::Real: std::fmt::Display,
{
    let shape = mat.shape();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            let v = mat[[i, j]];
            print!("{:>10.4} {:+.4}i  ", v.re(), v.im(),);
        }
        println!();
    }
    println!();
}

fn matmul<T: ComplexFloat>(a: &DSlice<T, 2>, b: &DSlice<T, 2>, c: &mut DSlice<T, 2>) {
    for (mut ci, ai) in c.rows_mut().into_iter().zip(a.rows()) {
        for (aik, bk) in ai.expr().into_iter().zip(b.rows()) {
            for (cij, bkj) in ci.expr_mut().into_iter().zip(bk) {
                *cij = (*aik) * (*bkj) + *cij;
            }
        }
    }
}

#[test]
fn test_backend_svd_square_matrix() {
    // test_svd_square_matrix(&Faer);
    test_svd_square_matrix(&Lapack);
}

fn test_svd_square_matrix(bd: &impl SVD<f64>) {
    let n = 3;
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * i[1]) as f64);

    let Ok((s, u, vt)) = bd.svd(&mut a.clone()).eval::<Dense, Dense, Dense>() else {
        panic!("SVD failed");
    };

    assert_eq!(*s.shape(), (n, n));
    assert_eq!(*u.shape(), (n, n));
    assert_eq!(*vt.shape(), (n, n));

    // U * S * Vt ≈ A
    let sigma = {
        let mut sigma = DTensor::<f64, 2>::zeros([n, n]);
        for i in 0..n {
            sigma[[i, i]] = s[[0, i]];
        }
        sigma
    };
    let mut us = tensor![[0.;n];n];
    let mut usvt = tensor![[0.;n];n];

    println!("=== Σ (Sigma) ===");
    pretty_print(&sigma);
    println!("=== U ===");
    pretty_print(&u);
    println!("=== Vᵀ ===");
    pretty_print(&vt);

    matmul(&u, &sigma, &mut us);
    println!("=== U × Σ ===");
    pretty_print(&us);
    matmul(&us, &vt, &mut usvt);
    println!("=== U × Σ × Vᵀ  ===");
    pretty_print(&usvt);
    println!("=== A original ===");
    pretty_print(&a);

    for i in 0..n {
        for j in 0..n {
            assert_relative_eq!(a[[i, j]], usvt[[i, j]], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_backend_svd_rectangular_m_gt_n() {
    test_svd_rectangular_m_gt_n(&Lapack);
    // test_svd_rectangular_m_gt_n(&Faer);
}

fn test_svd_rectangular_m_gt_n(bd: &impl SVD<f64>) {
    let (m, n) = (4, 3);
    let a = DTensor::<f64, 2>::from_fn([m, n], |i| (i[0] * i[1]) as f64);

    let Ok((s, u, vt)) = bd.svd(&mut a.clone()).eval::<Dense, Dense, Dense>() else {
        panic!("SVD failed");
    };

    assert_eq!(*s.shape(), (m, n));
    assert_eq!(*u.shape(), (m, m));
    assert_eq!(*vt.shape(), (n, n));

    // Reconstruction
    let sigma = {
        let mut sigma = DTensor::<f64, 2>::zeros([m, n]);
        for i in 0..n {
            sigma[[i, i]] = s[[0, i]];
        }
        sigma
    };

    let mut us = tensor![[0.;n];m];
    let mut usvt = tensor![[0.;n];m];

    println!("=== Σ (Sigma) ===");
    pretty_print(&sigma);
    println!("=== U ===");
    pretty_print(&u);
    println!("=== Vᵀ ===");
    pretty_print(&vt);

    matmul(&u, &sigma, &mut us);
    println!("=== U × Σ ===");
    pretty_print(&us);
    matmul(&us, &vt, &mut usvt);
    println!("=== U × Σ × Vᵀ  ===");
    pretty_print(&usvt);
    println!("=== A original ===");
    pretty_print(&a);

    for i in 0..m {
        for j in 0..n {
            assert_relative_eq!(a[[i, j]], usvt[[i, j]], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_backend_big_square_matrix() {
    test_svd_big_square_matrix(&Lapack);
    // test_svd_big_square_matrix(&Faer);
}

fn test_svd_big_square_matrix(bd: &impl SVD<f64>) {
    let n = 200;
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * i[1]) as f64);

    let Ok((s, u, vt)) = bd.svd(&mut a.clone()).eval::<Dense, Dense, Dense>() else {
        panic!("SVD failed");
    };

    assert_eq!(*s.shape(), (n, n));
    assert_eq!(*u.shape(), (n, n));
    assert_eq!(*vt.shape(), (n, n));

    // U * S * Vt ≈ A
    let sigma = {
        let mut sigma = DTensor::<f64, 2>::zeros([n, n]);
        for i in 0..n {
            sigma[[i, i]] = s[[0, i]];
        }
        sigma
    };
    let mut us = tensor![[0.;n];n];
    let mut usvt = tensor![[0.;n];n];

    matmul(&u, &sigma, &mut us);
    matmul(&us, &vt, &mut usvt);
    for i in 0..n {
        for j in 0..n {
            assert_relative_eq!(a[[i, j]], usvt[[i, j]], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_svd_cplx_square_matrix() {
    let n = 3;

    let a = DTensor::<Complex<f64>, 2>::from_fn([n, n], |i| {
        Complex::new((i[0] * i[1]) as f64, i[1] as f64)
    });

    let Ok((s, u, vt)) = Lapack.svd(&mut a.clone()).eval::<Dense, Dense, Dense>() else {
        panic!("SVD failed");
    };

    assert_eq!(*s.shape(), (n, n));
    assert_eq!(*u.shape(), (n, n));
    assert_eq!(*vt.shape(), (n, n));

    // U * S * Vt ≈ A
    let sigma = {
        let mut sigma = DTensor::<Complex<f64>, 2>::zeros([n, n]);
        for i in 0..n {
            sigma[[i, i]] = s[[0, i]];
        }
        sigma
    };
    let mut us = tensor![[Complex::new(0.,0.);n];n];
    let mut usvt = tensor![[Complex::new(0.,0.);n];n];

    println!("=== Σ (Sigma) ===");
    pretty_print(&sigma);
    println!("=== U ===");
    pretty_print(&u);
    println!("=== Vᵀ ===");
    pretty_print(&vt);

    matmul(&u, &sigma, &mut us);
    println!("=== U × Σ ===");
    pretty_print(&us);
    matmul(&us, &vt, &mut usvt);
    println!("=== U × Σ × Vᵀ  ===");
    pretty_print(&usvt);
    println!("=== A original ===");
    pretty_print(&a);

    println!("{}", Complex::new(0., 1.) * Complex::new(0., 1.));
    for i in 0..n {
        for j in 0..n {
            println!("{i} {j}");
            assert_relative_eq!(
                Complex::norm(a[[i, j]]),
                Complex::norm(usvt[[i, j]]),
                epsilon = 1e-8
            );
        }
    }
}
