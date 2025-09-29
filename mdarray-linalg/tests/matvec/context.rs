use mdarray::{DTensor, tensor};

use mdarray_linalg::prelude::*;

use mdarray_linalg_blas::Blas;
use mdarray_linalg_naive::Naive;

#[test]
fn eval_and_overwrite() {
    let n = 3;
    let x = DTensor::<f64, 1>::from_elem(n, 1.);
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * n + i[1] + 1) as f64);
    let y_result = Blas.matvec(&a, &x).scale(2.).eval();
    let y = DTensor::<f64, 1>::from_fn([n], |i| 2. * (6. + i[0] as f64 * 9.));
    assert_eq!(y_result, y);

    let mut y_overwritten = DTensor::<f64, 1>::from_elem(n, 0.);
    Blas.matvec(&a, &x).scale(2.).overwrite(&mut y_overwritten);
    assert_eq!(y_overwritten, y);
}

#[test]
fn add_to_scaled() {
    let n = 3;
    let x = DTensor::<f64, 1>::from_elem(n, 1.);
    let mut x2 = DTensor::<f64, 1>::from_elem(n, 1.);
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * 2 + i[1] + 1) as f64);
    Blas.matvec(&a, &x).add_to_scaled(&mut x2, 2.);
    let y = DTensor::<f64, 1>::from_fn([n], |i| 2.0 * 1.0 + (6.0 + i[0] as f64 * 6.0));

    assert_eq!(x2, y);
}

#[test]
fn add_to() {
    let n = 3;
    let x = DTensor::<f64, 1>::from_elem(n, 1.);
    let mut x2 = DTensor::<f64, 1>::from_elem(n, 1.);
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[1] * 2 + i[0] + 1) as f64);
    Blas.matvec(&a, &x).add_to(&mut x2);
    let y = DTensor::<f64, 1>::from_fn([n], |i| 10. + 3. * i[0] as f64);
    assert_eq!(x2, y);
}

#[test]
fn add_outer_basic() {
    let m = 2;
    let n = 3;

    let x = DTensor::<f64, 1>::from_fn([m], |i| (i[0] + 1) as f64);
    let y = DTensor::<f64, 1>::from_fn([n], |i| 10f64.powi(i[0] as i32));
    let a = DTensor::<f64, 2>::from_fn([m, n], |i| if i[0] == i[1] { 1.0 } else { 0.0 });
    let beta = 2.0;
    let a_updated = Blas.matvec(&a, &x).add_outer(&y, beta);

    let expected = DTensor::<f64, 2>::from_fn([m, n], |i| {
        let (row, col) = (i[0], i[1]);
        let a_val = if row == col { 1.0 } else { 0.0 };
        a_val + beta * (x[[row]]) * (y[[col]])
    });

    assert_eq!(a_updated, expected);
}

#[test]
fn add_outer_sym() {
    let n = 3;

    let x = DTensor::<f64, 1>::from_fn([n], |i| (i[0] + 1) as f64);
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| {
        let (row, col) = (i[0], i[1]);
        if row == col { 2.0 } else { 1.0 }
    });
    let beta = 0.5;

    let a_updated = Blas
        .matvec(&a, &x)
        .add_outer_special(beta, Type::Sym, Triangle::Upper);

    let expected = DTensor::<f64, 2>::from_fn([n, n], |i| {
        let (row, col) = (i[0], i[1]);
        let a_val = if row == col { 2.0 } else { 1.0 };
        if row <= col {
            a_val + beta * (x[[row]]) * (x[[col]])
        } else {
            a_val
        }
    });

    assert_eq!(a_updated, expected);
}

#[test]
fn add_outer_her() {
    use num_complex::Complex64;

    let n = 3;

    let x = DTensor::<Complex64, 1>::from_fn([n], |i| {
        Complex64::new((i[0] + 1) as f64, (i[0] as f64) * 0.5)
    });

    let a = DTensor::<Complex64, 2>::from_fn([n, n], |i| {
        let (row, col) = (i[0], i[1]);
        if row == col {
            Complex64::new(2.0, 0.0)
        } else if row < col {
            Complex64::new(1.0, 0.5)
        } else {
            Complex64::new(1.0, -0.5)
        }
    });
    let beta = 0.3;

    let a_updated = Blas.matvec(&a, &x).add_outer_special(
        Complex64::new(beta, 0.0),
        Type::Her,
        Triangle::Upper,
    );

    let expected = DTensor::<Complex64, 2>::from_fn([n, n], |i| {
        let (row, col) = (i[0], i[1]);
        let a_val = if row == col {
            Complex64::new(2.0, 0.0)
        } else if row < col {
            Complex64::new(1.0, 0.5)
        } else {
            Complex64::new(1.0, -0.5)
        };

        if row <= col {
            a_val + Complex64::new(beta, 0.0) * x[[row]] * x[[col]].conj()
        } else {
            a_val
        }
    });

    assert_eq!(a_updated, expected);
}

#[test]
fn add_to_scaled_vecvec() {
    let n = 3;
    let alpha = 2.0;
    let x = DTensor::<f64, 1>::from_fn([n], |i| (i[0] + 1) as f64); // [1., 2., 3.]
    let mut y = DTensor::<f64, 1>::from_elem(n, 1.0); // [1., 1., 1.]

    Blas.add_to_scaled(alpha, &x, &mut y);

    let expected = DTensor::<f64, 1>::from_fn([n], |i| 1.0 + alpha * (i[0] + 1) as f64);
    assert_eq!(y, expected);
}

#[test]
fn dot_real() {
    let n = 3;
    let x = DTensor::<f64, 1>::from_fn([n], |i| (i[0] + 1) as f64); // [1., 2., 3.]
    let y = DTensor::<f64, 1>::from_fn([n], |i| (2 * (i[0] + 1)) as f64); // [2., 4., 6.]

    let result = Blas.dot(&x, &y);

    // dot(x, y) = 1*2 + 2*4 + 3*6 = 28
    assert_eq!(result, 28.0);
}

#[test]
fn dot_complex() {
    use num_complex::Complex64;
    let n = 3;
    let x = DTensor::<Complex64, 1>::from_fn([n], |i| Complex64::new((i[0] + 1) as f64, 0.)); // [1., 2., 3.]
    let y = DTensor::<Complex64, 1>::from_fn([n], |i| Complex64::new(0., (2 * (i[0] + 1)) as f64)); // [2i, 4i, 6i]

    let result = Blas.dot(&x, &y);
    let expected = Complex64::new(0.0, 28.0);

    assert_eq!(result, expected);
}

#[test]
fn dotc_complex() {
    use num_complex::Complex64;

    let n = 2;
    let x = DTensor::<Complex64, 1>::from_fn([n], |i| {
        Complex64::new((i[0] + 1) as f64, (i[0] + 2) as f64)
    }); // [(1+2i), (2+3i)]
    let y = DTensor::<Complex64, 1>::from_fn([n], |i| {
        Complex64::new((i[0] + 3) as f64, (i[0] + 4) as f64)
    }); // [(3+4i), (4+5i)]

    let result = Blas.dotc(&x, &y);

    println!("{:?}", result);

    // dotc(x, y) = conj(x1)*y1 + conj(x2)*y2
    let expected = x[[0]].conj() * y[[0]] + x[[1]].conj() * y[[1]];
    assert_eq!(result, expected);
}

#[test]
fn norm1_complex() {
    use num_complex::Complex64;

    let n = 3;
    let x = DTensor::<Complex64, 1>::from_fn([n], |i| {
        Complex64::new((i[0] + 1) as f64, (i[0] + 2) as f64)
    });
    // x = [1+2i, 2+3i, 3+4i]
    // norm1 = sum(|z_k|)
    let expected: f64 = x.iter().map(|z| z.re.abs() + z.im.abs()).sum();

    let result = Blas.norm1(&x);

    println!("{}", result);
    println!("{}", expected);

    assert!((result - expected).abs() < 1e-12);
}

#[test]
fn norm2_complex() {
    use num_complex::Complex64;

    let n = 3;
    let x = DTensor::<Complex64, 1>::from_fn([n], |i| {
        Complex64::new((i[0] + 1) as f64, (i[0] + 2) as f64)
    });
    // x = [1+2i, 2+3i, 3+4i]
    // norm2 = sqrt(sum(|z_k|²))
    let expected: f64 = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();

    let result = Blas.norm2(&x);

    assert!((result - expected).abs() < 1e-12);
}

#[test]
fn argmax_real() {
    use mdarray::DTensor;

    let bd = Naive;

    // ----- Empty tensor -----
    let x = DTensor::<f64, 1>::from_fn([0], |_| 0.0);
    let idx = bd.argmax(&x.view(..).into_dyn());
    println!("Empty: {:?}", idx);
    assert_eq!(idx, None);

    // ----- Scalar (rank 0) -----
    let x = tensor![42.];
    let idx = bd.argmax(&x.view(..)).unwrap();
    println!("Scalar: {:?}", idx);
    assert_eq!(idx, vec![0]); // Empty vec for scalar

    // ----- 1D -----
    let n = 5;
    let x = DTensor::<f64, 1>::from_fn([n], |i| (i[0] + 1) as f64); // [1., 2., 3., 4., 5.]
    let idx = bd.argmax(&x.view(..).into_dyn()).unwrap();
    println!("{:?}", idx);
    assert_eq!(idx, vec![4]);

    // ----- 2D -----
    let x = DTensor::<f64, 2>::from_fn([2, 3], |i| (i[0] * 3 + i[1]) as f64);

    // [[0., 1., 2.],
    //  [3., 4., 5.]]
    let idx = bd.argmax(&x.view(.., ..).into_dyn()).unwrap();
    println!("{:?}", idx);
    assert_eq!(idx, vec![1, 2]);

    // ----- 3D -----
    let x = DTensor::<f64, 3>::from_fn([2, 2, 2], |i| (i[0] * 4 + i[1] * 2 + i[2]) as f64);
    let idx = bd.argmax(&x.view(.., .., ..).into_dyn()).unwrap();
    println!("{:?}", idx);
    assert_eq!(idx, vec![1, 1, 1]);
}
