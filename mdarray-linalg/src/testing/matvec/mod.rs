use num_complex::Complex;

use mdarray::{DTensor, tensor};

use crate::prelude::*;

use crate::matmul::{Triangle, Type};

pub fn test_eval_and_overwrite(bd: impl MatVec<f64>) {
    let n = 3;
    let x = DTensor::<f64, 1>::from_elem(n, 1.);
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * n + i[1] + 1) as f64);
    let y_result = bd.matvec(&a, &x).scale(2.).eval();
    let y = DTensor::<f64, 1>::from_fn([n], |i| 2. * (6. + i[0] as f64 * 9.));
    assert_eq!(y_result, y);

    let mut y_overwritten = DTensor::<f64, 1>::from_elem(n, 0.);
    bd.matvec(&a, &x).scale(2.).overwrite(&mut y_overwritten);
    assert_eq!(y_overwritten, y);
}

pub fn test_add_to_scaled(bd: impl MatVec<f64>) {
    let n = 3;
    let x = DTensor::<f64, 1>::from_elem(n, 1.);
    let mut x2 = DTensor::<f64, 1>::from_elem(n, 1.);
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[0] * 2 + i[1] + 1) as f64);
    bd.matvec(&a, &x).add_to_scaled(&mut x2, 2.);
    let y = DTensor::<f64, 1>::from_fn([n], |i| 2.0 * 1.0 + (6.0 + i[0] as f64 * 6.0));

    assert_eq!(x2, y);
}

pub fn test_add_to(bd: impl MatVec<f64>) {
    let n = 3;
    let x = DTensor::<f64, 1>::from_elem(n, 1.);
    let mut x2 = DTensor::<f64, 1>::from_elem(n, 1.);
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| (i[1] * 2 + i[0] + 1) as f64);
    bd.matvec(&a, &x).add_to(&mut x2);
    let y = DTensor::<f64, 1>::from_fn([n], |i| 10. + 3. * i[0] as f64);
    assert_eq!(x2, y);
}

pub fn test_add_outer_basic(bd: impl MatVec<f64>) {
    let m = 2;
    let n = 3;

    let x = DTensor::<f64, 1>::from_fn([m], |i| (i[0] + 1) as f64);
    let y = DTensor::<f64, 1>::from_fn([n], |i| 10f64.powi(i[0] as i32));
    let a = DTensor::<f64, 2>::from_fn([m, n], |i| if i[0] == i[1] { 1.0 } else { 0.0 });
    let beta = 2.0;
    let a_updated = bd.matvec(&a, &x).add_outer(&y, beta);

    let expected = DTensor::<f64, 2>::from_fn([m, n], |i| {
        let (row, col) = (i[0], i[1]);
        let a_val = if row == col { 1.0 } else { 0.0 };
        a_val + beta * (x[[row]]) * (y[[col]])
    });

    assert_eq!(a_updated, expected);
}

pub fn test_add_outer_sym(bd: impl MatVec<f64>) {
    let n = 3;

    let x = DTensor::<f64, 1>::from_fn([n], |i| (i[0] + 1) as f64);
    let a = DTensor::<f64, 2>::from_fn([n, n], |i| {
        let (row, col) = (i[0], i[1]);
        if row == col { 2.0 } else { 1.0 }
    });
    let beta = 0.5;

    let a_updated = bd
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

pub fn test_add_outer_her(bd: impl MatVec<Complex<f64>>) {
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

    let a_updated =
        bd.matvec(&a, &x)
            .add_outer_special(Complex64::new(beta, 0.0), Type::Her, Triangle::Upper);

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

pub fn test_add_to_scaled_vecvec(bd: impl VecOps<f64>) {
    let n = 3;
    let alpha = 2.0;
    let x = DTensor::<f64, 1>::from_fn([n], |i| (i[0] + 1) as f64); // [1., 2., 3.]
    let mut y = DTensor::<f64, 1>::from_elem(n, 1.0); // [1., 1., 1.]

    bd.add_to_scaled(alpha, &x, &mut y);

    let expected = DTensor::<f64, 1>::from_fn([n], |i| 1.0 + alpha * (i[0] + 1) as f64);
    assert_eq!(y, expected);
}

pub fn test_dot_real(bd: impl VecOps<f64>) {
    let n = 3;
    let x = DTensor::<f64, 1>::from_fn([n], |i| (i[0] + 1) as f64); // [1., 2., 3.]
    let y = DTensor::<f64, 1>::from_fn([n], |i| (2 * (i[0] + 1)) as f64); // [2., 4., 6.]

    // dot(x, y) = 1*2 + 2*4 + 3*6 = 28
    assert_eq!(bd.dot(&x, &y), 28.0);
}

pub fn test_dot_complex(bd: impl VecOps<Complex<f64>>) {
    use num_complex::Complex64;
    let n = 3;
    let x = DTensor::<Complex64, 1>::from_fn([n], |i| Complex64::new((i[0] + 1) as f64, 0.)); // [1., 2., 3.]
    let y = DTensor::<Complex64, 1>::from_fn([n], |i| Complex64::new(0., (2 * (i[0] + 1)) as f64)); // [2i, 4i, 6i]

    let expected = Complex64::new(0.0, 28.0);

    assert_eq!(bd.dot(&x, &y), expected);
}

pub fn test_dotc_complex(bd: impl VecOps<Complex<f64>>) {
    use num_complex::Complex64;

    let n = 2;
    let x = DTensor::<Complex64, 1>::from_fn([n], |i| {
        Complex64::new((i[0] + 1) as f64, (i[0] + 2) as f64)
    }); // [(1+2i), (2+3i)]
    let y = DTensor::<Complex64, 1>::from_fn([n], |i| {
        Complex64::new((i[0] + 3) as f64, (i[0] + 4) as f64)
    }); // [(3+4i), (4+5i)]

    let result = bd.dotc(&x, &y);

    println!("{:?}", result);

    // dotc(x, y) = conj(x1)*y1 + conj(x2)*y2
    let expected = x[[0]].conj() * y[[0]] + x[[1]].conj() * y[[1]];
    assert_eq!(result, expected);
}

pub fn test_norm1_complex(bd: impl VecOps<Complex<f64>>) {
    use num_complex::Complex64;

    let n = 3;
    let x = DTensor::<Complex64, 1>::from_fn([n], |i| {
        Complex64::new((i[0] + 1) as f64, (i[0] + 2) as f64)
    });
    // x = [1+2i, 2+3i, 3+4i]
    // norm1 = sum(|z_k|)
    let expected: f64 = x.iter().map(|z| z.re.abs() + z.im.abs()).sum();

    let result = bd.norm1(&x);

    println!("{}", result);
    println!("{}", expected);

    assert!((result - expected).abs() < 1e-12);
}

pub fn test_norm2_complex(bd: impl VecOps<Complex<f64>>) {
    use num_complex::Complex64;

    let n = 3;
    let x = DTensor::<Complex64, 1>::from_fn([n], |i| {
        Complex64::new((i[0] + 1) as f64, (i[0] + 2) as f64)
    });
    // x = [1+2i, 2+3i, 3+4i]
    // norm2 = sqrt(sum(|z_k|²))
    let expected: f64 = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();

    let result = bd.norm2(&x);

    assert!((result - expected).abs() < 1e-12);
}

pub fn test_argmax_real(bd: impl Argmax<f64>) {
    use mdarray::DTensor;

    // ----- Empty tensor -----
    let x = DTensor::<f64, 1>::from_fn([0], |_| 0.0);
    let idx = bd.argmax(&x);
    println!("Empty: {:?}", idx);
    assert_eq!(idx, None);

    // ----- Scalar (rank 0) -----
    let x = tensor![42.];
    let idx = bd.argmax(&x).unwrap();
    println!("Scalar: {:?}", idx);
    assert_eq!(idx, vec![0]); // Empty vec for scalar

    // ----- 1D -----
    let n = 5;
    let x = DTensor::<f64, 1>::from_fn([n], |i| (i[0] + 1) as f64);
    let idx = bd.argmax(&x.view(..)).unwrap();
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

pub fn test_argmax_overwrite_real(bd: impl Argmax<f64>) {
    let mut output = Vec::new();

    // ----- Empty tensor -----
    let x = DTensor::<f64, 1>::from_fn([0], |_| 0.0);
    let success = bd.argmax_overwrite(&x, &mut output);
    assert_eq!(success, false);
    assert_eq!(output, vec![]);

    // ----- Scalar (rank 0) -----
    let x = tensor![42.];
    let success = bd.argmax_overwrite(&x, &mut output);
    assert_eq!(success, true);
    assert_eq!(output, vec![0]);

    // ----- 1D -----
    let n = 5;
    let x = DTensor::<f64, 1>::from_fn([n], |i| (i[0] + 1) as f64);
    let success = bd.argmax_overwrite(&x.view(..), &mut output);
    assert_eq!(success, true);
    assert_eq!(output, vec![4]);

    // ----- 2D -----
    let x = DTensor::<f64, 2>::from_fn([2, 3], |i| (i[0] * 3 + i[1]) as f64);
    // [[0., 1., 2.],
    //  [3., 4., 5.]]
    let success = bd.argmax_overwrite(&x.view(.., ..).into_dyn(), &mut output);
    assert_eq!(success, true);
    assert_eq!(output, vec![1, 2]);

    // ----- 3D -----
    let x = DTensor::<f64, 3>::from_fn([2, 2, 2], |i| (i[0] * 4 + i[1] * 2 + i[2]) as f64);
    let success = bd.argmax_overwrite(&x.view(.., .., ..).into_dyn(), &mut output);
    assert_eq!(success, true);
    assert_eq!(output, vec![1, 1, 1]);

    // ----- Test reuse of output buffer -----
    output = vec![99, 99, 99];
    let x = DTensor::<f64, 1>::from_fn([3], |i| (3 - i[0]) as f64);
    let success = bd.argmax_overwrite(&x.view(..), &mut output);
    assert_eq!(success, true);
    assert_eq!(output, vec![0]); // Should be cleared and contain only result
}
