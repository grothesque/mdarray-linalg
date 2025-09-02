use mdarray::DTensor;

use mdarray_linalg::{MatVec, prelude::*};

use mdarray_linalg_blas::Blas;

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
