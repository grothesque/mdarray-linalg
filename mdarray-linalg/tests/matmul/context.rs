use mdarray::{Tensor, expr, expr::Expression as _};
use openblas_src as _;

use mdarray_linalg::{MatMul, prelude::*};

use mdarray_linalg_blas::Blas;

use crate::common::example_matrix;

use mdarray_linalg::naive_matmul;

fn test_backend(bd: &impl MatMul<f64>) {
    let a = example_matrix([2, 3]).eval();
    let b = example_matrix([3, 4]).eval();
    let c_expr = || example_matrix([2, 4]);
    let mut c = c_expr().eval();
    let mut ab = Tensor::from_elem([a.dim(0), b.dim(1)], 0.0);
    naive_matmul(&a, &b, &mut ab);
    let ab = ab;

    assert!(bd.matmul(&a, &b).scale(3.0).eval() == (expr::fill(3.0) * &ab).eval());

    c.assign(c_expr());
    bd.matmul(&a, &b).add_to(&mut c);
    assert!(c == ab.clone() + c_expr());

    c.assign(c_expr());
    bd.matmul(&a, &b).add_to_scaled(&mut c, 2.0);
    assert!(c == ab.clone() + expr::fill(2.0) * c_expr());

    bd.matmul(&a, &b).overwrite(&mut c);
    assert!(c == ab);
}

#[test]
fn test_backends() {
    test_backend(&Blas);
}

#[test]
fn test_backend_default() {
    let _blas = Blas::default();
}
