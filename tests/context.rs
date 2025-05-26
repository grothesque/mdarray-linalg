use mdarray::{Tensor, expr, expr::Expression as _};
use mdarray_linalg_prototype::{context::Blas, traits::{MatMul, MatMulBuilder}};

extern crate openblas_src;

mod common;
use common::{example_matrix, naive_matmul};

#[test]
fn test_context() {
    let a = example_matrix([2, 3]).eval();
    let b = example_matrix([3, 4]).eval();
    let c_expr = || example_matrix([2, 4]);
    let mut c = c_expr().eval();
    let mut ab = Tensor::from_elem([a.dim(0), b.dim(1)], 0.0);
    naive_matmul(&a, &b, &mut ab);
    let ab = ab;

    assert!(Blas.matmul(&a, &b).scale(3.0).to_owned() == (expr::fill(3.0) * &ab).eval());

    c.assign(c_expr());
    Blas.matmul(&a, &b).add_to(&mut c);
    assert!(c == ab.clone() + c_expr());

    c.assign(c_expr());
    Blas.matmul(&a, &b).add_to_scaled(&mut c, 2.0);
    assert!(c == ab.clone() + expr::fill(2.0) * c_expr());

    Blas.matmul(&a, &b).overwrite(&mut c);
    assert!(c == ab);
}
