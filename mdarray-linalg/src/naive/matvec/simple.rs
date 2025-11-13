use mdarray::Layout;

use num_complex::ComplexFloat;

use mdarray::DSlice;

use crate::matmul::{Triangle, Type};

/// Performs naively A + α·x·yᵀ (or α·x·xᵀ or x·x†)
pub fn naive_outer<T: ComplexFloat, La: Layout, Lx: Layout, Ly: Layout>(
    a: &mut DSlice<T, 2, La>,
    x: &DSlice<T, 1, Lx>,
    y: &DSlice<T, 1, Ly>,
    alpha: T,
    ty: Option<Type>,
    tr: Option<Triangle>,
) {
    let m = x.shape().0;
    let n = y.shape().0;
    let (ma, na) = *a.shape();

    assert!(na == n, "Output shape must match input vector length");
    assert!(ma == m, "Output shape must match input vector length");

    match ty {
        None => {
            for i in 0..m {
                for j in 0..n {
                    a[[i, j]] = a[[i, j]] + alpha * x[[i]] * y[[j]];
                }
            }
        }
        Some(ty_) => {
            match ty_ {
                Type::Sym | Type::Tri => {
                    let tr_ = tr.unwrap_or(Triangle::Upper);
                    // Symmetric: A := α·x·xᵀ + A
                    match tr_ {
                        Triangle::Upper => {
                            for i in 0..n {
                                for j in i..n {
                                    a[[i, j]] = a[[i, j]] + alpha * x[[i]] * x[[j]];
                                }
                            }
                        }
                        Triangle::Lower => {
                            for i in 0..n {
                                for j in 0..=i {
                                    a[[i, j]] = a[[i, j]] + alpha * x[[i]] * x[[j]];
                                }
                            }
                        }
                    }
                }
                Type::Her => {
                    let tr_ = tr.unwrap_or(Triangle::Upper);
                    // Hermitian: A := α·x·x† + A (conjugate transpose)
                    match tr_ {
                        Triangle::Upper => {
                            for i in 0..n {
                                for j in i..n {
                                    a[[i, j]] = a[[i, j]] + alpha * x[[i]] * x[[j]].conj();
                                }
                            }
                        }
                        Triangle::Lower => {
                            for i in 0..n {
                                for j in 0..=i {
                                    a[[i, j]] = a[[i, j]] + alpha * x[[i]] * x[[j]].conj();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
