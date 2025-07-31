use mdarray::{DSlice, DTensor};
use num_complex::ComplexFloat;

pub fn pretty_print<T: ComplexFloat + std::fmt::Display>(mat: &DTensor<T, 2>)
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

pub fn naive_matmul<T: ComplexFloat>(a: &DSlice<T, 2>, b: &DSlice<T, 2>, c: &mut DSlice<T, 2>) {
    for (mut ci, ai) in c.rows_mut().into_iter().zip(a.rows()) {
        for (aik, bk) in ai.expr().into_iter().zip(b.rows()) {
            for (cij, bkj) in ci.expr_mut().into_iter().zip(bk) {
                *cij = (*aik) * (*bkj) + *cij;
            }
        }
    }
}
