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

pub fn into_i32<T>(x: T) -> i32
where
    T: TryInto<i32>,
    <T as TryInto<i32>>::Error: std::fmt::Debug,
{
    x.try_into().expect("dimension must fit into i32")
}

#[macro_export]
macro_rules! get_dims {
    ( $( $matrix:expr ),+ ) => {
        (
            $(
                {
                    let shape = $matrix.shape();
                    (into_i32(shape.0), into_i32(shape.1))
                }
            ),*
        )
    };
}

// Make sure that matrix shapes are compatible with C = A * B, and return the dimensions (m, n, k)
// where C is (m x n), and k is the common dimension of A and B.
pub fn dims3(
    a_shape: &(usize, usize),
    b_shape: &(usize, usize),
    c_shape: &(usize, usize),
) -> (i32, i32, i32) {
    let (m, k) = *a_shape;
    let (k2, n) = *b_shape;
    let (m2, n2) = *c_shape;

    assert!(m == m2, "a and c must agree in number of rows");
    assert!(n == n2, "b and c must agree in number of columns");
    assert!(
        k == k2,
        "a's number of columns must be equal to b's number of rows"
    );

    (into_i32(m), into_i32(n), into_i32(k))
}

pub fn dims2(a_shape: &(usize, usize), b_shape: &(usize, usize)) -> (i32, i32) {
    let (m, k) = *a_shape;
    let (k2, n) = *b_shape;

    assert!(
        k == k2,
        "a's number of columns must be equal to b's number of rows"
    );

    (into_i32(m), into_i32(n))
}

#[macro_export]
macro_rules! trans_stride {
    ($x:expr, $same_order:expr, $other_order:expr) => {{
        if $x.stride(1) == 1 {
            ($same_order, into_i32($x.stride(0)))
        } else {
            {
                assert!($x.stride(0) == 1, stringify!($x must be contiguous in one dimension));
                ($other_order, into_i32($x.stride(1)))
            }
        }
    }};
}
