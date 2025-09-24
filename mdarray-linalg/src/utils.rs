use mdarray::{DSlice, DTensor, Layout, tensor};
use num_complex::ComplexFloat;

/// Displays a numeric mdarray in a human-readable format (NumPy-style)
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

/// Textbook implementation of matrix multiplication, useful for
/// debugging and simple tests without relying on a backend
pub fn naive_matmul<T: ComplexFloat>(a: &DSlice<T, 2>, b: &DSlice<T, 2>, c: &mut DSlice<T, 2>) {
    for (mut ci, ai) in c.rows_mut().into_iter().zip(a.rows()) {
        for (aik, bk) in ai.expr().into_iter().zip(b.rows()) {
            for (cij, bkj) in ci.expr_mut().into_iter().zip(bk) {
                *cij = (*aik) * (*bkj) + *cij;
            }
        }
    }
}

/// Safely casts a value to i32
pub fn into_i32<T>(x: T) -> i32
where
    T: TryInto<i32>,
    <T as TryInto<i32>>::Error: std::fmt::Debug,
{
    x.try_into().expect("dimension must fit into i32")
}

/// Returns the dimensions of an arbitrary number of matrices (e.g.,
/// A, B, C → (ma, na), (mb, nb), (mc, nc))
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

/// Make sure that matrix shapes are compatible with C = A * B, and
/// return the dimensions (m, n, k) safely cast to `i32`, where C is (m
/// x n), and k is the common dimension of A and B
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

/// Make sure that matrix shapes are compatible with A * B, and return
/// the dimensions (m, n) safely cast to `i32`
pub fn dims2(a_shape: &(usize, usize), b_shape: &(usize, usize)) -> (i32, i32) {
    let (m, k) = *a_shape;
    let (k2, n) = *b_shape;

    assert!(
        k == k2,
        "a's number of columns must be equal to b's number of rows"
    );

    (into_i32(m), into_i32(n))
}

/// Handles different stride layouts by selecting the correct memory
/// order and stride for contiguous arrays
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

/// Transposes a matrix in-place.
/// - For square matrices: swaps elements across the main diagonal.
/// - For rectangular matrices: reshuffles data in a temporary buffer so that the
///   same (rows, cols) slice now represents the transposed layout.
/// Dimensions stay the same, only the memory ordering changes.
pub fn transpose_in_place<T, L>(c: &mut DSlice<T, 2, L>)
where
    T: ComplexFloat + Default,
    L: Layout,
{
    let (m, n) = *c.shape();

    if n == m {
        for i in 0..m {
            for j in (i + 1)..n {
                c.swap(i * n + j, j * n + i);
            }
        }
    } else {
        let mut result = tensor![[T::default(); m]; n];
        for j in 0..n {
            for i in 0..m {
                result[j * m + i] = c[i * n + j];
            }
        }
        for j in 0..n {
            for i in 0..m {
                c[j * m + i] = result[j * m + i];
            }
        }
    }
}
