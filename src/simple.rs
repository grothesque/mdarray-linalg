//! Simple function-based interface to linear algebra libraries

use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use mdarray::{DSlice, Layout};

use crate::BlasScalar;

fn into_i32<T>(x: T) -> i32
where
    T: TryInto<i32>,
    <T as TryInto<i32>>::Error: std::fmt::Debug,
{
    x.try_into().expect("dimension must fit into i32")
}

// Make sure that matrix shapes are compatible with C = A * B, and return the dimensions (m, n, k)
// where C is (m x n), and k is the common dimension of A and B.
fn dims(
    a_shape: &(usize, usize),
    b_shape: &(usize, usize),
    c_shape: &(usize, usize),
) -> (i32, i32, i32) {
    let (m, k) = *a_shape;
    let (k2, n) = *b_shape;
    let (m2, n2) = *c_shape;

    assert!(m == m2, "a and c must agree in number of rows");
    assert!(n == n2, "b and c must agree in number of columns");
    assert!(k == k2, "a's number of columns must be equal to b's number of rows");

    (into_i32(m), into_i32(n), into_i32(k))
}

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

#[inline]
pub fn gemm<T, La, Lb, Lc>(
    alpha: T,
    a: &DSlice<T, 2, La>,
    b: &DSlice<T, 2, Lb>,
    beta: T,
    c: &mut DSlice<T, 2, Lc>,
) where
    T: BlasScalar,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
{
    let (m, n, k) = dims(a.shape(), b.shape(), c.shape());

    let row_major = c.stride(1) == 1;
    assert!(row_major || c.stride(0) == 1, "c must be contiguous in one dimension");

    let (same_order, other_order) = if row_major {
        (CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans)
    } else {
        (CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans)
    };
    let (a_trans, a_stride) = trans_stride!(a, same_order, other_order);
    let (b_trans, b_stride) = trans_stride!(b, same_order, other_order);

    let c_stride = into_i32(c.stride(if row_major { 0 } else { 1 } ));

    // SAFETY: All assumptions have been verified above.
    unsafe {
        T::cblas_gemm(
            if row_major {
                CBLAS_LAYOUT::CblasRowMajor
            } else {
                CBLAS_LAYOUT::CblasColMajor
            },
            a_trans,
            b_trans,
            m, n, k,
            alpha,
            a.as_ptr(), a_stride,
            b.as_ptr(), b_stride,
            beta,
            c.as_mut_ptr(), c_stride,
        );
    }
}
