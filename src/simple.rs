/// Simple function-based interface to linear algebra libraries

use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use mdarray::{DSlice, Layout};

fn into_i32<T>(x: T) -> i32
where
    T: TryInto<i32>,
    <T as TryInto<i32>>::Error: std::fmt::Debug,
{
    x.try_into().expect("dimension must fit into i32")
}

#[inline]
pub fn gemm<La, Lb, Lc>(
    alpha: f64,
    a: &DSlice<f64, 2, La>,
    b: &DSlice<f64, 2, Lb>,
    beta: f64,
    c: &mut DSlice<f64, 2, Lc>,
) where
    La: Layout,
    Lb: Layout,
    Lc: Layout,
{
    let (m, k) = *a.shape();
    let (k2, n) = *b.shape();
    let (m2, n2) = *c.shape();

    assert!(m == m2, "a and c must agree in number of rows");
    assert!(n == n2, "b and c must agree in number of columns");
    assert!(k == k2, "a's number of columns must be equal to b's number of rows");

    let m: i32 = into_i32(m);
    let n: i32 = into_i32(n);
    let k: i32 = into_i32(k);

    let row_major = c.stride(1) == 1;
    assert!(row_major || c.stride(0) == 1, "c must be contiguous in one dimension");

    let (same_order, other_order) = if row_major {
        (CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans)
    } else {
        (CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans)
    };

    let (a_trans, a_stride) = if a.stride(1) == 1 {
        (same_order, into_i32(a.stride(0)))
    } else {
        assert!(a.stride(0) == 1, "a must be contiguous in one dimension");
        (other_order, into_i32(a.stride(1)))
    };

    let (b_trans, b_stride) = if b.stride(1) == 1 {
        (same_order, into_i32(b.stride(0)))
    } else {
        assert!(b.stride(0) == 1, "b must be contiguous in one dimension");
        (other_order, into_i32(b.stride(1)))
    };

    let c_stride = into_i32(c.stride(if row_major { 0 } else { 1 } ));

    // SAFETY: All assumptions have been verified above.
    unsafe {
        cblas_sys::cblas_dgemm(
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
