//! Simple function-based interface to linear algebra libraries

use std::mem::MaybeUninit;

use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use mdarray::{Array, Dense, Dim, Layout, Slice};
use mdarray_linalg::{dims3, into_i32, trans_stride};
use num_complex::ComplexFloat;

use super::scalar::BlasScalar;

pub(super) fn gemm<T, La, Lb, Lc, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    c: &mut Slice<T, (D0, D2), Lc>,
) where
    T: BlasScalar + ComplexFloat,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let (m, n, k) = dims3(*a.shape(), *b.shape(), *c.shape());

    if a.stride(0) != 1 && a.stride(1) != 1 {
        let a_owned: Array<T, (D0, D1)> = a.to_array();
        return gemm(alpha, &a_owned, b, beta, c);
    }

    if b.stride(0) != 1 && b.stride(1) != 1 {
        let b_owned: Array<T, (D1, D2)> = b.to_array();
        return gemm(alpha, a, &b_owned, beta, c);
    }

    let row_major = c.stride(1) == 1;
    assert!(
        row_major || c.stride(0) == 1,
        "c must be contiguous in one dimension"
    );

    let (same_order, other_order) = if row_major {
        (CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans)
    } else {
        (CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans)
    };
    let (a_trans, a_stride) = trans_stride!(a, same_order, other_order);
    let (b_trans, b_stride) = trans_stride!(b, same_order, other_order);

    let c_stride = into_i32(c.stride(if row_major { 0 } else { 1 }));

    unsafe {
        T::cblas_gemm(
            if row_major {
                CBLAS_LAYOUT::CblasRowMajor
            } else {
                CBLAS_LAYOUT::CblasColMajor
            },
            a_trans,
            b_trans,
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            a_stride,
            b.as_ptr(),
            b_stride,
            beta,
            c.as_mut_ptr(),
            c_stride,
        )
    }
}

pub(super) fn gemm_uninit<T, La, Lb, Lc, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    mut c: Array<MaybeUninit<T>, (D0, D2)>,
) -> Array<T, (D0, D2)>
where
    T: BlasScalar + ComplexFloat,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let (m, n, k) = dims3(*a.shape(), *b.shape(), *c.shape());

    if a.stride(0) != 1 && a.stride(1) != 1 {
        let a_owned: Array<T, (D0, D1)> = a.to_array();
        return gemm_uninit::<T, Dense, Lb, Lc, D0, D1, D2>(alpha, &a_owned, b, beta, c);
    }

    if b.stride(0) != 1 && b.stride(1) != 1 {
        let b_owned: Array<T, (D1, D2)> = b.to_array();
        return gemm_uninit::<T, La, Dense, Lc, D0, D1, D2>(alpha, a, &b_owned, beta, c);
    }

    debug_assert!(c.stride(1) == 1);

    let same_order = CBLAS_TRANSPOSE::CblasNoTrans;
    let other_order = CBLAS_TRANSPOSE::CblasTrans;

    let (a_trans, a_stride) = trans_stride!(a, same_order, other_order);
    let (b_trans, b_stride) = trans_stride!(b, same_order, other_order);

    let c_stride = into_i32(c.stride(0));

    unsafe {
        T::cblas_gemm(
            CBLAS_LAYOUT::CblasRowMajor,
            a_trans,
            b_trans,
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            a_stride,
            b.as_ptr(),
            b_stride,
            beta,
            c.as_mut_ptr() as *mut T,
            c_stride,
        );

        c.assume_init()
    }
}
