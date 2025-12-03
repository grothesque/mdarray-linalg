//! Simple function-based interface to linear algebra libraries

use std::mem::MaybeUninit;

use cblas_sys::{CBLAS_DIAG, CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_TRANSPOSE, CBLAS_UPLO};
use mdarray::{DSlice, DTensor, Dim, Layout, Slice, Tensor};
use mdarray_linalg::{dims2, dims3, into_i32, trans_stride};
use num_complex::ComplexFloat;

use super::scalar::BlasScalar;

pub fn gemm<T, La, Lb, Lc, D0, D1, D2>(
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

pub fn gemm_uninit<T, La, Lb, Lc, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    mut c: Tensor<MaybeUninit<T>, (D0, D2)>,
) -> Tensor<T, (D0, D2)>
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

pub fn symm<T, La, Lb, Lc, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    c: &mut Slice<T, (D0, D2), Lc>,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
) where
    T: BlasScalar + ComplexFloat,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let (m, n, _) = dims3(*a.shape(), *b.shape(), *c.shape());

    let row_major = c.stride(1) == 1;
    assert!(
        row_major || c.stride(0) == 1,
        "c must be contiguous in one dimension"
    );

    let lda = match side {
        CBLAS_SIDE::CblasLeft => m,
        _ => n,
    };

    let ldb = n;
    let ldc = n;

    unsafe {
        T::cblas_symm(
            if row_major {
                CBLAS_LAYOUT::CblasRowMajor
            } else {
                CBLAS_LAYOUT::CblasColMajor
            },
            side,
            uplo,
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        )
    }
}

pub fn symm_uninit<T, La, Lb, Lc, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    mut c: Tensor<MaybeUninit<T>, (D0, D2)>,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
) -> Tensor<T, (D0, D2)>
where
    T: BlasScalar + ComplexFloat,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let (m, n, _) = dims3(*a.shape(), *b.shape(), *c.shape());

    debug_assert!(c.stride(1) == 1);

    let lda = match side {
        CBLAS_SIDE::CblasLeft => m,
        _ => n,
    };

    let ldb = n;
    let ldc = n;

    unsafe {
        T::cblas_symm(
            CBLAS_LAYOUT::CblasRowMajor,
            side,
            uplo,
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr() as *mut T,
            ldc,
        );

        c.assume_init()
    }
}

pub fn hemm<T, La, Lb, Lc, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    c: &mut Slice<T, (D1, D2), Lc>,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
) where
    T: BlasScalar + ComplexFloat,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let (m, n, _) = dims3(*a.shape(), *b.shape(), *c.shape());

    let row_major = c.stride(1) == 1;
    assert!(
        row_major || c.stride(0) == 1,
        "c must be contiguous in one dimension"
    );

    let lda = match side {
        CBLAS_SIDE::CblasLeft => m,
        _ => n,
    };

    let ldb = n;
    let ldc = n;

    unsafe {
        T::cblas_hemm(
            if row_major {
                CBLAS_LAYOUT::CblasRowMajor
            } else {
                CBLAS_LAYOUT::CblasColMajor
            },
            side,
            uplo,
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        )
    }
}

pub fn hemm_uninit<T, La, Lb, Lc, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    mut c: Tensor<MaybeUninit<T>, (D0, D2)>,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
) -> Tensor<T, (D0, D2)>
where
    T: BlasScalar + ComplexFloat,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let (m, n, _) = dims3(*a.shape(), *b.shape(), *c.shape());

    debug_assert!(c.stride(1) == 1);

    let lda = match side {
        CBLAS_SIDE::CblasLeft => m,
        _ => n,
    };

    let ldb = n;
    let ldc = n;

    unsafe {
        T::cblas_hemm(
            CBLAS_LAYOUT::CblasRowMajor,
            side,
            uplo,
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr() as *mut T,
            ldc,
        );

        c.assume_init()
    }
}

pub fn trmm<T, La, Lb, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &mut Slice<T, (D1, D2), Lb>,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
) where
    T: BlasScalar + ComplexFloat,
    La: Layout,
    Lb: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let (m, n) = dims2(*a.shape(), *b.shape());

    let row_major = b.stride(1) == 1;
    assert!(
        row_major || b.stride(0) == 1,
        "b must be contiguous in one dimension"
    );

    let (same_order, other_order) = if row_major {
        (CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans)
    } else {
        (CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans)
    };
    let (a_trans, a_stride) = trans_stride!(a, same_order, other_order);

    let b_stride = into_i32(b.stride(if row_major { 0 } else { 1 }));

    unsafe {
        T::cblas_trmm(
            if row_major {
                CBLAS_LAYOUT::CblasRowMajor
            } else {
                CBLAS_LAYOUT::CblasColMajor
            },
            side,
            uplo,
            a_trans,
            CBLAS_DIAG::CblasNonUnit,
            m,
            n,
            alpha,
            a.as_ptr(),
            a_stride,
            b.as_mut_ptr(),
            b_stride,
        )
    }
}

pub fn trmm_uninit<T, La, Lb, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    mut b: Tensor<MaybeUninit<T>, (D1, D2)>,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
) -> Tensor<T, (D1, D2)>
where
    T: BlasScalar + ComplexFloat,
    La: Layout,
    Lb: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let (m, n) = dims2(*a.shape(), *b.shape());

    debug_assert!(b.stride(1) == 1);

    let same_order = CBLAS_TRANSPOSE::CblasNoTrans;
    let other_order = CBLAS_TRANSPOSE::CblasTrans;

    let (a_trans, a_stride) = trans_stride!(a, same_order, other_order);

    let b_stride = into_i32(b.stride(0));

    unsafe {
        T::cblas_trmm(
            CBLAS_LAYOUT::CblasRowMajor,
            side,
            uplo,
            a_trans,
            CBLAS_DIAG::CblasNonUnit,
            m,
            n,
            alpha,
            a.as_ptr(),
            a_stride,
            b.as_mut_ptr() as *mut T,
            b_stride,
        );

        b.assume_init()
    }
}
