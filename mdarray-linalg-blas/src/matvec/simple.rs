use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_UPLO};
use mdarray::{DSlice, Layout, Shape, Slice};
use mdarray_linalg::{into_i32, trans_stride};
use num_complex::ComplexFloat;

use num_complex::Complex;
use std::any::TypeId;

use super::scalar::BlasScalar;

pub fn gemv<T, La, Lx, Ly>(
    alpha: T,
    a: &DSlice<T, 2, La>,
    x: &DSlice<T, 1, Lx>,
    beta: T,
    y: &mut DSlice<T, 1, Ly>,
) where
    T: BlasScalar + ComplexFloat,
    La: Layout,
    Lx: Layout,
    Ly: Layout,
{
    let (m, n) = *a.shape();

    if a.stride(1) == 1 {
        assert_eq!(x.len(), n, "x length must match number of columns in a");
    } else {
        assert_eq!(x.len(), m, "x length must match number of rows in a");
    }

    assert_eq!(
        y.len(),
        if a.stride(1) == 1 { m } else { n },
        "y length must match the output dimension"
    );

    let row_major = a.stride(1) == 1;
    assert!(
        row_major || a.stride(0) == 1,
        "a must be contiguous in one dimension"
    );

    let (same_order, other_order) = if row_major {
        (CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans)
    } else {
        (CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans)
    };
    let (a_trans, a_stride) = trans_stride!(a, same_order, other_order);

    let x_inc = into_i32(x.stride(0));
    let y_inc = into_i32(y.stride(0));

    unsafe {
        T::cblas_gemv(
            if row_major {
                CBLAS_LAYOUT::CblasRowMajor
            } else {
                CBLAS_LAYOUT::CblasColMajor
            },
            a_trans,
            into_i32(m),
            into_i32(n),
            alpha,
            a.as_ptr(),
            a_stride,
            x.as_ptr(),
            x_inc,
            beta,
            y.as_mut_ptr(),
            y_inc,
        )
    }
}

pub fn ger<T, La, Lx, Ly>(
    beta: T,
    x: &DSlice<T, 1, Lx>,
    y: &DSlice<T, 1, Ly>,
    a: &mut DSlice<T, 2, La>,
) where
    T: BlasScalar + ComplexFloat,
    La: Layout,
    Lx: Layout,
    Ly: Layout,
{
    let (m, n) = *a.shape();

    assert_eq!(x.len(), m, "x length must match number of rows in a");
    assert_eq!(y.len(), n, "y length must match number of columns in a");

    let x_inc = into_i32(x.stride(0));
    let y_inc = into_i32(y.stride(0));

    let row_major = a.stride(1) == 1;
    assert!(
        row_major || a.stride(0) == 1,
        "a must be contiguous in one dimension"
    );

    let lda = if row_major { into_i32(n) } else { into_i32(m) };

    unsafe {
        T::cblas_ger(
            if row_major {
                CBLAS_LAYOUT::CblasRowMajor
            } else {
                CBLAS_LAYOUT::CblasColMajor
            },
            into_i32(m),
            into_i32(n),
            beta,
            x.as_ptr(),
            x_inc,
            y.as_ptr(),
            y_inc,
            a.as_mut_ptr(),
            lda,
        )
    }
}

pub fn scal<T, Lx>(alpha: T, x: &mut DSlice<T, 1, Lx>)
where
    T: BlasScalar + ComplexFloat,
    Lx: Layout,
{
    let n = into_i32(x.len());
    let incx = into_i32(x.stride(0));

    unsafe { T::cblas_scal(n, alpha, x.as_mut_ptr(), incx) }
}

pub fn syr<T, Lx, La>(uplo: CBLAS_UPLO, alpha: T, x: &DSlice<T, 1, Lx>, a: &mut DSlice<T, 2, La>)
where
    T: BlasScalar + ComplexFloat,
    Lx: Layout,
    La: Layout,
{
    let (m, n) = *a.shape();
    assert_eq!(m, n, "Matrix a must be square for symmetric update");
    assert_eq!(x.len(), n, "x length must match matrix dimension");

    let row_major = a.stride(1) == 1;
    assert!(
        row_major || a.stride(0) == 1,
        "a must be contiguous in one dimension"
    );

    let x_inc = into_i32(x.stride(0));
    let lda = if row_major { into_i32(n) } else { into_i32(m) };

    unsafe {
        T::cblas_syr(
            if row_major {
                CBLAS_LAYOUT::CblasRowMajor
            } else {
                CBLAS_LAYOUT::CblasColMajor
            },
            uplo,
            into_i32(n),
            alpha,
            x.as_ptr(),
            x_inc,
            a.as_mut_ptr(),
            lda,
        )
    }
}

pub fn her<T, Lx, La>(
    uplo: CBLAS_UPLO,
    alpha: T::Real,
    x: &DSlice<T, 1, Lx>,
    a: &mut DSlice<T, 2, La>,
) where
    T: BlasScalar + ComplexFloat,
    Lx: Layout,
    La: Layout,
{
    let (m, n) = *a.shape();
    assert_eq!(m, n, "Matrix a must be square for hermitian update");
    assert_eq!(x.len(), n, "x length must match matrix dimension");

    let row_major = a.stride(1) == 1;
    assert!(
        row_major || a.stride(0) == 1,
        "a must be contiguous in one dimension"
    );

    let x_inc = into_i32(x.stride(0));
    let lda = if row_major { into_i32(n) } else { into_i32(m) };

    unsafe {
        T::cblas_her(
            if row_major {
                CBLAS_LAYOUT::CblasRowMajor
            } else {
                CBLAS_LAYOUT::CblasColMajor
            },
            uplo,
            into_i32(n),
            alpha,
            x.as_ptr(),
            x_inc,
            a.as_mut_ptr(),
            lda,
        )
    }
}

pub fn asum<T, Lx>(x: &DSlice<T, 1, Lx>) -> T::Real
where
    T: BlasScalar + ComplexFloat,
    Lx: Layout,
{
    let n = into_i32(x.len());
    let incx = into_i32(x.stride(0));

    unsafe { T::cblas_asum(n, x.as_ptr(), incx) }
}

pub fn axpy<T, Lx, Ly>(alpha: T, x: &DSlice<T, 1, Lx>, y: &mut DSlice<T, 1, Ly>)
where
    T: BlasScalar + ComplexFloat,
    Lx: Layout,
    Ly: Layout,
{
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = into_i32(x.len());
    let incx = into_i32(x.stride(0));
    let incy = into_i32(y.stride(0));

    unsafe { T::cblas_axpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy) }
}

pub fn nrm2<T, Lx>(x: &DSlice<T, 1, Lx>) -> T::Real
where
    T: BlasScalar + ComplexFloat,
    Lx: Layout,
{
    let n = into_i32(x.len());
    let incx = into_i32(x.stride(0));

    unsafe { T::cblas_nrm2(n, x.as_ptr(), incx) }
}

pub fn dotu<T, Lx, Ly>(x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T
where
    T: BlasScalar + ComplexFloat + 'static,
    Lx: Layout,
    Ly: Layout,
{
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = into_i32(x.len());
    let incx = into_i32(x.stride(0));
    let incy = into_i32(y.stride(0));

    let mut result = T::zero();

    if TypeId::of::<T>() == TypeId::of::<Complex<f32>>()
        || TypeId::of::<T>() == TypeId::of::<Complex<f64>>()
    {
        unsafe {
            T::cblas_dotu_sub(n, x.as_ptr(), incx, y.as_ptr(), incy, &mut result);
        }
    } else {
        unsafe {
            result = T::cblas_dot(n, x.as_ptr(), incx, y.as_ptr(), incy);
        }
    }

    result
}

pub fn dotc<T, Lx, Ly>(x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T
where
    T: BlasScalar + ComplexFloat + 'static,
    Lx: Layout,
    Ly: Layout,
{
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = into_i32(x.len());
    let incx = into_i32(x.stride(0));
    let incy = into_i32(y.stride(0));

    let mut result = T::zero();

    if TypeId::of::<T>() == TypeId::of::<Complex<f32>>()
        || TypeId::of::<T>() == TypeId::of::<Complex<f64>>()
    {
        unsafe {
            T::cblas_dotc_sub(n, x.as_ptr(), incx, y.as_ptr(), incy, &mut result);
        }
    } else {
        unsafe {
            result = T::cblas_dot(n, x.as_ptr(), incx, y.as_ptr(), incy);
        }
    }

    result
}

pub fn amax<T, S, L>(x: &Slice<T, S, L>) -> usize
where
    T: BlasScalar + ComplexFloat + 'static,
    S: Shape,
    L: Layout,
{
    assert!(!x.is_empty(), "Cannot find amax of empty slice");

    let n = into_i32(x.len());
    let incx = if x.rank() == 1 {
        into_i32(x.stride(0))
    } else {
        1
    };

    (unsafe { T::cblas_amax(n, x.as_ptr(), incx) } as usize)
}
