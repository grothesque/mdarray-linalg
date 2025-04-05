/// Trait abstracting the four BLAS scalar types

use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use num_complex::Complex;
use paste::paste;

pub trait BlasScalar {
    unsafe fn cblas_gemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
        ldc: i32,
    );
}

// CBLAS passes real factors by value but complex factors by reference.
// This is a helper for the following macro.
macro_rules! by_value_or_by_reference {
    (s, $var:ident) => {
        $var
    };
    (d, $var:ident) => {
        $var
    };
    (c, $var:ident) => {
        & $var as *const _ as *const _
    };
    (z, $var:ident) => {
        & $var as *const _ as *const _
    };
}

macro_rules! impl_complex_float {
    ($t:ty, $prefix:ident) => {
        impl BlasScalar for $t {
            #[inline]
            unsafe fn cblas_gemm(
                layout: CBLAS_LAYOUT,
                transa: CBLAS_TRANSPOSE,
                transb: CBLAS_TRANSPOSE,
                m: i32,
                n: i32,
                k: i32,
                alpha: Self,
                a: *const Self,
                lda: i32,
                b: *const Self,
                ldb: i32,
                beta: Self,
                c: *mut Self,
                ldc: i32,
            ) {
                // SAFETY: The pointer casts are safe because Complex<T> is memory layout
                // compatible with an array [T; 2].  Other than that, this block is just
                // a transparent wrapper.
                unsafe {
                    paste! {
                        cblas_sys:: [< cblas_ $prefix gemm >](
                            layout, transa, transb,
                            m, n, k,
                            by_value_or_by_reference!($prefix, alpha),
                            a as *const _,
                            lda,
                            b as *const _,
                            ldb,
                            by_value_or_by_reference!($prefix, beta),
                            c as *mut _,
                            ldc,
                        );
                    }
                }
            }
        }
    };
}

impl_complex_float!(f32, s);
impl_complex_float!(f64, d);
impl_complex_float!(Complex<f32>, c);
impl_complex_float!(Complex<f64>, z);
