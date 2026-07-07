// This file is auto-generated. Do not edit manually.
//! Abstracting the BLAS scalar types
use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use num_complex::{Complex, ComplexFloat};

#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) trait BlasScalar: Sized + ComplexFloat {
    /// # Safety
    /// Calls must respect BLAS conventions.
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
    ) where
        Self: Sized,
    {
        unimplemented!("")
    }
}

impl BlasScalar for f32 {
    unsafe fn cblas_gemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    ) {
        unsafe {
            cblas_sys::cblas_sgemm(
                layout,
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                beta,
                c as *mut _,
                ldc,
            )
        }
    }
}

impl BlasScalar for f64 {
    unsafe fn cblas_gemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    ) {
        unsafe {
            cblas_sys::cblas_dgemm(
                layout,
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                beta,
                c as *mut _,
                ldc,
            )
        }
    }
}

impl BlasScalar for Complex<f32> {
    unsafe fn cblas_gemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: i32,
        b: *const Complex<f32>,
        ldb: i32,
        beta: Complex<f32>,
        c: *mut Complex<f32>,
        ldc: i32,
    ) {
        unsafe {
            cblas_sys::cblas_cgemm(
                layout,
                transa,
                transb,
                m,
                n,
                k,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                &beta as *const _ as *const _,
                c as *mut _,
                ldc,
            )
        }
    }

}

impl BlasScalar for Complex<f64> {
    unsafe fn cblas_gemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: i32,
        b: *const Complex<f64>,
        ldb: i32,
        beta: Complex<f64>,
        c: *mut Complex<f64>,
        ldc: i32,
    ) {
        unsafe {
            cblas_sys::cblas_zgemm(
                layout,
                transa,
                transb,
                m,
                n,
                k,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                &beta as *const _ as *const _,
                c as *mut _,
                ldc,
            )
        }
    }

}
