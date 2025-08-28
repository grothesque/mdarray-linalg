// This file is auto-generated. Do not edit manually.
//! Abstracting the BLAS scalar types
use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_SIDE, CBLAS_UPLO, CBLAS_DIAG};
use num_complex::Complex;


#[allow(clippy::too_many_arguments, unused_variables)]
pub trait BlasScalar {
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
         ldc: i32
         ) where Self: Sized {unimplemented!("")}
    unsafe fn cblas_symm(
         layout: CBLAS_LAYOUT,
         side: CBLAS_SIDE,
         uplo: CBLAS_UPLO,
         m: i32,
         n: i32,
         alpha: Self,
         a: *const Self,
         lda: i32,
         b: *const Self,
         ldb: i32,
         beta: Self,
         c: *mut Self,
         ldc: i32
         ) where Self: Sized {unimplemented!("")}
    unsafe fn cblas_trmm(
         layout: CBLAS_LAYOUT,
         side: CBLAS_SIDE,
         uplo: CBLAS_UPLO,
         transa: CBLAS_TRANSPOSE,
         diag: CBLAS_DIAG,
         m: i32,
         n: i32,
         alpha: Self,
         a: *const Self,
         lda: i32,
         b: *mut Self,
         ldb: i32
         ) where Self: Sized {unimplemented!("")}
    unsafe fn cblas_hemm(
         layout: CBLAS_LAYOUT,
         side: CBLAS_SIDE,
         uplo: CBLAS_UPLO,
         m: i32,
         n: i32,
         alpha: Self,
         a: *const Self,
         lda: i32,
         b: *const Self,
         ldb: i32,
         beta: Self,
         c: *mut Self,
         ldc: i32
         ) where Self: Sized {unimplemented!("")}
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
        ldc: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_sgemm(
                layout,
                transa,
                transb,
                m,
                n,
                k,
                alpha as f32,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                beta as f32,
                c as *mut _,
                ldc
                )
        }
    }
    
    unsafe fn cblas_symm(
        layout: CBLAS_LAYOUT,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_ssymm(
                layout,
                side,
                uplo,
                m,
                n,
                alpha as f32,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                beta as f32,
                c as *mut _,
                ldc
                )
        }
    }
    
    unsafe fn cblas_trmm(
        layout: CBLAS_LAYOUT,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *mut f32,
        ldb: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_strmm(
                layout,
                side,
                uplo,
                transa,
                diag,
                m,
                n,
                alpha as f32,
                a as *const _,
                lda,
                b as *mut _,
                ldb
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
        ldc: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_dgemm(
                layout,
                transa,
                transb,
                m,
                n,
                k,
                alpha as f64,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                beta as f64,
                c as *mut _,
                ldc
                )
        }
    }
    
    unsafe fn cblas_symm(
        layout: CBLAS_LAYOUT,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_dsymm(
                layout,
                side,
                uplo,
                m,
                n,
                alpha as f64,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                beta as f64,
                c as *mut _,
                ldc
                )
        }
    }
    
    unsafe fn cblas_trmm(
        layout: CBLAS_LAYOUT,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *mut f64,
        ldb: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_dtrmm(
                layout,
                side,
                uplo,
                transa,
                diag,
                m,
                n,
                alpha as f64,
                a as *const _,
                lda,
                b as *mut _,
                ldb
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
        ldc: i32
        ) -> () {
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
                ldc
                )
        }
    }
    
    unsafe fn cblas_symm(
        layout: CBLAS_LAYOUT,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        m: i32,
        n: i32,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: i32,
        b: *const Complex<f32>,
        ldb: i32,
        beta: Complex<f32>,
        c: *mut Complex<f32>,
        ldc: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_csymm(
                layout,
                side,
                uplo,
                m,
                n,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                &beta as *const _ as *const _,
                c as *mut _,
                ldc
                )
        }
    }
    
    unsafe fn cblas_trmm(
        layout: CBLAS_LAYOUT,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: i32,
        n: i32,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: i32,
        b: *mut Complex<f32>,
        ldb: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_ctrmm(
                layout,
                side,
                uplo,
                transa,
                diag,
                m,
                n,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                b as *mut _,
                ldb
                )
        }
    }
    
    unsafe fn cblas_hemm(
        layout: CBLAS_LAYOUT,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        m: i32,
        n: i32,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: i32,
        b: *const Complex<f32>,
        ldb: i32,
        beta: Complex<f32>,
        c: *mut Complex<f32>,
        ldc: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_chemm(
                layout,
                side,
                uplo,
                m,
                n,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                &beta as *const _ as *const _,
                c as *mut _,
                ldc
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
        ldc: i32
        ) -> () {
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
                ldc
                )
        }
    }
    
    unsafe fn cblas_symm(
        layout: CBLAS_LAYOUT,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        m: i32,
        n: i32,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: i32,
        b: *const Complex<f64>,
        ldb: i32,
        beta: Complex<f64>,
        c: *mut Complex<f64>,
        ldc: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_zsymm(
                layout,
                side,
                uplo,
                m,
                n,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                &beta as *const _ as *const _,
                c as *mut _,
                ldc
                )
        }
    }
    
    unsafe fn cblas_trmm(
        layout: CBLAS_LAYOUT,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: i32,
        n: i32,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: i32,
        b: *mut Complex<f64>,
        ldb: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_ztrmm(
                layout,
                side,
                uplo,
                transa,
                diag,
                m,
                n,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                b as *mut _,
                ldb
                )
        }
    }
    
    unsafe fn cblas_hemm(
        layout: CBLAS_LAYOUT,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        m: i32,
        n: i32,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: i32,
        b: *const Complex<f64>,
        ldb: i32,
        beta: Complex<f64>,
        c: *mut Complex<f64>,
        ldc: i32
        ) -> () {
        unsafe {
            cblas_sys::cblas_zhemm(
                layout,
                side,
                uplo,
                m,
                n,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                &beta as *const _ as *const _,
                c as *mut _,
                ldc
                )
        }
    }
    
}
