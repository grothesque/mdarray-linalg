// This file is auto-generated. Do not edit manually.
//! Abstracting the BLAS scalar types
use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_SIDE, CBLAS_UPLO, CBLAS_DIAG};
use num_complex::{Complex, ComplexFloat};

#[allow(clippy::too_many_arguments, unused_variables)]
pub trait BlasScalar: Sized + ComplexFloat {
    
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_dotu_sub(
         n: i32,
         x: *const Self,
         incx: i32,
         y: *const Self,
         incy: i32,
         mut dotu: Self
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_dotc_sub(
         n: i32,
         x: *const Self,
         incx: i32,
         y: *const Self,
         incy: i32,
         mut dotc: Self
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_nrm2(
         n: i32,
         x: *const Self,
         incx: i32
         ) -> Self::Real where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_asum(
         n: i32,
         x: *const Self,
         incx: i32
         ) -> Self::Real where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_swap(
         n: i32,
         x: *mut Self,
         incx: i32,
         y: *mut Self,
         incy: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_copy(
         n: i32,
         x: *const Self,
         incx: i32,
         y: *mut Self,
         incy: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_axpy(
         n: i32,
         alpha: Self,
         x: *const Self,
         incx: i32,
         y: *mut Self,
         incy: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_scal(
         n: i32,
         alpha: Self,
         x: *mut Self,
         incx: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_rscal(
         n: i32,
         alpha: Self::Real,
         x: *mut Self,
         incx: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_gemv(
         layout: CBLAS_LAYOUT,
         transa: CBLAS_TRANSPOSE,
         m: i32,
         n: i32,
         alpha: Self,
         a: *const Self,
         lda: i32,
         x: *const Self,
         incx: i32,
         beta: Self,
         y: *mut Self,
         incy: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_trmv(
         layout: CBLAS_LAYOUT,
         uplo: CBLAS_UPLO,
         transa: CBLAS_TRANSPOSE,
         diag: CBLAS_DIAG,
         n: i32,
         a: *const Self,
         lda: i32,
         x: *mut Self,
         incx: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_symv(
         layout: CBLAS_LAYOUT,
         uplo: CBLAS_UPLO,
         n: i32,
         alpha: Self,
         a: *const Self,
         lda: i32,
         x: *const Self,
         incx: i32,
         beta: Self,
         y: *mut Self,
         incy: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_ger(
         layout: CBLAS_LAYOUT,
         m: i32,
         n: i32,
         alpha: Self,
         x: *const Self,
         incx: i32,
         y: *const Self,
         incy: i32,
         a: *mut Self,
         lda: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_syr(
         layout: CBLAS_LAYOUT,
         uplo: CBLAS_UPLO,
         n: i32,
         alpha: Self,
         x: *const Self,
         incx: i32,
         a: *mut Self,
         lda: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_syr2(
         layout: CBLAS_LAYOUT,
         uplo: CBLAS_UPLO,
         n: i32,
         alpha: Self,
         x: *const Self,
         incx: i32,
         y: *const Self,
         incy: i32,
         a: *mut Self,
         lda: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_geru(
         layout: CBLAS_LAYOUT,
         m: i32,
         n: i32,
         alpha: Self,
         x: *const Self,
         incx: i32,
         y: *const Self,
         incy: i32,
         a: *mut Self,
         lda: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_gerc(
         layout: CBLAS_LAYOUT,
         m: i32,
         n: i32,
         alpha: Self,
         x: *const Self,
         incx: i32,
         y: *const Self,
         incy: i32,
         a: *mut Self,
         lda: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_syrk(
         layout: CBLAS_LAYOUT,
         uplo: CBLAS_UPLO,
         trans: CBLAS_TRANSPOSE,
         n: i32,
         k: i32,
         alpha: Self,
         a: *const Self,
         lda: i32,
         beta: Self,
         c: *mut Self,
         ldc: i32
         ) where Self: Sized {unimplemented!("")}
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_syr2k(
         layout: CBLAS_LAYOUT,
         uplo: CBLAS_UPLO,
         trans: CBLAS_TRANSPOSE,
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
}

impl BlasScalar for f32 {
    
    unsafe fn cblas_nrm2(
	n: i32,
        x: *const f32,
        incx: i32
        ) -> f32 {
        unsafe {
            cblas_sys::cblas_snrm2(
		n,
                x as *const _,
                incx
                )
        }
    }
    
    unsafe fn cblas_asum(
	n: i32,
        x: *const f32,
        incx: i32
        ) -> f32 {
        unsafe {
            cblas_sys::cblas_sasum(
		n,
                x as *const _,
                incx
                )
        }
    }
    
    unsafe fn cblas_swap(
	n: i32,
        x: *mut f32,
        incx: i32,
        y: *mut f32,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_sswap(
		n,
                x as *mut _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_copy(
	n: i32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_scopy(
		n,
                x as *const _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_axpy(
	n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_saxpy(
		n,
                alpha,
                x as *const _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_scal(
	n: i32,
        alpha: f32,
        x: *mut f32,
        incx: i32
        ) {
        unsafe {
            cblas_sys::cblas_sscal(
		n,
                alpha,
                x as *mut _,
                incx
                )
        }
    }
    
    unsafe fn cblas_gemv(
	layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_sgemv(
		layout,
                transa,
                m,
                n,
                alpha,
                a as *const _,
                lda,
                x as *const _,
                incx,
                beta,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_trmv(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        n: i32,
        a: *const f32,
        lda: i32,
        x: *mut f32,
        incx: i32
        ) {
        unsafe {
            cblas_sys::cblas_strmv(
		layout,
                uplo,
                transa,
                diag,
                n,
                a as *const _,
                lda,
                x as *mut _,
                incx
                )
        }
    }
    
    unsafe fn cblas_symv(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_ssymv(
		layout,
                uplo,
                n,
                alpha,
                a as *const _,
                lda,
                x as *const _,
                incx,
                beta,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_ger(
	layout: CBLAS_LAYOUT,
        m: i32,
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
        a: *mut f32,
        lda: i32
        ) {
        unsafe {
            cblas_sys::cblas_sger(
		layout,
                m,
                n,
                alpha,
                x as *const _,
                incx,
                y as *const _,
                incy,
                a as *mut _,
                lda
                )
        }
    }
    
    unsafe fn cblas_syr(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        a: *mut f32,
        lda: i32
        ) {
        unsafe {
            cblas_sys::cblas_ssyr(
		layout,
                uplo,
                n,
                alpha,
                x as *const _,
                incx,
                a as *mut _,
                lda
                )
        }
    }
    
    unsafe fn cblas_syr2(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
        a: *mut f32,
        lda: i32
        ) {
        unsafe {
            cblas_sys::cblas_ssyr2(
		layout,
                uplo,
                n,
                alpha,
                x as *const _,
                incx,
                y as *const _,
                incy,
                a as *mut _,
                lda
                )
        }
    }
    
    unsafe fn cblas_syrk(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32
        ) {
        unsafe {
            cblas_sys::cblas_ssyrk(
		layout,
                uplo,
                trans,
                n,
                k,
                alpha,
                a as *const _,
                lda,
                beta,
                c as *mut _,
                ldc
                )
        }
    }
    
    unsafe fn cblas_syr2k(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
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
        ) {
        unsafe {
            cblas_sys::cblas_ssyr2k(
		layout,
                uplo,
                trans,
                n,
                k,
                alpha,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                beta,
                c as *mut _,
                ldc
                )
        }
    }
    
}

impl BlasScalar for f64 {
    
    unsafe fn cblas_nrm2(
	n: i32,
        x: *const f64,
        incx: i32
        ) -> f64 {
        unsafe {
            cblas_sys::cblas_dnrm2(
		n,
                x as *const _,
                incx
                )
        }
    }
    
    unsafe fn cblas_asum(
	n: i32,
        x: *const f64,
        incx: i32
        ) -> f64 {
        unsafe {
            cblas_sys::cblas_dasum(
		n,
                x as *const _,
                incx
                )
        }
    }
    
    unsafe fn cblas_swap(
	n: i32,
        x: *mut f64,
        incx: i32,
        y: *mut f64,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_dswap(
		n,
                x as *mut _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_copy(
	n: i32,
        x: *const f64,
        incx: i32,
        y: *mut f64,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_dcopy(
		n,
                x as *const _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_axpy(
	n: i32,
        alpha: f64,
        x: *const f64,
        incx: i32,
        y: *mut f64,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_daxpy(
		n,
                alpha,
                x as *const _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_scal(
	n: i32,
        alpha: f64,
        x: *mut f64,
        incx: i32
        ) {
        unsafe {
            cblas_sys::cblas_dscal(
		n,
                alpha,
                x as *mut _,
                incx
                )
        }
    }
    
    unsafe fn cblas_gemv(
	layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_dgemv(
		layout,
                transa,
                m,
                n,
                alpha,
                a as *const _,
                lda,
                x as *const _,
                incx,
                beta,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_trmv(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        n: i32,
        a: *const f64,
        lda: i32,
        x: *mut f64,
        incx: i32
        ) {
        unsafe {
            cblas_sys::cblas_dtrmv(
		layout,
                uplo,
                transa,
                diag,
                n,
                a as *const _,
                lda,
                x as *mut _,
                incx
                )
        }
    }
    
    unsafe fn cblas_symv(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_dsymv(
		layout,
                uplo,
                n,
                alpha,
                a as *const _,
                lda,
                x as *const _,
                incx,
                beta,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_ger(
	layout: CBLAS_LAYOUT,
        m: i32,
        n: i32,
        alpha: f64,
        x: *const f64,
        incx: i32,
        y: *const f64,
        incy: i32,
        a: *mut f64,
        lda: i32
        ) {
        unsafe {
            cblas_sys::cblas_dger(
		layout,
                m,
                n,
                alpha,
                x as *const _,
                incx,
                y as *const _,
                incy,
                a as *mut _,
                lda
                )
        }
    }
    
    unsafe fn cblas_syr(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        n: i32,
        alpha: f64,
        x: *const f64,
        incx: i32,
        a: *mut f64,
        lda: i32
        ) {
        unsafe {
            cblas_sys::cblas_dsyr(
		layout,
                uplo,
                n,
                alpha,
                x as *const _,
                incx,
                a as *mut _,
                lda
                )
        }
    }
    
    unsafe fn cblas_syr2(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        n: i32,
        alpha: f64,
        x: *const f64,
        incx: i32,
        y: *const f64,
        incy: i32,
        a: *mut f64,
        lda: i32
        ) {
        unsafe {
            cblas_sys::cblas_dsyr2(
		layout,
                uplo,
                n,
                alpha,
                x as *const _,
                incx,
                y as *const _,
                incy,
                a as *mut _,
                lda
                )
        }
    }
    
    unsafe fn cblas_syrk(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32
        ) {
        unsafe {
            cblas_sys::cblas_dsyrk(
		layout,
                uplo,
                trans,
                n,
                k,
                alpha,
                a as *const _,
                lda,
                beta,
                c as *mut _,
                ldc
                )
        }
    }
    
    unsafe fn cblas_syr2k(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
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
        ) {
        unsafe {
            cblas_sys::cblas_dsyr2k(
		layout,
                uplo,
                trans,
                n,
                k,
                alpha,
                a as *const _,
                lda,
                b as *const _,
                ldb,
                beta,
                c as *mut _,
                ldc
                )
        }
    }
    
}

impl BlasScalar for Complex<f32> {
    
    unsafe fn cblas_dotu_sub(
	n: i32,
        x: *const Complex<f32>,
        incx: i32,
        y: *const Complex<f32>,
        incy: i32,
        mut dotu: Complex<f32>
        ) {
        unsafe {
            cblas_sys::cblas_cdotu_sub(
		n,
                x as *const _,
                incx,
                y as *const _,
                incy,
                &mut dotu as *mut _ as *mut _
                )
        }
    }
    
    unsafe fn cblas_dotc_sub(
	n: i32,
        x: *const Complex<f32>,
        incx: i32,
        y: *const Complex<f32>,
        incy: i32,
        mut dotc: Complex<f32>
        ) {
        unsafe {
            cblas_sys::cblas_cdotc_sub(
		n,
                x as *const _,
                incx,
                y as *const _,
                incy,
                &mut dotc as *mut _ as *mut _
                )
        }
    }
    
    unsafe fn cblas_nrm2(
	n: i32,
        x: *const Complex<f32>,
        incx: i32
        ) -> f32 {
        unsafe {
            cblas_sys::cblas_scnrm2(
		n,
                x as *const _,
                incx
                )
        }
    }
    
    unsafe fn cblas_asum(
	n: i32,
        x: *const Complex<f32>,
        incx: i32
        ) -> f32 {
        unsafe {
            cblas_sys::cblas_scasum(
		n,
                x as *const _,
                incx
                )
        }
    }
    
    unsafe fn cblas_swap(
	n: i32,
        x: *mut Complex<f32>,
        incx: i32,
        y: *mut Complex<f32>,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_cswap(
		n,
                x as *mut _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_copy(
	n: i32,
        x: *const Complex<f32>,
        incx: i32,
        y: *mut Complex<f32>,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_ccopy(
		n,
                x as *const _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_axpy(
	n: i32,
        alpha: Complex<f32>,
        x: *const Complex<f32>,
        incx: i32,
        y: *mut Complex<f32>,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_caxpy(
		n,
                &alpha as *const _ as *const _,
                x as *const _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_scal(
	n: i32,
        alpha: Complex<f32>,
        x: *mut Complex<f32>,
        incx: i32
        ) {
        unsafe {
            cblas_sys::cblas_cscal(
		n,
                &alpha as *const _ as *const _,
                x as *mut _,
                incx
                )
        }
    }
    
    unsafe fn cblas_rscal(
	n: i32,
        alpha: f32,
        x: *mut Complex<f32>,
        incx: i32
        ) {
        unsafe {
            cblas_sys::cblas_csscal(
		n,
                alpha,
                x as *mut _,
                incx
                )
        }
    }
    
    unsafe fn cblas_gemv(
	layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: i32,
        x: *const Complex<f32>,
        incx: i32,
        beta: Complex<f32>,
        y: *mut Complex<f32>,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_cgemv(
		layout,
                transa,
                m,
                n,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                x as *const _,
                incx,
                &beta as *const _ as *const _,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_trmv(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        n: i32,
        a: *const Complex<f32>,
        lda: i32,
        x: *mut Complex<f32>,
        incx: i32
        ) {
        unsafe {
            cblas_sys::cblas_ctrmv(
		layout,
                uplo,
                transa,
                diag,
                n,
                a as *const _,
                lda,
                x as *mut _,
                incx
                )
        }
    }
    
    unsafe fn cblas_geru(
	layout: CBLAS_LAYOUT,
        m: i32,
        n: i32,
        alpha: Complex<f32>,
        x: *const Complex<f32>,
        incx: i32,
        y: *const Complex<f32>,
        incy: i32,
        a: *mut Complex<f32>,
        lda: i32
        ) {
        unsafe {
            cblas_sys::cblas_cgeru(
		layout,
                m,
                n,
                &alpha as *const _ as *const _,
                x as *const _,
                incx,
                y as *const _,
                incy,
                a as *mut _,
                lda
                )
        }
    }
    
    unsafe fn cblas_gerc(
	layout: CBLAS_LAYOUT,
        m: i32,
        n: i32,
        alpha: Complex<f32>,
        x: *const Complex<f32>,
        incx: i32,
        y: *const Complex<f32>,
        incy: i32,
        a: *mut Complex<f32>,
        lda: i32
        ) {
        unsafe {
            cblas_sys::cblas_cgerc(
		layout,
                m,
                n,
                &alpha as *const _ as *const _,
                x as *const _,
                incx,
                y as *const _,
                incy,
                a as *mut _,
                lda
                )
        }
    }
    
    unsafe fn cblas_syrk(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
        n: i32,
        k: i32,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: i32,
        beta: Complex<f32>,
        c: *mut Complex<f32>,
        ldc: i32
        ) {
        unsafe {
            cblas_sys::cblas_csyrk(
		layout,
                uplo,
                trans,
                n,
                k,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                &beta as *const _ as *const _,
                c as *mut _,
                ldc
                )
        }
    }
    
    unsafe fn cblas_syr2k(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
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
        ) {
        unsafe {
            cblas_sys::cblas_csyr2k(
		layout,
                uplo,
                trans,
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
    
}

impl BlasScalar for Complex<f64> {
    
    unsafe fn cblas_dotu_sub(
	n: i32,
        x: *const Complex<f64>,
        incx: i32,
        y: *const Complex<f64>,
        incy: i32,
        mut dotu: Complex<f64>
        ) {
        unsafe {
            cblas_sys::cblas_zdotu_sub(
		n,
                x as *const _,
                incx,
                y as *const _,
                incy,
                &mut dotu as *mut _ as *mut _
                )
        }
    }
    
    unsafe fn cblas_dotc_sub(
	n: i32,
        x: *const Complex<f64>,
        incx: i32,
        y: *const Complex<f64>,
        incy: i32,
        mut dotc: Complex<f64>
        ) {
        unsafe {
            cblas_sys::cblas_zdotc_sub(
		n,
                x as *const _,
                incx,
                y as *const _,
                incy,
                &mut dotc as *mut _ as *mut _
                )
        }
    }
    
    unsafe fn cblas_nrm2(
	n: i32,
        x: *const Complex<f64>,
        incx: i32
        ) -> f64 {
        unsafe {
            cblas_sys::cblas_dznrm2(
		n,
                x as *const _,
                incx
                )
        }
    }
    
    unsafe fn cblas_asum(
	n: i32,
        x: *const Complex<f64>,
        incx: i32
        ) -> f64 {
        unsafe {
            cblas_sys::cblas_dzasum(
		n,
                x as *const _,
                incx
                )
        }
    }
    
    unsafe fn cblas_swap(
	n: i32,
        x: *mut Complex<f64>,
        incx: i32,
        y: *mut Complex<f64>,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_zswap(
		n,
                x as *mut _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_copy(
	n: i32,
        x: *const Complex<f64>,
        incx: i32,
        y: *mut Complex<f64>,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_zcopy(
		n,
                x as *const _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_axpy(
	n: i32,
        alpha: Complex<f64>,
        x: *const Complex<f64>,
        incx: i32,
        y: *mut Complex<f64>,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_zaxpy(
		n,
                &alpha as *const _ as *const _,
                x as *const _,
                incx,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_scal(
	n: i32,
        alpha: Complex<f64>,
        x: *mut Complex<f64>,
        incx: i32
        ) {
        unsafe {
            cblas_sys::cblas_zscal(
		n,
                &alpha as *const _ as *const _,
                x as *mut _,
                incx
                )
        }
    }
    
    unsafe fn cblas_rscal(
	n: i32,
        alpha: f64,
        x: *mut Complex<f64>,
        incx: i32
        ) {
        unsafe {
            cblas_sys::cblas_zdscal(
		n,
                alpha,
                x as *mut _,
                incx
                )
        }
    }
    
    unsafe fn cblas_gemv(
	layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: i32,
        x: *const Complex<f64>,
        incx: i32,
        beta: Complex<f64>,
        y: *mut Complex<f64>,
        incy: i32
        ) {
        unsafe {
            cblas_sys::cblas_zgemv(
		layout,
                transa,
                m,
                n,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                x as *const _,
                incx,
                &beta as *const _ as *const _,
                y as *mut _,
                incy
                )
        }
    }
    
    unsafe fn cblas_trmv(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        n: i32,
        a: *const Complex<f64>,
        lda: i32,
        x: *mut Complex<f64>,
        incx: i32
        ) {
        unsafe {
            cblas_sys::cblas_ztrmv(
		layout,
                uplo,
                transa,
                diag,
                n,
                a as *const _,
                lda,
                x as *mut _,
                incx
                )
        }
    }
    
    unsafe fn cblas_geru(
	layout: CBLAS_LAYOUT,
        m: i32,
        n: i32,
        alpha: Complex<f64>,
        x: *const Complex<f64>,
        incx: i32,
        y: *const Complex<f64>,
        incy: i32,
        a: *mut Complex<f64>,
        lda: i32
        ) {
        unsafe {
            cblas_sys::cblas_zgeru(
		layout,
                m,
                n,
                &alpha as *const _ as *const _,
                x as *const _,
                incx,
                y as *const _,
                incy,
                a as *mut _,
                lda
                )
        }
    }
    
    unsafe fn cblas_gerc(
	layout: CBLAS_LAYOUT,
        m: i32,
        n: i32,
        alpha: Complex<f64>,
        x: *const Complex<f64>,
        incx: i32,
        y: *const Complex<f64>,
        incy: i32,
        a: *mut Complex<f64>,
        lda: i32
        ) {
        unsafe {
            cblas_sys::cblas_zgerc(
		layout,
                m,
                n,
                &alpha as *const _ as *const _,
                x as *const _,
                incx,
                y as *const _,
                incy,
                a as *mut _,
                lda
                )
        }
    }
    
    unsafe fn cblas_syrk(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
        n: i32,
        k: i32,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: i32,
        beta: Complex<f64>,
        c: *mut Complex<f64>,
        ldc: i32
        ) {
        unsafe {
            cblas_sys::cblas_zsyrk(
		layout,
                uplo,
                trans,
                n,
                k,
                &alpha as *const _ as *const _,
                a as *const _,
                lda,
                &beta as *const _ as *const _,
                c as *mut _,
                ldc
                )
        }
    }
    
    unsafe fn cblas_syr2k(
	layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
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
        ) {
        unsafe {
            cblas_sys::cblas_zsyr2k(
		layout,
                uplo,
                trans,
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
    
}
