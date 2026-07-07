//! Abstracting the BLAS scalar types
use cblas_sys::{CBLAS_INDEX, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use num_complex::{Complex, ComplexFloat};

#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) trait BlasScalar: Sized + ComplexFloat {
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_dotu_or_dot(
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
    ) -> Self;

    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_dotc_or_dot(
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
    ) -> Self;

    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_amax(n: i32, x: *const Self, incx: i32) -> CBLAS_INDEX
    where
        Self: Sized,
    {
        unimplemented!("")
    }

    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_nrm2(n: i32, x: *const Self, incx: i32) -> Self::Real
    where
        Self: Sized,
    {
        unimplemented!("")
    }
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_asum(n: i32, x: *const Self, incx: i32) -> Self::Real
    where
        Self: Sized,
    {
        unimplemented!("")
    }
    /// # Safety
    /// Calls must respect BLAS conventions.
    unsafe fn cblas_axpy(n: i32, alpha: Self, x: *const Self, incx: i32, y: *mut Self, incy: i32)
    where
        Self: Sized,
    {
        unimplemented!("")
    }
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
        incy: i32,
    ) where
        Self: Sized,
    {
        unimplemented!("")
    }
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
        lda: i32,
    ) where
        Self: Sized,
    {
        unimplemented!("")
    }
}

impl BlasScalar for f32 {
    unsafe fn cblas_dotu_or_dot(
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
    ) -> Self {
        unsafe { cblas_sys::cblas_sdot(n, x, incx, y, incy) }
    }

    unsafe fn cblas_dotc_or_dot(
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
    ) -> Self {
        unsafe { cblas_sys::cblas_sdot(n, x, incx, y, incy) }
    }

    unsafe fn cblas_amax(n: i32, x: *const f32, incx: i32) -> CBLAS_INDEX {
        unsafe { cblas_sys::cblas_isamax(n, x as *const _, incx) }
    }


    unsafe fn cblas_nrm2(n: i32, x: *const f32, incx: i32) -> f32 {
        unsafe { cblas_sys::cblas_snrm2(n, x as *const _, incx) }
    }

    unsafe fn cblas_asum(n: i32, x: *const f32, incx: i32) -> f32 {
        unsafe { cblas_sys::cblas_sasum(n, x as *const _, incx) }
    }



    unsafe fn cblas_axpy(n: i32, alpha: f32, x: *const f32, incx: i32, y: *mut f32, incy: i32) {
        unsafe { cblas_sys::cblas_saxpy(n, alpha, x as *const _, incx, y as *mut _, incy) }
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
        incy: i32,
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
                incy,
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
        lda: i32,
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
                lda,
            )
        }
    }

}

impl BlasScalar for f64 {
    unsafe fn cblas_dotu_or_dot(
        n: i32,
        x: *const f64,
        incx: i32,
        y: *const f64,
        incy: i32,
    ) -> Self {
        unsafe { cblas_sys::cblas_ddot(n, x, incx, y, incy) }
    }

    unsafe fn cblas_dotc_or_dot(
        n: i32,
        x: *const f64,
        incx: i32,
        y: *const f64,
        incy: i32,
    ) -> Self {
        unsafe { cblas_sys::cblas_ddot(n, x, incx, y, incy) }
    }

    unsafe fn cblas_amax(n: i32, x: *const f64, incx: i32) -> CBLAS_INDEX {
        unsafe { cblas_sys::cblas_idamax(n, x as *const _, incx) }
    }


    unsafe fn cblas_nrm2(n: i32, x: *const f64, incx: i32) -> f64 {
        unsafe { cblas_sys::cblas_dnrm2(n, x as *const _, incx) }
    }

    unsafe fn cblas_asum(n: i32, x: *const f64, incx: i32) -> f64 {
        unsafe { cblas_sys::cblas_dasum(n, x as *const _, incx) }
    }



    unsafe fn cblas_axpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32) {
        unsafe { cblas_sys::cblas_daxpy(n, alpha, x as *const _, incx, y as *mut _, incy) }
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
        incy: i32,
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
                incy,
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
        lda: i32,
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
                lda,
            )
        }
    }

}
use num_traits::Zero;
impl BlasScalar for Complex<f32> {
    unsafe fn cblas_dotu_or_dot(
        n: i32,
        x: *const Complex<f32>,
        incx: i32,
        y: *const Complex<f32>,
        incy: i32,
    ) -> Self {
        let mut result = Self::zero();
        unsafe {
            cblas_sys::cblas_cdotu_sub(
                n,
                x as *const _,
                incx,
                y as *const _,
                incy,
                &mut result as *mut Complex<f32> as *mut [f32; 2],
            );
        }
        result
    }

    unsafe fn cblas_dotc_or_dot(
        n: i32,
        x: *const Complex<f32>,
        incx: i32,
        y: *const Complex<f32>,
        incy: i32,
    ) -> Self {
        let mut result = Self::zero();
        unsafe {
            cblas_sys::cblas_cdotc_sub(
                n,
                x as *const _,
                incx,
                y as *const _,
                incy,
                &mut result as *mut Complex<f32> as *mut [f32; 2],
            );
        }
        result
    }

    unsafe fn cblas_amax(n: i32, x: *const Complex<f32>, incx: i32) -> CBLAS_INDEX {
        unsafe { cblas_sys::cblas_icamax(n, x as *const _, incx) }
    }



    unsafe fn cblas_nrm2(n: i32, x: *const Complex<f32>, incx: i32) -> f32 {
        unsafe { cblas_sys::cblas_scnrm2(n, x as *const _, incx) }
    }

    unsafe fn cblas_asum(n: i32, x: *const Complex<f32>, incx: i32) -> f32 {
        unsafe { cblas_sys::cblas_scasum(n, x as *const _, incx) }
    }



    unsafe fn cblas_axpy(
        n: i32,
        alpha: Complex<f32>,
        x: *const Complex<f32>,
        incx: i32,
        y: *mut Complex<f32>,
        incy: i32,
    ) {
        unsafe {
            cblas_sys::cblas_caxpy(
                n,
                &alpha as *const _ as *const _,
                x as *const _,
                incx,
                y as *mut _,
                incy,
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
        incy: i32,
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
                incy,
            )
        }
    }


    unsafe fn cblas_ger(
        layout: CBLAS_LAYOUT,
        m: i32,
        n: i32,
        alpha: Complex<f32>,
        x: *const Complex<f32>,
        incx: i32,
        y: *const Complex<f32>,
        incy: i32,
        a: *mut Complex<f32>,
        lda: i32,
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
                lda,
            )
        }
    }

}

impl BlasScalar for Complex<f64> {
    unsafe fn cblas_dotu_or_dot(
        n: i32,
        x: *const Complex<f64>,
        incx: i32,
        y: *const Complex<f64>,
        incy: i32,
    ) -> Self {
        let mut result = Self::zero();
        unsafe {
            cblas_sys::cblas_zdotu_sub(
                n,
                x as *const _,
                incx,
                y as *const _,
                incy,
                &mut result as *mut Complex<f64> as *mut [f64; 2],
            );
        }
        result
    }

    unsafe fn cblas_dotc_or_dot(
        n: i32,
        x: *const Complex<f64>,
        incx: i32,
        y: *const Complex<f64>,
        incy: i32,
    ) -> Self {
        let mut result = Self::zero();
        unsafe {
            cblas_sys::cblas_zdotc_sub(
                n,
                x as *const _,
                incx,
                y as *const _,
                incy,
                &mut result as *mut Complex<f64> as *mut [f64; 2],
            );
        }
        result
    }

    unsafe fn cblas_amax(n: i32, x: *const Complex<f64>, incx: i32) -> CBLAS_INDEX {
        unsafe { cblas_sys::cblas_izamax(n, x as *const _, incx) }
    }



    unsafe fn cblas_nrm2(n: i32, x: *const Complex<f64>, incx: i32) -> f64 {
        unsafe { cblas_sys::cblas_dznrm2(n, x as *const _, incx) }
    }

    unsafe fn cblas_asum(n: i32, x: *const Complex<f64>, incx: i32) -> f64 {
        unsafe { cblas_sys::cblas_dzasum(n, x as *const _, incx) }
    }



    unsafe fn cblas_axpy(
        n: i32,
        alpha: Complex<f64>,
        x: *const Complex<f64>,
        incx: i32,
        y: *mut Complex<f64>,
        incy: i32,
    ) {
        unsafe {
            cblas_sys::cblas_zaxpy(
                n,
                &alpha as *const _ as *const _,
                x as *const _,
                incx,
                y as *mut _,
                incy,
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
        incy: i32,
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
                incy,
            )
        }
    }


    unsafe fn cblas_ger(
        layout: CBLAS_LAYOUT,
        m: i32,
        n: i32,
        alpha: Complex<f64>,
        x: *const Complex<f64>,
        incx: i32,
        y: *const Complex<f64>,
        incy: i32,
        a: *mut Complex<f64>,
        lda: i32,
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
                lda,
            )
        }
    }

}
