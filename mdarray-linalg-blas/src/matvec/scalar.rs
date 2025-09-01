// This file is auto-generated. Do not edit manually.
//! Abstracting the BLAS scalar types
use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_SIDE, CBLAS_UPLO, CBLAS_DIAG};
use num_complex::Complex;

#[allow(clippy::too_many_arguments, unused_variables)]
pub trait BlasScalar {
    
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
}

impl BlasScalar for f32 {
    
}

impl BlasScalar for f64 {
    
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
    
}

impl BlasScalar for Complex<f32> {
    
}

impl BlasScalar for Complex<f64> {
    
}
