use num_complex::Complex;
use paste::paste;

#[allow(clippy::too_many_arguments)]
pub trait LapackScalar {
    unsafe fn lapack_getrf(
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
}

macro_rules! impl_lapack_scalar {
    ($t:ty, $prefix:ident) => {
        impl LapackScalar for $t {
            #[inline]
            unsafe fn lapack_getrf(
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                ipiv: *mut i32,
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                        lapack_sys::[<$prefix getrf_>](
                            &m as *const i32,
                            &n as *const i32,
                            a as *mut _,
                            &lda as *const i32,
                            ipiv as *mut i32,
                            info as *mut i32,
                        );
                    }
                }
            }
        }
    };
}

impl_lapack_scalar!(f32, s);
impl_lapack_scalar!(f64, d);
impl_lapack_scalar!(Complex<f32>, c);
impl_lapack_scalar!(Complex<f64>, z);
