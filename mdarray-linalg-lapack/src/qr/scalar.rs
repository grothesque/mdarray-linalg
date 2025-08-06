use num_complex::Complex;
use paste::paste;

#[allow(clippy::too_many_arguments)]
pub trait LapackScalar {
    unsafe fn lapack_geqrf(
        m: i32,
        n: i32,
        a: *mut Self,
        tau: *mut Self,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    );

    unsafe fn lapack_orgqr(
        m: i32,
        min_mn: i32,
        a: *mut Self,
        tau: *mut Self,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    );
}

macro_rules! impl_lapack_scalar {
    ($t:ty, $prefix:ident, $suffix:ident) => {
        impl LapackScalar for $t {
            #[inline]
            unsafe fn lapack_geqrf(
                m: i32,
                n: i32,
                a: *mut Self,
                tau: *mut Self,
                work: *mut Self,
                lwork: i32,
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                            lapack_sys::[<$prefix geqrf_>](
                    &m as *const i32,
                    &n as *const i32,
                    a as *mut _,
                    &m as *const i32,
                    tau as *mut _,
                    work as *mut _,
                    &lwork as *const i32,
                    info as  *mut i32,
                            );
                        }
                }
            }
            unsafe fn lapack_orgqr(
                m: i32,
                min_mn: i32,
                a: *mut Self,
                tau: *mut Self,
                work: *mut Self,
                lwork: i32,
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                                lapack_sys::[<$prefix $suffix gqr_>](
                                    &m as *const i32,
                    &m as *const i32,
                    &min_mn as *const i32,
                    a as *mut _,
                    &m,
                    tau as *mut _,
                    work as *mut _,
                    &lwork,
                    info as *mut i32,
                                    );
                                }
                }
            }
        }
    };
}

impl_lapack_scalar!(f32, s, or);
impl_lapack_scalar!(f64, d, or);
impl_lapack_scalar!(Complex<f32>, c, un);
impl_lapack_scalar!(Complex<f64>, z, un);
