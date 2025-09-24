use num_complex::Complex;
use paste::paste;

#[allow(clippy::too_many_arguments)]
pub trait LapackScalar {
    unsafe fn lapack_getrf(m: i32, n: i32, a: *mut Self, lda: i32, ipiv: *mut i32, info: *mut i32);

    unsafe fn lapack_getri(
        n: i32,
        a: *mut Self,
        lda: i32,
        ipiv: *const i32,
        work: *mut Self,
        lwork: i32,
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

            #[inline]
            unsafe fn lapack_getri(
                n: i32,
                a: *mut Self,
                lda: i32,
                ipiv: *const i32,
                work: *mut Self,
                lwork: i32,
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                        lapack_sys::[<$prefix getri_>](
                            &n as *const i32,
                            a as *mut _,
                            &lda as *const i32,
                            ipiv as *const _,
                            work as *mut _,
                            &lwork as *const i32,
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

pub trait Workspace {
    type RworkType;
    type Elem;
    fn lwork_from_query(query: &Self::Elem) -> i32;
    fn allocate(lwork: i32) -> Vec<Self::Elem>;
}

macro_rules! impl_needs_rwork {
    ($type:ty, $elem:ty, no_rwork) => {
        impl Workspace for $type {
            type RworkType = ();
            type Elem = $elem;

            fn lwork_from_query(query: &Self::Elem) -> i32 {
                *query as i32
            }

            fn allocate(lwork: i32) -> Vec<Self::Elem> {
                vec![<$elem>::default(); lwork as usize]
            }
        }
    };

    ($type:ty, $elem:ty, $rwork:ty) => {
        impl Workspace for $type {
            type RworkType = $rwork;
            type Elem = $elem;

            fn lwork_from_query(query: &Self::Elem) -> i32 {
                query.re as i32
            }

            fn allocate(lwork: i32) -> Vec<Self::Elem> {
                vec![<$elem>::default(); lwork as usize]
            }
        }
    };
}

impl_needs_rwork!(f32, f32, no_rwork);
impl_needs_rwork!(f64, f64, no_rwork);
impl_needs_rwork!(Complex<f32>, Complex<f32>, f32);
impl_needs_rwork!(Complex<f64>, Complex<f64>, f64);
