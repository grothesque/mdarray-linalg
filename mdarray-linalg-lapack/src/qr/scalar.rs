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

macro_rules! impl_lapack_real {
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
                    &m as *const i32,
                    tau as *mut _,
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

macro_rules! lapack_sys_cast {
    (c) => {
        lapack_sys::lapack_complex_float
    };
    (z) => {
        lapack_sys::lapack_complex_double
    };
}

macro_rules! impl_lapack_cplx {
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
                    a as *mut lapack_sys_cast!($prefix),
                    &m as *const i32,
                    tau as *mut lapack_sys_cast!($prefix),
                    work as *mut lapack_sys_cast!($prefix),
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
                    a as *mut lapack_sys_cast!($prefix),
                    &m as *const i32,
                    tau as *mut lapack_sys_cast!($prefix),
                    work as *mut lapack_sys_cast!($prefix),
                    &lwork as *const i32,
                    info as *mut i32,
                                    );
                                }
                }
            }
        }
    };
}

impl_lapack_real!(f32, s, or);
impl_lapack_real!(f64, d, or);
impl_lapack_cplx!(Complex<f32>, c, un);
impl_lapack_cplx!(Complex<f64>, z, un);

pub trait NeedsRwork {
    type RworkType;
    type Elem;
    fn rwork_len(m: i32, n: i32) -> usize;
    fn lwork_from_query(query: &Self::Elem) -> i32;
    fn allocate(lwork: i32) -> Vec<Self::Elem>;
}

macro_rules! impl_needs_rwork {
    ($type:ty, $elem:ty, no_rwork) => {
        impl NeedsRwork for $type {
            type RworkType = ();
            type Elem = $elem;

            fn rwork_len(_: i32, _: i32) -> usize {
                unimplemented!()
            }

            fn lwork_from_query(query: &Self::Elem) -> i32 {
                *query as i32
            }

            fn allocate(lwork: i32) -> Vec<Self::Elem> {
                vec![<$elem>::default(); lwork as usize]
            }
        }
    };

    ($type:ty, $elem:ty, $rwork:ty) => {
        impl NeedsRwork for $type {
            type RworkType = $rwork;
            type Elem = $elem;

            fn rwork_len(_: i32, _: i32) -> usize {
                unimplemented!()
            }

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
