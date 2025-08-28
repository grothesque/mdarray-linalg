use num_complex::Complex;
use paste::paste;

#[allow(clippy::too_many_arguments)]
pub trait LapackScalar {
    unsafe fn lapack_gesdd(
        jobz: i8,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        s: *mut Self,
        u: *mut Self,
        ldu: i32,
        vt: *mut Self,
        ldvt: i32,
        work: *mut Self,
        lwork: i32,
        rwork: *mut Self,
        iwork: *mut i32,
        info: *mut i32,
    );

    unsafe fn lapack_gesvd(
        jobu: i8,
        jobvt: i8,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        s: *mut Self,
        u: *mut Self,
        ldu: i32,
        vt: *mut Self,
        ldvt: i32,
        work: *mut Self,
        lwork: i32,
        rwork: *mut Self,
        info: *mut i32,
    );
}

macro_rules! impl_lapack_scalar_real {
    ($t:ty, $prefix:ident) => {
        impl LapackScalar for $t {
            #[inline]
            unsafe fn lapack_gesdd(
                jobz: i8,
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                s: *mut Self,
                u: *mut Self,
                ldu: i32,
                vt: *mut Self,
                ldvt: i32,
                work: *mut Self,
                lwork: i32,
                _rwork: *mut Self, // unused
                iwork: *mut i32,
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                        lapack_sys::[<$prefix gesdd_>](
                            &jobz as *const i8,
                            &m as *const i32,
                            &n as *const i32,
                            a as *mut _,
                            &lda as *const i32,
                            s as *mut _,
                            u as *mut _,
                            &ldu as *const i32,
                            vt as *mut _,
                            &ldvt as *const i32,
                            work as *mut _,
                            &lwork as *const i32,
                            iwork as *mut i32,
                            info as *mut i32,
                        );
                    }
                }
            }

            #[inline]
            unsafe fn lapack_gesvd(
                jobu: i8,
                jobvt: i8,
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                s: *mut Self,
                u: *mut Self,
                ldu: i32,
                vt: *mut Self,
                ldvt: i32,
                work: *mut Self,
                lwork: i32,
                _rwork: *mut Self,
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                            lapack_sys::[<$prefix gesvd_>](
                                &jobu as *const i8,
                    &jobvt as *const i8,
                                &m as *const i32,
                                &n as *const i32,
                                a as *mut _,
                                &lda as *const i32,
                                s as *mut _,
                                u as *mut _,
                                &ldu as *const i32,
                                vt as *mut _,
                                &ldvt as *const i32,
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

macro_rules! sv_cast {
    (c) => {
        f32
    };
    (z) => {
        f64
    };
}

macro_rules! impl_lapack_scalar_cplx {
    ($t:ty, $prefix:ident) => {
        impl LapackScalar for $t {
            // type Real = <$t as num_complex::ComplexFloat>::Real;

            #[inline]
            unsafe fn lapack_gesdd(
                jobz: i8,
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                s: *mut Self,
                u: *mut Self,
                ldu: i32,
                vt: *mut Self,
                ldvt: i32,
                work: *mut Self,
                lwork: i32,
                rwork: *mut Self,
                iwork: *mut i32,
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                    lapack_sys::[<$prefix gesdd_>](
                        &jobz as *const i8,
                        &m as *const i32,
                        &n as *const i32,
                        a as *mut lapack_sys_cast!($prefix),
                        &lda as *const i32,
                        s as *mut sv_cast!($prefix),
                        u as *mut lapack_sys_cast!($prefix),
                        &ldu as *const i32,
                        vt as *mut lapack_sys_cast!($prefix),
                        &ldvt as *const i32,
                        work as *mut lapack_sys_cast!($prefix),
                        &lwork as *const i32,
                        rwork as *mut _,
                        iwork as *mut i32,
                        info as *mut i32,
                    );

                    }
                }
            }

            #[inline]
            unsafe fn lapack_gesvd(
                jobu: i8,
                jobvt: i8,
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                s: *mut Self,
                u: *mut Self,
                ldu: i32,
                vt: *mut Self,
                ldvt: i32,
                work: *mut Self,
                lwork: i32,
                rwork: *mut Self,
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                            lapack_sys::[<$prefix gesvd_>](
                                &jobu as *const i8,
                    &jobvt as *const i8,
                                &m as *const i32,
                                &n as *const i32,
                                a as *mut _,
                                &lda as *const i32,
                                s as *mut _,
                                u as *mut _,
                                &ldu as *const i32,
                                vt as *mut _,
                                &ldvt as *const i32,
                                work as *mut _,
                                &lwork as *const i32,
                                        rwork as *mut _,
                                info as *mut i32,
                            );
                        }
                }
            }
        }
    };
}

impl_lapack_scalar_real!(f32, s);
impl_lapack_scalar_real!(f64, d);
impl_lapack_scalar_cplx!(Complex<f32>, c);
impl_lapack_scalar_cplx!(Complex<f64>, z);

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
                0
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

            fn rwork_len(m: i32, n: i32) -> usize {
                5 * (m + n) as usize
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
