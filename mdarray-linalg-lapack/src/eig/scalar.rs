use num_complex::Complex;
use paste::paste;

#[allow(clippy::too_many_arguments)]
pub trait LapackScalar {
    // General eigenvalue decomposition (GEEV)
    unsafe fn lapack_geev(
        jobvl: i8,
        jobvr: i8,
        n: i32,
        a: *mut Self,
        lda: i32,
        wr: *mut Self,
        wi: *mut Self,
        vl: *mut Self,
        ldvl: i32,
        vr: *mut Self,
        ldvr: i32,
        work: *mut Self,
        lwork: i32,
        rwork: *mut Self,
        info: *mut i32,
    );

    // Hermitian/symmetric eigenvalue decomposition (SYEV/HEEV)
    unsafe fn lapack_syev(
        jobz: i8,
        uplo: i8,
        n: i32,
        a: *mut Self,
        lda: i32,
        w: *mut Self,
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
            unsafe fn lapack_geev(
                jobvl: i8,
                jobvr: i8,
                n: i32,
                a: *mut Self,
                lda: i32,
                wr: *mut Self,
                wi: *mut Self,
                vl: *mut Self,
                ldvl: i32,
                vr: *mut Self,
                ldvr: i32,
                work: *mut Self,
                lwork: i32,
                _rwork: *mut Self, // unused for real types
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                        lapack_sys::[<$prefix geev_>](
                            &jobvl as *const i8,
                            &jobvr as *const i8,
                            &n as *const i32,
                            a as *mut _,
                            &lda as *const i32,
                            wr as *mut _,
                            wi as *mut _,
                            vl as *mut _,
                            &ldvl as *const i32,
                            vr as *mut _,
                            &ldvr as *const i32,
                            work as *mut _,
                            &lwork as *const i32,
                            info as *mut i32,
                        );
                    }
                }
            }

            #[inline]
            unsafe fn lapack_syev(
                jobz: i8,
                uplo: i8,
                n: i32,
                a: *mut Self,
                lda: i32,
                w: *mut Self,
                work: *mut Self,
                lwork: i32,
                _rwork: *mut Self, // unused for real types
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                        lapack_sys::[<$prefix syev_>](
                            &jobz as *const i8,
                            &uplo as *const i8,
                            &n as *const i32,
                            a as *mut _,
                            &lda as *const i32,
                            w as *mut _,
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

macro_rules! real_cast {
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
            #[inline]
            unsafe fn lapack_geev(
                jobvl: i8,
                jobvr: i8,
                n: i32,
                a: *mut Self,
                lda: i32,
                wr: *mut Self,  // For complex, this holds the eigenvalues directly
                _wi: *mut Self, // unused for complex types
                vl: *mut Self,
                ldvl: i32,
                vr: *mut Self,
                ldvr: i32,
                work: *mut Self,
                lwork: i32,
                rwork: *mut Self,
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                        lapack_sys::[<$prefix geev_>](
                            &jobvl as *const i8,
                            &jobvr as *const i8,
                            &n as *const i32,
                            a as *mut lapack_sys_cast!($prefix),
                            &lda as *const i32,
                            wr as *mut lapack_sys_cast!($prefix),
                            vl as *mut lapack_sys_cast!($prefix),
                            &ldvl as *const i32,
                            vr as *mut lapack_sys_cast!($prefix),
                            &ldvr as *const i32,
                            work as *mut lapack_sys_cast!($prefix),
                            &lwork as *const i32,
                            rwork as *mut real_cast!($prefix),
                            info as *mut i32,
                        );
                    }
                }
            }

            #[inline]
            unsafe fn lapack_syev(
                jobz: i8,
                uplo: i8,
                n: i32,
                a: *mut Self,
                lda: i32,
                w: *mut Self,
                work: *mut Self,
                lwork: i32,
                rwork: *mut Self,
                info: *mut i32,
            ) {
                unsafe {
                    paste! {
                        lapack_sys::[<$prefix heev_>](
                            &jobz as *const i8,
                            &uplo as *const i8,
                            &n as *const i32,
                            a as *mut lapack_sys_cast!($prefix),
                            &lda as *const i32,
                            w as *mut real_cast!($prefix),
                            work as *mut lapack_sys_cast!($prefix),
                            &lwork as *const i32,
                            rwork as *mut real_cast!($prefix),
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
    fn rwork_len_geev(n: i32) -> usize;
    fn rwork_len_syev(n: i32) -> usize;
    fn lwork_from_query(query: &Self::Elem) -> i32;
    fn allocate(lwork: i32) -> Vec<Self::Elem>;
}

macro_rules! impl_needs_rwork {
    ($type:ty, $elem:ty, no_rwork) => {
        impl NeedsRwork for $type {
            type RworkType = ();
            type Elem = $elem;

            fn rwork_len_geev(_: i32) -> usize {
                0
            }

            fn rwork_len_syev(_: i32) -> usize {
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

            fn rwork_len_geev(n: i32) -> usize {
                (2 * n) as usize
            }

            fn rwork_len_syev(n: i32) -> usize {
                (3 * n - 2).max(1) as usize
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

pub trait ToComplex {
    type ComplexType;

    fn zero_complex() -> Self::ComplexType;
}

impl ToComplex for f32 {
    type ComplexType = Complex<f32>;

    fn zero_complex() -> Self::ComplexType {
        Complex::<f32>::default()
    }
}

impl ToComplex for f64 {
    type ComplexType = Complex<f64>;

    fn zero_complex() -> Self::ComplexType {
        Complex::<f64>::default()
    }
}

impl ToComplex for Complex<f32> {
    type ComplexType = Complex<f32>;

    fn zero_complex() -> Self::ComplexType {
        Complex::<f32>::default()
    }
}

impl ToComplex for Complex<f64> {
    type ComplexType = Complex<f64>;

    fn zero_complex() -> Self::ComplexType {
        Complex::<f64>::default()
    }
}
