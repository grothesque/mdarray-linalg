//! Experimental linear algebra (BLAS, LAPACK, etc.) bindings for mdarray

mod blas_scalar;
mod simple;

pub use blas_scalar::BlasScalar;
pub use simple::gemm;
