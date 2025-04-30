//! Experimental linear algebra (BLAS, LAPACK, etc.) bindings for mdarray

mod blas_scalar;
mod simple;

pub mod context;
pub mod traits;

pub use blas_scalar::BlasScalar;
pub use simple::{gemm, gemm_uninit};
