//! Linear algebra (BLAS, LAPACK, etc.) bindings for mdarray.
//!
//! This crate provides bindings for BLAS by implementing some of the traits defined in
//! mdarray-linalg.

pub mod matmul;

pub use context::Blas;
pub use simple::{gemm, gemm_uninit};
