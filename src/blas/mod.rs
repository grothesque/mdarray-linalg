mod scalar;
mod simple;
mod context;

pub use simple::{gemm, gemm_uninit};
pub use context::Blas;
