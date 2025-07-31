pub mod context;
pub mod scalar;
pub mod simple;

pub use context::Blas;
pub use simple::{gemm, gemm_uninit};
