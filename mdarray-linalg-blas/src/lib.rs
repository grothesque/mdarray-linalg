pub mod matmul;

pub use matmul::Blas;
pub use matmul::{gemm, gemm_uninit};

pub mod matvec;
