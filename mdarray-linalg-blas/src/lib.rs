pub mod matmul;

// pub use matmul::Blas;
#[derive(Default)]
pub struct Blas;
pub use matmul::{gemm, gemm_uninit};

pub mod matvec;
