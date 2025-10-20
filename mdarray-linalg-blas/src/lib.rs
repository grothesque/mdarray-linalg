pub mod matmul;
pub use matmul::{gemm, gemm_uninit};
pub mod matvec;

#[derive(Default)]
pub struct Blas;
