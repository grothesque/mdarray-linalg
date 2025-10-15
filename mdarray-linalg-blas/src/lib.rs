pub mod matmul;
pub use matmul::{gemm, gemm_uninit};
pub mod matvec;
pub mod tensordot;

#[derive(Default)]
pub struct Blas;
