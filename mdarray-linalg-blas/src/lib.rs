//! ```rust
//! use mdarray::{DTensor, tensor};
//! use mdarray_linalg::prelude::*; // Imports only traits
//!
//! use mdarray_linalg_blas::Blas;
//!
//! fn main() {
//!     // Declare two vectors
//!     let x = tensor![1., 2.];
//!     let y = tensor![2., 4.];
//!
//!     // Declare two matrices
//!     let a = tensor![[1., 2.], [3., 4.]];
//!     let b = tensor![[5., 6.], [7., 8.]];
//!
//!     // ----- Vector operations -----
//!     let dot_result = Blas.dot(&x, &y);
//!     println!("dot(x, y) = {}", dot_result);
//!
//!     let y_result = Blas.matvec(&a, &x).scale(2.).eval();
//!     println!("A * x * 2 = {:?}", y_result);
//!
//!     // ----- Matrix multiplication -----
//!     let c = Blas.matmul(&a, &b).eval();
//!     println!("A * B = {:?}", c);
//!
//!
//!     // ----- Contract: full contraction between two 3D tensors -----
//!     let a = tensor![
//!         [[1., 2.], [3., 4.]],
//!         [[5., 6.], [7., 8.]]
//!     ].into_dyn();
//!
//!     let b = tensor![
//!         [[9., 10.], [11., 12.]],
//!         [[13., 14.], [15., 16.]]
//!     ].into_dyn();
//!
//!     let result = Blas.contract_all(&a, &b).eval();
//!     println!("Full contraction result (tensordot over all axes): {:?}", result);
//! }
//! ```

pub mod matmul;
pub use matmul::{gemm, gemm_uninit};
pub mod matvec;

#[derive(Default)]
pub struct Blas;
