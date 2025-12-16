//! ```rust
//! use mdarray::{DTensor, tensor};
//! use mdarray_linalg::prelude::*; // Imports traits anonymously
//!
//! use mdarray_linalg_nalgebra::Nalgebra;
//!
//! // Declare two matrices
//! let a = tensor![[1., 2.], [3., 4.]];
//! let b = tensor![[5., 6.], [7., 8.]];
//!
//! // ----- Matrix multiplication -----
//! let c = Nalgebra.matmul(&a, &b).eval();
//! println!("A * B = {:?}", c);
//!
//! ```

pub mod matmul;

#[derive(Default)]
pub struct Nalgebra;
