//! Linear algebra (BLAS, LAPACK, etc.) bindings for mdarray.
//!
//! This is the main crate that contains the traits that are implemented by some mdarray-linalg-*
//! crates, e.g. mdarray-linalg.blas.

pub mod prelude;

mod matmul;
pub use matmul::*;

mod svd;
pub use svd::*;

mod utils;
pub use utils::*;
