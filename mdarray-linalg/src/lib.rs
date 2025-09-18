//! Linear algebra (BLAS, LAPACK, Faer, etc.) bindings for mdarray.
//!
//! This is the main crate that contains the traits that are implemented by some mdarray-linalg-*
//! crates, e.g. mdarray-linalg.blas.

pub mod prelude;

mod matmul;
pub use matmul::*;

mod qr;
pub use qr::*;

mod svd;
pub use svd::*;

mod utils;
pub use utils::*;

mod matvec;
pub use matvec::*;

mod prrlu;
pub use prrlu::*;

mod lu;
pub use lu::*;

mod eig;
pub use eig::*;
