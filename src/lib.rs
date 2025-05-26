//! Experimental linear algebra (BLAS, LAPACK, etc.) bindings for mdarray

pub mod blas;
pub mod prelude;

mod traits;
pub use traits::*;
