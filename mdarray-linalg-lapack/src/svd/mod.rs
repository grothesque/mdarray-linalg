mod context;
mod scalar;
mod simple;

pub use crate::get_dims;
pub use context::Lapack;
pub use simple::{dgesdd, dgesdd_uninit};
