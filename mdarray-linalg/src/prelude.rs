//! This module can be wildcard-imported to make the traits available without polluting the local
//! namespace.

pub use super::Eig as _;
pub use super::LU as _;
pub use super::PRRLU as _;
pub use super::QR as _;
pub use super::SVD as _;
pub use super::{MatMul as _, MatMulBuilder as _};
pub use super::{MatVec as _, MatVecBuilder as _, VecOps as _};
