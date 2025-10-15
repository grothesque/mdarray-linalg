//! This module can be wildcard-imported to make the traits available without polluting the local
//! namespace.

pub use super::eig::Eig;
pub use super::lu::LU;
pub use super::matmul::{MatMul, MatMulBuilder};
pub use super::matvec::{MatVec, MatVecBuilder, VecOps};
pub use super::prrlu::PRRLU;
pub use super::qr::QR;
pub use super::svd::SVD;
