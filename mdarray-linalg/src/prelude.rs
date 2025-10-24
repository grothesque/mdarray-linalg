//! This module can be wildcard-imported to make the traits available without polluting the local
//! namespace.

pub use super::eig::Eig;
pub use super::lu::LU;
pub use super::matmul::{ContractBuilder, MatMul, MatMulBuilder};
pub use super::matvec::{Argmax, MatVec, MatVecBuilder, VecOps};
pub use super::qr::QR;
pub use super::svd::SVD;
