//! This module can be wildcard-imported to make the traits available without polluting the local
//! namespace.

pub use super::eig::Eig as _;
pub use super::lu::LU as _;
pub use super::matmul::{ContractBuilder as _, MatMul as _, MatMulBuilder as _};
pub use super::matvec::{
    Argmax as _, MatVec as _, MatVecBuilder as _, Outer as _, OuterBuilder as _, VecOps as _,
};
pub use super::qr::QR as _;
pub use super::svd::SVD as _;
