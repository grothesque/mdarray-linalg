//! This module can be wildcard-imported to make the traits available without polluting the local
//! namespace.

pub use super::{
    eig::Eig as _,
    lu::LU as _,
    matmul::{ContractBuilder as _, MatMul as _, MatMulBuilder as _},
    matvec::{
        Argmax as _, MatVec as _, MatVecBuilder as _, Outer as _, OuterBuilder as _, VecOps as _,
    },
    qr::QR as _,
    svd::SVD as _,
};
