//! This module can be wildcard-imported to make the traits available without polluting the local
//! namespace.

pub use super::{
    Argmax as _, Contract as _, Eig as _, LU as _, MatVec as _, Outer as _, QR as _, SVD as _,
    Solve as _, VecOps as _,
    contract::{ContractBuilder as _, MatmulBuilder as _},
    matvec::{MatVecBuilder as _, OuterBuilder as _},
};
