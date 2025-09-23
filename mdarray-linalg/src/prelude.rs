//! This module can be wildcard-imported to make the traits available without polluting the local
//! namespace.

pub use super::QR as _;
pub use super::{Eig as _, EigDecomp, EigError, EigResult};
pub use super::{MatMul as _, MatMulBuilder as _};
pub use super::{MatVec as _, MatVecBuilder as _, VecOps as _};
pub use super::{MatVec, MatVecBuilder, VecOps};
pub use super::{PRRLU, PRRLUDecomp};
pub use super::{SVD as _, SVDDecomp, SVDError, SVDResult};
pub use super::{Triangle, Type};
