//! Generic (backend-agnosting) tests for mdarray-linalg
//!
//! This is used by the backends to easily express tests that are actually run.

pub mod common;
pub mod eig;
pub mod lu;
pub mod matmul;
pub mod matvec;
pub mod qr;
pub mod solve;
pub mod svd;
pub mod tensordot;
