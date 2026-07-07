//! # mdarray-linalg-lapack
//!
//! LAPACK backend for [`mdarray_linalg`].
//!
//! This crate provides the [`Lapack`] struct that implements the decomposition and
//! solver traits defined by `mdarray-linalg`, delegating computations to a LAPACK
//! implementation (e.g. OpenBLAS) via the `lapack-sys` and `cblas-sys` crates.
//!
//! Backend implementation modules are private.  Use [`Lapack`] together with the
//! operation traits from `mdarray_linalg::prelude::*`.
//!
//! ## Scope
//!
//! The LAPACK backend covers:
//!
//! - **Eigenvalue decomposition** — `eig`, `eig_full`, `eig_values`, `eigh`, `eigs`
//! - **Schur decomposition** — `schur`, `schur_complex`
//! - **SVD** — `svd`, `svd_thin`, `svd_s`
//! - **LU decomposition** — `lu`, `det`, `inv`
//! - **Cholesky decomposition** — `choleski`
//! - **QR decomposition** — `qr`
//! - **Linear system solving** — `solve`
//!
//! For basic matrix/vector operations (Level 1–3 BLAS) and tensor contractions,
//! use the `mdarray-linalg-blas` or `mdarray-linalg-faer` backends instead.
//!
//! ## Setup
//!
//! This crate binds to the LAPACK/BLAS ABI but does not choose a native library to link against.
//! This is left to the user.  For example, to use a system OpenBLAS installation:
//!
//! ```bash
//! cargo add mdarray mdarray-linalg mdarray-linalg-lapack
//! cargo add lapack-src --features openblas
//! cargo add openblas-src --features system
//! ```
//!
//! In one of your Rust crates, reference the provider so its link directives are
//! included:
//!
//! ```rust
//! extern crate lapack_src as _;
//! ```
//!
//! Other LAPACK providers may be used if they expose the symbols required by
//! `lapack-sys` and `cblas-sys`.
//!
//! ## Example
//!
//! All operations are accessed through the [`Lapack`] backend via the traits from
//! `mdarray_linalg::prelude::*`:
//!
//! ```rust
//! # extern crate lapack_src as _;
//! use mdarray::array;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg::eig::EigDecomp;
//! use mdarray_linalg::solve::Solve;
//! use mdarray_linalg::svd::SVDDecomp;
//! use mdarray_linalg_lapack::{Lapack, SVDConfig};
//!
//! // ----- Eigenvalue decomposition -----
//! let mut a = array![[1., 2.], [3., 4.]];
//! let EigDecomp {
//!     eigenvalues: lambda,
//!     right_eigenvectors,
//!     ..
//! } = Lapack::new().eig(&mut a.clone()).expect("Eigenvalue decomposition failed");
//!
//! println!("Eigenvalues: {:?}", lambda);
//! if let Some(v) = right_eigenvectors {
//!     println!("Right eigenvectors: {:?}", v);
//! }
//!
//! // ----- SVD (with configuration) -----
//! let mut a = array![[1., 2.], [3., 4.]];
//! let bd = Lapack::new().config_svd(SVDConfig::DivideConquer);
//! let SVDDecomp { s, u, vt } = bd.svd_thin(&mut a).expect("SVD failed");
//! println!("Singular values: {:?}", s);
//!
//! // ----- QR decomposition -----
//! let mut a = array![[12., -51., 4.], [6., 167., -68.], [-4., 24., -41.]];
//! let (q, r) = Lapack::new().qr(&mut a);
//! println!("Q: {:?}", q);
//! println!("R: {:?}", r);
//!
//! // ----- Solve linear system Ax = b -----
//! let mut a = array![[2., 1., 0.], [1., 3., 1.], [0., 1., 2.]];
//! let b = array![[1., 0., 0.], [2., 0., 0.], [1., 0., 0.]];
//! let result = Lapack::new().solve(&mut a, &b).expect("Solve failed");
//! println!("x = {:?}", result.x);
//! ```
//!
//! > **Note:** Decomposition routines (eig, svd, lu, etc.) **destroy the input matrix**.
//! > Always pass a clone if you need the original data.
//!
//! ## Supported types
//!
//! `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
//!
//! ## Troubleshooting
//!
//! Linking errors usually mean that no LAPACK/BLAS implementation was linked
//! into the final binary, or that the selected libraries are not in the
//! linker/runtime search path.  Add a source crate such as `lapack-src`,
//! reference it from Rust code, or provide equivalent link flags from your
//! application `build.rs`.

#![cfg_attr(docsrs, doc = concat!(
    "[mdarray_linalg]: https://docs.rs/mdarray-linalg/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg/",
))]
#![cfg_attr(not(docsrs), doc = "\
[mdarray_linalg]: ../mdarray_linalg/index.html
")]

#[cfg(test)]
extern crate lapack_src as _;

mod eig;
mod lu;
mod qr;
mod solve;
mod svd;

/// Configuration for the SVD algorithm.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub enum SVDConfig {
    /// Let the backend choose the best algorithm.
    #[default]
    Auto,
    /// Divide-and-conquer algorithm (faster for large matrices).
    DivideConquer,
    /// Jacobi algorithm (more accurate for small matrices).
    Jacobi,
}

/// Configuration for the QR decomposition.
#[derive(Default, Debug, Clone, Copy)]
pub enum QRConfig {
    /// Reduced QR: Q is M×K, R is K×N (where K = min(M, N)).
    #[default]
    Reduced,
    /// Complete QR: Q is M×M, R is M×N.
    Complete,
}

/// LAPACK backend.
///
/// Implements the decomposition and solver traits from [`mdarray_linalg`] by
/// delegating to LAPACK routines.  Unlike the zero-sized
/// `mdarray_linalg_blas::Blas`, `Lapack` carries configuration state
/// for algorithm selection.
///
/// # Configuration
///
/// Use the builder-style methods to select algorithms:
///
/// ```rust
/// use mdarray_linalg_lapack::{Lapack, SVDConfig};
///
/// let default_bd = Lapack::new();
/// let svd_bd = Lapack::new().config_svd(SVDConfig::DivideConquer);
/// ```
#[derive(Debug, Default, Clone)]
pub struct Lapack {
    svd_config: SVDConfig,
    qr_config: QRConfig,
}

impl Lapack {
    /// Creates a new `Lapack` backend with default configuration.
    pub fn new() -> Self {
        Self {
            svd_config: SVDConfig::default(),
            qr_config: QRConfig::default(),
        }
    }

    /// Selects the SVD algorithm.
    #[must_use]
    pub fn config_svd(mut self, config: SVDConfig) -> Self {
        self.svd_config = config;
        self
    }

    /// Selects the QR algorithm variant.
    #[must_use]
    pub fn config_qr(mut self, config: QRConfig) -> Self {
        self.qr_config = config;
        self
    }
}
