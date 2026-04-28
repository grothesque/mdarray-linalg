//! ```rust
//! use mdarray::{DArray, tensor};
//! use mdarray_linalg::prelude::*; // Import traits anonymously
//! use mdarray_linalg::{Naive, matmul, diag};
//! use mdarray_linalg::eig::EigDecomp;
//! use mdarray_linalg::svd::SVDDecomp;
//!
//! use mdarray_linalg_lapack::{Lapack, svd, eig};
//! use mdarray_linalg_lapack::SVDConfig;
//!
//! let a = tensor![[1., 2.], [3., 4.]];
//!
//! // ----- Eigenvalue decomposition -----
//! // Note: we must clone `a` here because decomposition routines destroy the input.
//! let bd = Lapack::new(); // Unlike Blas, Lapack is not a zero-sized backend so `new` must be called.
//! let EigDecomp {
//!     eigenvalues,
//!     right_eigenvectors,
//!     ..
//! } = bd.eig(&mut a.clone()).expect("Eigenvalue decomposition failed");
//!
//! println!("Eigenvalues: {:?}", eigenvalues);
//! if let Some(vectors) = right_eigenvectors {
//!     println!("Right eigenvectors: {:?}", vectors);
//! } // Or...
//! let (lambda, v) = eig!(&mut a.clone());
//!
//! // ----- Singular Value Decomposition (SVD) -----
//! let bd = Lapack::new().config_svd(SVDConfig::DivideConquer);
//! let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");
//! println!("Singular values: {:?}", s);
//! println!("Left singular vectors U: {:?}", u);
//! println!("Right singular vectors V^T: {:?}", vt); // Or...
//! let (s,u,vt) = svd!(&mut a.clone()); // Convenience macro that directly unpacks the SVD.
//! let b = matmul!(&u, &diag(&s), &vt);
//!
//! assert!(((a[[0,1]] - b[[0,1]]) as f64).abs() < 10e-10_f64);
//!
//!
//! // ----- QR Decomposition -----
//! let (m, n) = *a.shape();
//! let mut q = DArray::<f64, 2>::zeros([m, m]);
//! let mut r = DArray::<f64, 2>::zeros([m, n]);
//!
//! let bd = Lapack::new();
//! bd.qr_write(&mut a.clone(), &mut q, &mut r); //
//! println!("Q: {:?}", q);
//! println!("R: {:?}", r);
//! ```

pub mod eig;
pub mod lu;
pub mod qr;
pub mod solve;
pub mod svd;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub enum SVDConfig {
    #[default]
    Auto,
    DivideConquer,
    Jacobi,
}

#[derive(Debug, Default, Clone)]
pub struct Lapack {
    svd_config: SVDConfig,
    qr_config: QRConfig,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum QRConfig {
    #[default]
    Reduced, // Q: M×K, R: K×N
    Complete, // Q: M×M, R: M×N
}

impl Lapack {
    pub fn new() -> Self {
        Self {
            svd_config: SVDConfig::default(),
            qr_config: QRConfig::default(),
        }
    }

    pub fn config_svd(mut self, config: SVDConfig) -> Self {
        self.svd_config = config;
        self
    }

    pub fn config_qr(mut self, config: QRConfig) -> Self {
        self.qr_config = config;
        self
    }
}

/// Convenience macro for SVD decomposition that unwraps the result
/// directly.  Panics if the decomposition fails.
#[macro_export]
macro_rules! svd {
    ($a:expr) => {{
        let svdr = Lapack::default().svd_thin($a).expect("SVD failed");
        (svdr.s, svdr.u, svdr.vt)
    }};
    ($a:expr, full) => {{
        let svdr = Lapack::default().svd($a).expect("SVD failed");
        (svdr.s, svdr.u, svdr.vt)
    }};
}

/// Convenience macro for eigenvalue decomposition.
/// Panics if the decomposition fails.
#[macro_export]
macro_rules! eig {
    ($a:expr) => {{
        let eig = Lapack::default()
            .eig($a)
            .expect("Eigenvalue decomposition failed");

        let vectors = eig
            .right_eigenvectors
            .expect("Eigenvectors were not computed");

        (eig.eigenvalues, vectors)
    }};
}
