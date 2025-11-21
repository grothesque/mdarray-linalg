//! ```rust
//! use mdarray::{DTensor, tensor};
//! use mdarray_linalg::prelude::*; // Import traits anonymously
//! use mdarray_linalg::eig::EigDecomp;
//! use mdarray_linalg::svd::SVDDecomp;
//!
//! // Backends
//! use mdarray_linalg_lapack::Lapack;
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
//! }
//!
//! // ----- Singular Value Decomposition (SVD) -----
//! let bd = Lapack::new().config_svd(SVDConfig::DivideConquer);
//! let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");
//! println!("Singular values: {:?}", s);
//! println!("Left singular vectors U: {:?}", u);
//! println!("Right singular vectors V^T: {:?}", vt);
//!
//! // ----- QR Decomposition -----
//! let (m, n) = *a.shape();
//! let mut q = DTensor::<f64, 2>::zeros([m, m]);
//! let mut r = DTensor::<f64, 2>::zeros([m, n]);
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
    _qr_config: LapackQRConfig,
}

#[derive(Default, Debug, Clone)]
pub enum LapackQRConfig {
    #[default]
    Full,
    Pivoting,
    TallSkinny,
}

impl Lapack {
    pub fn new() -> Self {
        Self {
            svd_config: SVDConfig::default(),
            _qr_config: LapackQRConfig::default(),
        }
    }

    pub fn config_svd(mut self, config: SVDConfig) -> Self {
        self.svd_config = config;
        self
    }

    pub fn config_qr(self, _config: LapackQRConfig) -> Self {
        todo!()
    }
}
