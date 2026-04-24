//! ```rust
//! use mdarray::{DArray, tensor};
//! use mdarray_linalg::prelude::*; // Imports traits anonymously
//!
//! use mdarray_linalg::svd::SVDDecomp;
//! use mdarray_linalg_nalgebra::svd;
//! use mdarray_linalg::{Naive, matmul, diag};
//!
//! use mdarray_linalg_nalgebra::Nalgebra;
//!
//! // Declare two matrices
//! let a = tensor![[1., 2.], [3., 4.]];
//! let b = tensor![[5., 6.], [7., 8.]];
//!
//!
//! // ----- Singular Value Decomposition (SVD) -----
//! let bd = Nalgebra;
//! let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");
//! println!("Singular values: {:?}", s);
//! println!("Left singular vectors U: {:?}", u);
//! println!("Right singular vectors V^T: {:?}", vt);
//! let (s,u,vt) = svd!(&mut a.clone()); // Convenience macro that directly unpacks the SVD.
//! let b = matmul!(&u, &diag(&s), &vt);
//! assert!(((a[[0,1]] - b[[0,1]]) as f64).abs() < 10e-10_f64);
//! ```

pub mod svd;

#[derive(Default)]
pub struct Nalgebra;

/// Convenience macro for SVD decomposition that unwraps the result
/// directly.  Panics if the decomposition fails.
#[macro_export]
macro_rules! svd {
    ($a:expr) => {{
        let svdr = Nalgebra::default().svd($a).expect("SVD failed");
        (svdr.s, svdr.u, svdr.vt)
    }};
    ($a:expr, full) => {{
        let svdr = Nalgebra::default().svd($a).expect("SVD failed");
        (svdr.s, svdr.u, svdr.vt)
    }};
}
