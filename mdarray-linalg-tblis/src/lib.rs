//! # mdarray-linalg-tblis
//!
//! [TBLIS](https://github.com/devinamatthews/tblis) backend for [`mdarray_linalg`].
//!
//! This crate provides the [`Tblis`] struct that implements matrix multiplication and
//! tensor contraction traits from `mdarray-linalg`, delegating to the high-performance
//! C library TBLIS.  TBLIS is specifically designed for **dense tensor contractions**
//! and shines on large, high-rank tensors where BLAS-based backends would need costly
//! transpositions.
//!
//! ## Scope
//!
//! The TBLIS backend covers:
//!
//! - **Matrix multiplication** — `matmul`
//! - **Tensor contraction** — `contract_all`, `contract_n`, `contract_pairs`, `contract` (einsum)
//!
//! For vector operations, decompositions, or solving, use another backend
//! (`mdarray-linalg-blas`, `mdarray-linalg-faer`, etc.).
//!
//! ## Setup
//!
//! This crate binds to TBLIS but does not choose which native TBLIS library to link against.
//! This is left to the user.  For example, to link against an installed TBLIS library:
//!
//! ```bash
//! cargo add mdarray mdarray-linalg mdarray-linalg-tblis
//! cargo add tblis-src
//! ```
//!
//! In one of your Rust crates, reference the provider so its link directives are
//! included:
//!
//! ```rust
//! extern crate tblis_src as _;
//! ```
//!
//! If TBLIS is installed outside standard linker paths, set `TBLIS_DIR`,
//! `LD_LIBRARY_PATH`, or equivalent linker/runtime configuration.  If you need
//! to install TBLIS from source, see the
//! [upstream build guide](https://github.com/MatthewsResearchGroup/tblis/wiki/Building).
//! You may also provide equivalent link flags from your application `build.rs`.
//!
//! ## Example
//!
//! All operations are accessed through the [`Tblis`] backend:
//!
//! ```rust
//! # extern crate tblis_src as _;
//! use mdarray::array;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg_tblis::Tblis;
//!
//! // ----- Matrix multiplication -----
//! let a = array![[1., 2.], [3., 4.]];
//! let b = array![[5., 6.], [7., 8.]];
//!
//! let c = Tblis.matmul(&a, &b).eval();
//! assert_eq!(c, array![[19., 22.], [43., 50.]]);
//!
//! // ----- Tensor contraction: contract last n axes -----
//! let t1 = array![[1., 2.], [3., 4.]].into_dyn();
//! let t2 = array![[5., 6.], [7., 8.]].into_dyn();
//!
//! // Full contraction (contract_all) → scalar
//! let scalar = Tblis.contract_all(&t1, &t2);
//! assert_eq!(scalar, 70.0);  // 1·5 + 2·6 + 3·7 + 4·8
//!
//! // ----- Outer product (contract_n with n=0) -----
//! let x = array![1., 2.].into_dyn();
//! let y = array![3., 4.].into_dyn();
//! let outer = Tblis.contract_n(&x, &y, 0).eval();
//! assert_eq!(outer, array![[3., 4.], [6., 8.]].into_dyn());
//!
//! // ----- Einsum-style contraction -----
//! // ij,jk->ik  (matrix multiplication)
//! let result = Tblis
//!     .contract(&t1, &t2, &[0, 1], &[1, 2], &[0, 2])
//!     .eval();
//! assert_eq!(result, array![[19., 22.], [43., 50.]].into_dyn());
//!
//! // ij,ij->  (full contraction)
//! let result = Tblis
//!     .contract(&t1, &t2, &[0, 1], &[0, 1], &[])
//!     .eval();
//! assert_eq!(result.into_scalar(), 70.0);
//!
//! // ij,jk->ki  (matmul with transposed output)
//! let result = Tblis
//!     .contract(&t1, &t2, &[0, 1], &[1, 2], &[2, 0])
//!     .eval();
//! assert_eq!(result, array![[19., 43.], [22., 50.]].into_dyn());
//!
//! // i,j->ij  (outer product via einsum)
//! let result = Tblis
//!     .contract(&x, &y, &[0], &[1], &[0, 1])
//!     .eval();
//! assert_eq!(result, array![[3., 4.], [6., 8.]].into_dyn());
//!
//! // ----- Scaled addition: C = α·A·B + β·C -----
//! let mut c = array![[1., 1.], [1., 1.]];
//! Tblis.matmul(&a, &b).add_to_scaled(&mut c, 2.0);
//! // C = A·B + 2·C = [[19,22],[43,50]] + 2·[[1,1],[1,1]]
//! ```
//!
//! ## Supported types
//!
//! `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
//!
//! ## Troubleshooting
//!
//! Linking errors usually mean that the native TBLIS library was not linked
//! into the final binary, or that it is not in the linker/runtime search path.
//! Add `tblis-src`, reference it from Rust code, or provide equivalent link
//! flags from your application `build.rs`.

#![cfg_attr(docsrs, doc = concat!(
    "[mdarray_linalg]: https://docs.rs/mdarray-linalg/", env!("CARGO_PKG_VERSION"), "/mdarray_linalg/",
))]
#![cfg_attr(not(docsrs), doc = "\
[mdarray_linalg]: ../mdarray_linalg/index.html
")]

#[cfg(test)]
extern crate tblis_src as _;

pub mod matmul;

/// TBLIS backend.
///
/// Implements matrix multiplication and tensor contraction traits from
/// [`mdarray_linalg`] by delegating to the TBLIS C library.  TBLIS is
/// particularly efficient for **high-rank tensor contractions** where it
/// avoids the explicit transpositions required by BLAS-based approaches.
///
/// This is a zero-sized marker struct — all state is managed by the
/// underlying TBLIS library.
#[derive(Default)]
pub struct Tblis;
