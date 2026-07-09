//! Singular Value Decomposition (SVD)
//!
//! The matrix A is decomposed as `A = U * S * V^T` where:
//! - `s` contains the singular values (1D vector)
//! - `u` contains the left singular vectors (matrix U)
//! - `vt` contains the transposed right singular vectors (matrix V^T)
//!
//! Singular values are mathematically real. Backends choose the scalar type
//! used to represent them through [`SVD::SingularValue`].
//!```rust,ignore
//!// ----- Singular Value Decomposition (SVD) -----
//!use mdarray_linalg::svd::SVDDecomp;
//!use mdarray_linalg::prelude::*; // Import traits anonymously
//!use mdarray_linalg_backend::Backend; // Use the real backend here, Lapack, Faer, ...
//!
//!let bd = Backend::default();
//!let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");
//!// Or the shorter ...
//!let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");
//!```
use mdarray::{Array, Dim, Layout, Slice};
use thiserror::Error;

/// Error types related to singular value decomposition
#[derive(Debug, Error)]
pub enum SVDError {
    #[error("Backend error code: {0}")]
    BackendError(i32),

    #[error("Inconsistent U and VT: must be both Some or both None")]
    InconsistentUV,

    #[error("Backend failed to converge: {superdiagonals} superdiagonals did not converge to zero")]
    BackendDidNotConverge { superdiagonals: i32 },
}

/// Holds the results of a singular value decomposition, including
/// singular values and the left and right singular vectors.
///
/// `T` is the matrix scalar type, `S` is the singular-value scalar type,
/// and `D` is the matrix dimension type.
pub struct SVDDecomp<T, S, D: Dim> {
    pub s: Array<S, (D,)>,
    pub u: Array<T, (D, D)>,
    pub vt: Array<T, (D, D)>,
}

/// Singular value decomposition for matrix factorization and analysis
pub trait SVD<T, D: Dim> {
    /// Scalar type used for singular values.
    ///
    /// Singular values are mathematically real. Backends choose their
    /// representation; some current backends use the matrix scalar type `T`.
    type SingularValue;

    /// Compute full SVD with new allocated matrices
    fn svd<L: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
    ) -> Result<SVDDecomp<T, Self::SingularValue, D>, SVDError>;

    /// Compute thin SVD with new allocated matrices
    fn svd_thin<L: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
    ) -> Result<SVDDecomp<T, Self::SingularValue, D>, SVDError>;

    /// Compute only singular values with new allocated matrix
    fn svd_s<L: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
    ) -> Result<Array<Self::SingularValue, (D,)>, SVDError>;

    /// Compute SVD, overwriting existing matrices
    fn svd_write<L: Layout, Ls: Layout, Lu: Layout, Lvt: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<Self::SingularValue, (D,), Ls>,
        u: &mut Slice<T, (D, D), Lu>,
        vt: &mut Slice<T, (D, D), Lvt>,
    ) -> Result<(), SVDError>;

    /// Compute only singular values, overwriting existing matrix
    fn svd_write_s<L: Layout, Ls: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<Self::SingularValue, (D,), Ls>,
    ) -> Result<(), SVDError>;
}
