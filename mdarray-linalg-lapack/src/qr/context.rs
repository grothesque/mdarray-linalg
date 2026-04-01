//! QR Decomposition:
//!     A = Q * R
//! where:
//!     - A is m × n (input matrix)
//!     - Q is m × k (orthogonal matrix)
//!     - R is k × n (upper triangular matrix)
//!
//! This implementation supports two modes controlled by `QRConfig`, accessible via Lapack::default().config_qr(mode);:
//! - **Reduced** (default / economy mode): k = min(m, n)
//!   - Q is m × min(m,n)
//!   - R is min(m,n) × n
//! - **Complete** (full mode):
//!   - Q is m × m (orthogonal)
//!   - R is m × n (upper triangular, with zeros below the diagonal)
//!
//! The implementation wraps two LAPACK routines:
//! - `geqrf`: Computes the QR factorization in a compact (implicit) form using
//!   blocked Householder reflectors.
//! - `orgqr` / `ungqr`: Explicitly generates the orthogonal matrix Q from the
//!   reflectors returned by `geqrf`, producing either the thin Q (Reduced) or
//!   the square Q (Complete) depending on [`QRConfig`].

use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::qr::QR;
use num_complex::ComplexFloat;

use crate::QRConfig;

use super::{
    scalar::{LapackScalar, NeedsRwork},
    simple::geqrf,
};
use crate::Lapack;

impl<T, D0: Dim, D1: Dim> QR<T, D0, D1> for Lapack
where
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
    T::Real: Into<T>,
{
    fn qr_write<D2: Dim, L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        q: &mut Slice<T, (D0, D2), Lq>,
        r: &mut Slice<T, (D2, D1), Lr>,
    ) {
        geqrf(a, q, r, self.qr_config)
    }

    fn qr<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Array<T, (D0, usize)>, Array<T, (usize, D1)>) {
        let ash = *a.shape();
        let m = ash.dim(0);
        let n = ash.dim(1);
        let min_mn = m.min(n);

        let (q_rows, q_cols, r_rows, r_cols) = match self.qr_config {
            QRConfig::Reduced => (m, min_mn, min_mn, n),
            QRConfig::Complete => (m, m, m, n),
        };

        let q_shape = <(D0, usize) as Shape>::from_dims(&[q_rows, q_cols]);
        let r_shape = <(usize, D1) as Shape>::from_dims(&[r_rows, r_cols]);

        let mut q: Array<T, (D0, usize)> = Array::from_elem(q_shape, T::default());
        let mut r: Array<T, (usize, D1)> = Array::from_elem(r_shape, T::default());

        geqrf(a, &mut q, &mut r, self.qr_config);

        (q, r)
    }
}
