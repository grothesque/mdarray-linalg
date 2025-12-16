//! QR decomposition
use mdarray::{DSlice, DTensor, Dim, Layout, Slice};

/// QR decomposition for orthogonal-triangular factorization
pub trait QR<T, D0: Dim, D1: Dim> {
    /// Compute QR decomposition overwriting existing matrices
    fn qr_write<L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        q: &mut DSlice<T, 2, Lq>,
        r: &mut DSlice<T, 2, Lr>,
    );

    /// Compute QR decomposition with new allocated matrices
    fn qr<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> (DTensor<T, 2>, DTensor<T, 2>);
}
