//! QR decomposition
use mdarray::{DSlice, DTensor, Layout};

/// QR decomposition for orthogonal-triangular factorization
pub trait QR<T> {
    /// Compute QR decomposition overwriting existing matrices
    fn qr_overwrite<L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        q: &mut DSlice<T, 2, Lq>,
        r: &mut DSlice<T, 2, Lr>,
    );

    /// Compute QR decomposition with new allocated matrices
    fn qr<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> (DTensor<T, 2>, DTensor<T, 2>);
}
