//! QR decomposition
use mdarray::{Dim, Layout, Slice, Array};

/// QR decomposition for orthogonal-triangular factorization
pub trait QR<T, D0: Dim, D1: Dim> {
    /// Compute QR decomposition overwriting existing matrices
    fn qr_write<L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        q: &mut Slice<T, (D0, D1), Lq>,
        r: &mut Slice<T, (D0, D1), Lr>,
    );

    /// Compute QR decomposition with new allocated matrices
    fn qr<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Array<T, (D0, D1)>, Array<T, (D0, D1)>);
}
