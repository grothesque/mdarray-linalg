//! QR decomposition
//!```rust, ignore
//!use mdarray_linalg_backend::Backend; // Use the real backend here, Lapack, Faer, ...
//!let bd = Backend::default();
//!
//!let a = darray![[2.0_f64, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
//!
//!let (q,r) = bd.qr(&mut a.clone()); // A = QR
//!
//!```
use mdarray::{Array, Dim, Layout, Slice};

/// QR decomposition for orthogonal-triangular factorization
pub trait QR<T, D0: Dim, D1: Dim> {
    /// Compute QR decomposition overwriting existing matrices
    fn qr_write<D2: Dim, L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        q: &mut Slice<T, (D0, D2), Lq>,
        r: &mut Slice<T, (D2, D1), Lr>,
    );

    /// Compute QR decomposition with new allocated matrices
    fn qr<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Array<T, (D0, usize)>, Array<T, (usize, D1)>);
}
