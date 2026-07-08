//! QR Decomposition:
//!     A = Q * R
//! where:
//!     - A is m × n (input matrix)
//!     - Q is m × k (orthogonal/unitary matrix)
//!     - R is k × n (upper triangular matrix)
//!
//! `qr()` returns the reduced factorization where k = min(m, n):
//! Q is m × k and R is k × n.  `qr_write()` infers the requested mode
//! from the shapes of the provided output matrices, so callers can request
//! either reduced or complete QR by choosing output shapes.
//!
//! The implementation wraps nalgebra's Householder QR decomposition:
//! - reduced mode uses the explicit `q()` and `r()` factors,
//! - complete mode reconstructs the square Q from the stored reflectors and pads R with zeros.

use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::qr::QR;
use num_complex::ComplexFloat;
use num_traits::Zero;

use super::simple::{qr_complete, qr_reduced};
use crate::{Nalgebra, write_dmatrix};

impl<T, D0: Dim, D1: Dim> QR<T, D0, D1> for Nalgebra
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
{
    fn qr_write<D2: Dim, L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        q: &mut Slice<T, (D0, D2), Lq>,
        r: &mut Slice<T, (D2, D1), Lr>,
    ) {
        let m = a.shape().dim(0);
        let n = a.shape().dim(1);
        let k = m.min(n);
        let q_cols = q.shape().dim(1);
        let r_rows = r.shape().dim(0);

        assert_eq!(q_cols, r_rows, "q columns must match r rows");

        if q_cols == k {
            let (q_nalgebra, r_nalgebra) = qr_reduced(a);
            write_dmatrix(&q_nalgebra, q);
            write_dmatrix(&r_nalgebra, r);
        } else if q_cols == m {
            let (q_nalgebra, r_nalgebra) = qr_complete(a);
            write_dmatrix(&q_nalgebra, q);
            write_dmatrix(&r_nalgebra, r);
        } else {
            panic!("unsupported QR output shapes");
        }
    }

    fn qr<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Array<T, (D0, usize)>, Array<T, (usize, D1)>) {
        let m = a.shape().dim(0);
        let n = a.shape().dim(1);
        let k = m.min(n);

        let (q_nalgebra, r_nalgebra) = qr_reduced(a);

        let mut q = Array::from_elem(
            <(D0, usize) as Shape>::from_dims(&[m, k]),
            T::zero(),
        );
        let mut r = Array::from_elem(
            <(usize, D1) as Shape>::from_dims(&[k, n]),
            T::zero(),
        );

        write_dmatrix(&q_nalgebra, &mut q);
        write_dmatrix(&r_nalgebra, &mut r);

        (q, r)
    }
}
