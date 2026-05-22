use mdarray::{Dim, Layout, Shape, Slice};
use mdarray_linalg::lu::InvError;
use num_complex::ComplexFloat;
use num_traits::Zero;

use crate::to_dmatrix;

/// Compute the LU factors and permutation matrix.
pub fn lu<T, D0, D1, L>(
    a: &Slice<T, (D0, D1), L>,
) -> (nalgebra::DMatrix<T>, nalgebra::DMatrix<T>, nalgebra::DMatrix<T>)
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let lu = to_dmatrix(a).lu();
    let (p, l, u) = lu.unpack();

    let mut p_nalgebra = nalgebra::DMatrix::identity(a.shape().dim(0), a.shape().dim(0));
    p.permute_rows(&mut p_nalgebra);

    (l, u, p_nalgebra)
}

/// Compute the inverse of a square matrix.
pub fn inv<T, D0, D1, L>(a: &Slice<T, (D0, D1), L>) -> Result<nalgebra::DMatrix<T>, InvError>
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let m = a.shape().dim(0);
    let n = a.shape().dim(1);

    if m != n {
        return Err(InvError::NotSquare {
            rows: m as i32,
            cols: n as i32,
        });
    }

    to_dmatrix(a)
        .lu()
        .try_inverse()
        .ok_or(InvError::Singular { pivot: 0 })
}

/// Compute the determinant of a square matrix.
pub fn det<T, D0, D1, L>(a: &Slice<T, (D0, D1), L>) -> T
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    to_dmatrix(a).lu().determinant()
}

/// Compute the Cholesky factor of a square positive-definite matrix.
pub fn choleski<T, D0, D1, L>(a: &Slice<T, (D0, D1), L>) -> Result<nalgebra::DMatrix<T>, InvError>
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let m = a.shape().dim(0);
    let n = a.shape().dim(1);

    if m != n {
        return Err(InvError::NotSquare {
            rows: m as i32,
            cols: n as i32,
        });
    }

    to_dmatrix(a)
        .cholesky()
        .map(|chol| chol.unpack())
        .ok_or(InvError::NotPositiveDefinite { lpm: 0 })
}
