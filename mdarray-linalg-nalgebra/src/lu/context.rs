//! LU Decomposition with partial pivoting:
//!     P * A = L * U
//! where:
//!     - A is m × n (input matrix)
//!     - P is m × m (permutation matrix)
//!     - L is m × min(m,n) (lower triangular with unit diagonal)
//!     - U is min(m,n) × n (upper triangular)
//!
//! This module also provides matrix inversion, determinant computation, and Cholesky
//! decomposition through nalgebra's public decomposition API.

use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::lu::{InvError, InvResult, LU};
use num_complex::ComplexFloat;
use num_traits::Zero;

use super::simple::{choleski, det, inv, lu};
use crate::{Nalgebra, write_dmatrix};

impl<T, D0: Dim, D1: Dim> LU<T, D0, D1> for Nalgebra
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
{
    fn lu_write<L: Layout, Ll: Layout, Lu: Layout, Lp: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        l: &mut Slice<T, (D0, D0), Ll>,
        u: &mut Slice<T, (D0, D1), Lu>,
        p: &mut Slice<T, (D0, D0), Lp>,
    ) {
        let (l_nalgebra, u_nalgebra, p_nalgebra) = lu(a);
        write_dmatrix(&l_nalgebra, l);
        write_dmatrix(&u_nalgebra, u);
        write_dmatrix(&p_nalgebra, p);
    }

    fn lu<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (Array<T, (D0, D0)>, Array<T, (D0, D1)>, Array<T, (D0, D0)>) {
        let m = a.shape().dim(0);
        let n = a.shape().dim(1);
        let k = m.min(n);

        let mut l = Array::from_elem(<(D0, D0) as Shape>::from_dims(&[m, k]), T::zero());
        let mut u = Array::from_elem(<(D0, D1) as Shape>::from_dims(&[k, n]), T::zero());
        let mut p = Array::from_elem(<(D0, D0) as Shape>::from_dims(&[m, m]), T::zero());

        self.lu_write(a, &mut l, &mut u, &mut p);
        (l, u, p)
    }

    fn inv_write<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<(), InvError> {
        let inv_nalgebra = inv(a)?;
        write_dmatrix(&inv_nalgebra, a);
        Ok(())
    }

    fn inv<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> InvResult<T, D0, D1> {
        let m = a.shape().dim(0);
        let n = a.shape().dim(1);
        let inv_nalgebra = inv(a)?;
        let mut out = Array::from_elem(<(D0, D1) as Shape>::from_dims(&[m, n]), T::zero());
        write_dmatrix(&inv_nalgebra, &mut out);
        Ok(out)
    }

    fn det<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> T {
        let m = a.shape().dim(0);
        let n = a.shape().dim(1);
        assert_eq!(m, n, "determinant is only defined for square matrices");
        det(a)
    }

    fn choleski<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> InvResult<T, D0, D1> {
        let m = a.shape().dim(0);
        let n = a.shape().dim(1);
        let chol_nalgebra = choleski(a)?;
        let mut out = Array::from_elem(<(D0, D1) as Shape>::from_dims(&[m, n]), T::zero());
        write_dmatrix(&chol_nalgebra, &mut out);
        Ok(out)
    }

    fn choleski_write<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<(), InvError> {
        let chol_nalgebra = choleski(a)?;
        write_dmatrix(&chol_nalgebra, a);
        Ok(())
    }
}
