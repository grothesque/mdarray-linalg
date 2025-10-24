//! LU Decomposition:
//!     P * A = L * U
//! where:
//!     - A is m × n (input matrix)
//!     - P is m × m (permutation matrix, represented by pivot vector)
//!     - L is m × min(m,n) (lower triangular matrix with unit diagonal)
//!     - U is min(m,n) × n (upper triangular matrix)
//! This decomposition is used to solve linear systems, compute matrix determinants, and matrix inversion.
//! The function `getrf` (LAPACK) computes the LU factorization of a general m-by-n matrix A using partial pivoting.
//! The matrix L is lower triangular with unit diagonal, and U is upper triangular.
use super::simple::{getrf, getri, potrf};
use mdarray_linalg::{get_dims, ipiv_to_perm_mat, transpose_in_place};

use super::scalar::{LapackScalar, Workspace};
use mdarray::{DSlice, DTensor, Dense, Layout, tensor};
use mdarray_linalg::into_i32;
use mdarray_linalg::lu::{InvError, InvResult, LU};
use num_complex::ComplexFloat;

use crate::Lapack;

impl<T> LU<T> for Lapack
where
    T: ComplexFloat + Default + LapackScalar + Workspace,
    T::Real: Into<T>,
{
    fn lu_overwrite<L: Layout, Ll: Layout, Lu: Layout, Lp: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        l: &mut DSlice<T, 2, Ll>,
        u: &mut DSlice<T, 2, Lu>,
        p: &mut DSlice<T, 2, Lp>,
    ) {
        let (m, _) = get_dims!(a);
        let ipiv = getrf(a, l, u);

        let p_matrix = ipiv_to_perm_mat::<T>(&ipiv, m as usize);

        for i in 0..(m as usize) {
            for j in 0..(m as usize) {
                p[[i, j]] = p_matrix[[i, j]];
            }
        }
    }

    fn lu<L: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
    ) -> (DTensor<T, 2>, DTensor<T, 2>, DTensor<T, 2>) {
        let (m, n) = get_dims!(a);
        let min_mn = m.min(n);
        let mut l = tensor![[T::default(); min_mn as usize]; m as usize];
        let mut u = tensor![[T::default(); n as usize]; min_mn as usize];
        let ipiv = getrf::<_, Dense, Dense, T>(a, &mut l, &mut u);

        let p_matrix = ipiv_to_perm_mat::<T>(&ipiv, m as usize);

        (l, u, p_matrix)
    }

    fn inv_overwrite<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> Result<(), InvError> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(InvError::NotSquare { rows: m, cols: n });
        }

        let min_mn = m.min(n);
        let mut l = DTensor::<T, 2>::zeros([m as usize, min_mn as usize]);
        let mut u = DTensor::<T, 2>::zeros([min_mn as usize, n as usize]);
        let mut ipiv = getrf::<_, Dense, Dense, T>(a, &mut l, &mut u);

        match getri::<_, T>(a, &mut ipiv) {
            0 => Ok(()),
            i if i > 0 => Err(InvError::Singular { pivot: i }),
            i => Err(InvError::BackendError(i)),
        }
    }

    fn inv<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> InvResult<T> {
        let (m, n) = get_dims!(a);
        if m != n {
            return Err(InvError::NotSquare { rows: m, cols: n });
        }

        let mut a_inv = DTensor::<T, 2>::zeros([n as usize, n as usize]);
        for i in 0..n as usize {
            for j in 0..m as usize {
                a_inv[[i, j]] = a[[i, j]];
            }
        }

        let min_mn = m.min(n);
        let mut l = DTensor::<T, 2>::zeros([m as usize, min_mn as usize]);
        let mut u = DTensor::<T, 2>::zeros([min_mn as usize, n as usize]);
        let mut ipiv = getrf::<_, Dense, Dense, T>(&mut a_inv, &mut l, &mut u);

        match getri::<_, T>(&mut a_inv, &mut ipiv) {
            0 => Ok(a_inv),
            i if i > 0 => Err(InvError::Singular { pivot: i }),
            i => Err(InvError::BackendError(i)),
        }
    }

    fn det<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> T {
        let (m, n) = get_dims!(a);
        assert_eq!(m, n, "determinant is only defined for square matrices");

        let mut l = tensor![[T::default(); n as usize]; n as usize];
        let mut u = tensor![[T::default(); n as usize]; n as usize];

        let ipiv = getrf::<_, Dense, Dense, T>(a, &mut l, &mut u);

        let mut det = T::one();
        for i in 0..n as usize {
            det = det * u[[i, i]];
        }

        let mut sign = T::one();
        for (i, &pivot) in ipiv.iter().enumerate() {
            if (i as i32) != (pivot - 1) {
                sign = sign * (-T::one());
            }
        }
        det * sign
    }

    /// Computes the Cholesky decomposition, returning a lower-triangular matrix
    fn choleski<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> InvResult<T> {
        let (m, n) = get_dims!(a);
        assert_eq!(m, n, "Matrix must be square for Cholesky decomposition");

        let mut l = DTensor::<T, 2>::zeros([m as usize, n as usize]);

        match potrf::<_, T>(a, 'L') {
            0 => {
                for i in 0..(m as usize) {
                    for j in 0..(n as usize) {
                        if i >= j {
                            l[[i, j]] = a[[j, i]];
                        } else {
                            l[[i, j]] = T::zero();
                        }
                    }
                }
                Ok(l)
            }
            i if i > 0 => Err(InvError::NotPositiveDefinite { lpm: i }),
            i => Err(InvError::BackendError(i)),
        }
    }

    /// Computes the Cholesky decomposition in-place, overwriting the input matrix
    fn choleski_overwrite<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> Result<(), InvError> {
        let (m, n) = get_dims!(a);
        assert_eq!(m, n, "Matrix must be square for Cholesky decomposition");

        match potrf::<_, T>(a, 'L') {
            0 => {
                transpose_in_place(a);
                Ok(())
            }
            i if i > 0 => Err(InvError::NotPositiveDefinite { lpm: i }),
            i => Err(InvError::BackendError(i)),
        }
    }
}
