//! Linear System Solver using LU decomposition (GESV):
//!     AX = B
//! where:
//!     - A is n × n (square coefficient matrix, overwritten with LU factorization)
//!     - X is n × nrhs (solution matrix)
//!     - B is n × nrhs (right-hand side matrix, overwritten with solution)
//!     - P is n × n (permutation matrix from LU decomposition)
//!
//! The function `gesv` (LAPACK) solves a system of linear equations AX = B using LU decomposition with partial pivoting.
//! It computes the LU factorization of A and then uses it to solve the linear system.
//! The matrix A is overwritten by its LU factorization, and B is overwritten by the solution X.

use super::simple::gesv;
use mdarray_linalg::get_dims;

use super::scalar::LapackScalar;
use mdarray::{DSlice, Dense, Layout, tensor};
use mdarray_linalg::into_i32;
use mdarray_linalg::ipiv_to_perm_mat;
use mdarray_linalg::solve::{Solve, SolveError, SolveResult, SolveResultType};
use num_complex::ComplexFloat;

use crate::Lapack;

impl<T> Solve<T> for Lapack
where
    T: ComplexFloat + Default + LapackScalar,
    T::Real: Into<T>,
{
    fn solve_overwrite<La: Layout, Lb: Layout, Lp: Layout>(
        &self,
        a: &mut DSlice<T, 2, La>,
        b: &mut DSlice<T, 2, Lb>,
        p: &mut DSlice<T, 2, Lp>,
    ) -> Result<(), SolveError> {
        let ipiv = gesv::<_, Lb, T>(a, b).unwrap();
        let (n, _) = *a.shape();
        let p_matrix = ipiv_to_perm_mat(&ipiv, n);
        for i in 0..n {
            for j in 0..n {
                p[[i, j]] = p_matrix[[i, j]];
            }
        }
        Ok(())
    }

    fn solve<La: Layout, Lb: Layout>(
        &self,
        a: &mut DSlice<T, 2, La>,
        b: &DSlice<T, 2, Lb>,
    ) -> SolveResultType<T> {
        let ((n, _), (_, nrhs)) = get_dims!(a, b);

        let mut b_copy = tensor![[T::default(); nrhs as usize]; n as usize];
        for i in 0..(n as usize) {
            for j in 0..(nrhs as usize) {
                b_copy[[i, j]] = b[[i, j]];
            }
        }

        match gesv::<_, Dense, T>(a, &mut b_copy) {
            Ok(ipiv) => Ok(SolveResult {
                x: b_copy,
                p: ipiv_to_perm_mat(&ipiv, n as usize),
            }),
            Err(e) => Err(e),
        }
    }
}
