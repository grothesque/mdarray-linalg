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

use mdarray::{Dense, Dim, Layout, Shape, Slice, Tensor};
use mdarray_linalg::{
    ipiv_to_perm_mat,
    solve::{Solve, SolveError, SolveResult, SolveResultType},
};
use num_complex::ComplexFloat;

use super::{scalar::LapackScalar, simple::gesv};
use crate::Lapack;

impl<T, D0: Dim, D1: Dim> Solve<T, D0, D1> for Lapack
where
    T: ComplexFloat + Default + LapackScalar,
    T::Real: Into<T>,
{
    fn solve_write<La: Layout, Lb: Layout, Lp: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), La>,
        b: &mut Slice<T, (D0, D1), Lb>,
        p: &mut Slice<T, (D0, D1), Lp>,
    ) -> Result<(), SolveError> {
        let ipiv = gesv::<_, Lb, T, D0, D1>(a, b).unwrap();
        let ash = *a.shape();
        let n = ash.dim(0);

        let p_matrix = ipiv_to_perm_mat::<T, usize, usize>(&ipiv, n);
        for i in 0..n {
            for j in 0..n {
                p[[i, j]] = p_matrix[[i, j]];
            }
        }
        Ok(())
    }

    fn solve<La: Layout, Lb: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), La>,
        b: &Slice<T, (D0, D1), Lb>,
    ) -> SolveResultType<T, D0, D1> {
        let ash = *a.shape();
        let bsh = *b.shape();

        let n = ash.dim(0);
        let nrhs = bsh.dim(1);

        let mut b_copy =
            Tensor::from_elem(<(D0, D1) as Shape>::from_dims(&[n, nrhs]), T::default());

        for i in 0..n {
            for j in 0..nrhs {
                b_copy[[i, j]] = b[[i, j]];
            }
        }

        match gesv::<_, Dense, T, D0, D1>(a, &mut b_copy) {
            Ok(ipiv) => Ok(SolveResult {
                x: b_copy,
                p: ipiv_to_perm_mat(&ipiv, n),
            }),
            Err(e) => Err(e),
        }
    }
}
