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
use mdarray::{DSlice, DTensor, Dense, Layout, tensor};
use mdarray_linalg::Solve;
use mdarray_linalg::{SolveError, SolveResult, SolveResultType, into_i32};
use num_complex::ComplexFloat;

use crate::Lapack;

/// Convert pivot indices to permutation matrix
fn ipiv_to_permutation_matrix<T: ComplexFloat>(ipiv: &[i32], n: usize) -> DTensor<T, 2> {
    let mut p = tensor![[T::zero(); n]; n];

    // Initialize as identity matrix
    for i in 0..n {
        p[[i, i]] = T::one();
    }

    // Apply row swaps according to LAPACK's ipiv convention
    for i in 0..ipiv.len() {
        let pivot_row = (ipiv[i] - 1) as usize; // LAPACK uses 1-based indexing
        if pivot_row != i {
            for j in 0..n {
                let temp = p[[i, j]];
                p[[i, j]] = p[[pivot_row, j]];
                p[[pivot_row, j]] = temp;
            }
        }
    }

    p
}

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
        gesv(a, b, p).map(|_| ())
    }

    fn solve<La: Layout, Lb: Layout>(
        &self,
        a: &mut DSlice<T, 2, La>,
        b: &DSlice<T, 2, Lb>,
    ) -> SolveResultType<T> {
        let ((n, _), (_, nrhs)) = get_dims!(a, b);

        // Create a copy of b since gesv overwrites it
        let mut b_copy = tensor![[T::default(); nrhs as usize]; n as usize];
        for i in 0..(n as usize) {
            for j in 0..(nrhs as usize) {
                b_copy[[i, j]] = b[[i, j]];
            }
        }

        // Create permutation matrix
        let mut p = tensor![[T::zero(); n as usize]; n as usize];

        match gesv::<_, Dense, Dense, T>(a, &mut b_copy, &mut p) {
            Ok(_ipiv) => Ok(SolveResult { x: b_copy, p }),
            Err(e) => Err(e),
        }
    }
}
