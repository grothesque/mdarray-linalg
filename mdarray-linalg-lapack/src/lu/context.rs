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

use super::simple::getrf;
use mdarray_linalg::get_dims;

use super::scalar::LapackScalar;
use mdarray::{DSlice, DTensor, Dense, Layout, tensor};
use mdarray_linalg::LU;
use mdarray_linalg::into_i32;
use num_complex::ComplexFloat;
use std::mem::MaybeUninit;

use crate::Lapack;

/// Convert pivot indices to permutation matrix
fn ipiv_to_permutation_matrix<T: ComplexFloat>(ipiv: &[i32], m: usize) -> DTensor<T, 2> {
    let mut p = tensor![[T::zero(); m]; m];

    // Initialize as identity matrix
    for i in 0..m {
        p[[i, i]] = T::one();
    }

    // Apply row swaps according to LAPACK's ipiv convention
    for i in 0..ipiv.len() {
        let pivot_row = (ipiv[i] - 1) as usize; // LAPACK uses 1-based indexing
        if pivot_row != i {
            // Swap rows i and pivot_row
            for j in 0..m {
                let temp = p[[i, j]];
                p[[i, j]] = p[[pivot_row, j]];
                p[[pivot_row, j]] = temp;
            }
        }
    }

    p
}

impl<T> LU<T> for Lapack
where
    T: ComplexFloat + Default + LapackScalar,
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

        // Convert pivot indices to permutation matrix
        let p_matrix = ipiv_to_permutation_matrix::<T>(&ipiv, m as usize);

        // Copy to output permutation matrix
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

        // Convert pivot indices to permutation matrix
        let p_matrix = ipiv_to_permutation_matrix::<T>(&ipiv, m as usize);

        (l, u, p_matrix)
    }
}
