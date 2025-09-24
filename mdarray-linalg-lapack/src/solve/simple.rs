use super::scalar::LapackScalar;
use mdarray::{DSlice, DTensor, Layout};
use mdarray_linalg::{get_dims, into_i32};
use num_complex::ComplexFloat;

pub fn gesv<La: Layout, Lb: Layout, Lp: Layout, T: ComplexFloat + Default + LapackScalar>(
    a: &mut DSlice<T, 2, La>,
    b: &mut DSlice<T, 2, Lb>,
    p: &mut DSlice<T, 2, Lp>,
) -> Vec<i32>
where
    T::Real: Into<T>,
{
    let ((n_a, m_a), (n_b, nrhs), (n_p, m_p)) = get_dims!(a, b, p);

    assert_eq!(n_a, m_a, "Matrix A must be square");
    assert_eq!(n_b, n_a, "B must have same number of rows as A");
    assert_eq!(n_p, n_a, "Permutation matrix P must have same size as A");
    assert_eq!(m_p, n_a, "Permutation matrix P must be square");

    let n = n_a;
    let mut ipiv = vec![0i32; n as usize];
    let mut info = 0;

    // Convert A to column-major format
    let mut a_col_major = DTensor::<T, 2>::zeros([n as usize, n as usize]);
    for i in 0..(n as usize) {
        for j in 0..(n as usize) {
            a_col_major[[j, i]] = a[[i, j]];
        }
    }

    // Convert B to column-major format
    let mut b_col_major = DTensor::<T, 2>::zeros([nrhs as usize, n as usize]);
    for i in 0..(n as usize) {
        for j in 0..(nrhs as usize) {
            b_col_major[[j, i]] = b[[i, j]];
        }
    }

    unsafe {
        T::lapack_gesv(
            n,
            nrhs,
            a_col_major.as_mut_ptr(),
            n, // lda
            ipiv.as_mut_ptr(),
            b_col_major.as_mut_ptr(),
            n, // ldb
            &mut info,
        );
    }

    // Copy solution back to B (row-major)
    for i in 0..(n as usize) {
        for j in 0..(nrhs as usize) {
            b[[i, j]] = b_col_major[[j, i]];
        }
    }

    // Build permutation matrix P from ipiv
    // Initialize P as identity matrix
    for i in 0..(n as usize) {
        for j in 0..(n as usize) {
            if i == j {
                p[[i, j]] = T::one();
            } else {
                p[[i, j]] = T::zero();
            }
        }
    }

    // Apply permutations from ipiv (LAPACK uses 1-based indexing)
    for k in 0..(n as usize) {
        let pivot_row = (ipiv[k] - 1) as usize; // Convert to 0-based
        if pivot_row != k {
            // Swap rows k and pivot_row in P
            for j in 0..(n as usize) {
                let temp = p[[k, j]];
                p[[k, j]] = p[[pivot_row, j]];
                p[[pivot_row, j]] = temp;
            }
        }
    }

    ipiv
}
