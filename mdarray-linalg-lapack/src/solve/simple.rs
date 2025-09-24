use super::scalar::LapackScalar;
use mdarray::{DSlice, DTensor, Layout};
use mdarray_linalg::{SolveError, get_dims, into_i32};
use num_complex::ComplexFloat;

pub fn gesv<La: Layout, Lb: Layout, Lp: Layout, T: ComplexFloat + Default + LapackScalar>(
    a: &mut DSlice<T, 2, La>,
    b: &mut DSlice<T, 2, Lb>,
    p: &mut DSlice<T, 2, Lp>,
) -> Result<Vec<i32>, SolveError>
where
    T::Real: Into<T>,
{
    let ((n_a, m_a), (n_b, nrhs), (n_p, m_p)) = get_dims!(a, b, p);

    if n_a != m_a {
        return Err(SolveError::InvalidDimensions);
    }
    if n_b != n_a {
        return Err(SolveError::InvalidDimensions);
    }
    if n_p != n_a || m_p != n_a {
        return Err(SolveError::InvalidDimensions);
    }

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

    // Handle LAPACK return codes
    match info {
        0 => {
            // Success: copy solution back and build permutation matrix

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

            Ok(ipiv)
        }
        i if i > 0 => {
            // Matrix is singular: U(i,i) is exactly zero
            Err(SolveError::SingularMatrix { diagonal: i })
        }
        i if i < 0 => {
            // Invalid argument
            Err(SolveError::BackendError(i))
        }
        _ => {
            // Should not happen but handle it
            Err(SolveError::BackendError(info))
        }
    }
}
