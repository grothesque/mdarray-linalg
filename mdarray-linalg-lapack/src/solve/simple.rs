use super::scalar::LapackScalar;
use mdarray::{DSlice, Layout};
use mdarray_linalg::{SolveError, get_dims, into_i32, to_col_major, transpose_in_place};
use num_complex::ComplexFloat;

pub fn gesv<La: Layout, Lb: Layout, T: ComplexFloat + Default + LapackScalar>(
    a: &mut DSlice<T, 2, La>,
    b: &mut DSlice<T, 2, Lb>,
) -> Result<Vec<i32>, SolveError>
where
    T::Real: Into<T>,
{
    let ((n_a, m_a), (n_b, nrhs)) = get_dims!(a, b);

    if n_a != m_a {
        return Err(SolveError::InvalidDimensions);
    }
    if n_b != n_a {
        return Err(SolveError::InvalidDimensions);
    }

    let n = n_a;
    let mut ipiv = vec![0i32; n as usize];
    let mut info = 0;

    transpose_in_place(a);

    let mut b_col_major = to_col_major(b);

    unsafe {
        T::lapack_gesv(
            n,
            nrhs,
            a.as_mut_ptr(),
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
            // Copy solution back to B (row-major)
            for i in 0..(n as usize) {
                for j in 0..(nrhs as usize) {
                    b[[i, j]] = b_col_major[[j, i]];
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
