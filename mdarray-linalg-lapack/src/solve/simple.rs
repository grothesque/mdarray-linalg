use mdarray::{Dim, Layout, Shape, Slice};
use mdarray_linalg::{into_i32, solve::SolveError, to_col_major, transpose_in_place};
use num_complex::ComplexFloat;

use super::scalar::LapackScalar;

pub fn gesv<La: Layout, Lb: Layout, T: ComplexFloat + Default + LapackScalar, D0, D1>(
    a: &mut Slice<T, (D0, D1), La>,
    b: &mut Slice<T, (D0, D1), Lb>,
) -> Result<Vec<i32>, SolveError>
where
    T::Real: Into<T>,
    D0: Dim,
    D1: Dim,
{
    let ash = *a.shape();
    let (n_a, m_a) = (ash.dim(0), ash.dim(1));

    let bsh = *b.shape();
    let (n_b, nrhs) = (bsh.dim(0), bsh.dim(1));

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

    dbg!("ici");

    unsafe {
        T::lapack_gesv(
            into_i32(n),
            into_i32(nrhs),
            a.as_mut_ptr(),
            into_i32(n), // lda
            ipiv.as_mut_ptr(),
            b_col_major.as_mut_ptr(),
            into_i32(n), // ldb
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
