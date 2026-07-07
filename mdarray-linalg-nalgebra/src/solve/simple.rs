use mdarray::{Dim, Layout, Shape, Slice};
use mdarray_linalg::solve::SolveError;
use num_complex::ComplexFloat;
use num_traits::Zero;

use crate::to_dmatrix;

/// Validate the dimensions of a linear system A * X = B.
pub(super) fn validate_solve_dims<T, D0, D1, La, Lb>(
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D0, D1), Lb>,
) -> Result<(), SolveError>
where
    D0: Dim,
    D1: Dim,
    La: Layout,
    Lb: Layout,
{
    let m = a.shape().dim(0);
    let n = a.shape().dim(1);
    let b_rows = b.shape().dim(0);

    if m != n || b_rows != m {
        return Err(SolveError::InvalidDimensions);
    }

    Ok(())
}

/// Solve A * X = B and return the packed LU matrix, the solution, and P.
pub(super) fn solve<T, D0, D1, La, Lb>(
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D0, D1), Lb>,
) -> Result<(nalgebra::DMatrix<T>, nalgebra::DMatrix<T>, nalgebra::DMatrix<T>), SolveError>
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
    D0: Dim,
    D1: Dim,
    La: Layout,
    Lb: Layout,
{
    validate_solve_dims(a, b)?;

    let n = a.shape().dim(0);
    let lu = to_dmatrix(a).lu();
    let packed_lu = lu.lu_internal().clone_owned();

    let x = lu.solve(&to_dmatrix(b)).ok_or_else(|| {
        let diagonal = (0..n)
            .find(|&i| packed_lu[(i, i)].is_zero())
            .map(|i| i as i32 + 1)
            .unwrap_or(0);

        SolveError::SingularMatrix { diagonal }
    })?;

    let mut p = nalgebra::DMatrix::identity(n, n);
    lu.p().permute_rows(&mut p);

    Ok((packed_lu, x, p))
}
