use mdarray::{Dim, Layout, Shape, Slice};
use mdarray_linalg::solve::SolveError;
use num_complex::ComplexFloat;
use num_traits::Zero;

use crate::to_dmatrix;

/// Validate the dimensions of a linear system A * X = B.
pub(super) fn validate_solve_dims<T, D, R, La, Lb>(
    a: &Slice<T, (D, D), La>,
    b: &Slice<T, (D, R), Lb>,
) -> Result<(), SolveError>
where
    D: Dim,
    R: Dim,
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

/// Solve A * X = B and return the packed LU matrix and solution.
pub(super) fn solve<T, D, R, La, Lb>(
    a: &Slice<T, (D, D), La>,
    b: &Slice<T, (D, R), Lb>,
) -> Result<(nalgebra::DMatrix<T>, nalgebra::DMatrix<T>), SolveError>
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
    D: Dim,
    R: Dim,
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

    Ok((packed_lu, x))
}
