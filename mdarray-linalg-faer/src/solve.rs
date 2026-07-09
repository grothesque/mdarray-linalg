// Linear system solver:
//     A * X = B
// where:
//     - A is m × m         (square coefficient matrix)
//     - B is m × n         (right-hand side matrix)
//     - X is m × n         (solution matrix)

use faer::linalg::solvers::Solve as FaerSolve;
use faer_traits::ComplexField;
use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::solve::{Solve, SolveError};
use num_complex::ComplexFloat;

use crate::{Faer, into_faer, into_faer_mut};

impl<T, D: Dim> Solve<T, D> for Faer
where
    T: ComplexFloat
        + ComplexField
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>,
{
    /// Solves linear system AX = B with new allocated solution matrix.
    fn solve<R: Dim, La: Layout, Lb: Layout>(
        &self,
        a: &mut Slice<T, (D, D), La>,
        b: &Slice<T, (D, R), Lb>,
    ) -> Result<Array<T, (D, R)>, SolveError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let bsh = *b.shape();
        let (b_m, b_n) = (bsh.dim(0), bsh.dim(1));

        if m != n {
            return Err(SolveError::InvalidDimensions);
        }

        if b_m != m {
            return Err(SolveError::InvalidDimensions);
        }

        let a_faer = into_faer_mut(a);

        let solver = a_faer.partial_piv_lu();

        let b_faer = into_faer(b);
        let x_faer = solver.solve(b_faer);

        let mut x_mda =
            Array::from_elem(<(D, R) as Shape>::from_dims(&[m, b_n]), T::default());

        let mut x_faer_mut = into_faer_mut(&mut x_mda);
        for i in 0..m {
            for j in 0..b_n {
                x_faer_mut[(i, j)] = x_faer[(i, j)];
            }
        }

        Ok(x_mda)
    }

    /// Solves linear system AX = B, overwriting B with the solution X.
    fn solve_write<R: Dim, La: Layout, Lb: Layout>(
        &self,
        a: &mut Slice<T, (D, D), La>,
        b: &mut Slice<T, (D, R), Lb>,
    ) -> Result<(), SolveError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let bsh = *b.shape();
        let (b_m, b_n) = (bsh.dim(0), bsh.dim(1));

        if m != n {
            return Err(SolveError::InvalidDimensions);
        }

        if b_m != m {
            return Err(SolveError::InvalidDimensions);
        }

        let a_faer = into_faer(a);

        let solver = a_faer.partial_piv_lu();

        let b_faer = into_faer(b).to_owned();
        let x_faer = solver.solve(b_faer);

        let mut b_faer_mut = into_faer_mut(b);
        for i in 0..m {
            for j in 0..b_n {
                b_faer_mut[(i, j)] = x_faer[(i, j)];
            }
        }

        Ok(())
    }
}
