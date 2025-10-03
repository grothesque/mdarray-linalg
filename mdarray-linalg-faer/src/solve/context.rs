// Linear system solver using LU decomposition:
//     A * X = B
// is solved by computing the LU decomposition with partial pivoting:
//     P * A = L * U
// then solving:
//     L * Y = P * B  (forward substitution)
//     U * X = Y      (backward substitution)
// where:
//     - A is m × m         (square coefficient matrix, overwritten with LU)
//     - B is m × n         (right-hand side matrix)
//     - X is m × n         (solution matrix)
//     - P is m × m         (permutation matrix)
//     - L is m × m         (lower triangular with ones on diagonal)
//     - U is m × m         (upper triangular)

use faer_traits::ComplexField;
use mdarray::{DSlice, Layout, tensor};
use mdarray_linalg::{
    Solve, SolveError, SolveResult, SolveResultType, identity, into_faer, into_faer_mut,
};
use num_complex::ComplexFloat;

use faer::linalg::solvers::Solve as FaerSolve;

use crate::Faer;

impl<T> Solve<T> for Faer
where
    T: ComplexFloat
        + ComplexField
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
{
    /// Solves linear system AX = B with new allocated solution matrix
    /// A is modified (overwritten with LU decomposition)
    /// Returns the solution X and P the permutation matrix (identity in that case), or error
    fn solve<La: Layout, Lb: Layout>(
        &self,
        a: &mut DSlice<T, 2, La>,
        b: &DSlice<T, 2, Lb>,
    ) -> SolveResultType<T> {
        let (m, n) = *a.shape();
        let (b_m, b_n) = *b.shape();

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

        let mut x_mda = tensor![[T::default(); b_n]; m];
        let mut x_faer_mut = into_faer_mut(&mut x_mda);
        for i in 0..m {
            for j in 0..b_n {
                x_faer_mut[(i, j)] = x_faer[(i, j)];
            }
        }

        let p_mda = identity(m); // No permutation with this routine

        Ok(SolveResult { x: x_mda, p: p_mda })
    }

    /// Solves linear system AX = b overwriting existing matrices
    /// A is overwritten with its LU decomposition
    /// B is overwritten with the solution X
    /// P is filled with the permutation matrix such that P*A = L*U (here P = identity)
    /// Returns Ok(()) on success, Err(SolveError) on failure
    fn solve_overwrite<La: Layout, Lb: Layout, Lp: Layout>(
        &self,
        a: &mut DSlice<T, 2, La>,
        b: &mut DSlice<T, 2, Lb>,
        p: &mut DSlice<T, 2, Lp>,
    ) -> Result<(), SolveError> {
        let (m, n) = *a.shape();
        let (b_m, b_n) = *b.shape();

        if m != n {
            return Err(SolveError::InvalidDimensions);
        }

        if b_m != m {
            return Err(SolveError::InvalidDimensions);
        }

        let par = faer::get_global_parallelism();
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

        let mut p_faer = into_faer_mut(p);
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    p_faer[(i, j)] = T::zero();
                } else {
                    p_faer[(i, j)] = T::one();
                }
            }
        }

        Ok(())
    }
}
