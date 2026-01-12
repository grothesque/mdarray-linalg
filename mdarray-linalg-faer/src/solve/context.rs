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

use faer::linalg::solvers::Solve as FaerSolve;
use faer_traits::ComplexField;
use mdarray::{Dim, Layout, Shape, Slice, Tensor};
use mdarray_linalg::{
    identity,
    solve::{Solve, SolveError, SolveResult, SolveResultType},
};
use num_complex::ComplexFloat;

use crate::{Faer, into_faer, into_faer_mut};

impl<T, D0: Dim, D1: Dim> Solve<T, D0, D1> for Faer
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
        a: &mut Slice<T, (D0, D1), La>,
        b: &Slice<T, (D0, D1), Lb>,
    ) -> SolveResultType<T, D0, D1> {
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

        let mut x_mda = Tensor::from_elem(<(D0, D1) as Shape>::from_dims(&[m, b_n]), T::default());

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
    fn solve_write<La: Layout, Lb: Layout, Lp: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), La>,
        b: &mut Slice<T, (D0, D1), Lb>,
        p: &mut Slice<T, (D0, D1), Lp>,
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

        let _par = faer::get_global_parallelism();
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
