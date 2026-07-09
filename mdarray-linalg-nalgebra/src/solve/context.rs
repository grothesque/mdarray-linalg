//! Linear system solver using LU decomposition:
//!     A * X = B
//! where:
//!     - A is n × n (square coefficient matrix)
//!     - X is n × nrhs (solution matrix)
//!     - B is n × nrhs (right-hand side matrix)
//!     - P is n × n (permutation matrix from partial pivoting)
//!
//! The implementation uses nalgebra's LU decomposition with partial row pivoting.
//! `solve()` returns a freshly allocated solution matrix, while `solve_write()` also writes the
//! packed LU factors back into `a` and stores the dense permutation matrix into `p`.

use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::solve::{Solution, Solve, SolveError};
use num_complex::ComplexFloat;
use num_traits::Zero;

use super::simple::solve;
use crate::{Nalgebra, write_dmatrix};

impl<T, D: Dim> Solve<T, D> for Nalgebra
where
    T: nalgebra::ComplexField + ComplexFloat + Zero + Copy,
{
    fn solve_write<R: Dim, La: Layout, Lb: Layout, Lp: Layout>(
        &self,
        a: &mut Slice<T, (D, D), La>,
        b: &mut Slice<T, (D, R), Lb>,
        p: &mut Slice<T, (D, D), Lp>,
    ) -> Result<(), SolveError> {
        let (packed_lu, x, p_nalgebra) = solve(a, b)?;
        write_dmatrix(&packed_lu, a);
        write_dmatrix(&x, b);
        write_dmatrix(&p_nalgebra, p);
        Ok(())
    }

    fn solve<R: Dim, La: Layout, Lb: Layout>(
        &self,
        a: &mut Slice<T, (D, D), La>,
        b: &Slice<T, (D, R), Lb>,
    ) -> Result<Solution<T, D, R>, SolveError> {
        let n = a.shape().dim(0);
        let nrhs = b.shape().dim(1);

        let (packed_lu, x_nalgebra, p_nalgebra) = solve(a, b)?;
        write_dmatrix(&packed_lu, a);

        let mut x = Array::from_elem(<(D, R) as Shape>::from_dims(&[n, nrhs]), T::zero());
        let mut p = Array::from_elem(<(D, D) as Shape>::from_dims(&[n, n]), T::zero());

        write_dmatrix(&x_nalgebra, &mut x);
        write_dmatrix(&p_nalgebra, &mut p);

        Ok(Solution { x, p })
    }
}
