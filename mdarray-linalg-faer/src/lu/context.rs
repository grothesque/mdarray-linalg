// LU Decomposition with partial pivoting:
//     P * A = L * U
// where:
//     - A is m × n         (input matrix)
//     - P is m × m        (permutation matrix)
//     - L is m × m        (lower triangular with ones on diagonal)
//     - U is m × n         (upper triangular/trapezoidal matrix)

use dyn_stack::{MemBuffer, MemStack};
use faer_traits::ComplexField;
use mdarray::{Dim, Layout, Shape, Slice, Tensor};
use mdarray_linalg::lu::{InvError, InvResult, LU};
use num_complex::ComplexFloat;

use super::simple::lu_faer;
use crate::{Faer, into_faer_mut};

impl<T, D0: Dim, D1: Dim> LU<T, D0, D1> for Faer
where
    T: ComplexFloat
        + ComplexField
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
{
    /// Computes LU decomposition with new allocated matrices: L, U, P (permutation matrix)
    fn lu<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
    ) -> (
        Tensor<T, (D0, D0)>,
        Tensor<T, (D0, D1)>,
        Tensor<T, (D0, D0)>,
    ) {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        let min_mn = m.min(n);

        // Create shapes for L, U, and P matrices
        let l_shape = <(D0, D0) as Shape>::from_dims(&[m, min_mn]);
        let u_shape = <(D0, D1) as Shape>::from_dims(&[min_mn, n]);
        let p_shape = <(D0, D0) as Shape>::from_dims(&[m, m]);

        let mut l_mda = Tensor::from_elem(l_shape, T::default());
        let mut u_mda = Tensor::from_elem(u_shape, T::default());
        let mut p_mda = Tensor::from_elem(p_shape, T::default());

        lu_faer(a, &mut l_mda, &mut u_mda, &mut p_mda);

        (l_mda, u_mda, p_mda)
    }

    /// Computes LU decomposition overwriting existing matrices
    fn lu_write<L: Layout, Ll: Layout, Lu: Layout, Lp: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        l: &mut Slice<T, (D0, D0), Ll>,
        u: &mut Slice<T, (D0, D1), Lu>,
        p: &mut Slice<T, (D0, D0), Lp>,
    ) {
        lu_faer::<T, D0, D1, L, Ll, Lu, Lp>(a, l, u, p);
    }

    /// Computes inverse with new allocated matrix
    fn inv<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> InvResult<T, D0, D1> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(InvError::NotSquare {
                rows: m as i32,
                cols: n as i32,
            });
        }

        let par = faer::get_global_parallelism();
        let mut a_faer = into_faer_mut(a);

        let mut row_perm_fwd = vec![0usize; m];
        let mut row_perm_bwd = vec![0usize; m];

        faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            a_faer.as_mut(),
            &mut row_perm_fwd,
            &mut row_perm_bwd,
            par,
            MemStack::new(&mut MemBuffer::new(
                faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T>(
                    m,
                    n,
                    par,
                    faer::prelude::default(),
                ),
            )),
            faer::prelude::default(),
        );

        let l_mat = a_faer.as_ref();
        let u_mat = a_faer.as_ref();

        let perm = unsafe {
            faer::perm::Perm::new_unchecked(
                row_perm_fwd.into_boxed_slice(),
                row_perm_bwd.into_boxed_slice(),
            )
        };

        let mut inv_mat = Tensor::<T, (D0, D1)>::from_elem(ash, T::zero());
        let mut inv_mat_faer = into_faer_mut(&mut inv_mat);

        faer::linalg::lu::partial_pivoting::inverse::inverse(
            inv_mat_faer.as_mut(),
            l_mat,
            u_mat,
            perm.as_ref(),
            par,
            MemStack::new(&mut MemBuffer::new(
                faer::linalg::lu::partial_pivoting::inverse::inverse_scratch::<usize, T>(m, par),
            )),
        );
        Ok(inv_mat)
    }

    /// Computes inverse overwriting the input matrix
    fn inv_write<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<(), InvError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(InvError::NotSquare {
                rows: m as i32,
                cols: n as i32,
            });
        }

        let par = faer::get_global_parallelism();
        let mut a_faer = into_faer_mut(a);

        let mut row_perm_fwd = vec![0usize; m];
        let mut row_perm_bwd = vec![0usize; m];

        faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            a_faer.as_mut(),
            &mut row_perm_fwd,
            &mut row_perm_bwd,
            par,
            MemStack::new(&mut MemBuffer::new(
                faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T>(
                    m,
                    n,
                    par,
                    faer::prelude::default(),
                ),
            )),
            faer::prelude::default(),
        );

        let l_mat = a_faer.as_ref();
        let u_mat = a_faer.as_ref();

        let perm = unsafe {
            faer::perm::Perm::new_unchecked(
                row_perm_fwd.into_boxed_slice(),
                row_perm_bwd.into_boxed_slice(),
            )
        };

        let mut inv_mat = faer::Mat::<T>::zeros(m, n);

        faer::linalg::lu::partial_pivoting::inverse::inverse(
            inv_mat.as_mut(),
            l_mat,
            u_mat,
            perm.as_ref(),
            par,
            MemStack::new(&mut MemBuffer::new(
                faer::linalg::lu::partial_pivoting::inverse::inverse_scratch::<usize, T>(m, par),
            )),
        );

        for i in 0..m {
            for j in 0..n {
                a_faer[(i, j)] = inv_mat[(i, j)];
            }
        }

        Ok(())
    }

    /// Computes the determinant of a square matrix. Panics if the matrix is non-square.
    fn det<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> T {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        assert_eq!(m, n, "determinant is only defined for square matrices");
        let a_faer = into_faer_mut(a);
        a_faer.determinant()
    }

    /// Computes the Cholesky decomposition, returning a lower-triangular matrix
    fn choleski<L: Layout>(&self, _a: &mut Slice<T, (D0, D1), L>) -> InvResult<T, D0, D1> {
        todo!("choleski will be implemented later")
    }

    /// Computes the Cholesky decomposition in-place, overwriting the input matrix
    fn choleski_write<L: Layout>(&self, _a: &mut Slice<T, (D0, D1), L>) -> Result<(), InvError> {
        todo!("choleski_write will be implemented later")
    }
}
