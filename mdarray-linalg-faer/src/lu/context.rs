// LU Decomposition with partial pivoting:
//     P * A = L * U
// where:
//     - A is m × n         (input matrix)
//     - P is m × m        (permutation matrix)
//     - L is m × m        (lower triangular with ones on diagonal)
//     - U is m × n         (upper triangular/trapezoidal matrix)

use dyn_stack::{MemBuffer, MemStack};

use super::simple::lu_faer;
use faer_traits::ComplexField;
use mdarray::{DSlice, DTensor, Layout, tensor};
use mdarray_linalg::{InvError, InvResult, LU};
use num_complex::ComplexFloat;

use crate::{Faer, into_faer_mut, into_mdarray};

impl<T> LU<T> for Faer
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
        a: &mut DSlice<T, 2, L>,
    ) -> (DTensor<T, 2>, DTensor<T, 2>, DTensor<T, 2>) {
        let (m, n) = *a.shape();
        let min_mn = m.min(n);
        let mut l_mda = tensor![[T::default(); min_mn]; m ];
        let mut u_mda = tensor![[T::default(); n ]; min_mn];
        let mut p_mda = tensor![[T::default(); m]; m];

        lu_faer(a, &mut l_mda, &mut u_mda, &mut p_mda);

        (l_mda, u_mda, p_mda)
    }

    /// Computes LU decomposition overwriting existing matrices
    fn lu_overwrite<L: Layout, Ll: Layout, Lu: Layout, Lp: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        l: &mut DSlice<T, 2, Ll>,
        u: &mut DSlice<T, 2, Lu>,
        p: &mut DSlice<T, 2, Lp>,
    ) {
        lu_faer::<T, L, Ll, Lu, Lp>(a, l, u, p);
    }

    /// Computes inverse with new allocated matrix
    fn inv<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> InvResult<T> {
        let (m, n) = *a.shape();

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
        Ok(into_mdarray(inv_mat))
    }

    /// Computes inverse overwriting the input matrix
    fn inv_overwrite<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> Result<(), InvError> {
        let (m, n) = *a.shape();

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
    fn det<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> T {
        let (m, n) = *a.shape();
        assert_eq!(m, n, "determinant is only defined for square matrices");
        let a_faer = into_faer_mut(a);
        a_faer.determinant()
    }

    /// Computes the Cholesky decomposition, returning a lower-triangular matrix
    fn choleski<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> InvResult<T> {
        todo!("choleski will be implemented later")
    }

    /// Computes the Cholesky decomposition in-place, overwriting the input matrix
    fn choleski_overwrite<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> Result<(), InvError> {
        todo!("choleski_overwrite will be implemented later")
    }
}
