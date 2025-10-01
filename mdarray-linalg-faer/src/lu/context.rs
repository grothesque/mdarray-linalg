// LU Decomposition with partial pivoting:
//     P * A = L * U
// where:
//     - A is m × n         (input matrix)
//     - P is m × m        (permutation matrix)
//     - L is m × m        (lower triangular with ones on diagonal)
//     - U is m × n         (upper triangular/trapezoidal matrix)

use super::simple::lu_faer;
use faer_traits::ComplexField;
use mdarray::{DSlice, DTensor, Layout, tensor};
use mdarray_linalg::{InvError, InvResult, LU};
use num_complex::ComplexFloat;

use crate::Faer;

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
        // let mut l_mda = tensor![[T::default(); m]; m];
        // let mut u_mda = tensor![[T::default(); n]; m];
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
        todo!("inv will be implemented later")
    }

    /// Computes inverse overwriting the input matrix
    fn inv_overwrite<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> Result<(), InvError> {
        todo!("inv_overwrite will be implemented later")
    }

    /// Computes the determinant of a square matrix. Panics if the matrix is non-square.
    fn det<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> T {
        todo!("det will be implemented later")
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
