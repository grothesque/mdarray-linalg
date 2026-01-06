//! LU Decomposition:
//!     P * A = L * U
//! where:
//!     - A is m × n (input matrix)
//!     - P is m × m (permutation matrix, represented by pivot vector)
//!     - L is m × min(m,n) (lower triangular matrix with unit diagonal)
//!     - U is min(m,n) × n (upper triangular matrix)
//! This decomposition is used to solve linear systems, compute matrix determinants, and matrix inversion.
//! The function `getrf` (LAPACK) computes the LU factorization of a general m-by-n matrix A using partial pivoting.
//! The matrix L is lower triangular with unit diagonal, and U is upper triangular.
use mdarray::{Dense, Dim, Layout, Shape, Slice, Tensor};
use mdarray_linalg::{
    get_dims, into_i32, ipiv_to_perm_mat,
    lu::{InvError, InvResult, LU},
    transpose_in_place,
};
use num_complex::ComplexFloat;

use super::{
    scalar::{LapackScalar, Workspace},
    simple::{getrf, getri, potrf},
};
use crate::Lapack;

impl<T, D0: Dim, D1: Dim> LU<T, D0, D1> for Lapack
where
    T: ComplexFloat + Default + LapackScalar + Workspace,
    T::Real: Into<T>,
{
    fn lu_write<L: Layout, Ll: Layout, Lu: Layout, Lp: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        l: &mut Slice<T, (D0, D0), Ll>,
        u: &mut Slice<T, (D0, D1), Lu>,
        p: &mut Slice<T, (D0, D0), Lp>,
    ) {
        let ash = *a.shape();
        let m = ash.dim(0);

        let ipiv = getrf(a, l, u);

        let p_matrix = ipiv_to_perm_mat::<T, D0, D1>(&ipiv, m);

        for i in 0..m {
            for j in 0..m {
                p[[i, j]] = p_matrix[[i, j]];
            }
        }
    }

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

        let l_shape = <(D0, D0) as Shape>::from_dims(&[m, min_mn]);
        let u_shape = <(D0, D1) as Shape>::from_dims(&[min_mn, n]);

        let mut l = Tensor::from_elem(l_shape, T::default());
        let mut u = Tensor::from_elem(u_shape, T::default());

        let ipiv = getrf::<T, D0, D1, _, _, _>(a, &mut l, &mut u);

        let p_matrix = ipiv_to_perm_mat::<T, D0, D0>(&ipiv, m);

        (l, u, p_matrix)
    }

    fn inv_write<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<(), InvError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(InvError::NotSquare {
                rows: into_i32(m),
                cols: into_i32(n),
            });
        }

        let min_mn = m.min(n);

        let l_shape = <(D0, D0) as Shape>::from_dims(&[m, min_mn]);
        let u_shape = <(D0, D1) as Shape>::from_dims(&[min_mn, n]);

        let mut l = Tensor::from_elem(l_shape, T::default());
        let mut u = Tensor::from_elem(u_shape, T::default());
        let mut ipiv = getrf::<T, D0, D1, _, _, _>(a, &mut l, &mut u);

        match getri::<T, D0, D1, _>(a, &mut ipiv) {
            0 => Ok(()),
            i if i > 0 => Err(InvError::Singular { pivot: i }),
            i => Err(InvError::BackendError(i)),
        }
    }

    fn inv<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> InvResult<T, D0, D1> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(InvError::NotSquare {
                rows: into_i32(m),
                cols: into_i32(n),
            });
        }

        let mut a_inv = Tensor::<T, (D0, D1)>::zeros(ash);

        // let mut a_inv_mut = a_inv.view_mut(.., ..);

        for i in 0..n {
            for j in 0..m {
                a_inv[[i, j]] = a[[i, j]];
            }
        }

        let min_mn = m.min(n);

        let l_shape = <(D0, D0) as Shape>::from_dims(&[m, min_mn]);
        let u_shape = <(D0, D1) as Shape>::from_dims(&[min_mn, n]);

        let mut l = Tensor::from_elem(l_shape, T::default());
        let mut u = Tensor::from_elem(u_shape, T::default());
        let mut ipiv = getrf::<T, D0, D1, _, _, _>(&mut a_inv, &mut l, &mut u);

        match getri::<T, D0, D1, L>(a, &mut ipiv) {
            0 => Ok(a.to_tensor()),
            i if i > 0 => Err(InvError::Singular { pivot: i }),
            i => Err(InvError::BackendError(i)),
        }
    }

    fn det<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> T {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));
        assert_eq!(m, n, "determinant is only defined for square matrices");

        let l_shape = <(D0, D0) as Shape>::from_dims(&[n, n]);
        let u_shape = <(D0, D1) as Shape>::from_dims(&[n, n]);
        let mut l = Tensor::from_elem(l_shape, T::default());
        let mut u = Tensor::from_elem(u_shape, T::default());

        let ipiv = getrf::<T, D0, D1, _, _, _>(a, &mut l, &mut u);

        let mut det = T::one();
        for i in 0..n {
            det = det * u[[i, i]];
        }

        let mut sign = T::one();
        for (i, &pivot) in ipiv.iter().enumerate() {
            if (i as i32) != (pivot - 1) {
                sign = sign * (-T::one());
            }
        }
        det * sign
    }

    /// Computes the Cholesky decomposition, returning a lower-triangular matrix
    fn choleski<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> InvResult<T, D0, D1> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));
        assert_eq!(m, n, "Matrix must be square for Cholesky decomposition");

        let mut l = Tensor::<T, (D0, D1)>::zeros(ash);

        match potrf::<T, D0, D1, _>(a, 'L') {
            0 => {
                for i in 0..m {
                    for j in 0..n {
                        if i >= j {
                            l[[i, j]] = a[[j, i]];
                        } else {
                            l[[i, j]] = T::zero();
                        }
                    }
                }
                Ok(l)
            }
            i if i > 0 => Err(InvError::NotPositiveDefinite { lpm: i }),
            i => Err(InvError::BackendError(i)),
        }
    }

    /// Computes the Cholesky decomposition in-place, overwriting the input matrix
    fn choleski_write<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<(), InvError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));
        assert_eq!(m, n, "Matrix must be square for Cholesky decomposition");

        match potrf::<T, D0, D1, _>(a, 'L') {
            0 => {
                transpose_in_place(a);
                Ok(())
            }
            i if i > 0 => Err(InvError::NotPositiveDefinite { lpm: i }),
            i => Err(InvError::BackendError(i)),
        }
    }
}
