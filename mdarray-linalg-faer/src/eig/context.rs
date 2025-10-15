// Eigenvalue Decomposition:
//     A * V = V * Λ  (right eigenvectors)
//     W^H * A = Λ * W^H  (left eigenvectors)
// where:
//     - A is n × n         (input square matrix)
//     - V is n × n         (right eigenvectors as columns)
//     - W is n × n         (left eigenvectors as columns)
//     - Λ is n × n         (diagonal matrix with eigenvalues)
//
// For Hermitian/Symmetric matrices:
//     A = Q * Λ * Q^H
// where:
//     - Q is n × n         (orthogonal/unitary eigenvectors)
//     - Λ is n × n         (diagonal matrix with real eigenvalues)
//
// Schur Decomposition:
//     A = Z * T * Z^H
// where:
//     - Z is n × n         (unitary Schur vectors)
//     - T is n × n         (upper triangular for complex, quasi-upper triangular for real)

use faer_traits::ComplexField;
use mdarray::{DSlice, Dense, Layout, tensor};
use mdarray_linalg::eig::{Eig, EigDecomp, EigError, EigResult, SchurError, SchurResult};
use num_complex::{Complex, ComplexFloat};

use crate::{Faer, into_faer, into_faer_mut};

macro_rules! complex_from_faer {
    ($val:expr, $t:ty) => {{
        let re: <$t as ComplexFloat>::Real = unsafe { std::mem::transmute_copy(&($val.re)) };
        let im: <$t as ComplexFloat>::Real = unsafe { std::mem::transmute_copy(&($val.im)) };
        Complex::new(re, im)
    }};
}

impl<T> Eig<T> for Faer
where
    T: ComplexFloat
        + ComplexField
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
{
    /// Compute eigenvalues and right eigenvectors with new allocated matrices
    /// The matrix `A` satisfies: `A * v = λ * v` where v are the right eigenvectors
    fn eig<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        let (m, n) = *a.shape();

        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let a_faer = into_faer(a);
        let eig_result = a_faer.eigen();

        match eig_result {
            Ok(eig) => {
                let eigenvalues = eig.S();
                let right_vecs = eig.U();

                let x = T::default();
                let mut eigenvalues_mda = tensor![[Complex::new(x.re(), x.re()); n]; 1];
                let mut right_vecs_mda = tensor![[Complex::new(x.re(), x.re());n]; n];

                for i in 0..n {
                    eigenvalues_mda[[0, i]] = complex_from_faer!(&eigenvalues[i], T);
                }

                for i in 0..n {
                    for j in 0..n {
                        right_vecs_mda[[i, j]] = complex_from_faer!(&right_vecs[(i, j)], T);
                    }
                }

                Ok(EigDecomp {
                    eigenvalues: eigenvalues_mda,
                    left_eigenvectors: None,
                    right_eigenvectors: Some(right_vecs_mda),
                })
            }
            Err(_) => Err(EigError::BackendDidNotConverge { iterations: 0 }),
        }
    }

    // /// Compute eigenvalues and both left/right eigenvectors with new allocated matrices
    // /// The matrix A satisfies: `A * vr = λ * vr` and `vl^H * A = λ * vl^H`
    // /// where `vr` are right eigenvectors and `vl` are left eigenvectors
    // fn eig_full<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
    //     let (m, n) = *a.shape();

    //     if m != n {
    //         return Err(EigError::NotSquareMatrix);
    //     }

    //     let a_faer = into_faer(a);

    //     let eig_result = a_faer.eigen();

    //     match eig_result {
    //         Ok(eig) => {
    //             let eigenvalues = eig.S();
    //             let right_vecs = eig.U();

    //             let x = T::default();
    //             let mut eigenvalues_mda = tensor![[Complex::new(x.re(), x.re()); n]; 1];
    //             let mut right_vecs_mda = tensor![[Complex::new(x.re(), x.re());n]; n];

    //             for i in 0..n {
    //                 eigenvalues_mda[[0, i]] = complex_from_faer!(&eigenvalues[i], T);
    //             }

    //             let mut right_vecs_faer = into_faer_mut(&mut right_vecs_mda);
    //             for i in 0..n {
    //                 for j in 0..n {
    //                     right_vecs_faer[(i, j)] = complex_from_faer!(right_vecs[(i, j)], T);
    //                 }
    //             }

    //             let mut left_vecs_mda = tensor![[Complex::new(x.re(), x.re());n]; n];

    //             let mut left_vecs_faer = into_faer_mut(&mut left_vecs_mda);
    //             for i in 0..n {
    //                 for j in 0..n {
    //                     left_vecs_faer[(i, j)] = complex_from_faer!(right_vecs[(i, j)].conj(), T);
    //                 }
    //             }

    //             Ok(EigDecomp {
    //                 eigenvalues: eigenvalues_mda,
    //                 left_eigenvectors: Some(left_vecs_mda),
    //                 right_eigenvectors: Some(right_vecs_mda),
    //             })
    //         }
    //         Err(_) => Err(EigError::BackendDidNotConverge { iterations: 0 }),
    //     }
    // }

    fn eig_full<L: Layout>(&self, _a: &mut DSlice<T, 2, L>) -> Result<EigDecomp<T>, EigError> {
        todo!();
        // let (m, n) = *a.shape();
        // if m != n {
        //     return Err(EigError::NotSquareMatrix);
        // }

        // let par = faer::get_global_parallelism();

        // let x = T::default();

        // let mut eigenvalues_mda = tensor![[Complex::new(x.re(), x.re()); n]; 1];
        // let mut right_vecs_mda = tensor![[Complex::new(x.re(), x.re()); n]; n];
        // let mut left_vecs_mda = tensor![[Complex::new(x.re(), x.re()); n]; n];

        // let a_faer = into_faer_mut(a);

        // let params = <faer::linalg::evd::EvdParams as faer::Auto<T>>::auto();

        // // let eig_result = if TypeId::of::<T>() == TypeId::of::<Complex<f32>>()
        // //     || TypeId::of::<T>() == TypeId::of::<Complex<f64>>()
        // // {
        // let eig_result = if true {
        //     let mut eigenvalues_faer = into_faer_mut(&mut eigenvalues_mda);
        //     let mut right_vecs_faer = into_faer_mut(&mut right_vecs_mda);
        //     let mut left_vecs_faer = into_faer_mut(&mut left_vecs_mda);

        //     let a_faer_complex: MatRef<'_, Complex<<T as faer::traits::ComplexField>::Real>> = unsafe {
        //         faer::hacks::coerce::<_, MatRef<'_, Complex<<T as faer::traits::ComplexField>::Real>>>(
        //             a_faer,
        //         )
        //     };

        //     let mut left_vecs_complex: MatMut<
        //         '_,
        //         Complex<<T as faer::traits::ComplexField>::Real>,
        //     > = unsafe {
        //         faer::hacks::coerce::<_, MatMut<'_, Complex<<T as faer::traits::ComplexField>::Real>>>(
        //             left_vecs_faer,
        //         )
        //     };

        //     let mut right_vecs_complex: MatMut<
        //         '_,
        //         Complex<<T as faer::traits::ComplexField>::Real>,
        //     > = unsafe {
        //         faer::hacks::coerce::<_, MatMut<'_, Complex<<T as faer::traits::ComplexField>::Real>>>(
        //             right_vecs_faer,
        //         )
        //     };

        //     let mut stack_buf = MemBuffer::new(faer::linalg::evd::evd_scratch::<T>(
        //         n,
        //         ComputeEigenvectors::Yes,
        //         ComputeEigenvectors::Yes,
        //         par,
        //         params.into(),
        //     ));
        //     let stack = MemStack::new(&mut stack_buf);

        //     let col0 = eigenvalues_faer.col_mut(0);

        //     let col0_as_matmut: MatMut<'_, Complex<<T as faer::traits::ComplexField>::Real>> = unsafe {
        //         faer::hacks::coerce::<_, MatMut<'_, Complex<<T as faer::traits::ComplexField>::Real>>>(
        //             col0,
        //         )
        //     };

        //     let diag_mut = col0_as_matmut.diagonal_mut();

        //     faer::linalg::evd::evd_cplx::<<T as faer::traits::ComplexField>::Real>(
        //         a_faer_complex,
        //         diag_mut,
        //         Some(left_vecs_complex.as_mut()),
        //         Some(right_vecs_complex.as_mut()),
        //         par,
        //         stack,
        //         params.into(),
        //     )
        // } else {
        //     let mut s_re_mda = tensor![[x.re(); n]; 1];
        //     let mut s_im_mda = tensor![[x.re(); n]; 1];

        //     let mut right_vecs_faer = into_faer_mut(&mut right_vecs_mda);
        //     let mut left_vecs_faer = into_faer_mut(&mut left_vecs_mda);
        //     let mut s_re_faer = into_faer_mut(&mut s_re_mda);
        //     let mut s_im_faer = into_faer_mut(&mut s_im_mda);

        //     let a_faer_real: MatRef<'_, <T as faer::traits::ComplexField>::Real> = unsafe {
        //         faer::hacks::coerce::<_, MatRef<'_, <T as faer::traits::ComplexField>::Real>>(
        //             a_faer,
        //         )
        //     };

        //     println!("ici");

        //     let mut left_vecs_real: MatMut<'_, <T as faer::traits::ComplexField>::Real> = unsafe {
        //         faer::hacks::coerce::<_, MatMut<'_, <T as faer::traits::ComplexField>::Real>>(
        //             left_vecs_faer,
        //         )
        //     };

        //     let mut right_vecs_real: MatMut<'_, <T as faer::traits::ComplexField>::Real> = unsafe {
        //         faer::hacks::coerce::<_, MatMut<'_, <T as faer::traits::ComplexField>::Real>>(
        //             right_vecs_faer,
        //         )
        //     };

        //     let mut stack_buf = MemBuffer::new(faer::linalg::evd::evd_scratch::<T>(
        //         n,
        //         ComputeEigenvectors::Yes,
        //         ComputeEigenvectors::Yes,
        //         par,
        //         params.into(),
        //     ));
        //     let stack = MemStack::new(&mut stack_buf);

        //     let s_re_col0 = s_re_faer.col_mut(0);
        //     let s_re_as_matmut: MatMut<'_, <T as faer::traits::ComplexField>::Real> = unsafe {
        //         faer::hacks::coerce::<_, MatMut<'_, <T as faer::traits::ComplexField>::Real>>(
        //             s_re_col0,
        //         )
        //     };
        //     let s_re_diag = s_re_as_matmut.diagonal_mut();

        //     let s_im_col0 = s_im_faer.col_mut(0);
        //     let s_im_as_matmut: MatMut<'_, <T as faer::traits::ComplexField>::Real> = unsafe {
        //         faer::hacks::coerce::<_, MatMut<'_, <T as faer::traits::ComplexField>::Real>>(
        //             s_im_col0,
        //         )
        //     };
        //     let s_im_diag = s_im_as_matmut.diagonal_mut();

        //     let result = faer::linalg::evd::evd_real::<<T as faer::traits::ComplexField>::Real>(
        //         a_faer_real,
        //         s_re_diag,
        //         s_im_diag,
        //         Some(left_vecs_real.as_mut()),
        //         Some(right_vecs_real.as_mut()),
        //         par,
        //         stack,
        //         params.into(),
        //     );

        //     if result.is_ok() {
        //         for i in 0..n {
        //             eigenvalues_mda[[0, i]] = Complex::new(s_re_mda[[0, i]], s_im_mda[[0, i]]);
        //         }
        //     }

        //     result
        // };

        // match eig_result {
        //     Ok(_) => Ok(EigDecomp {
        //         eigenvalues: eigenvalues_mda,
        //         left_eigenvectors: Some(left_vecs_mda),
        //         right_eigenvectors: Some(right_vecs_mda),
        //     }),
        //     Err(_) => Err(EigError::BackendDidNotConverge { iterations: 0 }),
        // }
    }
    /// Compute only eigenvalues with new allocated vectors
    fn eig_values<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        let (m, n) = *a.shape();

        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let a_faer = into_faer(a);

        let eigenvalues_result = a_faer.eigenvalues();

        match eigenvalues_result {
            Ok(eigenvalues) => {
                let x = T::default();
                let mut eigenvalues_mda = tensor![[Complex::new(x.re(), x.re()); n]; 1];

                for i in 0..n {
                    eigenvalues_mda[[0, i]] = complex_from_faer!(&eigenvalues[i], T);
                }

                Ok(EigDecomp {
                    eigenvalues: eigenvalues_mda,
                    left_eigenvectors: None,
                    right_eigenvectors: None,
                })
            }
            Err(_) => Err(EigError::BackendDidNotConverge { iterations: 0 }),
        }
    }

    /// Compute eigenvalues and eigenvectors of a Hermitian matrix (input should be complex)
    fn eigh<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        let (m, n) = *a.shape();

        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let a_faer = into_faer(a);

        let eig_result = a_faer.self_adjoint_eigen(faer::Side::Lower);

        match eig_result {
            Ok(eig) => {
                let eigenvalues = eig.S();
                let eigenvectors = eig.U();

                let x = T::default();
                let mut eigenvalues_mda = tensor![[Complex::new(x.re(), x.re()); n]; 1];

                let mut eigenvalues_faer = into_faer_mut(&mut eigenvalues_mda);
                for i in 0..n {
                    eigenvalues_faer[(0, i)] = Complex::new(eigenvalues[i].re(), x.re());
                }

                let mut right_vecs_mda = tensor![[Complex::new(x.re(), x.re());n]; n];

                let mut eigenvectors_faer = into_faer_mut(&mut right_vecs_mda);
                for i in 0..n {
                    for j in 0..n {
                        let val = eigenvectors[(i, j)];
                        eigenvectors_faer[(i, j)] = Complex::new(val.re(), val.im());
                    }
                }

                Ok(EigDecomp {
                    eigenvalues: eigenvalues_mda,
                    left_eigenvectors: None,
                    right_eigenvectors: Some(right_vecs_mda),
                })
            }
            Err(_) => Err(EigError::BackendDidNotConverge { iterations: 0 }),
        }
    }

    /// Compute eigenvalues and eigenvectors of a symmetric matrix (input should be real)
    fn eigs<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> EigResult<T> {
        self.eigh(a)
    }

    /// Compute Schur decomposition with new allocated matrices
    fn schur<L: Layout>(&self, _a: &mut DSlice<T, 2, L>) -> SchurResult<T> {
        todo!();
    }

    /// Compute Schur decomposition overwriting existing matrices
    fn schur_overwrite<L: Layout>(
        &self,
        _a: &mut DSlice<T, 2, L>,
        _t: &mut DSlice<T, 2, Dense>,
        _z: &mut DSlice<T, 2, Dense>,
    ) -> Result<(), SchurError> {
        todo!();
    }

    /// Compute Schur (complex) decomposition with new allocated matrices
    fn schur_complex<L: Layout>(&self, _a: &mut DSlice<T, 2, L>) -> SchurResult<T> {
        todo!();
    }

    /// Compute Schur (complex) decomposition overwriting existing matrices
    fn schur_complex_overwrite<L: Layout>(
        &self,
        _a: &mut DSlice<T, 2, L>,
        _t: &mut DSlice<T, 2, Dense>,
        _z: &mut DSlice<T, 2, Dense>,
    ) -> Result<(), SchurError> {
        todo!();
    }
}
