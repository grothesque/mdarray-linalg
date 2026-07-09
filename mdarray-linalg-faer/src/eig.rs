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

use dyn_stack::{MemBuffer, MemStack};
use faer_traits::ComplexField;
use mdarray::{Array, Dense, Dim, Layout, Shape, Slice};
use mdarray_linalg::eig::{Eig, EigDecomp, EigError, EighDecomp, SchurDecomp, SchurError};
use num_complex::{Complex, ComplexFloat};

use crate::{Faer, into_faer, into_faer_diag_mut, into_faer_mut};

macro_rules! complex_from_faer {
    ($val:expr, $t:ty) => {{
        // SAFETY: for the scalar types supported by this backend, faer and num_complex use the
        // same real component type, so this is a bitwise-preserving reinterpretation.
        let re: <$t as ComplexFloat>::Real = unsafe { std::mem::transmute_copy(&($val.re)) };
        // SAFETY: same rationale as above for the imaginary component.
        let im: <$t as ComplexFloat>::Real = unsafe { std::mem::transmute_copy(&($val.im)) };
        Complex::new(re, im)
    }};
}

// Faer exposes the Hessenberg reduction publicly, but not the final Schur QR step.
// We keep the same A = Z * T * Z^H interface with the reduced Hessenberg form.
fn schur_faer_in_place<T, D0: Dim, D1: Dim, L: Layout, Lz: Layout>(
    t: &mut Slice<T, (D0, D1), L>,
    z: &mut Slice<T, (D0, D1), Lz>,
) -> Result<(), SchurError>
where
    T: ComplexFloat
        + ComplexField
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>,
{
    let ash = *t.shape();
    let (m, n) = (ash.dim(0), ash.dim(1));

    if m != n {
        return Err(SchurError::NotSquareMatrix);
    }

    for i in 0..n {
        for j in 0..n {
            z[[i, j]] = if i == j { T::one() } else { T::zero() };
        }
    }

    if n <= 1 {
        return Ok(());
    }

    let par = faer::get_global_parallelism();
    let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n - 1, n - 1);
    let mut householder = faer::Mat::<T>::zeros(bs, n - 1);

    {
        let mut t_faer = into_faer_mut(t);
        faer::linalg::evd::hessenberg::hessenberg_in_place(
            t_faer.as_mut(),
            householder.as_mut(),
            par,
            MemStack::new(&mut MemBuffer::new(
                faer::linalg::evd::hessenberg::hessenberg_in_place_scratch::<T>(
                    n,
                    bs,
                    par,
                    faer::prelude::default(),
                ),
            )),
            faer::prelude::default(),
        );
    }

    {
        let t_faer = into_faer(t);
        let mut z_faer = into_faer_mut(z);
        faer::linalg::householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
            t_faer.submatrix(1, 0, n - 1, n - 1),
            householder.as_ref(),
            faer::Conj::No,
            z_faer.as_mut().submatrix_mut(1, 1, n - 1, n - 1),
            par,
            MemStack::new(&mut MemBuffer::new(
                faer::linalg::householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(
                    n - 1,
                    bs,
                    n - 1,
                ),
            )),
        );
    }

    for j in 0..n {
        for i in j + 2..n {
            t[[i, j]] = T::zero();
        }
    }

    Ok(())
}

fn swap_matrices<T, D0: Dim, D1: Dim, L0: Layout, L1: Layout>(
    a: &mut Slice<T, (D0, D1), L0>,
    b: &mut Slice<T, (D0, D1), L1>,
) {
    let ash = *a.shape();
    let (m, n) = (ash.dim(0), ash.dim(1));

    for i in 0..m {
        for j in 0..n {
            std::mem::swap(&mut a[[i, j]], &mut b[[i, j]]);
        }
    }
}

impl<T, D0: Dim, D1: Dim> Eig<T, D0, D1> for Faer
where
    T: ComplexFloat
        + ComplexField
        + Default
        + std::convert::From<<T as num_complex::ComplexFloat>::Real>,
    Complex<<T as ComplexFloat>::Real>: ComplexFloat
        + ComplexField
        + Default
        + std::convert::From<<T as ComplexFloat>::Real>
        + std::convert::From<<Complex<<T as ComplexFloat>::Real> as ComplexFloat>::Real>,
{
    type SpectralScalar = Complex<<T as ComplexFloat>::Real>;
    type RealScalar = <T as ComplexFloat>::Real;

    /// Compute eigenvalues and right eigenvectors with new allocated matrices
    /// The matrix `A` satisfies: `A * v = λ * v` where v are the right eigenvectors
    fn eig<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<EigDecomp<Self::SpectralScalar, D0, D1>, EigError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

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
                let ash1 = <(D0,) as Shape>::from_dims(&[n]);
                let mut eigenvalues_mda = Array::from_elem(ash1, Complex::new(x.re(), x.re()));
                let mut right_vecs_mda = Array::from_elem(ash, Complex::new(x.re(), x.re()));

                for i in 0..n {
                    eigenvalues_mda[i] = complex_from_faer!(&eigenvalues[i], T);
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
    // fn eig_full<L: Layout>(
    //     &self,
    //     a: &mut Slice<T, (D0, D1), L>,
    // ) -> Result<EigDecomp<Complex<T::Real>, D0, D1>, EigError> {
    //     let ash = *a.shape();
    //     let (m, n) = (ash.dim(0), ash.dim(1));

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

    fn eig_full<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<EigDecomp<Self::SpectralScalar, D0, D1>, EigError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let par = faer::get_global_parallelism();
        let x = T::default();
        let xr = x.re();
        // SAFETY: for the scalar types supported by this backend, both traits expose the same
        // concrete real scalar, so this preserves the bit pattern without changing layout.
        let xr_faer: <T as faer_traits::ComplexField>::Real = unsafe {
            std::mem::transmute_copy(&xr)
        };
        let ash1 = <(D0,) as Shape>::from_dims(&[n]);
        let mut eigenvalues_mda = Array::from_elem(ash1, Complex::new(xr, xr));
        let a_faer = into_faer(a);

        if T::IS_REAL {
            let mut s_re_mda = Array::<<T as faer_traits::ComplexField>::Real, (D0,)>::from_elem(
                ash1,
                xr_faer.clone(),
            );
            let mut s_im_mda = Array::<<T as faer_traits::ComplexField>::Real, (D0,)>::from_elem(
                ash1,
                xr_faer.clone(),
            );
            let mut left_vecs_tmp = Array::<
                <T as faer_traits::ComplexField>::Real,
                (D0, D1),
            >::from_elem(ash, xr_faer.clone());
            let mut right_vecs_tmp = Array::<
                <T as faer_traits::ComplexField>::Real,
                (D0, D1),
            >::from_elem(ash, xr_faer.clone());
            let mut left_vecs_mda = Array::from_elem(ash, Complex::new(xr, xr));
            let mut right_vecs_mda = Array::from_elem(ash, Complex::new(xr, xr));

            let a_faer_real: faer::MatRef<'_, <T as faer_traits::ComplexField>::Real> = unsafe {
                // SAFETY: this branch is only taken for real scalar types, so `T` has the same
                // in-memory representation as its real component and the matrix layout is unchanged.
                faer::hacks::coerce::<_, faer::MatRef<'_, <T as faer_traits::ComplexField>::Real>>(
                    a_faer,
                )
            };

            let params = <faer::linalg::evd::EvdParams as faer::Auto<
                <T as faer_traits::ComplexField>::Real,
            >>::auto();

            let result = faer::linalg::evd::evd_real::<<T as faer_traits::ComplexField>::Real>(
                a_faer_real,
                into_faer_diag_mut(&mut s_re_mda),
                into_faer_diag_mut(&mut s_im_mda),
                Some(into_faer_mut(&mut left_vecs_tmp)),
                Some(into_faer_mut(&mut right_vecs_tmp)),
                par,
                MemStack::new(&mut MemBuffer::new(faer::linalg::evd::evd_scratch::<
                    <T as faer_traits::ComplexField>::Real,
                >(
                    n,
                    faer::linalg::evd::ComputeEigenvectors::Yes,
                    faer::linalg::evd::ComputeEigenvectors::Yes,
                    par,
                    params.into(),
                ))),
                params.into(),
            );

            match result {
                Ok(_) => {
                    for i in 0..n {
                        // SAFETY: the temporary storage uses faer's real scalar, which matches
                        // num_complex's real scalar for the supported backend types.
                        let re: <T as ComplexFloat>::Real = unsafe {
                            std::mem::transmute_copy(&s_re_mda[i])
                        };
                        // SAFETY: same rationale as above for the imaginary part.
                        let im: <T as ComplexFloat>::Real = unsafe {
                            std::mem::transmute_copy(&s_im_mda[i])
                        };
                        eigenvalues_mda[i] = Complex::new(re, im);
                    }

                    let mut j = 0_usize;
                    while j < n {
                        let imag_is_zero = s_im_mda[j] == xr_faer;
                        if imag_is_zero {
                            for i in 0..n {
                                // SAFETY: the temporary eigenvector matrices are real in this
                                // branch, and their scalar matches the public real scalar type.
                                let vr: <T as ComplexFloat>::Real = unsafe {
                                    std::mem::transmute_copy(&right_vecs_tmp[[i, j]])
                                };
                                // SAFETY: same rationale as above for the left eigenvector entry.
                                let vl: <T as ComplexFloat>::Real = unsafe {
                                    std::mem::transmute_copy(&left_vecs_tmp[[i, j]])
                                };
                                right_vecs_mda[[i, j]] = Complex::new(vr, xr);
                                left_vecs_mda[[i, j]] = Complex::new(vl, xr);
                            }
                            j += 1;
                        } else {
                            for i in 0..n {
                                // SAFETY: LAPACK-like real output convention from faer stores the
                                // real and imaginary parts in adjacent real matrices. The scalar
                                // type matches the public real scalar for supported backends.
                                let re_right: <T as ComplexFloat>::Real = unsafe {
                                    std::mem::transmute_copy(&right_vecs_tmp[[i, j]])
                                };
                                // SAFETY: same rationale as above.
                                let im_right: <T as ComplexFloat>::Real = unsafe {
                                    std::mem::transmute_copy(&right_vecs_tmp[[i, j + 1]])
                                };
                                // SAFETY: same rationale as above.
                                let re_left: <T as ComplexFloat>::Real = unsafe {
                                    std::mem::transmute_copy(&left_vecs_tmp[[i, j]])
                                };
                                // SAFETY: same rationale as above.
                                let im_left: <T as ComplexFloat>::Real = unsafe {
                                    std::mem::transmute_copy(&left_vecs_tmp[[i, j + 1]])
                                };

                                right_vecs_mda[[i, j]] = Complex::new(re_right, im_right);
                                right_vecs_mda[[i, j + 1]] = Complex::new(re_right, -im_right);
                                left_vecs_mda[[i, j]] = Complex::new(re_left, im_left);
                                left_vecs_mda[[i, j + 1]] = Complex::new(re_left, -im_left);
                            }
                            j += 2;
                        }
                    }

                    Ok(EigDecomp {
                        eigenvalues: eigenvalues_mda,
                        left_eigenvectors: Some(left_vecs_mda),
                        right_eigenvectors: Some(right_vecs_mda),
                    })
                }
                Err(_) => Err(EigError::BackendDidNotConverge { iterations: 0 }),
            }
        } else {
            let mut eigenvalues_tmp = Array::<
                Complex<<T as faer_traits::ComplexField>::Real>,
                (D0,),
            >::from_elem(ash1, Complex::new(xr_faer.clone(), xr_faer.clone()));
            let mut left_vecs_tmp = Array::<
                Complex<<T as faer_traits::ComplexField>::Real>,
                (D0, D1),
            >::from_elem(ash, Complex::new(xr_faer.clone(), xr_faer.clone()));
            let mut right_vecs_tmp = Array::<
                Complex<<T as faer_traits::ComplexField>::Real>,
                (D0, D1),
            >::from_elem(ash, Complex::new(xr_faer.clone(), xr_faer.clone()));
            let mut left_vecs_mda = Array::from_elem(ash, Complex::new(xr, xr));
            let mut right_vecs_mda = Array::from_elem(ash, Complex::new(xr, xr));

            let a_faer_cplx: faer::MatRef<'_, Complex<<T as faer_traits::ComplexField>::Real>> = unsafe {
                // SAFETY: this branch is only taken for complex scalar types, so `T` has the same
                // in-memory representation as `num_complex::Complex<Real>` expected by faer here.
                faer::hacks::coerce::<_, faer::MatRef<'_, Complex<<T as faer_traits::ComplexField>::Real>>>(
                    a_faer,
                )
            };

            let params = <faer::linalg::evd::EvdParams as faer::Auto<
                Complex<<T as faer_traits::ComplexField>::Real>,
            >>::auto();

            let result = faer::linalg::evd::evd_cplx::<<T as faer_traits::ComplexField>::Real>(
                a_faer_cplx,
                into_faer_diag_mut(&mut eigenvalues_tmp),
                Some(into_faer_mut(&mut left_vecs_tmp)),
                Some(into_faer_mut(&mut right_vecs_tmp)),
                par,
                MemStack::new(&mut MemBuffer::new(faer::linalg::evd::evd_scratch::<
                    Complex<<T as faer_traits::ComplexField>::Real>,
                >(
                    n,
                    faer::linalg::evd::ComputeEigenvectors::Yes,
                    faer::linalg::evd::ComputeEigenvectors::Yes,
                    par,
                    params.into(),
                ))),
                params.into(),
            );

            match result {
                Ok(_) => {
                    for i in 0..n {
                        eigenvalues_mda[i] = complex_from_faer!(&eigenvalues_tmp[i], T);
                        for j in 0..n {
                            left_vecs_mda[[i, j]] = complex_from_faer!(&left_vecs_tmp[[i, j]], T);
                            right_vecs_mda[[i, j]] = complex_from_faer!(&right_vecs_tmp[[i, j]], T);
                        }
                    }

                    Ok(EigDecomp {
                        eigenvalues: eigenvalues_mda,
                        left_eigenvectors: Some(left_vecs_mda),
                        right_eigenvectors: Some(right_vecs_mda),
                    })
                }
                Err(_) => Err(EigError::BackendDidNotConverge { iterations: 0 }),
            }
        }
    }

    /// Compute only eigenvalues with new allocated vectors
    fn eig_values<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<Array<Self::SpectralScalar, (D0,)>, EigError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(EigError::NotSquareMatrix);
        }

        let a_faer = into_faer(a);

        let eigenvalues_result = a_faer.eigenvalues();

        match eigenvalues_result {
            Ok(eigenvalues) => {
                let x = T::default();
                let ash1 = <(D0,) as Shape>::from_dims(&[n]);
                let mut eigenvalues_mda = Array::from_elem(ash1, Complex::new(x.re(), x.re()));

                for i in 0..n {
                    eigenvalues_mda[i] = complex_from_faer!(&eigenvalues[i], T);
                }

                Ok(eigenvalues_mda)
            }
            Err(_) => Err(EigError::BackendDidNotConverge { iterations: 0 }),
        }
    }

    /// Compute eigenvalues and eigenvectors of a Hermitian matrix (input should be complex)
    fn eigh<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<EighDecomp<T, Self::RealScalar, D0, D1>, EigError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

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
                let ash1 = <(D0,) as Shape>::from_dims(&[n]);
                let mut eigenvalues_mda = Array::from_elem(ash1, x.re());
                let mut eigenvectors_mda = Array::from_elem(ash, T::default());

                for i in 0..n {
                    eigenvalues_mda[i] = eigenvalues[i].re();
                }

                let mut eigenvectors_faer = into_faer_mut(&mut eigenvectors_mda);
                for i in 0..n {
                    for j in 0..n {
                        eigenvectors_faer[(i, j)] = eigenvectors[(i, j)];
                    }
                }

                Ok(EighDecomp {
                    eigenvalues: eigenvalues_mda,
                    eigenvectors: eigenvectors_mda,
                })
            }
            Err(_) => Err(EigError::BackendDidNotConverge { iterations: 0 }),
        }
    }

    /// Compute Schur decomposition with new allocated matrices
    fn schur<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<SchurDecomp<T, D0, D1>, SchurError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        let mut t = a.to_tensor();
        let mut z = Array::from_elem(ash, T::zero());
        schur_faer_in_place(&mut t, &mut z)?;

        Ok(SchurDecomp { t, z })
    }

    /// Compute Schur decomposition overwriting existing matrices
    fn schur_write<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        t: &mut Slice<T, (D0, D1), Dense>,
        z: &mut Slice<T, (D0, D1), Dense>,
    ) -> Result<(), SchurError> {
        schur_faer_in_place(a, z)?;
        swap_matrices(a, t);
        Ok(())
    }

    /// Compute Schur (complex) decomposition with new allocated matrices
    fn schur_complex<L: Layout>(&self, a: &mut Slice<T, (D0, D1), L>) -> Result<SchurDecomp<Self::SpectralScalar, D0, D1>, SchurError> {
        let ash = *a.shape();
        let (m, n) = (ash.dim(0), ash.dim(1));

        if m != n {
            return Err(SchurError::NotSquareMatrix);
        }

        let zero = T::default().re();
        let shape = <(D0, D1) as Shape>::from_dims(&[m, n]);
        let mut t = Array::from_fn(shape, |idx| {
            let x = a[idx];
            Complex::new(x.re(), x.im())
        });
        let mut z = Array::from_elem(shape, Complex::new(zero, zero));
        schur_faer_in_place(&mut t, &mut z)?;

        Ok(SchurDecomp { t, z })
    }

    /// Compute Schur (complex) decomposition overwriting existing matrices
    fn schur_complex_write<L: Layout>(
        &self,
        a: &mut Slice<T, (D0, D1), L>,
        t: &mut Slice<Self::SpectralScalar, (D0, D1), Dense>,
        z: &mut Slice<Self::SpectralScalar, (D0, D1), Dense>,
    ) -> Result<(), SchurError> {
        let SchurDecomp { t: t_result, z: z_result } = self.schur_complex(a)?;
        for (dst, src) in t.iter_mut().zip(t_result.iter()) {
            *dst = *src;
        }
        for (dst, src) in z.iter_mut().zip(z_result.iter()) {
            *dst = *src;
        }
        Ok(())
    }
}
