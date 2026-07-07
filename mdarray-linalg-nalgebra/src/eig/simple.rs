use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::eig::{EigError, SchurDecomp, SchurError};
use num_complex::{Complex, ComplexFloat};
use num_traits::Zero;

use crate::{
    to_complex_dmatrix, to_dmatrix, write_complex_dmatrix, write_complex_dvector, write_dmatrix,
};

/// Compute a complex eigendecomposition from the complex Schur form.
pub(super) fn eigendecomp<T, D0, D1, L>(
    a: &Slice<T, (D0, D1), L>,
) -> Result<(nalgebra::DVector<Complex<T::Real>>, nalgebra::DMatrix<Complex<T::Real>>), EigError>
where
    T: ComplexFloat,
    T::Real: nalgebra::RealField + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let dim = a.shape().dim(0);
    let schur = to_complex_dmatrix(a).schur();
    let (mut eigenvectors, mut schur_form) = schur.unpack();

    for j in 1..dim {
        for i in 0..j {
            let diff = schur_form[(i, i)] - schur_form[(j, j)];
            if diff.is_zero() && !schur_form[(i, j)].is_zero() {
                return Err(EigError::BackendError(-1));
            }

            let z = -schur_form[(i, j)] / diff;

            for k in (j + 1)..dim {
                let row_jk = schur_form[(j, k)];
                schur_form[(i, k)] -= z * row_jk;
            }

            for k in 0..dim {
                let vec_ki = eigenvectors[(k, i)];
                eigenvectors[(k, j)] += z * vec_ki;
            }
        }
    }

    for i in 0..dim {
        let _ = eigenvectors.column_mut(i).normalize_mut();
    }

    Ok((schur_form.diagonal(), eigenvectors))
}

pub(super) fn eig_values_from_complex_matrix<R, D>(
    values: &nalgebra::DVector<Complex<R>>,
) -> Array<Complex<R>, (D,)>
where
    R: nalgebra::RealField + Copy,
    D: Dim,
{
    let mut out = Array::from_elem(
        <(D,) as Shape>::from_dims(&[values.len()]),
        Complex::new(R::zero(), R::zero()),
    );
    write_complex_dvector(values, &mut out);
    out
}

pub(super) fn eig_vectors_from_complex_matrix<R, D0, D1>(
    vectors: &nalgebra::DMatrix<Complex<R>>,
) -> Array<Complex<R>, (D0, D1)>
where
    R: nalgebra::RealField + Copy,
    D0: Dim,
    D1: Dim,
{
    let shape = <(D0, D1) as Shape>::from_dims(&[vectors.nrows(), vectors.ncols()]);
    let mut out = Array::from_elem(shape, Complex::new(R::zero(), R::zero()));
    write_complex_dmatrix(vectors, &mut out);
    out
}

pub(super) fn schur<T, D0, D1, L>(a: &Slice<T, (D0, D1), L>) -> Result<SchurDecomp<T, D0, D1>, SchurError>
where
    T: ComplexFloat + Copy + Zero + nalgebra::ComplexField<RealField = T::Real>,
    T::Real: nalgebra::RealField + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let schur = to_dmatrix(a).schur();
    let (z_nalgebra, t_nalgebra) = schur.unpack();
    let shape = *a.shape();

    let mut t = Array::from_elem(shape, T::zero());
    let mut z = Array::from_elem(shape, T::zero());

    write_dmatrix(&t_nalgebra, &mut t);
    write_dmatrix(&z_nalgebra, &mut z);

    Ok(SchurDecomp { t, z })
}
