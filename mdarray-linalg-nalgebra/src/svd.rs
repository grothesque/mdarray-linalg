use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::svd::{SVD, SVDDecomp, SVDError};
use num_complex::ComplexFloat;
use num_traits::Zero;

use crate::{Nalgebra, to_dmatrix};

/// Copy nalgebra singular values into an mdarray vector.
fn write_singular_values<T, D, L>(
    singular_values: &nalgebra::DVector<T::Real>,
    s: &mut Slice<T, (D,), L>,
) where
    T: ComplexFloat + nalgebra::ComplexField<RealField = T::Real>,
    T::Real: nalgebra::RealField + Copy,
    D: Dim,
    L: Layout,
{
    assert_eq!(s.len(), singular_values.len());

    for (dst, src) in s.iter_mut().zip(singular_values.iter()) {
        *dst = <T as nalgebra::ComplexField>::from_real(*src);
    }
}

fn svd_values_impl<T, D, L, Ls>(
    a: &Slice<T, (D, D), L>,
    s: &mut Slice<T, (D,), Ls>,
) -> Result<(), SVDError>
where
    T: ComplexFloat + Copy + Zero + nalgebra::ComplexField<RealField = T::Real>,
    T::Real: nalgebra::RealField + Copy,
    D: Dim,
    L: Layout,
    Ls: Layout,
{
    let svd = to_dmatrix(a).svd(false, false);
    write_singular_values::<T, D, Ls>(&svd.singular_values, s);
    Ok(())
}

/// Copy a possibly thin nalgebra factor into a larger mdarray output.
fn write_dmatrix_padded<T, D0, D1, L>(src: &nalgebra::DMatrix<T>, dst: &mut Slice<T, (D0, D1), L>)
where
    T: nalgebra::Scalar + Zero + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    assert!(src.nrows() <= dst.shape().dim(0));
    assert!(src.ncols() <= dst.shape().dim(1));

    dst.fill(T::zero());
    for i in 0..src.nrows() {
        for j in 0..src.ncols() {
            dst[[i, j]] = src[(i, j)];
        }
    }
}

fn svd_full_impl<T, D, L, Ls, Lu, Lvt>(
    a: &Slice<T, (D, D), L>,
    s: &mut Slice<T, (D,), Ls>,
    u: &mut Slice<T, (D, D), Lu>,
    vt: &mut Slice<T, (D, D), Lvt>,
) -> Result<(), SVDError>
where
    T: ComplexFloat + Copy + Zero + nalgebra::ComplexField<RealField = T::Real>,
    T::Real: nalgebra::RealField + Copy,
    D: Dim,
    L: Layout,
    Ls: Layout,
    Lu: Layout,
    Lvt: Layout,
{
    let svd = to_dmatrix(a).svd(true, true);
    write_singular_values::<T, D, Ls>(&svd.singular_values, s);

    let u_nalgebra = svd.u.ok_or(SVDError::BackendError(-1))?;
    let vt_nalgebra = svd.v_t.ok_or(SVDError::BackendError(-1))?;

    write_dmatrix_padded(&u_nalgebra, u);
    write_dmatrix_padded(&vt_nalgebra, vt);
    Ok(())
}

impl<T, D> SVD<T, D> for Nalgebra
where
    T: ComplexFloat + Copy + Zero + nalgebra::ComplexField<RealField = T::Real>,
    T::Real: nalgebra::RealField + Copy,
    D: Dim,
{
    type SingularValue = T;

    fn svd<L: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
    ) -> Result<SVDDecomp<T, Self::SingularValue, D>, SVDError> {
        let shape = *a.shape();
        let m = shape.dim(0);
        let n = shape.dim(1);
        let min_mn = m.min(n);

        let mut s = Array::<T, (D,)>::from_elem(<(D,) as Shape>::from_dims(&[min_mn]), T::zero());
        let mut u = Array::<T, (D, D)>::from_elem(<(D, D) as Shape>::from_dims(&[m, m]), T::zero());
        let mut vt =
            Array::<T, (D, D)>::from_elem(<(D, D) as Shape>::from_dims(&[n, n]), T::zero());

        svd_full_impl(a, &mut s, &mut u, &mut vt)?;

        Ok(SVDDecomp { s, u, vt })
    }

    fn svd_thin<L: Layout>(
        &self,
        _a: &mut Slice<T, (D, D), L>,
    ) -> Result<SVDDecomp<T, Self::SingularValue, D>, SVDError> {
        unimplemented!()
    }

    fn svd_s<L: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
    ) -> Result<Array<Self::SingularValue, (D,)>, SVDError> {
        let shape = *a.shape();
        let min_mn = shape.dim(0).min(shape.dim(1));
        let mut s = Array::<T, (D,)>::from_elem(<(D,) as Shape>::from_dims(&[min_mn]), T::zero());

        svd_values_impl(a, &mut s)?;
        Ok(s)
    }

    fn svd_write<L: Layout, Ls: Layout, Lu: Layout, Lvt: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<Self::SingularValue, (D,), Ls>,
        u: &mut Slice<T, (D, D), Lu>,
        vt: &mut Slice<T, (D, D), Lvt>,
    ) -> Result<(), SVDError> {
        svd_full_impl(a, s, u, vt)
    }

    fn svd_write_s<L: Layout, Ls: Layout>(
        &self,
        a: &mut Slice<T, (D, D), L>,
        s: &mut Slice<Self::SingularValue, (D,), Ls>,
    ) -> Result<(), SVDError> {
        svd_values_impl(a, s)
    }
}
