use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;
use std::num::NonZero;

use std::time::Instant;

use faer::Mat;
use faer::linalg::matmul::matmul;
use faer_traits::ComplexField;

use faer::{Accum, Conj, Par, mat, unzip, zip};
use mdarray::Mapping;
use mdarray::StridedMapping;
use mdarray::View;
use mdarray::{DSlice, DTensor, Dense, DenseMapping, Layout, Strided, tensor};
use num_complex::ComplexFloat;

use num_traits::One;

use mdarray_linalg::{MatMul, MatMulBuilder};
use num_cpus;

pub struct Faer;

/// Converts a `DSlice<T, 2, L>` (from `mdarray`) into a `faer::MatRef<'static, T>`.
/// This function **does not copy** any data.
fn into_faer<T, L: Layout>(mat: &DSlice<T, 2, L>) -> faer::mat::MatRef<'static, T> {
    let (nrows, ncols) = *mat.shape();
    let strides = (mat.stride(0), mat.stride(1));

    // SAFETY:
    // We are constructing a MatRef from raw parts. This requires that:
    // - `mat.as_ptr()` points to a valid matrix of size `nrows x ncols`
    // - The given strides correctly describe the memory layout
    unsafe { faer::MatRef::from_raw_parts(mat.as_ptr(), nrows, ncols, strides.0, strides.1) }
}

/// Converts a `DSlice<T, 2, L>` (from `mdarray`) into a `faer::MatMut<'static, T>`.
/// This function **does not copy** any data.
fn into_faer_mut<T, L: Layout>(mat: &mut DSlice<T, 2, L>) -> faer::mat::MatMut<'static, T> {
    let (nrows, ncols) = *mat.shape();
    let strides = (mat.stride(0), mat.stride(1));

    // SAFETY:
    // We are constructing a MatMut from raw parts. This requires that:
    // - `mat.as_mut_ptr()` points to a valid mutable matrix of size `nrows x ncols`
    // - The given strides correctly describe the memory layout
    unsafe {
        faer::MatMut::from_raw_parts_mut(
            mat.as_mut_ptr() as *mut _,
            nrows,
            ncols,
            strides.0,
            strides.1,
        )
    }
}

/// Converts a `faer::Mat<T>` into a `DTensor<T, 2>` (from `mdarray`) by constructing
/// a strided view over the matrix memory. This function **does not copy** any data.
fn into_mdarray<T: std::clone::Clone>(mat: faer::Mat<T>) -> DTensor<T, 2> {
    // Manually dropping to avoid a double free: DTensor will take ownership of the data,
    // so we must prevent Rust from automatically dropping the original matrix.
    let mut mat = ManuallyDrop::new(mat);

    let (nrows, ncols) = (mat.nrows(), mat.ncols());

    // faer and mdarray have different memory layouts; we need to construct a
    // strided mapping explicitly to describe the layout of `mat` to mdarray.
    let mapping = StridedMapping::new((nrows, ncols), &[mat.row_stride(), mat.col_stride()]);

    // SAFETY:
    // We use `new_unchecked` because the memory layout in faer isn't guaranteed
    // to satisfy mdarray's internal invariants automatically.
    // `from_raw_parts` isn't usable here due to layout incompatibilities.
    let view_strided: View<'_, _, (usize, usize), Strided> =
        unsafe { mdarray::View::new_unchecked(mat.as_ptr_mut(), mapping) };

    DTensor::<T, 2>::from(view_strided)
}

struct FaerMatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
{
    alpha: T,
    a: &'a DSlice<T, 2, La>,
    b: &'a DSlice<T, 2, Lb>,
    par: Par,
}

impl<'a, T, La, Lb> FaerMatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + ComplexField + One + 'static,
{
    pub fn parallelize(mut self) -> Self {
        // Alternative ??? : use faer::get_global_parallelism()
        self.par = Par::Rayon(NonZero::new(num_cpus::get()).unwrap());
        self
    }
}

impl<'a, T, La, Lb> MatMulBuilder<'a, T, La, Lb> for FaerMatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + ComplexField + One + 'static,
{
    fn parallelize(mut self) -> Self {
        // Alternative ?????
        self.par = Par::Rayon(NonZero::new(num_cpus::get()).unwrap());
        self
    }

    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> DTensor<T, 2> {
        let (ma, _) = *self.a.shape();
        let (_, nb) = *self.b.shape();

        let a_faer = into_faer(self.a);
        let b_faer = into_faer(self.b);

        let mut c_faer = Mat::<T>::zeros(ma, nb);

        matmul(
            &mut c_faer,
            Accum::Replace,
            a_faer,
            b_faer,
            self.alpha,
            self.par,
        );

        into_mdarray::<T>(c_faer)
    }

    fn overwrite<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>) {
        let mut c_faer = into_faer_mut(c);
        matmul(
            &mut c_faer,
            Accum::Replace,
            into_faer(self.a),
            into_faer(self.b),
            self.alpha,
            self.par,
        );
    }

    fn add_to<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>) {
        let mut c_faer = into_faer_mut(c);
        matmul(
            &mut c_faer,
            Accum::Add,
            into_faer(self.a),
            into_faer(self.b),
            self.alpha,
            self.par,
        );
    }

    fn add_to_scaled<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>, beta: T) {
        let mut c_faer = into_faer_mut(c);
        matmul(
            &mut c_faer,
            Accum::Add,
            into_faer(self.a),
            into_faer(self.b),
            self.alpha,
            self.par,
        );
        todo!(); // multiplication by beta not implemented in faer ?
    }
}

impl<T> MatMul<T> for Faer
where
    T: ComplexFloat + ComplexField + One + 'static,
{
    fn matmul<'a, La, Lb>(
        &self,
        a: &'a DSlice<T, 2, La>,
        b: &'a DSlice<T, 2, Lb>,
    ) -> impl MatMulBuilder<'a, T, La, Lb>
    where
        La: Layout,
        Lb: Layout,
    {
        FaerMatMulBuilder {
            alpha: T::one(),
            a,
            b,
            par: Par::Seq,
        }
    }
}
