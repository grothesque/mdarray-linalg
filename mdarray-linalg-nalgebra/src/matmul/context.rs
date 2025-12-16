use std::fmt::Debug;
use std::ops::{AddAssign, MulAssign};

use mdarray::{Dim, DynRank, Layout, Slice, Tensor};
use mdarray_linalg::matmul::{
    _contract, Axes, ContractBuilder, MatMul, MatMulBuilder, Side, Triangle, Type,
};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};

use crate::Nalgebra;
use matamorph::mut_::MataConvertMut;
use matamorph::own::MataConvertOwn;
use matamorph::ref_::MataConvertRef;

struct NalgebraMatMulBuilder<'a, T, La, Lb, D0, D1, D2>
where
    La: Layout,
    Lb: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    alpha: T,
    a: &'a Slice<T, (D0, D1), La>,
    b: &'a Slice<T, (D1, D2), Lb>,
}

struct NalgebraContractBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
{
    alpha: T,
    a: &'a Slice<T, DynRank, La>,
    b: &'a Slice<T, DynRank, Lb>,
    axes: Axes,
}

impl<'a, T, La, Lb, D0, D1, D2> NalgebraMatMulBuilder<'a, T, La, Lb, D0, D1, D2>
where
    La: Layout,
    Lb: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
    T: Debug + ComplexFloat + One + 'static + MulAdd<Output = T>,
{
    #[allow(dead_code)]
    pub fn parallelize(mut self) -> Self {
        self
    }
}

// pub fn into_nalgebra<T: Clone + nalgebra::Scalar, D0: Dim, D1: Dim>(
//     mat: &mut Tensor<T, (D0, D1)>,
// ) -> nalgebra::DMatrix<T> {
//     let (rows, cols) = mat.shape();
//     let m = rows.size();
//     let n = cols.size();

//     let vec = unsafe { Vec::from_raw_parts(mat.as_mut_ptr(), m * n, m * n) };

//     nalgebra::DMatrix::from_fn(m, n, |i, j| vec[i * n + j].clone())
// }

pub fn into_nalgebra<T: Clone + nalgebra::Scalar, D0: Dim, D1: Dim>(
    mat: &Tensor<T, (D0, D1)>,
) -> nalgebra::DMatrix<T> {
    let (rows, cols) = mat.shape();
    let m = rows.size();
    let n = cols.size();

    nalgebra::DMatrix::from_fn(m, n, |i, j| mat[[i, j]].clone())
}

pub fn into_nalgebra_ref<T: Clone + nalgebra::Scalar, D0: Dim, D1: Dim, L: Layout>(
    mat: &Slice<T, (D0, D1), L>,
) -> nalgebra::DMatrixView<'_, T> {
    let (rows, cols) = mat.shape();
    let m = rows.size();
    let n = cols.size();

    let row_stride = mat.stride(1) as usize;
    let col_stride = mat.stride(0) as usize;

    let required_size = if m > 0 && n > 0 {
        (m - 1) * row_stride + (n - 1) * col_stride + 1
    } else {
        0
    };

    let s = unsafe { core::slice::from_raw_parts(mat.as_ptr(), required_size) };

    nalgebra::base::DMatrixView::from_slice_with_strides_generic(
        s,
        nalgebra::Dim::from_usize(m),
        nalgebra::Dim::from_usize(n),
        nalgebra::Dim::from_usize(row_stride),
        nalgebra::Dim::from_usize(col_stride),
    )
}
pub fn into_nalgebra_mut<T: Clone + nalgebra::Scalar, D0: Dim, D1: Dim, L: Layout>(
    mat: &mut Slice<T, (D0, D1), L>,
) -> nalgebra::DMatrixViewMut<'_, T> {
    let (rows, cols) = mat.shape();
    let m = rows.size();
    let n = cols.size();

    let row_stride = mat.stride(1) as usize;
    let col_stride = mat.stride(0) as usize;

    let required_size = row_stride * col_stride * m * n;

    let s = unsafe { core::slice::from_raw_parts_mut(mat.as_mut_ptr(), required_size) };

    nalgebra::base::DMatrixViewMut::from_slice_with_strides_generic(
        s,
        nalgebra::Dim::from_usize(m),
        nalgebra::Dim::from_usize(n),
        nalgebra::Dim::from_usize(row_stride),
        nalgebra::Dim::from_usize(col_stride),
    )
}
impl<'a, T, La, Lb, D0, D1, D2> MatMulBuilder<'a, T, La, Lb, D0, D1, D2>
    for NalgebraMatMulBuilder<'a, T, La, Lb, D0, D1, D2>
where
    La: Layout,
    Lb: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
    T: ComplexFloat + One + Debug + AddAssign + MulAssign + 'static,
    // mdarray::View<'a, T, (D0, D1), La>: MataConvertRef<'a, T>,
    // mdarray::View<'a, T, (D1, D2), Lb>: MataConvertRef<'a, T>,
    // Tensor<T, (D0, D2)>: MataConvertRef<'a, T>,
{
    fn parallelize(mut self) -> Self {
        self
    }

    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Tensor<T, (D0, D2)> {
        let (ma, _) = *self.a.shape();
        let (_, nb) = *self.b.shape();

        // let a_nalgebra = self.a.view(.., ..).to_nalgebra();
        // let b_nalgebra = self.b.view(.., ..).to_nalgebra();

        let mut a_tensor = self.a.to_tensor();
        let mut b_tensor = self.b.to_tensor();

        let a_nalgebra = into_nalgebra(&mut a_tensor);
        let b_nalgebra = into_nalgebra(&mut b_tensor);

        let mut c = Tensor::<T, (D0, D2)>::from_elem((ma, nb), T::zero());
        let mut c_nalgebra = into_nalgebra_mut(&mut c);

        // c_nalgebra.transpose_mut();

        c_nalgebra.gemm(self.alpha, &a_nalgebra, &b_nalgebra, T::zero());

        // c_nalgebra.transpose_mut();

        c
    }

    fn write<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        let a_nalgebra = into_nalgebra_ref(self.a);
        let b_nalgebra = into_nalgebra_ref(self.b);

        let mut c_nalgebra = into_nalgebra_mut(c);

        c_nalgebra.gemm(self.alpha, &a_nalgebra, &b_nalgebra, T::zero());
    }

    fn add_to<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        todo!()
        // let mut c_nalgebra = c.to_nalgebra();

        // let a_nalgebra = self.a.view(.., ..).to_nalgebra();
        // let b_nalgebra = self.b.view(.., ..).to_nalgebra();

        // a_nalgebra.mul_to(&b_nalgebra, c_nalgebra);
    }

    fn add_to_scaled<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>, _beta: T) {
        todo!()
        // let mut c_nalgebra = c.to_nalgebra();

        // let a_nalgebra = self.a.view(.., ..).to_nalgebra();
        // let b_nalgebra = self.b.view(.., ..).to_nalgebra();

        // a_nalgebra.mul_to(&b_nalgebra, c_nalgebra);
    }

    fn special(self, _lr: Side, _type_of_matrix: Type, _tr: Triangle) -> Tensor<T, (D0, D2)> {
        self.eval()
    }
}

impl<'a, T, La, Lb> ContractBuilder<'a, T, La, Lb> for NalgebraContractBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Debug + AddAssign + MulAssign + 'static,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Tensor<T, DynRank> {
        _contract(Nalgebra, self.a, self.b, self.axes, self.alpha)
    }

    fn write(self, _c: &mut Slice<T>) {
        todo!()
    }
}

impl<T> MatMul<T> for Nalgebra
where
    T: ComplexFloat + One + MulAdd<Output = T> + Debug + AddAssign + MulAssign + 'static,
{
    fn matmul<'a, La, Lb, D0, D1, D2>(
        &self,
        a: &'a Slice<T, (D0, D1), La>,
        b: &'a Slice<T, (D1, D2), Lb>,
    ) -> impl MatMulBuilder<'a, T, La, Lb, D0, D1, D2>
    where
        La: Layout,
        Lb: Layout,
        D0: Dim,
        D1: Dim,
        D2: Dim,
    {
        NalgebraMatMulBuilder {
            alpha: T::one(),
            a,
            b,
        }
    }

    /// Contracts all axes of the first tensor with all axes of the second tensor.
    fn contract_all<'a, La, Lb>(
        &self,
        a: &'a Slice<T, DynRank, La>,
        b: &'a Slice<T, DynRank, Lb>,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout,
    {
        NalgebraContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::All,
        }
    }

    /// Contracts the last `n` axes of the first tensor with the first `n` axes of the second tensor.
    /// # Example
    /// For two matrices (2D tensors), `contract_n(1)` performs standard matrix multiplication.
    fn contract_n<'a, La, Lb>(
        &self,
        a: &'a Slice<T, DynRank, La>,
        b: &'a Slice<T, DynRank, Lb>,
        n: usize,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout,
    {
        NalgebraContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::LastFirst { k: (n) },
        }
    }

    /// Specifies exactly which axes to contract_all.
    /// # Example
    /// `specific([1, 2], [3, 4])` contracts axis 1 and 2 of `a`
    /// with axes 3 and 4 of `b`.
    fn contract<'a, La, Lb>(
        &self,
        a: &'a Slice<T, DynRank, La>,
        b: &'a Slice<T, DynRank, Lb>,
        axes_a: impl Into<Box<[usize]>>,
        axes_b: impl Into<Box<[usize]>>,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout,
    {
        NalgebraContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(axes_a.into(), axes_b.into()),
        }
    }
}
