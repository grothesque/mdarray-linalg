use num_traits::Zero;
use std::ops::{Add, Mul};

use mdarray::{DSlice, DTensor, Layout, Shape, Slice};
use num_complex::ComplexFloat;

use mdarray_linalg::matmul::{Triangle, Type};
use mdarray_linalg::matvec::{MatVec, MatVecBuilder, VecOps};

use crate::Naive;

struct BlasMatVecBuilder<'a, T, La, Lx>
where
    La: Layout,
    Lx: Layout,
{
    alpha: T,
    a: &'a DSlice<T, 2, La>,
    x: &'a DSlice<T, 1, Lx>,
}

impl<'a, T, La, Lx> MatVecBuilder<'a, T, La, Lx> for BlasMatVecBuilder<'a, T, La, Lx>
where
    La: Layout,
    Lx: Layout,
    T: ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    fn parallelize(self) -> Self {
        self
    }

    fn scale(mut self, alpha: T) -> Self {
        self.alpha = alpha * self.alpha;
        self
    }

    fn eval(self) -> DTensor<T, 1> {
        let mut _y = DTensor::<T, 1>::from_elem(self.x.len(), 0.into().into());
        // gemv(self.alpha, self.a, self.x, 0.into().into(), &mut y);
        // y
        todo!()
    }

    fn overwrite<Ly: Layout>(self, _y: &mut DSlice<T, 1, Ly>) {
        // gemv(self.alpha, self.a, self.x, 0.into().into(), y);
        todo!()
    }

    fn add_to<Ly: Layout>(self, _y: &mut DSlice<T, 1, Ly>) {
        // gemv(self.alpha, self.a, self.x, 1.into().into(), y);
        todo!()
    }

    fn add_to_scaled<Ly: Layout>(self, _y: &mut DSlice<T, 1, Ly>, _beta: T) {
        // gemv(self.alpha, self.a, self.x, beta, y);
        todo!()
    }

    fn add_outer<Ly: Layout>(self, _y: &DSlice<T, 1, Ly>, _beta: T) -> DTensor<T, 2> {
        let mut a_copy = DTensor::<T, 2>::from_elem(*self.a.shape(), 0.into().into());
        a_copy.assign(self.a);

        // Apply scale factor to preserve builder pattern logic: the alpha parameter
        // may have been modified before this call, so we must scale the matrix
        // before applying the rank-1 update. Unlike gemm operations, this requires
        // a separate pass since BLAS lacks a direct matrix-scalar multiplication.

        // if self.alpha != 1.into().into() {
        //     a_copy = a_copy.map(|x| x * self.alpha);
        // }

        // ger(beta, self.x, y, &mut a_copy);
        // a_copy
        todo!()
    }

    fn add_outer_special(self, _beta: T, _ty: Type, _tr: Triangle) -> DTensor<T, 2> {
        let mut a_copy = DTensor::<T, 2>::from_elem(*self.a.shape(), 0.into().into());
        a_copy.assign(self.a);

        // if self.alpha != 1.into().into() {
        //     a_copy = a_copy.map(|x| x * self.alpha);
        // }

        // let cblas_uplo = match tr {
        //     Triangle::Lower => CBLAS_UPLO::CblasLower,
        //     Triangle::Upper => CBLAS_UPLO::CblasUpper,
        // };

        // match ty {
        //     Type::Her => her(cblas_uplo, beta.re(), self.x, &mut a_copy),
        //     Type::Sym => syr(cblas_uplo, beta, self.x, &mut a_copy),
        //     Type::Tri => {
        //         ger(beta, self.x, self.x, &mut a_copy);
        //     }
        // }
        // a_copy
        todo!()
    }
}

impl<T> MatVec<T> for Naive
where
    T: ComplexFloat,
    i8: Into<T::Real>,
    T::Real: Into<T>,
{
    fn matvec<'a, La, Lx>(
        &self,
        a: &'a DSlice<T, 2, La>,
        x: &'a DSlice<T, 1, Lx>,
    ) -> impl MatVecBuilder<'a, T, La, Lx>
    where
        La: Layout,
        Lx: Layout,
    {
        BlasMatVecBuilder {
            alpha: 1.into().into(),
            a,
            x,
        }
    }
}

impl<T: ComplexFloat + 'static + PartialOrd + Add<Output = T> + Mul<Output = T> + Zero + Copy>
    VecOps<T> for Naive
{
    fn add_to_scaled<Lx: Layout, Ly: Layout>(
        &self,
        _alpha: T,
        _x: &DSlice<T, 1, Lx>,
        _y: &mut DSlice<T, 1, Ly>,
    ) {
        todo!()
        // axpy(alpha, x, y);
    }

    fn dot<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T {
        let mut result = T::zero();
        for (elem_x, elem_y) in std::iter::zip(x.into_iter(), y.into_iter()) {
            result = result + *elem_x * (*elem_y);
        }
        result
    }

    fn dotc<Lx: Layout, Ly: Layout>(&self, _x: &DSlice<T, 1, Lx>, _y: &DSlice<T, 1, Ly>) -> T {
        todo!()
        // dotc(x, y)
    }

    fn norm2<Lx: Layout>(&self, _x: &DSlice<T, 1, Lx>) -> T::Real {
        todo!()
        // nrm2(x)
    }

    fn norm1<Lx: Layout>(&self, _x: &DSlice<T, 1, Lx>) -> T::Real
    where
        T: ComplexFloat,
    {
        todo!()
        // asum(x)
    }

    // fn argmax<Lx: Layout, S: Shape>(&self, x: &Slice<T, S, Lx>) -> Option<Vec<usize>> {
    //     if x.is_empty() {
    //         return None;
    //     }

    //     if x.rank() == 0 {
    //         return Some(Vec::new());
    //     }

    //     let mut max_flat_idx = 0;
    //     let mut max_val = x.iter().next().unwrap();

    //     for (flat_idx, val) in x.iter().enumerate().skip(1) {
    //         if val > max_val {
    //             max_val = val;
    //             max_flat_idx = flat_idx;
    //         }
    //     }

    //     Some(unravel_index(x, max_flat_idx))
    // }

    fn argmax_overwrite<Lx: Layout, S: Shape>(
        &self,
        x: &Slice<T, S, Lx>,
        output: &mut Vec<usize>,
    ) -> bool {
        output.clear();

        if x.is_empty() {
            return false;
        }

        if x.rank() == 0 {
            return true;
        }

        let mut max_flat_idx = 0;
        let mut max_val = x.iter().next().unwrap();

        for (flat_idx, val) in x.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_flat_idx = flat_idx;
            }
        }

        let indices = unravel_index(x, max_flat_idx);
        output.extend_from_slice(&indices);
        true
    }

    fn argmax<Lx: Layout, S: Shape>(&self, x: &Slice<T, S, Lx>) -> Option<Vec<usize>> {
        let mut result = Vec::new();
        if self.argmax_overwrite(x, &mut result) {
            Some(result)
        } else {
            None
        }
    }

    fn copy<Lx: Layout, Ly: Layout>(&self, _x: &DSlice<T, 1, Lx>, _y: &mut DSlice<T, 1, Ly>) {
        todo!()
    }
    fn scal<Lx: Layout>(&self, _alpha: T, _x: &mut DSlice<T, 1, Lx>) {
        todo!()
    }
    fn swap<Lx: Layout, Ly: Layout>(&self, _x: &mut DSlice<T, 1, Lx>, _y: &mut DSlice<T, 1, Ly>) {
        todo!()
    }
    fn rot<Lx: Layout, Ly: Layout>(
        &self,
        _x: &mut DSlice<T, 1, Lx>,
        _y: &mut DSlice<T, 1, Ly>,
        _c: T::Real,
        _s: T,
    ) where
        T: ComplexFloat,
    {
        todo!()
    }
}

pub fn unravel_index<T, S: Shape, L: Layout>(x: &Slice<T, S, L>, mut flat: usize) -> Vec<usize> {
    let rank = x.rank();

    assert!(
        flat < x.len(),
        "flat index out of bounds: {} >= {}",
        flat,
        x.len()
    );

    let mut coords = vec![0usize; rank];

    for i in (0..rank).rev() {
        let dim = x.shape().dim(i);
        coords[i] = flat % dim;
        flat /= dim;
    }

    coords
}
