use std::num::NonZero;

use faer::Mat;
use faer::linalg::matmul::matmul;
use faer_traits::ComplexField;

use faer::{Accum, Par};
use mdarray::{DSlice, DTensor, Layout};
use num_complex::ComplexFloat;

use num_traits::One;

use mdarray_linalg::{
    MatMul, MatMulBuilder, Side, Triangle, Type, into_faer, into_faer_mut, into_mdarray,
};
use num_cpus;

use crate::Faer;

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
    #[allow(dead_code)]
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

    fn add_to_scaled<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>, _beta: T) {
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

    fn special(self, _lr: Side, _type_of_matrix: Type, _tr: Triangle) -> DTensor<T, 2> {
        self.eval()
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
