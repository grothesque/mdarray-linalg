use std::num::NonZero;

use faer::{
    Accum, Conj, Par,
    linalg::matmul::{dot::inner_prod, matmul},
};
use faer_traits::ComplexField;
use mdarray::{Array, Dim, Layout, Shape, Slice};
use mdarray_linalg::matvec::{MatVec, MatVecBuilder, Outer, OuterBuilder, VecOps};
use num_complex::{Complex32, Complex64, ComplexFloat};
use num_traits::{One, Zero};

use crate::{Faer, into_faer, into_faer_col, into_faer_col_mut, into_faer_mut, into_faer_row};

trait FaerVectorScalar: ComplexFloat + ComplexField {
    fn from_faer_real(real: <Self as ComplexField>::Real) -> <Self as ComplexFloat>::Real;
}

impl FaerVectorScalar for f32 {
    fn from_faer_real(real: <Self as ComplexField>::Real) -> <Self as ComplexFloat>::Real {
        real
    }
}

impl FaerVectorScalar for f64 {
    fn from_faer_real(real: <Self as ComplexField>::Real) -> <Self as ComplexFloat>::Real {
        real
    }
}

impl FaerVectorScalar for Complex32 {
    fn from_faer_real(real: <Self as ComplexField>::Real) -> <Self as ComplexFloat>::Real {
        real
    }
}

impl FaerVectorScalar for Complex64 {
    fn from_faer_real(real: <Self as ComplexField>::Real) -> <Self as ComplexFloat>::Real {
        real
    }
}

struct FaerMatVecBuilder<'a, T, La, Lx, D0, D1>
where
    La: Layout,
    Lx: Layout,
    D0: Dim,
    D1: Dim,
{
    alpha: T,
    a: &'a Slice<T, (D0, D1), La>,
    x: &'a Slice<T, (D1,), Lx>,
    par: Par,
}

struct FaerOuterBuilder<'a, T, Lx, Ly, Dx, Dy>
where
    Lx: Layout,
    Ly: Layout,
    Dx: Dim,
    Dy: Dim,
{
    alpha: T,
    x: &'a Slice<T, (Dx,), Lx>,
    y: &'a Slice<T, (Dy,), Ly>,
    par: Par,
}

impl<'a, T, La, Lx, D0, D1> MatVecBuilder<'a, T, La, Lx, D0, D1>
    for FaerMatVecBuilder<'a, T, La, Lx, D0, D1>
where
    La: Layout,
    Lx: Layout,
    D0: Dim,
    D1: Dim,
    T: FaerVectorScalar + One,
{
    fn parallelize(mut self) -> Self {
        self.par = Par::Rayon(NonZero::new(num_cpus::get()).unwrap());
        self
    }

    fn scale(mut self, alpha: T) -> Self {
        self.alpha *= alpha;
        self
    }

    fn eval(self) -> Array<T, (D0,)> {
        let m = self.a.shape().dim(0);
        let n = self.a.shape().dim(1);
        let x_len = self.x.shape().dim(0);
        assert_eq!(n, x_len, "Matrix columns must match vector length");

        let shape = <(D0,) as Shape>::from_dims(&[m]);
        let mut y = Array::<T, (D0,)>::from_elem(shape, T::zero());
        self.write(&mut y);
        y
    }

    fn write<Ly: Layout>(self, y: &mut Slice<T, (D0,), Ly>) {
        let m = self.a.shape().dim(0);
        let n = self.a.shape().dim(1);
        let x_len = self.x.shape().dim(0);
        let y_len = y.shape().dim(0);
        assert_eq!(n, x_len, "Matrix columns must match vector length");
        assert_eq!(m, y_len, "Matrix rows must match output vector length");

        matmul(
            into_faer_col_mut(y),
            Accum::Replace,
            into_faer(self.a),
            into_faer_col(self.x),
            self.alpha,
            self.par,
        );
    }

    fn add_to_vec<Ly: Layout>(self, y: &mut Slice<T, (D0,), Ly>) {
        let m = self.a.shape().dim(0);
        let n = self.a.shape().dim(1);
        let x_len = self.x.shape().dim(0);
        let y_len = y.shape().dim(0);
        assert_eq!(n, x_len, "Matrix columns must match vector length");
        assert_eq!(m, y_len, "Matrix rows must match output vector length");

        matmul(
            into_faer_col_mut(y),
            Accum::Add,
            into_faer(self.a),
            into_faer_col(self.x),
            self.alpha,
            self.par,
        );
    }

    fn add_to_scaled_vec<Ly: Layout>(self, y: &mut Slice<T, (D0,), Ly>, beta: T) {
        let m = self.a.shape().dim(0);
        let n = self.a.shape().dim(1);
        let x_len = self.x.shape().dim(0);
        let y_len = y.shape().dim(0);
        assert_eq!(n, x_len, "Matrix columns must match vector length");
        assert_eq!(m, y_len, "Matrix rows must match output vector length");

        for yi in y.iter_mut() {
            *yi = beta * *yi;
        }

        matmul(
            into_faer_col_mut(y),
            Accum::Add,
            into_faer(self.a),
            into_faer_col(self.x),
            self.alpha,
            self.par,
        );
    }
}

impl<T, D0: Dim, D1: Dim> MatVec<T, D0, D1> for Faer
where
    T: FaerVectorScalar + One,
{
    fn matvec<'a, La, Lx>(
        &self,
        a: &'a Slice<T, (D0, D1), La>,
        x: &'a Slice<T, (D1,), Lx>,
    ) -> impl MatVecBuilder<'a, T, La, Lx, D0, D1>
    where
        La: Layout,
        Lx: Layout,
    {
        FaerMatVecBuilder {
            alpha: T::one(),
            a,
            x,
            par: if self.parallelize {
                Par::Rayon(NonZero::new(num_cpus::get()).unwrap())
            } else {
                Par::Seq
            },
        }
    }
}

impl<T, D: Dim> VecOps<T, D> for Faer
where
    T: FaerVectorScalar + Zero + Copy,
{
    fn add_to_scaled<Lx: Layout, Ly: Layout>(
        &self,
        _alpha: T,
        _x: &Slice<T, (D,), Lx>,
        _y: &mut Slice<T, (D,), Ly>,
    ) {
        unimplemented!();
    }

    fn dot<Lx: Layout, Ly: Layout>(&self, x: &Slice<T, (D,), Lx>, y: &Slice<T, (D,), Ly>) -> T {
        assert_eq!(
            x.shape().dim(0),
            y.shape().dim(0),
            "Vectors must have same length"
        );
        inner_prod(into_faer_row(x), Conj::No, into_faer_col(y), Conj::No)
    }

    fn dotc<Lx: Layout, Ly: Layout>(&self, x: &Slice<T, (D,), Lx>, y: &Slice<T, (D,), Ly>) -> T {
        assert_eq!(
            x.shape().dim(0),
            y.shape().dim(0),
            "Vectors must have same length"
        );
        inner_prod(into_faer_row(x), Conj::Yes, into_faer_col(y), Conj::No)
    }

    fn norm2<Lx: Layout>(&self, x: &Slice<T, (D,), Lx>) -> <T as ComplexFloat>::Real {
        T::from_faer_real(into_faer_col(x).norm_l2())
    }

    fn norm1<Lx: Layout>(&self, x: &Slice<T, (D,), Lx>) -> <T as ComplexFloat>::Real
    where
        T: ComplexFloat,
    {
        T::from_faer_real(into_faer_col(x).norm_l1())
    }

    fn rot<Lx: Layout, Ly: Layout>(
        &self,
        _x: &mut Slice<T, (D,), Lx>,
        _y: &mut Slice<T, (D,), Ly>,
        _c: <T as ComplexFloat>::Real,
        _s: T,
    ) where
        T: ComplexFloat,
    {
        unimplemented!();
    }
}

impl<T, Dx, Dy> Outer<T, Dx, Dy> for Faer
where
    T: FaerVectorScalar + One,
    Dx: Dim,
    Dy: Dim,
{
    fn outer<'a, Lx, Ly>(
        &self,
        x: &'a Slice<T, (Dx,), Lx>,
        y: &'a Slice<T, (Dy,), Ly>,
    ) -> impl OuterBuilder<'a, T, Lx, Ly, Dx, Dy>
    where
        Lx: Layout,
        Ly: Layout,
    {
        FaerOuterBuilder {
            alpha: T::one(),
            x,
            y,
            par: if self.parallelize {
                Par::Rayon(NonZero::new(num_cpus::get()).unwrap())
            } else {
                Par::Seq
            },
        }
    }
}

impl<'a, T, Lx, Ly, Dx, Dy> OuterBuilder<'a, T, Lx, Ly, Dx, Dy>
    for FaerOuterBuilder<'a, T, Lx, Ly, Dx, Dy>
where
    Lx: Layout,
    Ly: Layout,
    Dx: Dim,
    Dy: Dim,
    T: FaerVectorScalar,
{
    fn scale(mut self, alpha: T) -> Self {
        self.alpha *= alpha;
        self
    }

    fn eval(self) -> Array<T, (Dx, Dy)> {
        let m = self.x.shape().dim(0);
        let n = self.y.shape().dim(0);
        let shape = <(Dx, Dy) as Shape>::from_dims(&[m, n]);
        let mut a = Array::<T, (Dx, Dy)>::from_elem(shape, T::zero());
        self.write(&mut a);
        a
    }

    fn write<La: Layout>(self, a: &mut Slice<T, (Dx, Dy), La>) {
        let m = a.shape().dim(0);
        let n = a.shape().dim(1);
        assert_eq!(m, self.x.shape().dim(0), "Output rows must match x length");
        assert_eq!(n, self.y.shape().dim(0), "Output cols must match y length");

        matmul(
            into_faer_mut(a),
            Accum::Replace,
            into_faer_col(self.x),
            into_faer_row(self.y),
            self.alpha,
            self.par,
        );
    }

    fn add_to<La: Layout>(self, a: &mut Slice<T, (Dx, Dy), La>) {
        let m = a.shape().dim(0);
        let n = a.shape().dim(1);
        assert_eq!(m, self.x.shape().dim(0), "Output rows must match x length");
        assert_eq!(n, self.y.shape().dim(0), "Output cols must match y length");

        matmul(
            into_faer_mut(a),
            Accum::Add,
            into_faer_col(self.x),
            into_faer_row(self.y),
            self.alpha,
            self.par,
        );
    }
}
