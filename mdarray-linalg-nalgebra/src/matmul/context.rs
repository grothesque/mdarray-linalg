use std::iter::Sum;
use std::ops::AddAssign;

use mdarray::{Array, Dim, DynRank, Layout, Shape, Slice};
use mdarray_linalg::matmul::{
    _contract, _hypercontract, einsum_to_contract_axes, Axes, Contract, ContractBuilder,
    MatMulBuilder, Side, Triangle, Type,
};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign};

use super::simple::gemm;
use crate::{Nalgebra, to_dmatrix, write_dmatrix};

/// Rebuild the missing half of a structured matrix before dispatching to nalgebra.
fn copy_special_matrix<T, D0, D1, L>(
    a: &Slice<T, (D0, D1), L>,
    ty: &Type,
    tr: &Triangle,
) -> nalgebra::DMatrix<T>
where
    T: nalgebra::Scalar + ComplexFloat + Zero + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let rows = a.shape().dim(0);
    let cols = a.shape().dim(1);
    assert_eq!(
        rows, cols,
        "special matrix operations require a square matrix"
    );

    let mut out = nalgebra::DMatrix::from_element(rows, cols, T::zero());

    for i in 0..rows {
        for j in 0..cols {
            let stored = match tr {
                Triangle::Upper => i <= j,
                Triangle::Lower => i >= j,
            };

            out[(i, j)] = if stored {
                a[[i, j]]
            } else {
                match ty {
                    Type::Sym => a[[j, i]],
                    Type::Her => a[[j, i]].conj(),
                    Type::Tri => T::zero(),
                }
            };
        }
    }

    out
}

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

struct NalgebraContractBuilder<'a, T, La, Lb, Sa, Sb>
where
    La: Layout,
    Lb: Layout,
    Sa: Shape,
    Sb: Shape,
{
    alpha: T,
    a: &'a Slice<T, Sa, La>,
    b: &'a Slice<T, Sb, Lb>,
    axes: Axes<'a>,
    einsum: bool,
    einsum_axes_a: Option<Vec<usize>>,
    einsum_axes_b: Option<Vec<usize>>,
    current_output_labels: Option<Vec<u8>>,
    requested_output_labels: Option<Vec<u8>>,
}

impl<'a, T, La, Lb, D0, D1, D2> MatMulBuilder<'a, T, D0, D1, D2, La, Lb>
    for NalgebraMatMulBuilder<'a, T, La, Lb, D0, D1, D2>
where
    T: nalgebra::Scalar + ComplexFloat + Zero + One + ClosedAddAssign + ClosedMulAssign + Copy,
    La: Layout,
    Lb: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha *= factor;
        self
    }

    fn eval(self) -> Array<T, (D0, D2)> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let mut c = Array::<T, (D0, D2)>::from_elem((m, n), T::zero());
        gemm(self.alpha, self.a, self.b, T::zero(), &mut c);
        c
    }

    fn write<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        gemm(self.alpha, self.a, self.b, T::zero(), c);
    }

    fn add_to<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        gemm(self.alpha, self.a, self.b, T::one(), c);
    }

    fn add_to_scaled<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>, beta: T) {
        gemm(self.alpha, self.a, self.b, beta, c);
    }

    fn special(self, lr: Side, ty: Type, tr: Triangle) -> Array<T, (D0, D2)> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let mut c_nalgebra = nalgebra::DMatrix::from_element(m.size(), n.size(), T::zero());

        let a_nalgebra = match lr {
            Side::Left => copy_special_matrix(self.a, &ty, &tr),
            Side::Right => to_dmatrix(self.a),
        };
        let b_nalgebra = match lr {
            Side::Left => to_dmatrix(self.b),
            Side::Right => copy_special_matrix(self.b, &ty, &tr),
        };

        c_nalgebra.gemm(self.alpha, &a_nalgebra, &b_nalgebra, T::zero());

        let mut c = Array::<T, (D0, D2)>::from_elem((m, n), T::zero());
        write_dmatrix(&c_nalgebra, &mut c);
        c
    }
}

impl<'a, T, La, Lb, Sa, Sb> ContractBuilder<'a, T, Sa, Sb, La, Lb>
    for NalgebraContractBuilder<'a, T, La, Lb, Sa, Sb>
where
    T: nalgebra::Scalar
        + ComplexFloat
        + Zero
        + One
        + ClosedAddAssign
        + ClosedMulAssign
        + Copy
        + MulAdd<Output = T>
        + AddAssign
        + Sum,
    La: Layout,
    Lb: Layout,
    Sa: Shape,
    Sb: Shape,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha *= factor;
        self
    }

    fn eval(self) -> Array<T, DynRank> {
        if self.einsum {
            let a = self.a.to_array().into_dyn();
            let b = self.b.to_array().into_dyn();

            let axes_a = self
                .einsum_axes_a
                .as_deref()
                .expect("missing einsum axis labels for A");
            let axes_b = self
                .einsum_axes_b
                .as_deref()
                .expect("missing einsum axis labels for B");

            let mut result = _hypercontract(Nalgebra::default(), a.expr(), b.expr(), axes_a, axes_b);

            if let (Some(current), Some(requested)) = (
                self.current_output_labels.as_deref(),
                self.requested_output_labels.as_deref(),
            )
                && current != requested
            {
                    let perm: Vec<usize> = requested
                        .iter()
                        .map(|label| {
                            current
                                .iter()
                                .position(|cur| cur == label)
                                .expect("output label not present in contraction result")
                        })
                        .collect();
                    result = result.permute(perm).to_tensor().into_dyn();
            }

            if self.alpha != T::one() {
                result = result.map(|x| x * self.alpha).into_dyn();
            }

            result
        } else {
            _contract(Nalgebra::default(), self.a, self.b, self.axes, self.alpha)
        }
    }

    fn write<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>) {
        let result = self.eval();
        assert_eq!(c.rank(), result.rank(), "output rank mismatch");
        for i in 0..c.rank() {
            assert_eq!(c.dim(i), result.dim(i), "output shape mismatch on axis {i}");
        }
        for (dst, src) in c.iter_mut().zip(result.iter()) {
            *dst = *src;
        }
    }

    fn add_to<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>) {
        self.add_to_scaled(c, T::one())
    }

    fn add_to_scaled<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>, beta: T) {
        let result = self.eval();
        assert_eq!(c.rank(), result.rank(), "output rank mismatch");
        for i in 0..c.rank() {
            assert_eq!(c.dim(i), result.dim(i), "output shape mismatch on axis {i}");
        }
        for (dst, src) in c.iter_mut().zip(result.iter()) {
            *dst = beta * *dst + *src;
        }
    }
}

impl<T> Contract<T> for Nalgebra
where
    T: nalgebra::Scalar
        + ComplexFloat
        + Zero
        + One
        + ClosedAddAssign
        + ClosedMulAssign
        + Copy
        + MulAdd<Output = T>
        + AddAssign
        + Sum,
{
    fn matmul<'a, D0, D1, D2, La, Lb>(
        &self,
        a: &'a Slice<T, (D0, D1), La>,
        b: &'a Slice<T, (D1, D2), Lb>,
    ) -> impl MatMulBuilder<'a, T, D0, D1, D2, La, Lb>
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

    fn contract_all<'a, Sa, Sb, La, Lb>(
        &self,
        a: &'a Slice<T, Sa, La>,
        b: &'a Slice<T, Sb, Lb>,
    ) -> T
    where
        T: 'a,
        Sa: Shape,
        Sb: Shape,
        La: Layout,
        Lb: Layout,
    {
        _contract(Nalgebra::default(), a, b, Axes::All, T::one()).into_scalar()
    }

    fn contract_n<'a, Sa, Sb, La, Lb>(
        &self,
        a: &'a Slice<T, Sa, La>,
        b: &'a Slice<T, Sb, Lb>,
        n: usize,
    ) -> impl ContractBuilder<'a, T, Sa, Sb, La, Lb>
    where
        T: 'a,
        Sa: Shape,
        Sb: Shape,
        La: Layout,
        Lb: Layout,
    {
        NalgebraContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::LastFirst { k: n },
            einsum: false,
            einsum_axes_a: None,
            einsum_axes_b: None,
            current_output_labels: None,
            requested_output_labels: None,
        }
    }

    fn contract_pairs<'a, Sa, Sb, La, Lb>(
        &self,
        a: &'a Slice<T, Sa, La>,
        b: &'a Slice<T, Sb, Lb>,
        axes_a: &'a [usize],
        axes_b: &'a [usize],
    ) -> impl ContractBuilder<'a, T, Sa, Sb, La, Lb>
    where
        T: 'a,
        Sa: Shape,
        Sb: Shape,
        La: Layout,
        Lb: Layout,
    {
        NalgebraContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::Specific(axes_a, axes_b),
            einsum: false,
            einsum_axes_a: None,
            einsum_axes_b: None,
            current_output_labels: None,
            requested_output_labels: None,
        }
    }

    fn contract<'a, Sa, Sb, La, Lb>(
        &self,
        a: &'a Slice<T, Sa, La>,
        b: &'a Slice<T, Sb, Lb>,
        indices_a: &'a [u8],
        indices_b: &'a [u8],
        indices_c: &'a [u8],
    ) -> impl ContractBuilder<'a, T, Sa, Sb, La, Lb>
    where
        T: 'a,
        Sa: Shape,
        Sb: Shape,
        La: Layout,
        Lb: Layout,
    {
        assert_eq!(indices_a.len(), a.rank(), "einsum indices_a length must match A rank");
        assert_eq!(indices_b.len(), b.rank(), "einsum indices_b length must match B rank");

        let free: std::collections::HashSet<u8> = indices_c.iter().copied().collect();
        let current_output_labels: Vec<u8> = indices_a
            .iter()
            .chain(indices_b.iter())
            .copied()
            .filter(|label| free.contains(label))
            .collect();
        let (einsum_axes_a, einsum_axes_b) =
            einsum_to_contract_axes(indices_a, indices_b, indices_c);

        NalgebraContractBuilder {
            alpha: T::one(),
            a,
            b,
            axes: Axes::SpecificOwned(Vec::new(), Vec::new()),
            einsum: true,
            einsum_axes_a: Some(einsum_axes_a),
            einsum_axes_b: Some(einsum_axes_b),
            current_output_labels: Some(current_output_labels),
            requested_output_labels: Some(indices_c.to_vec()),
        }
    }
}
