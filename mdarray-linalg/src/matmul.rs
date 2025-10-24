//! Matrix multiplication and specialized products for triangular and Hermitian matrices.
//! Tensor contraction (generalized dot product) for arbitrary rank tensors.
//!```rust
//!use mdarray::tensor;
//!use mdarray_linalg::prelude::*;
//!use mdarray_linalg::Naive;
//!
//!let a = tensor![[1., 2.], [3., 4.]].into_dyn(); // requires dynamic tensor
//!let b = tensor![[5., 6.], [7., 8.]].into_dyn();
//!
//!let expected_all = tensor![[70.0]].into_dyn();
//!let result_all = Naive.contract_all(&a, &b).eval();
//!let result_contract_k = Naive.contract_n(&a, &b, 2).eval();
//!assert_eq!(result_contract_k, expected_all);
//!
//!let expected_matmul = tensor![[19., 22.], [43., 50.]].into_dyn();
//!let result_specific = Naive
//!    .contract(&a, &b, vec![1], vec![0])
//!    .eval();
//!assert_eq!(result_specific, expected_matmul);
//!```
use num_complex::ComplexFloat;
use num_traits::{One, Zero};

use mdarray::{DSlice, DTensor, DynRank, Layout, Slice, Tensor};

/// Specifies whether the left or right matrix has the special property
pub enum Side {
    Left,
    Right,
}

/// Identifies the structural type of a matrix (Hermitian, symmetric, or triangular)
pub enum Type {
    Sym,
    Her,
    Tri,
}

/// Specifies whether a matrix is lower or upper triangular
pub enum Triangle {
    Upper,
    Lower,
}

/// Matrix-matrix multiplication and related operations
pub trait MatMul<T: One> {
    fn matmul<'a, La, Lb>(
        &self,
        a: &'a DSlice<T, 2, La>,
        b: &'a DSlice<T, 2, Lb>,
    ) -> impl MatMulBuilder<'a, T, La, Lb>
    where
        T: One,
        La: Layout,
        Lb: Layout;

    /// Contracts all axes of the first tensor with all axes of the second tensor.
    fn contract_all<'a, La, Lb>(
        &self,
        a: &'a Slice<T, DynRank, La>,
        b: &'a Slice<T, DynRank, Lb>,
    ) -> impl ContractBuilder<'a, T, La, Lb>
    where
        T: 'a,
        La: Layout,
        Lb: Layout;

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
        Lb: Layout;

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
        Lb: Layout;
}

/// Builder interface for configuring matrix-matrix operations
pub trait MatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: 'a,
    La: 'a,
    Lb: 'a,
{
    /// Enable parallelization.
    fn parallelize(self) -> Self;

    /// Multiplies the result by a scalar factor.
    fn scale(self, factor: T) -> Self;

    /// Returns a new owned tensor containing the result.
    fn eval(self) -> DTensor<T, 2>;

    /// Overwrites the provided slice with the result.
    fn overwrite<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>);

    /// Adds the result to the provided slice.
    fn add_to<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>);

    /// Adds the result to the provided slice after scaling the slice by `beta`
    /// (i.e. C := beta * C + result).
    fn add_to_scaled<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>, beta: T);

    /// Computes a matrix product where the first operand is a special
    /// matrix (symmetric, Hermitian, or triangular) and the other is
    /// general.
    ///
    /// The special matrix is always treated as `A`. `lr` determines the multiplication order:
    /// - `Side::Left`  : C := alpha * A * B
    /// - `Side::Right` : C := alpha * B * A
    ///
    /// # Parameters
    /// * `lr` - side of multiplication (left or right)
    /// * `type_of_matrix` - special matrix type: `Sym`, `Her`, or `Tri`
    /// * `tr` - triangle containing stored data: `Upper` or `Lower`
    ///
    /// Only the specified triangle needs to be stored for symmetric/Hermitian matrices;
    /// for triangular matrices it specifies which half is used.
    ///
    /// # Returns
    /// A new tensor with the result.
    fn special(self, lr: Side, type_of_matrix: Type, tr: Triangle) -> DTensor<T, 2>;
}

/// Builder interface for configuring tensor contraction operations
pub trait ContractBuilder<'a, T, La, Lb>
where
    T: 'a,
    La: Layout,
    Lb: Layout,
{
    /// Multiplies the result by a scalar factor.
    fn scale(self, factor: T) -> Self;

    /// Returns a new owned tensor containing the result.
    fn eval(self) -> Tensor<T, DynRank>;

    /// Overwrites the provided tensor with the result.
    fn overwrite(self, c: &mut Slice<T>);
}

pub enum Axes {
    All,
    LastFirst { k: usize },
    Specific(Box<[usize]>, Box<[usize]>),
}

/// Helper for implementing contraction through matrix multiplication
pub fn _contract<T: Zero + ComplexFloat, La: Layout, Lb: Layout>(
    bd: impl MatMul<T>,
    a: &Slice<T, DynRank, La>,
    b: &Slice<T, DynRank, Lb>,
    axes: Axes,
    alpha: T,
) -> Tensor<T, DynRank> {
    let rank_a = a.rank();
    let rank_b = b.rank();

    let extract_shape = |s: &DynRank| match s {
        DynRank::Dyn(arr) => arr.clone(),
        DynRank::One(n) => Box::new([*n]),
    };
    let shape_a = extract_shape(a.shape());
    let shape_b = extract_shape(b.shape());

    let (axes_a, axes_b) = match axes {
        Axes::All => ((0..rank_a).collect(), (0..rank_b).collect()),
        Axes::LastFirst { k } => (((rank_a - k)..rank_a).collect(), (0..k).collect()),
        Axes::Specific(ax_a, ax_b) => (ax_a, ax_b),
    };

    assert_eq!(
        axes_a.len(),
        axes_b.len(),
        "Axis count mismatch: {} (tensor A) vs {} (tensor B)",
        axes_a.len(),
        axes_b.len()
    );

    axes_a.iter().zip(&axes_b).for_each(|(a_ax, b_ax)| {
        assert_eq!(
            shape_a[*a_ax], shape_b[*b_ax],
            "Dimension mismatch at contraction: A[axis {}] = {} ≠ B[axis {}] = {}",
            *a_ax, shape_a[*a_ax], *b_ax, shape_b[*b_ax]
        );
    });

    let compute_keep_axes = |rank: usize, axes: &[usize]| -> Vec<usize> {
        (0..rank).filter(|k| !axes.contains(k)).collect()
    };
    let keep_axes_a = compute_keep_axes(rank_a, &axes_a);
    let keep_axes_b = compute_keep_axes(rank_b, &axes_b);
    let compute_keep_shape = |axes: &[usize], shape: &[usize]| -> Vec<usize> {
        axes.iter().map(|&ax| shape[ax]).collect()
    };

    let mut keep_shape_a = compute_keep_shape(&keep_axes_a, &shape_a);
    let keep_shape_b = compute_keep_shape(&keep_axes_b, &shape_b);

    let compute_size =
        |axes: &[usize], shape: &[usize]| -> usize { axes.iter().map(|&k| shape[k]).product() };

    let contract_size_a = compute_size(&axes_a, &shape_a);
    let contract_size_b = compute_size(&axes_b, &shape_b);
    let keep_size_a = compute_size(&keep_axes_a, &shape_a);
    let keep_size_b = compute_size(&keep_axes_b, &shape_b);

    let order_a: Vec<usize> = keep_axes_a.iter().chain(axes_a.iter()).copied().collect();
    let order_b: Vec<usize> = axes_b.iter().chain(keep_axes_b.iter()).copied().collect();

    let trans_a = a.permute(order_a).to_tensor();
    let trans_b = b.permute(order_b).to_tensor();

    let a_resh = trans_a.reshape([keep_size_a, contract_size_a]);
    let b_resh = trans_b.reshape([contract_size_b, keep_size_b]);

    let ab_resh = bd.matmul(&a_resh, &b_resh).scale(alpha).eval();

    if keep_shape_a.is_empty() && keep_shape_b.is_empty() {
        ab_resh.to_owned().into_dyn()
    } else if keep_shape_a.is_empty() {
        ab_resh
            .view(0, ..)
            .reshape(keep_shape_a)
            .to_owned()
            .into_dyn()
            .into()
    } else if keep_shape_b.is_empty() {
        ab_resh
            .view(.., 0)
            .reshape(keep_shape_b)
            .to_owned()
            .into_dyn()
            .into()
    } else {
        keep_shape_a.extend(keep_shape_b);
        ab_resh.reshape(keep_shape_a).to_owned().into_dyn().into()
    }
}
