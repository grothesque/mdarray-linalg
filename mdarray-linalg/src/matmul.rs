//! Matrix multiplication and tensor contraction
//!
//!```rust
//!use mdarray::tensor;
//!use mdarray_linalg::prelude::*;
//!use mdarray_linalg::Naive;
//!
//!let a = tensor![[1., 2.], [3., 4.]];
//!let b = tensor![[5., 6.], [7., 8.]];
//!
//!// Standard matrix multiplication
//!let expected_matmul = tensor![[19., 22.], [43., 50.]];
//!let result = Naive.matmul(&a, &b).eval();
//!assert_eq!(result, expected_matmul);
//!
//!// Matrix multiplication with scalar factor
//!let result_scaled = Naive.matmul(&a, &b).scale(2.0).eval();
//!assert_eq!(result_scaled, expected_matmul.map(|x| x * 2.0));
//!
//!// Full contraction
//!let expected_all = 70.0;
//!let result_all = Naive.contract_all(&a, &b);
//!assert_eq!(result_all, expected_all);
//!
//!// Contract last n axes of a with first n axes of b
//!let expected_n = tensor![[19., 22.], [43., 50.]].into_dyn();
//!let result_contract_k = Naive.contract_n(&a, &b, 1).eval();
//!assert_eq!(result_contract_k, expected_n);
//!
//!// Contract specific axes (equivalent to matmul: contract axis 1 of a with axis 0 of b)
//!let expected_pairs = tensor![[19., 22.], [43., 50.]].into_dyn();
//!let result_specific = Naive
//!    .contract_pairs(&a, &b, &[1], &[0])
//!    .eval();
//!assert_eq!(result_specific, expected_pairs);
//!```

use std::iter::Sum;
use std::ops::AddAssign;

use mdarray::{Array, Dim, DynRank, Layout, Shape, Slice, View};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};

/// Tensor contraction and related operations
pub trait Contract<T: One + MulAdd<Output = T>> {
    fn matmul<'a, D0, D1, D2, La, Lb>(
        &self,
        a: &'a Slice<T, (D0, D1), La>,
        b: &'a Slice<T, (D1, D2), Lb>,
    ) -> impl MatMulBuilder<'a, T, D0, D1, D2, La, Lb>
    where
        T: One,
        D0: Dim,
        D1: Dim,
        D2: Dim,
        La: Layout,
        Lb: Layout;

    /// Contracts all axes of the first tensor with all axes of the second tensor.
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
        Lb: Layout;

    /// Contracts the last `n` axes of the first tensor with the first `n` axes of the second tensor.
    /// # Example
    /// For two matrices (2D tensors), `contract_n(1)` performs standard matrix multiplication.
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
        Lb: Layout;

    /// Contracts pairs of axes.
    /// # Example
    /// `specific([1, 2], [3, 4])` contracts axis 1 and 2 of `a`
    /// with axes 3 and 4 of `b`.
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
        Lb: Layout;

    /// Fully general contraction of two tensors, à la einsum.
    ///
    /// *New* indices in `indices_a` and `indices_b` *must* be subsequent integers starting with 0.
    /// For example, having `[0, 1, 1, 2]` or `[0, 1, 0, 2]` for `indices_a` is OK, but `[0, 2, 2, 3]` is not.
    /// Note that this is not limiting in any way.  Any legal einsum can be specified in this way.
    ///
    /// Note that this is a low-level operation.  The above restrictions allow to avoid runtime checks.
    /// We will add a more user-friendly higher-level wrapper.
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
        Lb: Layout;
}

/// Builder interface for configuring matrix-matrix operations
pub trait MatMulBuilder<'a, T, D0, D1, D2, La, Lb>
where
    T: 'a,
    D0: Dim,
    D1: Dim,
    D2: Dim,
    La: 'a + Layout,
    Lb: 'a + Layout,
{
    /// Multiplies the result by a scalar factor.
    fn scale(self, factor: T) -> Self;

    /// Returns a new owned tensor containing the result.
    fn eval(self) -> Array<T, (D0, D2)>;

    /// Overwrites the provided slice with the result.
    fn write<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>);

    /// Adds the result to the provided slice.
    fn add_to<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>);

    /// Adds the result to the provided slice after scaling the slice by `beta`
    /// (i.e. C := beta * C + result).
    fn add_to_scaled<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>, beta: T);
}

/// Builder interface for configuring tensor contraction operations
pub trait ContractBuilder<'a, T, Sa, Sb, La, Lb>
where
    T: 'a,
    La: Layout,
    Lb: Layout,
{
    /// Multiplies the result by a scalar factor.
    fn scale(self, factor: T) -> Self;

    /// Returns a new owned tensor containing the result.
    fn eval(self) -> Array<T, DynRank>;

    /// Overwrites the provided slice with the result.
    fn write<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>);

    /// Adds the result to the provided slice.
    fn add_to<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>);

    /// Adds the result to the provided slice after scaling the slice by `beta`
    /// (i.e. C := beta * C + result).
    fn add_to_scaled<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>, beta: T);
}

pub enum Axes<'a> {
    All,
    LastFirst { k: usize },
    Specific(&'a [usize], &'a [usize]),
    SpecificOwned(Vec<usize>, Vec<usize>),
}

pub struct ContractAxes {
    pub keep_size_a: usize,
    pub keep_size_b: usize,
    pub contract_size: usize,
    pub keep_shape_a: Vec<usize>,
    pub keep_shape_b: Vec<usize>,
    pub order_a: Vec<usize>,
    pub order_b: Vec<usize>,
}

/// Resolves the axis partition for a tensor contraction, avoiding
/// allocations when axes are already provided as slices
/// (`Axes::Specific`).
pub fn extract_axes<T, Sa, Sb, La, Lb>(
    axes: Axes,
    a: &Slice<T, Sa, La>,
    b: &Slice<T, Sb, Lb>,
) -> ContractAxes
where
    T: Zero + ComplexFloat + MulAdd<Output = T>,
    La: Layout,
    Lb: Layout,
    Sa: Shape,
    Sb: Shape,
{
    let rank_a = a.rank();
    let rank_b = b.rank();

    let axes_a_storage: Option<Vec<usize>>;
    let axes_b_storage: Option<Vec<usize>>;

    let (axes_a, axes_b): (&[usize], &[usize]) = match axes {
        Axes::All => {
            axes_a_storage = Some((0..rank_a).collect());
            axes_b_storage = Some((0..rank_b).collect());
            (
                axes_a_storage.as_deref().unwrap(),
                axes_b_storage.as_deref().unwrap(),
            )
        }
        Axes::LastFirst { k } => {
            axes_a_storage = Some(((rank_a - k)..rank_a).collect());
            axes_b_storage = Some((0..k).collect());
            (
                axes_a_storage.as_deref().unwrap(),
                axes_b_storage.as_deref().unwrap(),
            )
        }
        Axes::Specific(ax_a, ax_b) => (ax_a, ax_b),
        Axes::SpecificOwned(ax_a, ax_b) => {
            axes_a_storage = Some(ax_a);
            axes_b_storage = Some(ax_b);
            (
                axes_a_storage.as_deref().unwrap(),
                axes_b_storage.as_deref().unwrap(),
            )
        }
    };

    assert_eq!(
        axes_a.len(),
        axes_b.len(),
        "Axis count mismatch: {} (tensor A) vs {} (tensor B)",
        axes_a.len(),
        axes_b.len()
    );

    let mut contract_size = 1;
    for (&a_ax, &b_ax) in axes_a.iter().zip(axes_b) {
        assert_eq!(
            a.dim(a_ax),
            b.dim(b_ax),
            "Dimension mismatch at contraction: A[axis {}] = {} ≠ B[axis {}] = {}",
            a_ax,
            a.dim(a_ax),
            b_ax,
            b.dim(b_ax)
        );
        contract_size *= a.dim(a_ax);
    }

    let mut keep_shape_a = Vec::new();
    let mut keep_size_a = 1;
    let mut order_a = Vec::with_capacity(rank_a);

    for i in 0..rank_a {
        if !axes_a.contains(&i) {
            keep_shape_a.push(a.dim(i));
            keep_size_a *= a.dim(i);
            order_a.push(i);
        }
    }
    order_a.extend_from_slice(axes_a);

    let mut keep_shape_b = Vec::new();
    let mut keep_size_b = 1;
    let mut order_b = Vec::with_capacity(rank_b);
    order_b.extend_from_slice(axes_b);

    for i in 0..rank_b {
        if !axes_b.contains(&i) {
            keep_shape_b.push(b.dim(i));
            keep_size_b *= b.dim(i);
            order_b.push(i);
        }
    }

    ContractAxes {
        keep_size_a,
        keep_size_b,
        contract_size,
        keep_shape_a,
        keep_shape_b,
        order_a,
        order_b,
    }
}

#[macro_export]
macro_rules! prepare_contraction {
    ($axes:expr, $a:expr, $b:expr) => {{
        let ContractAxes {
            keep_size_a,
            keep_size_b,
            contract_size,
            keep_shape_a,
            keep_shape_b,
            order_a,
            order_b,
            ..
        } = extract_axes($axes, $a, $b);

        let trans_a = $a.permute(order_a).to_tensor();
        let a_2d = trans_a.reshape([keep_size_a, contract_size]).to_tensor();

        let trans_b = $b.permute(order_b).to_tensor();
        let b_2d = trans_b.reshape([contract_size, keep_size_b]).to_tensor(); // TODO remove this useless copy

        (a_2d, b_2d, keep_shape_a, keep_shape_b)
    }};
}

#[macro_export]
macro_rules! finish_contraction {
    ($ab:expr, $keep_shape_a:expr, $keep_shape_b:expr) => {{
        let mut keep_shape_a = $keep_shape_a;
        let keep_shape_b = $keep_shape_b;

        if keep_shape_a.is_empty() && keep_shape_b.is_empty() {
            $ab.into_dyn()
        } else if keep_shape_a.is_empty() {
            $ab.view(0, ..)
                .reshape(keep_shape_b)
                .to_owned()
                .into_dyn()
                .into()
        } else if keep_shape_b.is_empty() {
            $ab.view(.., 0)
                .reshape(keep_shape_a)
                .to_owned()
                .into_dyn()
                .into()
        } else {
            keep_shape_a.extend(keep_shape_b);
            $ab.reshape(keep_shape_a).to_owned().into_dyn().into()
        }
    }};
}

/// Helper for implementing contraction through matrix multiplication.
/// Backends that implement `Contract` directly should call the macros
/// `prepare_contraction!` and `finish_contraction!` themselves.
pub fn _contract<T, La, Lb, Sa, Sb>(
    bd: impl Contract<T>,
    a: &Slice<T, Sa, La>,
    b: &Slice<T, Sb, Lb>,
    axes: Axes,
    alpha: T,
) -> Array<T, DynRank>
where
    T: Zero + ComplexFloat + MulAdd<Output = T>,
    La: Layout,
    Lb: Layout,
    Sa: Shape,
    Sb: Shape,
{
    let (a_2d, b_2d, keep_shape_a, keep_shape_b) = prepare_contraction!(axes, a, b);

    let ab = bd.matmul(&a_2d, &b_2d).scale(alpha).eval();

    finish_contraction!(ab, keep_shape_a, keep_shape_b)
}

pub fn _hypercontract<T>(
    bd: impl Contract<T>,
    a: View<'_, T, DynRank>,
    b: View<'_, T, DynRank>,
    axes_a_in: &[usize],
    axes_b_in: &[usize],
) -> Array<T, DynRank>
where
    T: Copy + Zero + One + MulAdd<Output = T> + ComplexFloat,
{
    let mut a_owned: Option<Array<T, DynRank>> = None;
    let mut b_owned: Option<Array<T, DynRank>> = None;

    let mut map_a: Vec<usize> = (0..a.shape().dims().len()).collect();
    let mut map_b: Vec<usize> = (0..b.shape().dims().len()).collect();

    let mut axes_a: Vec<usize> = Vec::new();
    let mut axes_b: Vec<usize> = Vec::new();

    for (&idx_a, &idx_b) in axes_a_in.iter().zip(axes_b_in.iter()) {
        let ax_a = {
            let view = a_owned
                .as_ref()
                .map(|o| o.expr())
                .unwrap_or_else(|| a.clone());
            let (arr, ax) = extract_hyperdiag(view, &[idx_a], &mut map_a);
            a_owned = Some(arr);
            ax
        };
        let ax_b = {
            let view = b_owned
                .as_ref()
                .map(|o| o.expr())
                .unwrap_or_else(|| b.clone());
            let (arr, ax) = extract_hyperdiag(view, &[idx_b], &mut map_b);
            b_owned = Some(arr);
            ax
        };
        axes_a.push(ax_a);
        axes_b.push(ax_b);
    }

    let final_a = a_owned
        .as_ref()
        .map(|o| o.expr())
        .unwrap_or_else(|| a.clone());
    let final_b = b_owned
        .as_ref()
        .map(|o| o.expr())
        .unwrap_or_else(|| b.clone());

    _contract(
        bd,
        &final_a,
        &final_b,
        Axes::SpecificOwned(axes_a, axes_b),
        T::one(),
    )
}

/// Generalized diagonal extraction along an arbitrary set of axes (zero-copy)
pub fn hyperdiagonal<'a, T, L: Layout>(
    a: View<'a, T, DynRank, L>,
    axes: &[usize],
) -> View<'a, T, DynRank, mdarray::Strided> {
    let axes_set: std::collections::BTreeSet<usize> = axes.iter().copied().collect(); // TODO: use vec instead

    let dims = a.shape().dims();
    let rank = dims.len();

    for &ax in &axes_set {
        assert!(ax < rank, "axis ({ax}) out of bounds for rank {rank}");
    }

    // All diagonal axes must have the same size.
    let m = dims[*axes_set.first().unwrap()];
    for &ax in &axes_set {
        let n = dims[ax];
        assert!(
            m == n,
            "all diagonal axes must have equal size, got {m} and {n}"
        );
    }

    // Build output shape: drop all diagonal axes, ...
    let mut out_dims: Vec<usize> = Vec::with_capacity(rank - axes_set.len() + 1);
    let mut out_strides: Vec<isize> = Vec::with_capacity(rank - axes_set.len() + 1);

    for (i, item) in dims.iter().enumerate() {
        if axes_set.contains(&i) {
            continue;
        }
        out_dims.push(*item);
        out_strides.push(a.stride(i));
    }

    // ... and append the diagonal axis as the last one.
    out_dims.push(m);
    out_strides.push(axes_set.iter().map(|&ax| a.stride(ax)).sum());

    let mapping = mdarray::StridedMapping::new(Shape::from_dims(&out_dims), &out_strides);

    // SAFETY:
    // - `a.as_ptr()` is valid and live for `'a` (inherited from input view).
    // - Let upper = Σ_{i<r} (dims[i] - 1) * stride(i) be the maximum offset
    //   reachable in `a`'s allocation. Any index over the output view is
    //   (a_0, ..., a_{r-|axes|-1}, k) with a_j < dims[j] for j ∉ axes_set
    //   and k < m = dims[ax] for all ax ∈ axes_set. The offset decomposes as:
    //   Σ_{j ∉ axes_set} a_j * stride(j)  +  k * Σ_{ax ∈ axes_set} stride(ax)
    //   Each term is bounded by (dims[j]-1)*stride(j), so the total ≤ upper.
    unsafe { View::new_unchecked(a.as_ptr(), mapping) }
}

/// Sum a tensor over an arbitrary subset of its axes.
/// Equivalent to a sequence of `np.sum(a, axis=s)` calls (with appropriate
/// index renumbering after each removal), but performed in a single pass.
pub fn hypersum<T, L: Layout>(a: &View<'_, T, DynRank, L>, axes: &[usize]) -> Array<T, DynRank>
where
    T: std::iter::Sum + Copy + Zero + std::ops::AddAssign,
{
    let axes_set: std::collections::BTreeSet<usize> = axes.iter().copied().collect();
    let dims = a.shape().dims();
    let rank = dims.len();

    for &ax in &axes_set {
        assert!(ax < rank, "axis ({ax}) out of bounds for rank {rank}");
    }

    let out_dims: Vec<usize> = (0..rank)
        .filter(|i| !axes_set.contains(i))
        .map(|i| dims[i])
        .collect();

    let mut out = Array::from_elem(out_dims, T::zero());

    for idx in odometer(dims) {
        let out_idx: Vec<usize> = (0..rank)
            .filter(|i| !axes_set.contains(i))
            .map(|i| idx[i])
            .collect();
        out[out_idx.as_slice()] += a[idx.as_slice()];
    }

    out
}

/// Row-major multi-index iterator over a tensor of the given shape.
///
/// Yields every index tuple (i₀, i₁, …, i_{r−1}) in lexicographic order
/// (last axis varies fastest — C-contiguous order).
fn odometer(dims: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    let mut first = true;

    (0..total).map(move |_| {
        if first {
            first = false;
        } else {
            for i in (0..dims.len()).rev() {
                idx[i] += 1;
                if idx[i] < dims[i] {
                    break;
                }
                idx[i] = 0;
            }
        }
        idx.clone()
    })
}

/// A hyper-edge encodes one Kronecker delta connecting axes of A, axes of B,
/// or both.
///
/// Each `Option<&[usize]>` is a list of axis indices.
///
///   (Some(a_axes), Some(b_axes))  — δ shared between A and B: the fused
///                                   diagonal axis becomes a contraction pair
///                                   in the final tensordot.
///   (Some(a_axes), None)          — δ attached to A only: partial trace on A
///                                   (diagonal extraction + marginal sum).
///   (None, Some(b_axes))          — same, but on B.
///   (None, None)                  — no-op.
type HyperEdge<'a> = (Option<&'a [usize]>, Option<&'a [usize]>);

/// Recompute the axis-position map after `hyperdiagonal` has been applied.
fn update_axis_map(axis_map: &[usize], diag_axes: &[usize], ndim_after: usize) -> Vec<usize> {
    let removed: Vec<usize> = {
        let mut v = diag_axes.to_vec();
        v.sort_unstable();
        v
    };
    let new_diag_pos = ndim_after - 1;

    axis_map
        .iter()
        .map(|&cur| {
            if diag_axes.contains(&cur) {
                new_diag_pos
            } else {
                let shift = removed.iter().filter(|&&r| r < cur).count();
                cur - shift
            }
        })
        .collect()
}

/// Process a `(Some, None)` or `(None, Some)` edge: diagonal-sum on one tensor.
///
/// This corresponds to a Kronecker delta that is only connected to one side of
/// the network.
fn apply_hypersum<T, L: Layout>(
    view: &View<'_, T, DynRank, L>,
    idx: &[usize],
    axis_map: &mut Vec<usize>,
) -> Array<T, DynRank>
where
    T: Copy + Zero + std::iter::Sum + std::ops::AddAssign,
{
    // Translate original axis indices to their current positions.
    let cur_axes: Vec<usize> = idx.iter().map(|&a| axis_map[a]).collect();

    if cur_axes.len() == 1 {
        let ax = cur_axes[0];
        let result = hypersum(view, &[ax]);
        for cur in axis_map.iter_mut() {
            if *cur > ax {
                *cur -= 1;
            }
        }
        result
    } else {
        let diag = hyperdiagonal(view.clone(), &cur_axes);

        let diag_ax = diag.shape().dims().len() - 1;
        *axis_map = update_axis_map(axis_map, &cur_axes, diag.shape().dims().len());

        let result = hypersum(&diag.into_dyn(), &[diag_ax]);
        for cur in axis_map.iter_mut() {
            if *cur > diag_ax {
                *cur -= 1;
            }
        }
        result
    }
}

/// Process a `(Some, Some)` edge: prepare the contraction axis for tensordot.
///
/// For a delta that bridges A and B, we do *not* contract immediately. Instead
/// we reduce the multi-axis delta to a single axis (by extracting the
/// generalized diagonal when needed) and return its current position so that
/// `hypercontract` can pass it to the final tensordot.
fn extract_hyperdiag<T, L: Layout>(
    view: View<'_, T, DynRank, L>,
    idx: &[usize],
    axis_map: &mut Vec<usize>,
) -> (Array<T, DynRank>, usize)
where
    T: Copy,
{
    // Translate original axis indices to their current positions.
    let cur_axes: Vec<usize> = idx.iter().map(|&a| axis_map[a]).collect();

    if cur_axes.len() == 1 {
        // Single-axis edge: no diagonal to extract, axis_map is unchanged.
        let ax = cur_axes[0];
        (view.to_owned().into(), ax)
    } else {
        let diag = hyperdiagonal(view, &cur_axes);
        let diag_ax = diag.shape().dims().len() - 1;
        *axis_map = update_axis_map(axis_map, &cur_axes, diag.shape().dims().len());
        (diag.to_owned().into(), diag_ax)
    }
}
