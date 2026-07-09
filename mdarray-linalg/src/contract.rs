//! Tensor contraction and matrix multiplication
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
//!let result_contract_n = Naive.contract_n(&a, &b, 1).eval();
//!assert_eq!(result_contract_n, expected_n);
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
pub trait Contract<T> {
    /// Matrix multiplication.
    ///
    /// ```rust
    /// use mdarray::tensor;
    /// use mdarray_linalg::{Naive, prelude::*};
    ///
    /// let a = tensor![[1., 2.], [3., 4.]];
    /// let b = tensor![[5., 6.], [7., 8.]];
    /// assert_eq!(Naive.matmul(&a, &b).eval(), tensor![[19., 22.], [43., 50.]]);
    /// ```
    fn matmul<'a, D0, D1, D2, La, Lb>(
        &self,
        a: &'a Slice<T, (D0, D1), La>,
        b: &'a Slice<T, (D1, D2), Lb>,
    ) -> impl MatmulBuilder<'a, T, D0, D1, D2, La, Lb>
    where
        D0: Dim,
        D1: Dim,
        D2: Dim,
        La: Layout,
        Lb: Layout;

    /// Contracts all axes of `a` with all axes of `b`.
    ///
    /// This is the full reduction case, i.e. a scalar result.
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

    /// Contracts the last `n` axes of `a` with the first `n` axes of `b`.
    ///
    /// For matrices, `contract_n(1)` is standard matrix multiplication.
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

    /// Contracts explicit pairs of axes.
    ///
    /// This is the structured contraction API.
    /// `contract_pairs(&a, &b, &[1], &[0])` is matrix multiplication for 2D inputs.
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
    /// ```rust
    /// use mdarray::array;
    /// use mdarray_linalg::{Naive, prelude::*};
    ///
    /// let a = array![[1., 2.], [3., 4.]].into_dyn();
    /// let b = array![[5., 6.], [7., 8.]].into_dyn();
    /// let c = Naive.contract(&a, &b, &[0, 1], &[1, 2], &[0, 2]).eval();
    /// assert_eq!(c, array![[19., 22.], [43., 50.]].into_dyn());
    /// ```
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
pub trait MatmulBuilder<'a, T, D0, D1, D2, La, Lb>
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

// The following exported items are unstable backend-implementation helpers.
// They are hidden from generated documentation and may be redesigned before
// the public API stabilizes.
#[doc(hidden)]
pub enum Axes<'a> {
    All,
    LastFirst { k: usize },
    Specific(&'a [usize], &'a [usize]),
    SpecificOwned(Vec<usize>, Vec<usize>),
}

#[doc(hidden)]
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
#[doc(hidden)]
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

#[doc(hidden)]
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

#[doc(hidden)]
#[macro_export]
macro_rules! finish_contraction {
    ($ab:expr, $keep_shape_a:expr, $keep_shape_b:expr) => {{
        let mut keep_shape_a = $keep_shape_a;
        let keep_shape_b = $keep_shape_b;

        if keep_shape_a.is_empty() && keep_shape_b.is_empty() {
            mdarray::Array::from_elem((), $ab.into_scalar()).into_dyn()
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
#[doc(hidden)]
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

#[doc(hidden)]
pub const FREE_AXIS: usize = usize::MAX;

/// General contraction on labeled axes.
///
/// `axes_a` and `axes_b` have one entry per original axis. Equal values belong
/// to the same contraction/diagonalization hyper-edge. `FREE_AXIS` marks axes
/// that remain in the output.
#[doc(hidden)]
pub fn _hypercontract<T>(
    bd: impl Contract<T>,
    a: View<'_, T, DynRank>,
    b: View<'_, T, DynRank>,
    axes_a: &[usize],
    axes_b: &[usize],
) -> Array<T, DynRank>
where
    T: Copy + Zero + One + Sum + AddAssign + MulAdd<Output = T> + ComplexFloat,
{
    assert_eq!(
        axes_a.len(),
        a.rank(),
        "hypercontract axes_a length ({}) must match A rank ({})",
        axes_a.len(),
        a.rank()
    );
    assert_eq!(
        axes_b.len(),
        b.rank(),
        "hypercontract axes_b length ({}) must match B rank ({})",
        axes_b.len(),
        b.rank()
    );

    let edges = axes_to_hyperedges(axes_a, axes_b);

    // Owned buffers for A and B. Allocated only when a transformation
    // (diagonal extraction or summation) actually modifies the tensor
    let mut a_owned: Option<Array<T, DynRank>> = None;
    let mut b_owned: Option<Array<T, DynRank>> = None;

    // map_x[i] = current position of original axis i in the transformed tensor.
    // Initialised to the identity: no transformation has occurred yet.
    let mut map_a: Vec<usize> = (0..a.shape().dims().len()).collect();
    let mut map_b: Vec<usize> = (0..b.shape().dims().len()).collect();

    // Contraction axis pairs accumulated from (Some, Some) edges,
    // consumed in one shot by the final contraction.
    let mut axes_a: Vec<usize> = Vec::new();
    let mut axes_b: Vec<usize> = Vec::new();

    for edge in &edges {
        let (idx_a, idx_b) = edge;

        match (idx_a, idx_b) {
            // Delta attached to A only: diagonal-sum on A, no contraction with B.
            (Some(axes), None) => {
                let view = a_owned
                    .as_ref()
                    .map(|o| o.expr())
                    .unwrap_or_else(|| a.clone());
                a_owned = Some(apply_hypersum(&view, axes, &mut map_a));
            }
            // Delta attached to B only: diagonal-sum on B, no contraction with A.
            (None, Some(axes)) => {
                let view = b_owned
                    .as_ref()
                    .map(|o| o.expr())
                    .unwrap_or_else(|| b.clone());
                b_owned = Some(apply_hypersum(&view, axes, &mut map_b));
            }
            // Delta bridges A and B: prepare one contraction axis on each side.
            // The actual dot-product is deferred to the final contraction below.
            (Some(axes_a_idx), Some(axes_b_idx)) => {
                let ax_a = {
                    let view = a_owned
                        .as_ref()
                        .map(|o| o.expr())
                        .unwrap_or_else(|| a.clone());
                    let (arr, ax) = extract_hyperdiag(view, axes_a_idx, &mut map_a);
                    a_owned = Some(arr);
                    ax
                };
                let ax_b = {
                    let view = b_owned
                        .as_ref()
                        .map(|o| o.expr())
                        .unwrap_or_else(|| b.clone());
                    let (arr, ax) = extract_hyperdiag(view, axes_b_idx, &mut map_b);
                    b_owned = Some(arr);
                    ax
                };
                axes_a.push(ax_a);
                axes_b.push(ax_b);
            }

            (None, None) => {}
        }
    }

    // Final contraction: contracts all (Some, Some) edge axes simultaneously.
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
#[doc(hidden)]
pub fn hyperdiagonal<'a, T, L: Layout>(
    a: View<'a, T, DynRank, L>,
    axes: &[usize],
) -> View<'a, T, DynRank, mdarray::Strided> {
    let mut axes_sorted = axes.to_vec();
    axes_sorted.sort_unstable();
    axes_sorted.dedup();

    let dims = a.shape().dims();
    let rank = dims.len();

    for &ax in &axes_sorted {
        assert!(ax < rank, "axis ({ax}) out of bounds for rank {rank}");
    }

    // All diagonal axes must have the same size.
    let m = dims[*axes_sorted.first().unwrap()];
    for &ax in &axes_sorted {
        let n = dims[ax];
        assert!(
            m == n,
            "all diagonal axes must have equal size, got {m} and {n}"
        );
    }

    // Build output shape: drop all diagonal axes, ...
    let mut out_dims: Vec<usize> = Vec::with_capacity(rank - axes_sorted.len() + 1);
    let mut out_strides: Vec<isize> = Vec::with_capacity(rank - axes_sorted.len() + 1);

    for (i, item) in dims.iter().enumerate() {
        if axes_sorted.binary_search(&i).is_ok() {
            continue;
        }
        out_dims.push(*item);
        out_strides.push(a.stride(i));
    }

    // ... and append the diagonal axis as the last one.
    out_dims.push(m);
    out_strides.push(axes_sorted.iter().map(|&ax| a.stride(ax)).sum());

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
#[doc(hidden)]
pub fn hypersum<T, L: Layout>(a: &View<'_, T, DynRank, L>, axes: &[usize]) -> Array<T, DynRank>
where
    T: std::iter::Sum + Copy + Zero + std::ops::AddAssign,
{
    let mut axes_sorted = axes.to_vec();
    axes_sorted.sort_unstable();
    axes_sorted.dedup();

    let dims = a.shape().dims();
    let rank = dims.len();

    for &ax in &axes_sorted {
        assert!(ax < rank, "axis ({ax}) out of bounds for rank {rank}");
    }

    let out_dims: Vec<usize> = (0..rank)
        .filter(|i| axes_sorted.binary_search(i).is_err())
        .map(|i| dims[i])
        .collect();

    let mut out = Array::from_elem(out_dims, T::zero());

    for idx in odometer(dims) {
        let out_idx: Vec<usize> = (0..rank)
            .filter(|i| axes_sorted.binary_search(i).is_err())
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

/// Process a `(Some, Some)` edge: prepare the contraction axis.
///
/// For a delta that bridges A and B, we do *not* contract immediately. Instead
/// we reduce the multi-axis delta to a single axis (by extracting the
/// generalized diagonal when needed) and return its current position so that
/// `hypercontract` can pass it to the final contraction.
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

fn axes_to_hyperedges(
    axes_a: &[usize],
    axes_b: &[usize],
) -> Vec<(Option<Vec<usize>>, Option<Vec<usize>>)> {
    let mut remap: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut edges: Vec<(Option<Vec<usize>>, Option<Vec<usize>>)> = Vec::new();

    for (axis, &label) in axes_a.iter().enumerate() {
        if label == FREE_AXIS {
            continue;
        }

        let edge = *remap.entry(label).or_insert_with(|| {
            edges.push((None, None));
            edges.len() - 1
        });
        edges[edge].0.get_or_insert_with(Vec::new).push(axis);
    }

    for (axis, &label) in axes_b.iter().enumerate() {
        if label == FREE_AXIS {
            continue;
        }

        let edge = *remap.entry(label).or_insert_with(|| {
            edges.push((None, None));
            edges.len() - 1
        });
        edges[edge].1.get_or_insert_with(Vec::new).push(axis);
    }

    edges
}

#[doc(hidden)]
pub fn einsum_to_contract_axes(
    indices_a: &[u8],
    indices_b: &[u8],
    indices_c: &[u8],
) -> (Vec<usize>, Vec<usize>) {
    let free: std::collections::HashSet<u8> = indices_c.iter().copied().collect();

    let axes_a = indices_a
        .iter()
        .map(|&label| {
            if free.contains(&label) {
                FREE_AXIS
            } else {
                label as usize
            }
        })
        .collect();

    let axes_b = indices_b
        .iter()
        .map(|&label| {
            if free.contains(&label) {
                FREE_AXIS
            } else {
                label as usize
            }
        })
        .collect();

    (axes_a, axes_b)
}
