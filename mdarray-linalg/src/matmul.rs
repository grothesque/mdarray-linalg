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
//!let expected_all = tensor![[70.0]].into_dyn();
//!let result_all = Naive.contract_all(&a, &b).eval();
//!let result_contract_k = Naive.contract_n(&a, &b, 2).eval();
//!assert_eq!(result_contract_k, expected_all);
//!
//!let expected_matmul = tensor![[19., 22.], [43., 50.]].into_dyn();
//!let result_specific = Naive
//!    .contract(&a, &b, &[1], &[0])
//!    .eval();
//!assert_eq!(result_specific, expected_matmul);
//!```
use mdarray::{Array, Dim, DynRank, Layout, Shape, Slice};
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
}

struct ContractAxes {
    keep_size_a: usize,
    keep_size_b: usize,
    contract_size: usize,
    keep_shape_a: Vec<usize>,
    keep_shape_b: Vec<usize>,
    order_a: Vec<usize>,
    order_b: Vec<usize>,
}

/// Resolves the axis partition for a tensor contraction, avoiding
/// allocations when axes are already provided as slices
/// (`Axes::Specific`).
fn extract_axes<T, Sa, Sb, La, Lb>(
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

/// Helper for implementing contraction through matrix multiplication
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
    // Contracts tensors `a` and `b` along the specified axes via matrix multiplication.
    // Each tensor's axes are partitioned into `keep_axes` and `contract_axes` (their union
    // covering all axes), computed by `extract_axes` which also validates dimension compatibility.
    // Both tensors are then permuted and reshaped into 2D matrices so that a single matmul
    // performs the contraction, and the result is reshaped back to `[keep_shape_a | keep_shape_b]`.
    let ContractAxes {
        keep_size_a,
        keep_size_b,
        contract_size,
        mut keep_shape_a,
        keep_shape_b,
        order_a,
        order_b,
        ..
    } = extract_axes(axes, a, b);

    let trans_a = a.permute(order_a).to_tensor();
    let trans_b = b.permute(order_b).to_tensor();

    let a_resh = trans_a.reshape([keep_size_a, contract_size]);
    let b_resh = trans_b.reshape([contract_size, keep_size_b]);

    let ab_resh = bd.matmul(&a_resh, &b_resh).scale(alpha).eval();

    if keep_shape_a.is_empty() && keep_shape_b.is_empty() {
        ab_resh.to_owned().into_dyn()
    } else if keep_shape_a.is_empty() {
        ab_resh
            .view(0, ..)
            .reshape(keep_shape_b)
            .to_owned()
            .into_dyn()
            .into()
    } else if keep_shape_b.is_empty() {
        ab_resh
            .view(.., 0)
            .reshape(keep_shape_a)
            .to_owned()
            .into_dyn()
            .into()
    } else {
        keep_shape_a.extend(keep_shape_b);
        ab_resh.reshape(keep_shape_a).to_owned().into_dyn().into()
    }
}
