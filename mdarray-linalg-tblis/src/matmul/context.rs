use mdarray::{Array, Dim, DynRank, Layout, Shape, Slice};
use mdarray_linalg::matmul::{Axes, Contract, ContractBuilder, MatMulBuilder};
use num_traits::{MulAdd, One, Zero};
use tblis::{
    containers::TblisTensor,
    einsum_impl::tblis_einsum,
    float_trait::TblisFloatAPI,
    tensor_ops::{TblisMultCfgBuilder, tblis_tensor_mult},
};

use crate::Tblis;

struct TblisMatMulBuilder<'a, T, D0, D1, D2, La, Lb>
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

struct TblisContractBuilder<'a, T, Sa, Sb, La, Lb>
where
    La: Layout,
    Lb: Layout,
    Sa: Shape,
    Sb: Shape,
{
    alpha: T,
    a: &'a Slice<T, Sa, La>,
    b: &'a Slice<T, Sb, Lb>,
    mode: ContractMode<'a>,
}

enum ContractMode<'a> {
    Structured { axes: Axes<'a> },
    Einsum {
        indices_a: &'a [u8],
        indices_b: &'a [u8],
        indices_c: &'a [u8],
    },
}

impl<'a, T, D0, D1, D2, La, Lb> MatMulBuilder<'a, T, D0, D1, D2, La, Lb>
    for TblisMatMulBuilder<'a, T, D0, D1, D2, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: TblisFloatAPI + Zero + One,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Array<T, (D0, D2)> {
        let (m, _) = *self.a.shape();
        let (_, n) = *self.b.shape();
        let mut c = Array::from_elem((m, n), T::zero());
        self.write(&mut c);
        c
    }

    fn write<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        self.add_to_scaled(c, T::zero())
    }

    fn add_to<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>) {
        self.add_to_scaled(c, T::one())
    }

    fn add_to_scaled<Lc: Layout>(self, c: &mut Slice<T, (D0, D2), Lc>, beta: T) {
        assert_eq!(self.a.dim(1), self.b.dim(0), "matrix inner dimensions must match");
        assert_eq!(c.dim(0), self.a.dim(0), "output row count mismatch");
        assert_eq!(c.dim(1), self.b.dim(1), "output column count mismatch");

        let a_t = slice_to_tblis_tensor(self.a);
        let b_t = slice_to_tblis_tensor(self.b);
        let mut c_t = slice_to_tblis_tensor_mut(c);
        let cfg = TblisMultCfgBuilder::default()
            .alpha(self.alpha)
            .beta(beta)
            .build()
            .unwrap();

        unsafe {
            tblis_tensor_mult(&a_t, "ik", &b_t, "kj", &mut c_t, "ij", Some(cfg));
        }
    }
}

impl<'a, T, Sa, Sb, La, Lb> ContractBuilder<'a, T, Sa, Sb, La, Lb>
    for TblisContractBuilder<'a, T, Sa, Sb, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: TblisFloatAPI + Zero + One + MulAdd<Output = T>,
    Sa: Shape,
    Sb: Shape,
{
    fn scale(mut self, factor: T) -> Self {
        self.alpha = self.alpha * factor;
        self
    }

    fn eval(self) -> Array<T, DynRank> {
        match &self.mode {
            ContractMode::Structured { axes } => {
                let (indices_a, indices_b, indices_c, shape_c) =
                    build_structured_subscripts(self.a, self.b, axes);
                let mut c = Array::from_elem(shape_c, T::zero());
                self.run_structured_into(&indices_a, &indices_b, &indices_c, &mut c, T::zero());
                // When all axes are contracted, TBLIS returns a scalar but the
                // API expects a 1×1 tensor (like numpy's np.tensordot).
                if c.rank() == 0 {
                    let scalar = c.iter().next().copied().unwrap_or(T::zero());
                    c = Array::from_elem([1, 1], scalar).into_dyn();
                }
                c
            }
            ContractMode::Einsum {
                indices_a,
                indices_b,
                indices_c,
            } => self.eval_einsum(indices_a, indices_b, indices_c),
        }
    }

    fn write<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>) {
        match &self.mode {
            ContractMode::Structured { axes } => {
                let (indices_a, indices_b, indices_c, shape_c) =
                    build_structured_subscripts(self.a, self.b, axes);
                assert_output_shape(c, &shape_c);
                self.run_structured_into(&indices_a, &indices_b, &indices_c, c, T::zero());
            }
            ContractMode::Einsum {
                indices_a,
                indices_b,
                indices_c,
            } => {
                let result = self.eval_einsum(indices_a, indices_b, indices_c);
                copy_result(c, &result);
            }
        }
    }

    fn add_to<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>) {
        self.add_to_scaled(c, T::one())
    }

    fn add_to_scaled<Sc: Shape, Lc: Layout>(self, c: &mut Slice<T, Sc, Lc>, beta: T) {
        match &self.mode {
            ContractMode::Structured { axes } => {
                let (indices_a, indices_b, indices_c, shape_c) =
                    build_structured_subscripts(self.a, self.b, axes);
                assert_output_shape(c, &shape_c);
                self.run_structured_into(&indices_a, &indices_b, &indices_c, c, beta);
            }
            ContractMode::Einsum {
                indices_a,
                indices_b,
                indices_c,
            } => {
                let result = self.eval_einsum(indices_a, indices_b, indices_c);
                add_result(c, &result, beta);
            }
        }
    }
}

impl<'a, T, Sa, Sb, La, Lb> TblisContractBuilder<'a, T, Sa, Sb, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: TblisFloatAPI + Zero + One + MulAdd<Output = T>,
    Sa: Shape,
    Sb: Shape,
{
    fn run_structured_into<Sc: Shape, Lc: Layout>(
        &self,
        indices_a: &str,
        indices_b: &str,
        indices_c: &str,
        c: &mut Slice<T, Sc, Lc>,
        beta: T,
    ) {
        let a_t = slice_to_tblis_tensor(self.a);
        let b_t = slice_to_tblis_tensor(self.b);
        let mut c_t = slice_to_tblis_tensor_mut(c);
        let cfg = TblisMultCfgBuilder::default()
            .alpha(self.alpha)
            .beta(beta)
            .build()
            .unwrap();

        unsafe {
            tblis_tensor_mult(&a_t, indices_a, &b_t, indices_b, &mut c_t, indices_c, Some(cfg));
        }
    }

    fn eval_einsum(&self, indices_a: &[u8], indices_b: &[u8], indices_c: &[u8]) -> Array<T, DynRank> {
        assert_eq!(indices_a.len(), self.a.rank(), "einsum indices_a length must match A rank");
        assert_eq!(indices_b.len(), self.b.rank(), "einsum indices_b length must match B rank");

        let subscripts = build_einsum_subscripts(indices_a, indices_b, indices_c);
        let a_t = slice_to_tblis_tensor(self.a);
        let b_t = slice_to_tblis_tensor(self.b);
        let operands = [&a_t, &b_t];

        let (vec, tsr) = unsafe {
            tblis_einsum(&subscripts, &operands, "optimal", None, true, None)
                .expect("tblis_einsum must allocate an output tensor")
        };

        let shape: Vec<usize> = tsr.shape.iter().map(|&d| d as usize).collect();
        let mut result = Array::from_elem(shape, T::zero());
        for (dst, src) in result.iter_mut().zip(vec.into_iter()) {
            *dst = self.alpha * src;
        }
        result
    }
}

impl<T> Contract<T> for Tblis
where
    T: TblisFloatAPI + Zero + One + MulAdd<Output = T>,
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
        TblisMatMulBuilder {
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
        // contract_n returns a 1×1 tensor when all axes are contracted;
        // extract the scalar from it.
        self.contract_n(a, b, a.rank())
            .eval()
            .iter()
            .next()
            .copied()
            .unwrap_or(T::zero())
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
        TblisContractBuilder {
            alpha: T::one(),
            a,
            b,
            mode: ContractMode::Structured {
                axes: Axes::LastFirst { k: n },
            },
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
        TblisContractBuilder {
            alpha: T::one(),
            a,
            b,
            mode: ContractMode::Structured {
                axes: Axes::Specific(axes_a, axes_b),
            },
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
        TblisContractBuilder {
            alpha: T::one(),
            a,
            b,
            mode: ContractMode::Einsum {
                indices_a,
                indices_b,
                indices_c,
            },
        }
    }
}

fn slice_to_tblis_tensor<T, S, L>(slice: &Slice<T, S, L>) -> TblisTensor<T>
where
    T: TblisFloatAPI,
    S: Shape,
    L: Layout,
{
    let shape: Vec<isize> = (0..slice.rank()).map(|i| slice.dim(i) as isize).collect();
    let stride: Vec<isize> = (0..slice.rank()).map(|i| slice.stride(i) as isize).collect();
    TblisTensor::new(slice.as_ptr() as *mut T, &shape, &stride)
}

fn slice_to_tblis_tensor_mut<T, S, L>(slice: &mut Slice<T, S, L>) -> TblisTensor<T>
where
    T: TblisFloatAPI,
    S: Shape,
    L: Layout,
{
    let shape: Vec<isize> = (0..slice.rank()).map(|i| slice.dim(i) as isize).collect();
    let stride: Vec<isize> = (0..slice.rank()).map(|i| slice.stride(i) as isize).collect();
    TblisTensor::new(slice.as_mut_ptr(), &shape, &stride)
}

fn resolve_axes<T, Sa, Sb, La, Lb>(
    axes: &Axes<'_>,
    a: &Slice<T, Sa, La>,
    b: &Slice<T, Sb, Lb>,
) -> (Vec<usize>, Vec<usize>)
where
    Sa: Shape,
    Sb: Shape,
    La: Layout,
    Lb: Layout,
{
    let rank_a = a.rank();
    let rank_b = b.rank();
    match axes {
        Axes::All => {
            assert_eq!(rank_a, rank_b, "full contraction requires equal ranks");
            ((0..rank_a).collect(), (0..rank_b).collect())
        }
        Axes::LastFirst { k } => {
            assert!(*k <= rank_a, "cannot contract {k} axes on A of rank {rank_a}");
            assert!(*k <= rank_b, "cannot contract {k} axes on B of rank {rank_b}");
            (((rank_a - *k)..rank_a).collect(), (0..*k).collect())
        }
        Axes::Specific(ax_a, ax_b) => (ax_a.to_vec(), ax_b.to_vec()),
        Axes::SpecificOwned(ax_a, ax_b) => (ax_a.clone(), ax_b.clone()),
    }
}

fn build_structured_subscripts<T, Sa, Sb, La, Lb>(
    a: &Slice<T, Sa, La>,
    b: &Slice<T, Sb, Lb>,
    axes: &Axes<'_>,
) -> (String, String, String, Vec<usize>)
where
    Sa: Shape,
    Sb: Shape,
    La: Layout,
    Lb: Layout,
{
    let (axes_a, axes_b) = resolve_axes(axes, a, b);
    assert_eq!(axes_a.len(), axes_b.len(), "axis count mismatch");

    let rank_a = a.rank();
    let rank_b = b.rank();
    let mut labels_a = vec![usize::MAX; rank_a];
    let mut labels_b = vec![usize::MAX; rank_b];
    let mut next_label = 0usize;

    for (&ax_a, &ax_b) in axes_a.iter().zip(axes_b.iter()) {
        assert!(ax_a < rank_a, "axis {ax_a} out of bounds for A rank {rank_a}");
        assert!(ax_b < rank_b, "axis {ax_b} out of bounds for B rank {rank_b}");
        assert_eq!(a.dim(ax_a), b.dim(ax_b), "dimension mismatch on contracted axes");
        assert_eq!(labels_a[ax_a], usize::MAX, "duplicate contracted axis in A");
        assert_eq!(labels_b[ax_b], usize::MAX, "duplicate contracted axis in B");
        labels_a[ax_a] = next_label;
        labels_b[ax_b] = next_label;
        next_label += 1;
    }

    let mut output_labels = Vec::new();
    let mut output_shape = Vec::new();

    for (ax, label) in labels_a.iter_mut().enumerate() {
        if *label == usize::MAX {
            *label = next_label;
            output_labels.push(next_label);
            output_shape.push(a.dim(ax));
            next_label += 1;
        }
    }

    for (ax, label) in labels_b.iter_mut().enumerate() {
        if *label == usize::MAX {
            *label = next_label;
            output_labels.push(next_label);
            output_shape.push(b.dim(ax));
            next_label += 1;
        }
    }

    let idx_a = labels_to_subscript(&labels_a);
    let idx_b = labels_to_subscript(&labels_b);
    let idx_c = labels_to_subscript(&output_labels);

    (idx_a, idx_b, idx_c, output_shape)
}

fn build_einsum_subscripts(indices_a: &[u8], indices_b: &[u8], indices_c: &[u8]) -> String {
    let mut unique = Vec::<u8>::new();
    for &label in indices_a.iter().chain(indices_b.iter()).chain(indices_c.iter()) {
        if !unique.contains(&label) {
            unique.push(label);
        }
    }
    assert!(unique.len() <= 128, "TBLIS backend supports at most 128 distinct einsum labels");

    let idx_a = labels_to_subscript(
        &indices_a
            .iter()
            .map(|label| unique.iter().position(|x| x == label).unwrap())
            .collect::<Vec<_>>(),
    );
    let idx_b = labels_to_subscript(
        &indices_b
            .iter()
            .map(|label| unique.iter().position(|x| x == label).unwrap())
            .collect::<Vec<_>>(),
    );
    let idx_c = labels_to_subscript(
        &indices_c
            .iter()
            .map(|label| unique.iter().position(|x| x == label).unwrap())
            .collect::<Vec<_>>(),
    );

    format!("{idx_a},{idx_b}->{idx_c}")
}

fn labels_to_subscript(labels: &[usize]) -> String {
    labels
        .iter()
        .map(|&label| {
            let code = 0x0100_u32 + label as u32;
            char::from_u32(code).expect("invalid generated label")
        })
        .collect()
}

fn assert_output_shape<T, S, L>(c: &Slice<T, S, L>, expected: &[usize])
where
    S: Shape,
    L: Layout,
{
    assert_eq!(c.rank(), expected.len(), "output rank mismatch");
    for (i, &dim) in expected.iter().enumerate() {
        assert_eq!(c.dim(i), dim, "output shape mismatch on axis {i}");
    }
}

fn copy_result<T, Sc, Lc>(c: &mut Slice<T, Sc, Lc>, result: &Array<T, DynRank>)
where
    T: Copy,
    Sc: Shape,
    Lc: Layout,
{
    assert_output_shape(c, result.shape().dims());
    for (dst, src) in c.iter_mut().zip(result.iter()) {
        *dst = *src;
    }
}

fn add_result<T, Sc, Lc>(c: &mut Slice<T, Sc, Lc>, result: &Array<T, DynRank>, beta: T)
where
    T: Copy + MulAdd<Output = T>,
    Sc: Shape,
    Lc: Layout,
{
    assert_output_shape(c, result.shape().dims());
    for (dst, src) in c.iter_mut().zip(result.iter()) {
        *dst = beta.mul_add(*dst, *src);
    }
}
