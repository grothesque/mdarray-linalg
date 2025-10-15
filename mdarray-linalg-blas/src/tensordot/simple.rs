use num_complex::ComplexFloat;
use num_traits::Zero;

use mdarray::{DynRank, Slice, Tensor};

use crate::Blas;

use crate::matmul::scalar;
use mdarray_linalg::matmul::{MatMul, MatMulBuilder};

pub enum Axes {
    All,
    LastFirst { k: isize },
    Specific(Box<[isize]>, Box<[isize]>),
}

pub fn tensordot<T: Zero + ComplexFloat + std::fmt::Debug + scalar::BlasScalar>(
    a: &Slice<T>,
    b: &Slice<T>,
    axes: Axes,
) -> Tensor<T> {
    let rank_a = a.rank() as isize;
    let rank_b = b.rank() as isize;

    let extract_shape = |s: &DynRank| match s {
        DynRank::Dyn(arr) => arr.clone(),
        DynRank::One(n) => Box::new([*n]),
    };
    let shape_a = extract_shape(a.shape());
    let shape_b = extract_shape(b.shape());

    let (mut axes_a, mut axes_b) = match axes {
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

    axes_a.iter_mut().for_each(|ax| {
        if *ax < 0 {
            *ax += rank_a
        }
    });
    axes_b.iter_mut().for_each(|ax| {
        if *ax < 0 {
            *ax += rank_b
        }
    });

    axes_a.iter().zip(&axes_b).for_each(|(a_ax, b_ax)| {
        assert_eq!(
            shape_a[*a_ax as usize], shape_b[*b_ax as usize],
            "Dimension mismatch at contraction: A[axis {}] = {} ≠ B[axis {}] = {}",
            *a_ax, shape_a[*a_ax as usize], *b_ax, shape_b[*b_ax as usize]
        );
    });

    let compute_keep_axes = |rank: isize, axes: &[isize]| -> Vec<isize> {
        (0..rank).filter(|k| !axes.contains(k)).collect()
    };
    let keep_axes_a = compute_keep_axes(rank_a, &axes_a);
    let keep_axes_b = compute_keep_axes(rank_b, &axes_b);
    let compute_keep_shape = |axes: &[isize], shape: &[usize]| -> Vec<usize> {
        axes.iter().map(|&ax| shape[ax as usize]).collect()
    };

    let mut keep_shape_a = compute_keep_shape(&keep_axes_a, &shape_a);
    let keep_shape_b = compute_keep_shape(&keep_axes_b, &shape_b);

    let compute_size = |axes: &[isize], shape: &[usize]| -> usize {
        axes.iter().map(|&k| shape[k as usize]).product()
    };

    let contract_size_a = compute_size(&axes_a, &shape_a);
    let contract_size_b = compute_size(&axes_b, &shape_b);
    let keep_size_a = compute_size(&keep_axes_a, &shape_a);
    let keep_size_b = compute_size(&keep_axes_b, &shape_b);

    let order_a: Vec<usize> = keep_axes_a
        .iter()
        .chain(axes_a.iter())
        .map(|&x| x as usize)
        .collect();

    let order_b: Vec<usize> = axes_b
        .iter()
        .chain(keep_axes_b.iter())
        .map(|&x| x as usize)
        .collect();

    let trans_a = a.permute(order_a).to_tensor();
    let trans_b = b.permute(order_b).to_tensor();

    let a_resh = trans_a.reshape([keep_size_a, contract_size_a]);
    let b_resh = trans_b.reshape([contract_size_b, keep_size_b]);

    let ab_resh = Blas.matmul(&a_resh, &b_resh).eval();

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
