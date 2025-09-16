use mdarray::{DSlice, DTensor, Layout, ViewMut};
use num_traits::{One, Signed};
use std::mem::swap;
use std::time::Instant;

pub fn find_pivot<T: PartialOrd + Signed, L: Layout>(
    mat: &DSlice<T, 2, L>,
    step: usize,
) -> (usize, usize) {
    let (m, n) = *mat.shape();
    let mut max = mat[[step, step]].abs();
    let mut pos = (step, step);

    for i in step..m {
        for j in step..n {
            let val = mat[[i, j]].abs();
            if val > max {
                max = val;
                pos = (i, j);
            }
        }
    }
    pos
}

pub fn swap_axis<T, L: Layout>(mat: &mut DSlice<T, 2, L>, axis: usize, i: usize, j: usize) {
    let other_dim = match axis {
        0 => mat.shape().1,
        1 => mat.shape().0,
        _ => panic!("Axis must be 0 (rows) or 1 (cols)"),
    };

    swap_axis_range(mat, axis, i, j, 0..other_dim);
}

pub fn swap_axis_range<T, L: Layout>(
    mat: &mut DSlice<T, 2, L>,
    axis: usize,
    i: usize,
    j: usize,
    range: std::ops::Range<usize>,
) {
    assert!(axis < 2, "Axis must be 0 (rows) or 1 (cols)");
    if i == j {
        return;
    }

    let (min_ij, max_ij) = if i < j { (i, j) } else { (j, i) };
    let (mut part1, mut part2) = mat.split_axis_at_mut(axis, min_ij + 1);

    let mut view_a = match axis {
        0 => part1.view_mut(min_ij, range.clone()),
        1 => part1.view_mut(range.clone(), min_ij),
        _ => unreachable!(),
    };

    let mut view_b = match axis {
        0 => part2.view_mut(max_ij - (min_ij + 1), range.clone()),
        1 => part2.view_mut(range.clone(), max_ij - (min_ij + 1)),
        _ => unreachable!(),
    };

    for (a, b) in view_a.iter_mut().zip(view_b.iter_mut()) {
        swap(a, b);
    }
}

pub fn minus_outer_pivot<
    T: Copy + std::ops::Mul<Output = T> + std::ops::Sub<Output = T> + Signed + PartialOrd,
    Lout: Layout,
>(
    a: Vec<T>,
    b: Vec<T>,
    out: &mut ViewMut<'_, T, (usize, usize), Lout>,
) -> (usize, usize) {
    let (m, n) = (a.len(), b.len());

    assert_eq!(
        out.shape(),
        &(m, n),
        "Output shape must match a.len() × b.len()"
    );

    let mut pos = (0, 0);
    let mut max = {
        out[[0, 0]] = out[[0, 0]] - a[0] * b[0];
        out[[0, 0]].abs()
    };

    for i in 0..m {
        for j in 0..n {
            if i == 0 && j == 0 {
                continue;
            }
            out[[i, j]] = out[[i, j]] - a[i] * b[j];
            let out_abs = out[[i, j]].abs();
            if out_abs > max {
                max = out_abs;
                pos = (i, j);
            }
        }
    }
    pos
}

pub fn eye<T: Clone + One + Default>(n: usize) -> DTensor<T, 2> {
    let mut a = DTensor::<T, 2>::zeros([n, n]);
    a.diag_mut(0).fill(T::one());
    a
}

pub fn gaussian_elimination_pivot<
    T: Copy
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Sub<Output = T>
        + Signed
        + PartialOrd,
    L: Layout,
>(
    work: &mut DSlice<T, 2, L>,
    lower: &mut DSlice<T, 2>,
    step: usize,
    pivot: T,
) -> (usize, usize) {
    let n = work.shape().0;
    if step < n - 1 {
        let multipliers: Vec<_> = work
            .view(step + 1.., step)
            .iter()
            .map(|x| *x / pivot)
            .collect();
        let wv = work.view(step, step..).iter().copied().collect();
        lower
            .view_mut(step + 1.., step)
            .iter_mut()
            .zip(multipliers.iter())
            .for_each(|(l, m)| *l = *m);
        minus_outer_pivot(multipliers, wv, &mut work.view_mut(step + 1.., step..))
    } else {
        (step, step)
    }
}

pub fn update_permutation_matrices<T>(
    p: &mut DSlice<T, 2>,
    q: &mut DSlice<T, 2>,
    lower: &mut DSlice<T, 2>,
    idx_pivot: (usize, usize),
    step: usize,
) {
    if idx_pivot.0 != step {
        swap_axis(p, 0, step, idx_pivot.0);
        if step > 0 {
            swap_axis_range(lower, 0, step, idx_pivot.0, 0..step);
        }
    }

    if idx_pivot.1 != step {
        swap_axis(q, 1, step, idx_pivot.1);
    }
}

pub fn prrlu<T, L: Layout>(
    a: &mut DSlice<T, 2, L>,
    p: &mut DSlice<T, 2>,
    q: &mut DSlice<T, 2>,
    lower: &mut DSlice<T, 2>,
    k: usize,
    epsilon: T,
) -> usize
where
    T: Copy
        + PartialOrd
        + Signed
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Neg<Output = T>
        + std::fmt::Debug,
{
    let (m, n) = *a.shape();

    let max_steps = k.min(n.min(m));

    let mut idx_pivot = find_pivot(a, 0);

    for step in 0..max_steps {
        swap_axis(a, 0, step, idx_pivot.0);
        swap_axis(a, 1, step, idx_pivot.1);

        update_permutation_matrices(p, q, lower, idx_pivot, step);

        let pivot_current = a[[step, step]];

        if is_pivot_too_small(pivot_current, epsilon) {
            return step;
        }

        idx_pivot = gaussian_elimination_pivot(a, lower, step, pivot_current);
        idx_pivot = (idx_pivot.0 + step + 1, idx_pivot.1 + step);
    }
    max_steps
}

pub fn is_pivot_too_small<T>(pivot: T, epsilon: T) -> bool
where
    T: Copy + PartialOrd + std::ops::Neg<Output = T>,
{
    pivot < epsilon && (-pivot) < epsilon
}
