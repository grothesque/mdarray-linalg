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

pub fn outer<T: Copy + std::ops::Mul<Output = T>>(
    a: &DSlice<T, 1>,
    b: &DSlice<T, 1>,
    out: &mut DSlice<T, 2>,
) {
    let (m, n) = (a.shape().0, b.shape().0);

    assert_eq!(
        out.shape(),
        &(m, n),
        "Output shape must match a.len() × b.len()"
    );

    for i in 0..m {
        for j in 0..n {
            out[[i, j]] = a[[i]] * b[[j]];
        }
    }
}

pub fn minus_outer<
    T: Copy + std::ops::Mul<Output = T> + std::ops::Sub<Output = T>,
    Lout: Layout,
>(
    a: Vec<T>,
    b: Vec<T>,
    out: &mut ViewMut<'_, T, (usize, usize), Lout>,
) {
    let (m, n) = (a.len(), b.len());

    assert_eq!(
        out.shape(),
        &(m, n),
        "Output shape must match a.len() × b.len()"
    );

    for i in 0..m {
        for j in 0..n {
            out[[i, j]] = out[[i, j]] - a[i] * b[j];
        }
    }
}

pub fn eye<T: Clone + One + Default>(n: usize) -> DTensor<T, 2> {
    let mut a = DTensor::<T, 2>::zeros([n, n]);
    a.diag_mut(0).fill(T::one());
    a
}

pub fn perform_elimination<
    T: Copy + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + std::ops::Sub<Output = T>,
    L: Layout,
>(
    work: &mut DSlice<T, 2, L>,
    lower: &mut DSlice<T, 2>,
    step: usize,
    pivot: T,
) {
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
        let start_minus_outer = Instant::now();
        minus_outer(multipliers, wv, &mut work.view_mut(step + 1.., step..));
        // println!("Time minus_outer = {:?}", start_minus_outer.elapsed());
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
        + std::ops::Neg<Output = T>,
{
    let (n, m) = *a.shape();

    let max_steps = k.min(n.saturating_sub(1).min(m));

    for step in 0..max_steps {
        let start_pivot = Instant::now();
        let idx_pivot = find_pivot(a, step);
        // println!("Time pivot = {:?}", start_pivot.elapsed());

        let start_swap = Instant::now();
        swap_axis(a, 0, step, idx_pivot.0);
        swap_axis(a, 1, step, idx_pivot.1);

        update_permutation_matrices(p, q, lower, idx_pivot, step);
        // println!("Time swap = {:?}", start_swap.elapsed());

        let pivot_current = a[[step, step]];
        if is_pivot_too_small(pivot_current, epsilon) {
            return step;
        }
        let start_elimination = Instant::now();
        perform_elimination(a, lower, step, pivot_current);
        // println!("Time elimination = {:?}", start_elimination.elapsed());
    }
    return max_steps;
}

pub fn is_pivot_too_small<T>(pivot: T, epsilon: T) -> bool
where
    T: Copy + PartialOrd + std::ops::Neg<Output = T>,
{
    pivot < epsilon && (-pivot) < epsilon
}
