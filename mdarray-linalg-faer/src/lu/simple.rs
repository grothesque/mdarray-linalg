use dyn_stack::{MemBuffer, MemStack};
use faer_traits::ComplexField;
use mdarray::{Dim, Layout, Shape, Slice};
use num_complex::ComplexFloat;

use crate::into_faer_mut;

pub fn lu_faer<
    T: ComplexFloat + ComplexField + Default + 'static,
    D0: Dim,
    D1: Dim,
    La: Layout,
    Ll: Layout,
    Lu: Layout,
    Lp: Layout,
>(
    a: &mut Slice<T, (D0, D1), La>,
    l_mda: &mut Slice<T, (D0, D0), Ll>,
    u_mda: &mut Slice<T, (D0, D1), Lu>,
    p_mda: &mut Slice<T, (D0, D0), Lp>,
) {
    let ash = *a.shape();
    let (m, n) = (ash.dim(0), ash.dim(1));

    let min_mn = m.min(n);
    let par = faer::get_global_parallelism();

    let mut lu_mat = into_faer_mut(a);

    let mut row_perm_fwd = vec![0usize; m];
    let mut row_perm_bwd = vec![0usize; m];

    // avoid a copy done in intern bu Faer
    faer::linalg::lu::partial_pivoting::factor::lu_in_place(
        lu_mat.as_mut(),
        &mut row_perm_fwd,
        &mut row_perm_bwd,
        par,
        MemStack::new(&mut MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T>(
                m,
                n,
                par,
                faer::prelude::default(),
            ),
        )),
        faer::prelude::default(),
    );

    let mut l_faer = into_faer_mut(l_mda);
    for i in 0..m {
        for j in 0..min_mn {
            if i == j {
                l_faer[(i, j)] = T::one();
            } else if i > j {
                l_faer[(i, j)] = lu_mat[(i, j)];
            } else {
                l_faer[(i, j)] = T::zero();
            }
        }
    }

    let mut u_faer = into_faer_mut(u_mda);
    for i in 0..min_mn {
        for j in 0..n {
            if i <= j {
                u_faer[(i, j)] = lu_mat[(i, j)];
            } else {
                u_faer[(i, j)] = T::zero();
            }
        }
    }

    let mut p_faer = into_faer_mut(p_mda);
    for i in 0..m {
        for j in 0..m {
            p_faer[(i, j)] = T::zero();
        }
    }
    for i in 0..m {
        let perm_idx = row_perm_fwd[i];
        p_faer[(i, perm_idx)] = T::one();
    }
}
