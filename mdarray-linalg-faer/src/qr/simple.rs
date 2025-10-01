use dyn_stack::{MemBuffer, MemStack};
use faer_traits::ComplexField;
use mdarray::{DSlice, Layout};
use mdarray_linalg::{into_faer, into_faer_mut};
use num_complex::ComplexFloat;

pub fn qr_faer<
    T: ComplexFloat + ComplexField + Default + 'static,
    La: Layout,
    Lq: Layout,
    Lr: Layout,
>(
    a: &DSlice<T, 2, La>,
    q_mda: Option<&mut DSlice<T, 2, Lq>>,
    r_mda: &mut DSlice<T, 2, Lr>,
) {
    let (m, n) = *a.shape();
    let rank = Ord::min(m, n);
    let par = faer::get_global_parallelism();

    let block_size = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(m, n);

    let mut qr_mat = into_faer(a).to_owned();
    let mut h_factor = faer::Mat::<T>::zeros(block_size, rank);

    let _ = faer::linalg::qr::no_pivoting::factor::qr_in_place(
        qr_mat.as_mut(),
        h_factor.as_mut(),
        par,
        MemStack::new(&mut MemBuffer::new(
            faer::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
                m,
                n,
                block_size,
                par,
                faer::prelude::default(),
            ),
        )),
        faer::prelude::default(),
    );

    let mut r_faer = into_faer_mut(r_mda);
    for i in 0..rank {
        for j in i..n {
            r_faer[(i, j)] = qr_mat[(i, j)];
        }
        for j in 0..i {
            r_faer[(i, j)] = T::zero();
        }
    }
    for i in rank..m {
        for j in 0..n {
            r_faer[(i, j)] = T::zero();
        }
    }

    if let Some(q) = q_mda {
        let mut q_faer = into_faer_mut(q);
        // TODO: check why this is necessary
        for i in 0..m {
            for j in 0..m {
                if i == j {
                    q_faer[(i, j)] = T::one();
                } else {
                    q_faer[(i, j)] = T::zero();
                }
            }
        }

        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                    qr_mat.as_ref(),
                    h_factor.as_ref(),
                    faer::Conj::No,
                    q_faer,
                    par,
                    MemStack::new(&mut MemBuffer::new(
                        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                            m,
                            block_size,
                            m,
                        )
                    )),
                );
    }
}
