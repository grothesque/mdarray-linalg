use dyn_stack::{MemBuffer, MemStack};
use faer_traits::ComplexField;
use mdarray::{DSlice, Layout};
use mdarray_linalg::SVDError;
use num_complex::ComplexFloat;

use crate::{into_faer, into_faer_diag_mut, into_faer_mut, into_faer_mut_transpose};

pub fn svd_faer<
    T: ComplexFloat + ComplexField + Default + 'static,
    La: Layout,
    Ls: Layout,
    Lu: Layout,
    Lvt: Layout,
>(
    a: &DSlice<T, 2, La>,
    s_mda: &mut DSlice<T, 2, Ls>,
    u_mda: Option<&mut DSlice<T, 2, Lu>>,
    vt_mda: Option<&mut DSlice<T, 2, Lvt>>,
) -> Result<(), SVDError> {
    let (m, n) = *a.shape();
    let a_faer = into_faer(a);
    let par = faer::get_global_parallelism();
    // let par = faer::Par::Seq; // Faster for small matrices

    match (u_mda, vt_mda) {
        (Some(x), Some(y)) => {
            let mut s_faer = into_faer_diag_mut(s_mda);
            let u_faer = into_faer_mut(x);
            let vt_faer = into_faer_mut_transpose(y);

            let ret = faer::linalg::svd::svd(
                a_faer,
                s_faer.as_mut(),
                Some(u_faer),
                Some(vt_faer),
                par,
                MemStack::new(&mut MemBuffer::new(faer::linalg::svd::svd_scratch::<T>(
                    m,
                    n,
                    faer::linalg::svd::ComputeSvdVectors::Full,
                    faer::linalg::svd::ComputeSvdVectors::Full,
                    par,
                    faer::prelude::default(),
                ))),
                faer::prelude::default(),
            );
            match ret {
                Ok(()) => Ok(()),
                Err(_) => Err(SVDError::BackendDidNotConverge {
                    superdiagonals: (0),
                }),
            }
        }
        (None, None) => {
            let mut s_faer = into_faer_diag_mut(s_mda);
            let ret = faer::linalg::svd::svd(
                a_faer,
                s_faer.as_mut(),
                None,
                None,
                par,
                MemStack::new(&mut MemBuffer::new(faer::linalg::svd::svd_scratch::<T>(
                    m,
                    n,
                    faer::linalg::svd::ComputeSvdVectors::No,
                    faer::linalg::svd::ComputeSvdVectors::No,
                    par,
                    faer::prelude::default(),
                ))),
                faer::prelude::default(),
            );
            match ret {
                Ok(()) => Ok(()),
                Err(_) => Err(SVDError::BackendDidNotConverge {
                    superdiagonals: (0),
                }),
            }
        }
        _ => Err(SVDError::InconsistentUV),
    }
}
