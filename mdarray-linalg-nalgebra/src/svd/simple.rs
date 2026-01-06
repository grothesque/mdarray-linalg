use dyn_stack::{MemBuffer, MemStack};
use mdarray::{DSlice, Dim, Layout, Shape, Slice};
use mdarray_linalg::svd::SVDError;
use nalgebra_traits::ComplexField;
use num_complex::ComplexFloat;

use crate::{
    into_nalgebra, into_nalgebra_diag_mut, into_nalgebra_mut, into_nalgebra_mut_transpose,
};

pub fn svd_nalgebra<
    T: ComplexFloat + ComplexField + Default + 'static,
    D: Dim,
    La: Layout,
    Ls: Layout,
    Lu: Layout,
    Lvt: Layout,
>(
    a: &Slice<T, (D, D), La>,
    s_mda: &mut Slice<T, (D, D), Ls>,
    u_mda: Option<&mut Slice<T, (D, D), Lu>>,
    vt_mda: Option<&mut Slice<T, (D, D), Lvt>>,
) -> Result<(), SVDError> {
    let ash = *a.shape();
    let (m, n) = (ash.dim(0), ash.dim(1));

    let a_nalgebra = into_nalgebra(a);
    let par = nalgebra::get_global_parallelism();
    // let par = nalgebra::Par::Seq; // Faster for small matrices

    match (u_mda, vt_mda) {
        (Some(x), Some(y)) => {
            let mut s_nalgebra = into_nalgebra_diag_mut(s_mda);
            let u_nalgebra = into_nalgebra_mut(x);
            let vt_nalgebra = into_nalgebra_mut_transpose(y);

            let ret = nalgebra::linalg::svd::svd(
                a_nalgebra,
                s_nalgebra.as_mut(),
                Some(u_nalgebra),
                Some(vt_nalgebra),
                par,
                MemStack::new(&mut MemBuffer::new(
                    nalgebra::linalg::svd::svd_scratch::<T>(
                        m,
                        n,
                        nalgebra::linalg::svd::ComputeSvdVectors::Full,
                        nalgebra::linalg::svd::ComputeSvdVectors::Full,
                        par,
                        nalgebra::prelude::default(),
                    ),
                )),
                nalgebra::prelude::default(),
            );
            match ret {
                Ok(()) => Ok(()),
                Err(_) => Err(SVDError::BackendDidNotConverge {
                    superdiagonals: (0),
                }),
            }
        }
        (None, None) => {
            let mut s_nalgebra = into_nalgebra_diag_mut(s_mda);
            let ret = nalgebra::linalg::svd::svd(
                a_nalgebra,
                s_nalgebra.as_mut(),
                None,
                None,
                par,
                MemStack::new(&mut MemBuffer::new(
                    nalgebra::linalg::svd::svd_scratch::<T>(
                        m,
                        n,
                        nalgebra::linalg::svd::ComputeSvdVectors::No,
                        nalgebra::linalg::svd::ComputeSvdVectors::No,
                        par,
                        nalgebra::prelude::default(),
                    ),
                )),
                nalgebra::prelude::default(),
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
