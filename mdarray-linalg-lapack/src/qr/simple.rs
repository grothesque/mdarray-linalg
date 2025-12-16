use mdarray::{DSlice, Dim, Layout, Shape, Slice};
use mdarray_linalg::{get_dims, into_i32, transpose_in_place};
use num_complex::ComplexFloat;

use super::scalar::{LapackScalar, NeedsRwork};

pub fn geqrf<
    La: Layout,
    Lq: Layout,
    Lr: Layout,
    D0: Dim,
    D1: Dim,
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
>(
    a: &mut Slice<T, (D0, D1), La>,
    q: &mut DSlice<T, 2, Lq>,
    r: &mut DSlice<T, 2, Lr>,
) where
    T::Real: Into<T>,
{
    let ash = *a.shape();
    let (m, n) = (into_i32(ash.dim(0)), into_i32(ash.dim(1)));

    let qsh = *q.shape();
    let (mq, nq) = (into_i32(ash.dim(0)), into_i32(ash.dim(1)));

    let rsh = *q.shape();
    let (mr, nr) = (into_i32(ash.dim(0)), into_i32(ash.dim(1)));

    let min_mn = m.min(n);

    assert_eq!(mq, nq, "Q must be square (m × m)");
    assert_eq!(mr, min_mn, "R must have min(m,n) rows");
    assert_eq!(nr, n, "R must have n columns");

    // Allocate tau (Householder scalars)
    let mut tau = vec![T::default(); min_mn as usize];

    let mut work = T::allocate(1);
    let lwork = -1;
    let mut info = 0;

    // Lapack works with column-major
    transpose_in_place(a);

    // Query optimal workspace size
    unsafe {
        T::lapack_geqrf(
            m,
            n,
            a.as_mut_ptr(),
            tau.as_mut_ptr(),
            work.as_mut_ptr() as *mut _,
            lwork,
            &mut info,
        );
    }

    let lwork = T::lwork_from_query(work.first().expect("Query buffer is empty"));
    let mut work = T::allocate(lwork);

    // Actual computation
    unsafe {
        T::lapack_geqrf(
            m,
            n,
            a.as_mut_ptr(),
            tau.as_mut_ptr(),
            work.as_mut_ptr() as *mut _,
            lwork,
            &mut info,
        );
    }

    for i in 0_usize..(min_mn as usize) {
        for j in i..(n as usize) {
            r[[i, j]] = a[[j, i]];
        }
    }

    let mut work = T::allocate(1);
    let lwork = -1;

    // Construct Q explicitly using orgqr/ungqr
    unsafe {
        T::lapack_orgqr(
            m,
            min_mn,
            a.as_mut_ptr() as *mut _,
            tau.as_mut_ptr() as *mut _,
            work.as_mut_ptr() as *mut _,
            lwork,
            &mut info,
        );
    }

    let lwork = T::lwork_from_query(work.first().expect("Query buffer is empty"));
    let mut work = T::allocate(lwork);

    unsafe {
        T::lapack_orgqr(
            m,
            min_mn,
            a.as_mut_ptr() as *mut _,
            tau.as_mut_ptr() as *mut _,
            work.as_mut_ptr() as *mut _,
            lwork,
            &mut info,
        );
    }

    for i in 0_usize..(m as usize) {
        for j in 0_usize..(m as usize) {
            q[[i, j]] = a[[j, i]];
        }
    }
}
