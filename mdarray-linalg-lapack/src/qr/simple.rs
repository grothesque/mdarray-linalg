use crate::QRConfig;
use mdarray::{Dim, Layout, Shape, Slice};
use mdarray_linalg::utils::into_i32;
use num_complex::ComplexFloat;

use super::scalar::{LapackScalar, NeedsRwork};

pub(super) fn geqrf<
    La: Layout,
    Lq: Layout,
    Lr: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
>(
    a: &mut Slice<T, (D0, D1), La>,
    q: &mut Slice<T, (D0, D2), Lq>,
    r: &mut Slice<T, (D2, D1), Lr>,
    mode: QRConfig,
) where
    T::Real: Into<T>,
{
    let ash = *a.shape();
    let (m, n) = (into_i32(ash.dim(0)), into_i32(ash.dim(1)));
    let min_mn = m.min(n);

    let ncols_q = match mode {
        QRConfig::Complete => m,
        QRConfig::Reduced => min_mn,
    };

    let qsh = *q.shape();
    let (mq, nq) = (into_i32(qsh.dim(0)), into_i32(qsh.dim(1)));

    let rsh = *r.shape();
    let (mr, nr) = (into_i32(rsh.dim(0)), into_i32(rsh.dim(1)));

    assert_eq!(mq, m, "Q must have the same number of rows as A");
    assert!(
        nq >= ncols_q,
        "Q has too few columns for the configured QR mode"
    );
    assert!(
        mr >= ncols_q,
        "R has too few rows for the configured QR mode"
    );
    assert_eq!(nr, n, "R must have the same number of columns as A");

    // Allocate tau (Householder scalars)
    let mut tau = vec![T::default(); min_mn as usize];

    let mut work = T::allocate(1);
    let lwork = -1;
    let mut info = 0;

    // Lapack works with column-major
    // transpose_in_place(a);

    // let mut a_col = vec![T::default(); (m * n) as usize];
    let a_col_size = (m as usize) * (m as usize).max(n as usize);
    let mut a_col = vec![T::default(); a_col_size];

    for i in 0..(m as usize) {
        for j in 0..(n as usize) {
            a_col[j * (m as usize) + i] = a[[i, j]];
        }
    }

    // Query optimal workspace size
    unsafe {
        T::lapack_geqrf(
            m,
            n,
            a_col.as_mut_ptr(),
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
            a_col.as_mut_ptr(),
            tau.as_mut_ptr(),
            work.as_mut_ptr() as *mut _,
            lwork,
            &mut info,
        );
    }

    assert_eq!(info, 0, "geqrf failed with info={}", info);

    for i in 0_usize..(mr as usize) {
        for j in 0_usize..(n as usize) {
            r[[i, j]] = if (i as i32) < min_mn && j >= i {
                a_col[j * (m as usize) + i]
            } else {
                T::default()
            };
        }
    }

    let mut work = T::allocate(1);
    let lwork = -1;

    // Construct Q explicitly using orgqr/ungqr
    unsafe {
        T::lapack_orgqr(
            m,
            ncols_q,
            min_mn,
            a_col.as_mut_ptr() as *mut _,
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
            ncols_q,
            min_mn,
            a_col.as_mut_ptr() as *mut _,
            tau.as_mut_ptr() as *mut _,
            work.as_mut_ptr() as *mut _,
            lwork,
            &mut info,
        );
    }

    assert_eq!(info, 0, "orgqr failed with info={}", info);

    for i in 0..(m as usize) {
        for j in 0..(ncols_q as usize) {
            q[[i, j]] = a_col[j * (m as usize) + i];
        }
    }
}
