use super::scalar::LapackScalar;
use mdarray::{DSlice, DTensor, Layout, tensor};
use mdarray_linalg::get_dims;
use num_complex::{Complex, ComplexFloat};
use std::mem::MaybeUninit;

pub fn into_i32<T>(x: T) -> i32
where
    T: TryInto<i32>,
    <T as TryInto<i32>>::Error: std::fmt::Debug,
{
    x.try_into().expect("dimension must fit into i32")
}

fn to_column_major<T, La: Layout>(a: &mut DSlice<T, 2, La>)
where
    T: ComplexFloat + Default,
    T::Real: Into<T>,
{
    let (rows, cols) = (a.shape().0, a.shape().1);
    let mut result = tensor![[T::default(); rows];cols];
    for j in 0..cols {
        for i in 0..rows {
            result[j * rows + i] = a[i * cols + j];
        }
    }

    for j in 0..cols {
        for i in 0..rows {
            a[j * rows + i] = result[j * rows + i];
        }
    }
}

pub fn geqrf<La: Layout, Lq: Layout, Lr: Layout, T: ComplexFloat + Default + LapackScalar>(
    a: &mut DSlice<T, 2, La>,
    q: &mut DSlice<T, 2, Lq>,
    r: &mut DSlice<T, 2, Lr>,
) where
    T::Real: Into<T>,
{
    let ((m, n), (mq, nq), (mr, nr)) = get_dims!(a, q, r);
    let min_mn = m.min(n);

    assert_eq!(mq, nq, "Q must be square (m × m)");
    assert_eq!(mr, min_mn, "R must have min(m,n) rows");
    assert_eq!(nr, n, "R must have n columns");

    // Allocate tau (Householder scalars)
    let mut tau = vec![T::default(); min_mn as usize];

    let mut work = [0.; 1];
    let lwork = -1;
    let mut info = 0;

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

    let lwork = work[0] as i32;

    let mut work = vec![T::default(); lwork as usize];

    transpose(a); // Lapack is col major

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

    transpose(a);

    for i in 0_usize..(min_mn as usize) {
        for j in i..(n as usize) {
            r[[i, j]] = a[[i, j]];
        }
    }

    transpose(a);

    //    Construct Q explicitly using orgqr/ungqr
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

    transpose(a);

    for i in 0_usize..(m as usize) {
        for j in 0_usize..(m as usize) {
            q[[i, j]] = a[[i, j]];
        }
    }
}

pub fn geqrf_uninit<La: Layout, Lq: Layout, Lr: Layout, T: ComplexFloat + Default + LapackScalar>(
    a: &mut DSlice<T, 2, La>,
    mut q: DTensor<MaybeUninit<T>, 2>,
    mut r: DTensor<MaybeUninit<T>, 2>,
) -> (DTensor<T, 2>, DTensor<T, 2>)
where
    T::Real: Into<T>,
{
    let ((m, n), (mq, nq), (mr, nr)) = get_dims!(a, q, r);
    let min_mn = m.min(n);

    assert_eq!(mq, nq, "Q must be square (m × m)");
    assert_eq!(mr, min_mn, "R must have min(m,n) rows");
    assert_eq!(nr, n, "R must have n columns");

    // Allocate tau (Householder scalars)
    let mut tau = vec![T::default(); min_mn as usize];

    let mut work = [0.];
    let lwork = -1i32;

    let mut info = 0;

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

    let lwork = work[0] as i32;
    let mut work = vec![T::default(); lwork as usize];

    transpose(a); // Lapack is col major
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

    transpose(a);

    for i in 0_usize..(min_mn as usize) {
        for j in i..(n as usize) {
            r[[i, j]].write(a[[i, j]]);
        }
    }

    transpose(a);

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
    transpose(a);

    for i in 0_usize..(m as usize) {
        for j in 0_usize..(m as usize) {
            q[[i, j]].write(a[[i, j]]);
        }
    }

    unsafe { (q.assume_init(), r.assume_init()) }
}

pub fn transpose<T, L>(c: &mut DSlice<T, 2, L>)
where
    T: ComplexFloat,
    L: Layout,
{
    let (m, n) = (*c.shape()).into();

    assert_eq!(
        m, n,
        "Transpose in-place only implemented for square matrices."
    );

    for i in 0..m {
        for j in (i + 1)..n {
            let tmp = c[[i, j]].clone();
            c[[i, j]] = c[[j, i]].clone();
            c[[j, i]] = tmp;
        }
    }
}
