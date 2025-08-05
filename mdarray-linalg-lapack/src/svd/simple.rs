use super::scalar::{LapackScalar, NeedsRwork};
use mdarray::{DSlice, DTensor, Layout, tensor};
use mdarray_linalg::{SVDError, SVDResult};
use mdarray_linalg::{get_dims, into_i32};
use num_complex::ComplexFloat;
use std::mem::MaybeUninit;
use std::ptr::null_mut;

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

pub fn dgesdd<
    La: Layout,
    Ls: Layout,
    Lu: Layout,
    Lvt: Layout,
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
>(
    a: &mut DSlice<T, 2, La>,
    s: &mut DSlice<T, 2, Ls>,
    mut u: Option<&mut DSlice<T, 2, Lu>>,
    mut vt: Option<&mut DSlice<T, 2, Lvt>>,
) -> Result<(), SVDError>
where
    T::Real: Into<T>,
{
    let (m, n) = get_dims!(a);
    let min_mn = m.min(n);
    let job = match (&u, &vt) {
        (Some(x), Some(y)) => {
            let ((mu, nu), (ms, ns), (mvt, nvt)) = get_dims!(x, s, y);
            assert_eq!(mu, nu, "U must be square (m × m)");
            assert_eq!(mvt, nvt, "VT must be square (n × n)");
            assert_eq!(ns, ms, "s must be square (min(m,n),min(m,n))");
            assert_eq!(
                ms, min_mn,
                "s must have min(m, n) rows (number of singular values)"
            );
            assert_eq!(mu, m, "U must have the same number of rows as A: U(m, m)");
            assert_eq!(
                nvt, n,
                "VT must have the same number of columns as A: VT(n, n)"
            );
            'A'
        }

        (None, None) => 'N',
        _ => return Err(SVDError::InconsistentUV),
    };

    let u_ptr: *mut T = u.as_mut().map_or(null_mut(), |x| x.as_mut_ptr());
    let vt_ptr: *mut T = vt.as_mut().map_or(null_mut(), |x| x.as_mut_ptr());

    let mut work = T::allocate(1);

    let lwork = -1i32;
    let mut iwork = vec![0i32; 8 * min_mn as usize];
    let mut info = 0;

    let row_major = a.stride(1) == 1;
    assert!(
        row_major || a.stride(0) == 1,
        "c must be contiguous in one dimension"
    );

    if row_major {
        to_column_major(a)
    };

    let mut rwork = vec![0.0; T::rwork_len(m, n)];

    unsafe {
        T::lapack_gesdd(
            job as i8,
            m,
            n,
            a.as_mut_ptr() as *mut _,
            m,
            s.as_mut_ptr() as *mut _,
            u_ptr as *mut _,
            m,
            vt_ptr as *mut _,
            n,
            work.as_mut_ptr() as *mut _,
            lwork,
            rwork.as_mut_ptr() as *mut _,
            iwork.as_mut_ptr() as *mut _,
            &mut info,
        );
    }

    let lwork = T::lwork_from_query(work.first().expect("Query buffer is empty"));
    let mut work = T::allocate(lwork);

    let lwork = lwork as usize;

    unsafe {
        T::lapack_gesdd(
            job as i8,
            m,
            n,
            a.as_mut_ptr() as *mut _,
            m,
            s.as_mut_ptr() as *mut _,
            u_ptr as *mut _,
            m,
            vt_ptr as *mut _,
            n,
            work.as_mut_ptr() as *mut _,
            lwork as i32,
            rwork.as_mut_ptr() as *mut _,
            iwork.as_mut_ptr() as *mut _,
            &mut info,
        );
    }

    if job == 'A' {
        math_transpose(u.unwrap());
        math_transpose(vt.unwrap());
    }

    if info < 0 {
        panic!(
            "Invalid argument to dgesdd: the {}-th parameter had an illegal value.",
            -info
        );
    } else if info > 0 {
        Err(SVDError::BackendDidNotConverge {
            superdiagonals: (info),
        })
    } else {
        Ok(())
    }
}

pub fn dgesdd_uninit<
    La: Layout,
    Lu: Layout,
    Ls: Layout,
    Lvt: Layout,
    T: ComplexFloat + Default + LapackScalar + NeedsRwork,
>(
    a: &mut DSlice<T, 2, La>,
    mut s: DTensor<MaybeUninit<T>, 2>,
    mut u: Option<DTensor<MaybeUninit<T>, 2>>,
    mut vt: Option<DTensor<MaybeUninit<T>, 2>>,
) -> SVDResult<T>
where
    T::Real: Into<T>,
{
    let (m, n) = get_dims!(a);

    let job = match (&u, &vt) {
        (Some(_), Some(_)) => 'A',
        (None, None) => 'N',
        _ => return Err(SVDError::InconsistentUV),
    };

    let mut work = T::allocate(1);
    let lwork = -1i32;
    let mut iwork = vec![0i32; 8 * m.min(n) as usize];
    let mut info = 0;

    let row_major = a.stride(1) == 1;
    assert!(
        row_major || a.stride(0) == 1,
        "c must be contiguous in one dimension"
    );

    if row_major {
        to_column_major(a)
    };

    let mut rwork = vec![0.0; T::rwork_len(m, n)];

    let u_ptr: *mut MaybeUninit<T> = u.as_mut().map_or(null_mut(), |x| x.as_mut_ptr());
    let vt_ptr: *mut MaybeUninit<T> = vt.as_mut().map_or(null_mut(), |x| x.as_mut_ptr());

    unsafe {
        T::lapack_gesdd(
            job as i8,
            m,
            n,
            a.as_mut_ptr() as *mut _,
            m,
            s.as_mut_ptr() as *mut _,
            u_ptr as *mut _,
            m,
            vt_ptr as *mut _,
            n,
            work.as_mut_ptr() as *mut _,
            lwork,
            rwork.as_mut_ptr() as *mut _,
            iwork.as_mut_ptr() as *mut _,
            &mut info,
        );
    }

    let lwork = T::lwork_from_query(work.first().expect("Query buffer is empty"));
    let mut work = T::allocate(lwork);

    unsafe {
        T::lapack_gesdd(
            job as i8,
            m,
            n,
            a.as_mut_ptr() as *mut _,
            m,
            s.as_mut_ptr() as *mut _,
            u_ptr as *mut _,
            m,
            vt_ptr as *mut _,
            n,
            work.as_mut_ptr() as *mut _,
            lwork,
            rwork.as_mut_ptr() as *mut _,
            iwork.as_mut_ptr() as *mut _,
            &mut info,
        );
    }

    if info < 0 {
        panic!(
            "Invalid argument to dgesdd: the {}-th parameter had an illegal value.",
            -info
        );
    } else if info > 0 {
        Err(SVDError::BackendDidNotConverge {
            superdiagonals: (info),
        })
    } else {
        Ok((
            unsafe { s.assume_init() },
            unsafe {
                if !u_ptr.is_null() {
                    let mut u = u.unwrap().assume_init();
                    math_transpose(&mut u);
                    u
                } else {
                    tensor![[T::default();1];1]
                }
            },
            unsafe {
                if !vt_ptr.is_null() {
                    let mut vt = vt.unwrap().assume_init();
                    math_transpose(&mut vt);
                    vt
                } else {
                    tensor![[T::default();1];1]
                }
            },
        ))
    }
}

pub fn math_transpose<T, L>(c: &mut DSlice<T, 2, L>)
where
    T: ComplexFloat,
    L: Layout,
{
    let (m, n) = *c.shape();

    assert_eq!(
        m, n,
        "Transpose in-place only implemented for square matrices."
    );

    for i in 0..m {
        for j in (i + 1)..n {
            let tmp = c[[i, j]];
            c[[i, j]] = c[[j, i]];
            c[[j, i]] = tmp;
        }
    }
}
