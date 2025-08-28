use super::scalar::{LapackScalar, NeedsRwork};
use crate::svd::SVDConfig;
use mdarray::{DSlice, DTensor, Layout, tensor};
use mdarray_linalg::{SVDError, SVDResult};

use mdarray_linalg::{get_dims, into_i32};
use num_complex::ComplexFloat;
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

pub fn gsvd<
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
    config: SVDConfig,
) -> Result<(), SVDError>
where
    T::Real: Into<T>,
{
    let (m, n) = get_dims!(a);
    let min_mn = m.min(n);

    // Determine which algorithm to use
    let use_divide_conquer = match config {
        SVDConfig::Auto => {
            // Auto: use divide and conquer for larger matrices, Jacobi for smaller ones
            min_mn > 100
        }
        SVDConfig::DivideConquer => true,
        SVDConfig::Jacobi => false,
    };

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

    // Create a backup copy of matrix A if we're in Auto mode and using divide-and-conquer
    // This allows fallback to gesvd with the original matrix if gesdd fails
    let a_backup = if use_divide_conquer && matches!(config, SVDConfig::Auto) {
        let mut ab = DTensor::<T, 2>::from_elem([m as usize, n as usize], T::default());
        for i in 0..(m as usize) {
            for j in 0..(n as usize) {
                ab[[i, j]] = a[[i, j]];
            }
        }
        Some(ab)
    } else {
        None
    };

    let info = if use_divide_conquer {
        call_gesdd(a, m, n, s.as_mut_ptr(), Some(u_ptr), Some(vt_ptr), job)
    } else {
        call_gesvd(a, m, n, s.as_mut_ptr(), Some(u_ptr), Some(vt_ptr), job)
    };

    if info < 0 {
        panic!(
            "Invalid argument to SVD: the {}-th parameter had an illegal value.",
            -info
        );
    } else if info > 0 && use_divide_conquer && (config == SVDConfig::Auto) {
        // If divide-and-conquer failed and the user asked for "Auto", fallback to Jacobi (gesvd).
        // This provides robustness since gesvd is generally more stable but slower than gesdd.
        // We restore the original matrix A from our backup since gesdd may have corrupted it.
        let mut backup = a_backup.unwrap();
        let info = call_gesvd(
            &mut backup,
            m,
            n,
            s.as_mut_ptr(),
            Some(u_ptr),
            Some(vt_ptr),
            job,
        );

        if info < 0 {
            panic!(
                "Invalid argument to fallback SVD: the {}-th parameter had an illegal value.",
                -info
            );
        } else if info > 0 {
            Err(SVDError::BackendDidNotConverge {
                superdiagonals: (info),
            })
        } else {
            if job == 'A' {
                math_transpose(u.unwrap());
                math_transpose(vt.unwrap());
            }
            Ok(())
        }
    } else if info > 0 {
        Err(SVDError::BackendDidNotConverge {
            superdiagonals: (info),
        })
    } else {
        if job == 'A' {
            math_transpose(u.unwrap());
            math_transpose(vt.unwrap());
        }
        Ok(())
    }
}

fn call_gesdd<T: ComplexFloat + Default + LapackScalar + NeedsRwork, La: Layout>(
    a: &mut DSlice<T, 2, La>,
    m: i32,
    n: i32,
    s_ptr: *mut T,
    u_ptr: Option<*mut T>,
    vt_ptr: Option<*mut T>,
    job: char,
) -> i32
where
    T::Real: Into<T>,
{
    let mut work = T::allocate(1);

    let lwork = -1i32;
    let mut iwork = vec![0i32; 8 * m.min(n) as usize];
    let mut info = 0;

    let row_major = a.stride(1) == 1;
    assert!(
        row_major || a.stride(0) == 1,
        "a must be contiguous in one dimension"
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
            s_ptr as *mut _,
            u_ptr.unwrap() as *mut _,
            m,
            vt_ptr.unwrap() as *mut _,
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
            s_ptr as *mut _,
            u_ptr.unwrap() as *mut _,
            m,
            vt_ptr.unwrap() as *mut _,
            n,
            work.as_mut_ptr() as *mut _,
            lwork as i32,
            rwork.as_mut_ptr() as *mut _,
            iwork.as_mut_ptr() as *mut _,
            &mut info,
        );
    }

    info
}

fn call_gesvd<T: ComplexFloat + Default + LapackScalar + NeedsRwork, La: Layout>(
    a: &mut DSlice<T, 2, La>,
    m: i32,
    n: i32,
    s_ptr: *mut T,
    u_ptr: Option<*mut T>,
    vt_ptr: Option<*mut T>,
    job: char,
) -> i32
where
    T::Real: Into<T>,
{
    let mut work = T::allocate(1);
    let lwork = -1i32;
    let mut info = 0;

    let row_major = a.stride(1) == 1;
    assert!(
        row_major || a.stride(0) == 1,
        "a must be contiguous in one dimension"
    );

    if row_major {
        to_column_major(a)
    };

    let mut rwork = vec![0.0; T::rwork_len(m, n)];

    // First call to query optimal workspace size
    unsafe {
        T::lapack_gesvd(
            job as i8,
            job as i8,
            m,
            n,
            a.as_mut_ptr() as *mut _,
            m,
            s_ptr as *mut _,
            u_ptr.unwrap_or(null_mut()) as *mut _,
            m,
            vt_ptr.unwrap_or(null_mut()) as *mut _,
            n,
            work.as_mut_ptr() as *mut _,
            lwork,
            rwork.as_mut_ptr() as *mut _,
            &mut info,
        );
    }

    let lwork = T::lwork_from_query(&work[0]);
    let mut work = T::allocate(lwork);

    // Second call with optimal workspace
    unsafe {
        T::lapack_gesvd(
            job as i8,
            job as i8,
            m,
            n,
            a.as_mut_ptr() as *mut _,
            m,
            s_ptr as *mut _,
            u_ptr.unwrap_or(null_mut()) as *mut _,
            m,
            vt_ptr.unwrap_or(null_mut()) as *mut _,
            n,
            work.as_mut_ptr() as *mut _,
            lwork,
            rwork.as_mut_ptr() as *mut _,
            &mut info,
        );
    }

    info
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
