use super::scalar::{LapackScalar, NeedsRwork};
use mdarray::{DSlice, DTensor, Layout, tensor};
use mdarray_linalg::EigError;

use mdarray_linalg::{get_dims, into_i32};
use num_complex::{Complex, ComplexFloat};
use std::ptr::null_mut;

fn to_column_major<T, La: Layout>(a: &mut DSlice<T, 2, La>)
where
    T: ComplexFloat + Default,
    T::Real: Into<T>,
{
    let (rows, cols) = (a.shape().0, a.shape().1);
    let mut result = tensor![[T::default(); rows]; cols];
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

pub fn geig<
    La: Layout,
    Ler: Layout,
    Lei: Layout,
    Lvl: Layout,
    Lvr: Layout,
    T: ComplexFloat + Default + LapackScalar + NeedsRwork<Elem = T>,
>(
    a: &mut DSlice<T, 2, La>,
    eigenvalues_real: &mut DSlice<T, 2, Ler>,
    eigenvalues_imag: &mut DSlice<T, 2, Lei>,
    mut left_eigenvectors: Option<&mut DSlice<T, 2, Lvl>>,
    mut right_eigenvectors: Option<&mut DSlice<T, 2, Lvr>>,
) -> Result<(), EigError>
where
    T::Real: Into<T>,
{
    let (m, n) = get_dims!(a);

    if m != n {
        return Err(EigError::NotSquareMatrix);
    }

    let jobvl = if left_eigenvectors.is_some() {
        'V'
    } else {
        'N'
    };
    let jobvr = if right_eigenvectors.is_some() {
        'V'
    } else {
        'N'
    };

    // Validate dimensions if eigenvectors are requested
    if let Some(ref vl) = left_eigenvectors {
        let (mvl, nvl) = get_dims!(vl);
        assert_eq!(
            mvl, n,
            "Left eigenvectors must have same number of rows as A"
        );
        assert_eq!(nvl, n, "Left eigenvectors must be square (n × n)");
    }

    if let Some(ref vr) = right_eigenvectors {
        let (mvr, nvr) = get_dims!(vr);
        assert_eq!(
            mvr, n,
            "Right eigenvectors must have same number of rows as A"
        );
        assert_eq!(nvr, n, "Right eigenvectors must be square (n × n)");
    }

    // Validate eigenvalue vectors dimensions
    let (mer, ner) = get_dims!(eigenvalues_real);
    let (mei, nei) = get_dims!(eigenvalues_imag);
    assert_eq!(mer, 1, "Real eigenvalues must be a row vector");
    assert_eq!(ner, n, "Real eigenvalues must have n elements");
    assert_eq!(mei, 1, "Imaginary eigenvalues must be a row vector");
    assert_eq!(nei, n, "Imaginary eigenvalues must have n elements");

    let vl_ptr: *mut T = left_eigenvectors
        .as_mut()
        .map_or(null_mut(), |x| x.as_mut_ptr());
    let vr_ptr: *mut T = right_eigenvectors
        .as_mut()
        .map_or(null_mut(), |x| x.as_mut_ptr());

    let info = call_geev(
        a,
        n,
        eigenvalues_real.as_mut_ptr(),
        eigenvalues_imag.as_mut_ptr(),
        vl_ptr,
        vr_ptr,
        jobvl,
        jobvr,
    );

    if info < 0 {
        panic!(
            "Invalid argument to EIG: the {}-th parameter had an illegal value.",
            -info
        );
    } else if info > 0 {
        Err(EigError::BackendDidNotConverge { iterations: info })
    } else {
        // Transpose eigenvectors if computed (LAPACK returns column-major)
        if let Some(ref mut vl) = left_eigenvectors {
            math_transpose(vl);
        }
        if let Some(ref mut vr) = right_eigenvectors {
            math_transpose(vr);
        }
        Ok(())
    }
}

pub fn geigh<
    La: Layout,
    Lw: Layout,
    Lv: Layout,
    T: ComplexFloat + Default + LapackScalar + NeedsRwork<Elem = T>,
>(
    a: &mut DSlice<T, 2, La>,
    eigenvalues: &mut DSlice<T, 2, Lw>,
    eigenvectors: &mut DSlice<T, 2, Lv>,
) -> Result<(), EigError>
where
    T::Real: Into<T>,
{
    let (m, n) = get_dims!(a);

    if m != n {
        return Err(EigError::NotSquareMatrix);
    }

    // Validate dimensions
    let (mw, nw) = get_dims!(eigenvalues);
    let (mv, nv) = get_dims!(eigenvectors);

    assert_eq!(mw, 1, "Eigenvalues must be a row vector");
    assert_eq!(nw, n, "Eigenvalues must have n elements");
    assert_eq!(mv, n, "Eigenvectors must have same number of rows as A");
    assert_eq!(nv, n, "Eigenvectors must be square (n × n)");

    let info = call_syev(
        a,
        n,
        eigenvalues.as_mut_ptr(),
        eigenvectors.as_mut_ptr(),
        'V', // Always compute eigenvectors for geigh
        'U', // Use upper triangle
    );

    if info < 0 {
        panic!(
            "Invalid argument to EIGH: the {}-th parameter had an illegal value.",
            -info
        );
    } else if info > 0 {
        Err(EigError::BackendDidNotConverge { iterations: info })
    } else {
        // Transpose eigenvectors (LAPACK returns column-major)
        math_transpose(eigenvectors);
        Ok(())
    }
}

fn call_geev<T: ComplexFloat + Default + LapackScalar + NeedsRwork<Elem = T>, La: Layout>(
    a: &mut DSlice<T, 2, La>,
    n: i32,
    wr_ptr: *mut T,
    wi_ptr: *mut T,
    vl_ptr: *mut T,
    vr_ptr: *mut T,
    jobvl: char,
    jobvr: char,
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
        to_column_major(a);
    }

    let mut rwork = vec![0.0; T::rwork_len_geev(n)];

    // First call to query optimal workspace size
    unsafe {
        T::lapack_geev(
            jobvl as i8,
            jobvr as i8,
            n,
            a.as_mut_ptr(),
            n,
            wr_ptr,
            wi_ptr,
            vl_ptr,
            n,
            vr_ptr,
            n,
            work.as_mut_ptr(),
            lwork,
            rwork.as_mut_ptr() as *mut _,
            &mut info,
        );
    }

    let lwork = T::lwork_from_query(&work[0]);
    let mut work = T::allocate(lwork);

    // Second call with optimal workspace
    unsafe {
        T::lapack_geev(
            jobvl as i8,
            jobvr as i8,
            n,
            a.as_mut_ptr(),
            n,
            wr_ptr,
            wi_ptr,
            vl_ptr,
            n,
            vr_ptr,
            n,
            work.as_mut_ptr(),
            lwork,
            rwork.as_mut_ptr() as *mut _,
            &mut info,
        );
    }

    info
}

fn call_syev<T: ComplexFloat + Default + LapackScalar + NeedsRwork<Elem = T>, La: Layout>(
    a: &mut DSlice<T, 2, La>,
    n: i32,
    w_ptr: *mut T,
    v_ptr: *mut T,
    jobz: char,
    uplo: char,
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
        to_column_major(a);
    }

    let mut rwork = vec![0.0; T::rwork_len_syev(n)];

    // First call to query optimal workspace size
    unsafe {
        T::lapack_syev(
            jobz as i8,
            uplo as i8,
            n,
            a.as_mut_ptr(),
            n,
            w_ptr,
            work.as_mut_ptr(),
            lwork,
            rwork.as_mut_ptr() as *mut _,
            &mut info,
        );
    }

    let lwork = T::lwork_from_query(&work[0]);
    let mut work = T::allocate(lwork);

    // Second call with optimal workspace
    unsafe {
        T::lapack_syev(
            jobz as i8,
            uplo as i8,
            n,
            a.as_mut_ptr(),
            n,
            w_ptr,
            work.as_mut_ptr(),
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
