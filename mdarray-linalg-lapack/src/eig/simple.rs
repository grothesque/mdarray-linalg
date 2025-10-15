use super::scalar::{LapackScalar, NeedsRwork};
use mdarray::{DSlice, Layout};
use mdarray_linalg::eig::{EigError, SchurError};
use mdarray_linalg::transpose_in_place;

use mdarray_linalg::{get_dims, into_i32};
use num_complex::ComplexFloat;
use std::ptr;
use std::ptr::null_mut;

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
            transpose_in_place(vl);
        }
        if let Some(ref mut vr) = right_eigenvectors {
            transpose_in_place(vr);
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
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
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
        transpose_in_place(a);
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
        transpose_in_place(a);
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

pub fn gees<
    La: Layout,
    Lwr: Layout,
    Lwi: Layout,
    Lvs: Layout,
    T: ComplexFloat + Default + LapackScalar + NeedsRwork<Elem = T>,
>(
    a: &mut DSlice<T, 2, La>,
    eigenvalues_real: &mut DSlice<T, 2, Lwr>,
    eigenvalues_imag: &mut DSlice<T, 2, Lwi>,
    schur_vectors: &mut DSlice<T, 2, Lvs>,
) -> Result<(), SchurError>
where
    T::Real: Into<T>,
{
    let (m, n) = get_dims!(a);

    if m != n {
        return Err(SchurError::NotSquareMatrix);
    }

    let jobvs = b'V';

    let (mvs, nvs) = get_dims!(schur_vectors);
    assert_eq!(mvs, n, "Schur vectors must have same number of rows as A");
    assert_eq!(nvs, n, "Schur vectors must be square (n × n)");

    let (mwr, nwr) = get_dims!(eigenvalues_real);
    let (mwi, nwi) = get_dims!(eigenvalues_imag);
    assert_eq!(mwr, 1, "Real eigenvalues must be a row vector");
    assert_eq!(nwr, n, "Real eigenvalues must have n elements");
    assert_eq!(mwi, 1, "Imaginary eigenvalues must be a row vector");
    assert_eq!(nwi, n, "Imaginary eigenvalues must have n elements");

    transpose_in_place(a);

    // Allocate workspace
    let mut sdim = 0i32;
    let lwork = {
        let mut work = vec![T::default(); 1];
        let mut info = 0;
        let rwork_len = T::rwork_len_gees(n);
        let mut rwork = vec![T::default(); rwork_len];
        let mut bwork_storage = if n > 0 {
            Some(vec![0i32; n as usize])
        } else {
            None
        }; // keep the vec alive

        let bwork: *mut i32 = bwork_storage
            .as_mut()
            .map_or(ptr::null_mut(), |v| v.as_mut_ptr());

        unsafe {
            T::lapack_gees(
                jobvs.try_into().unwrap(),
                b'N'.try_into().unwrap(), // No sorting
                ptr::null_mut(),          // Null select as *mut c_void
                n,
                a.as_mut_ptr(),
                n,
                &mut sdim,
                eigenvalues_real.as_mut_ptr(),
                eigenvalues_imag.as_mut_ptr(),
                schur_vectors.as_mut_ptr(),
                n,
                work.as_mut_ptr(),
                -1, // Query workspace size
                rwork.as_mut_ptr(),
                bwork,
                &mut info,
            );
        }

        if info != 0 {
            panic!("Error during workspace query: info = {info}");
        }
        T::lwork_from_query(&work[0])
    };

    let mut work = T::allocate(lwork);
    let rwork_len = T::rwork_len_gees(n);
    let mut rwork = vec![T::default(); rwork_len];
    let mut bwork_storage = if n > 0 {
        Some(vec![0i32; n as usize])
    } else {
        None
    }; // keep the vec alive

    let bwork: *mut i32 = bwork_storage
        .as_mut()
        .map_or(ptr::null_mut(), |v| v.as_mut_ptr());

    let mut info = 0;
    unsafe {
        T::lapack_gees(
            jobvs.try_into().unwrap(),
            b'N'.try_into().unwrap(), // No sorting
            ptr::null_mut(),          // Null select as *mut c_void
            n,
            a.as_mut_ptr(),
            n,
            &mut sdim,
            eigenvalues_real.as_mut_ptr(),
            eigenvalues_imag.as_mut_ptr(),
            schur_vectors.as_mut_ptr(),
            n,
            work.as_mut_ptr(),
            lwork,
            rwork.as_mut_ptr(),
            bwork,
            &mut info,
        );
    }

    if info < 0 {
        return Err(SchurError::BackendError(-info));
    } else if info > 0 {
        return Err(SchurError::BackendDidNotConverge { iterations: info });
    }
    Ok(())
}
pub fn gees_complex<
    La: Layout,
    Lw: Layout,
    Lvs: Layout,
    T: ComplexFloat + Default + LapackScalar + NeedsRwork<Elem = T>,
>(
    a: &mut DSlice<T, 2, La>,
    eigenvalues: &mut DSlice<T, 1, Lw>,    // shape (1, n)
    schur_vectors: &mut DSlice<T, 2, Lvs>, // shape (n, n)
) -> Result<(), SchurError> {
    let (m, n) = get_dims!(a);
    if m != n {
        return Err(SchurError::NotSquareMatrix);
    }

    let jobvs = b'V';

    // shape checks
    let (mvs, nvs) = get_dims!(schur_vectors);
    assert_eq!(mvs, n, "Schur vectors must have same number of rows as A");
    assert_eq!(nvs, n, "Schur vectors must be square (n × n)");

    transpose_in_place(a);

    // --- workspace query (lwork) ---
    let mut sdim: i32 = 0;

    let lwork = {
        let mut work = vec![T::default(); 1];
        let mut info = 0i32;

        // rwork is real workspace
        let mut rwork = vec![0.; 1];

        // keep bwork alive
        let mut bwork_storage = if n > 0 {
            Some(vec![0i32; n as usize])
        } else {
            None
        };
        let bwork: *mut i32 = bwork_storage
            .as_mut()
            .map_or(ptr::null_mut(), |v| v.as_mut_ptr());

        unsafe {
            T::lapack_gees(
                jobvs.try_into().unwrap(),
                b'N'.try_into().unwrap(), // no sorting
                ptr::null_mut(),          // select (unused)
                n,
                a.as_mut_ptr(),
                n,
                &mut sdim,
                eigenvalues.as_mut_ptr(),     // wr
                ptr::null_mut(),              // _wi (unused for complex)
                schur_vectors.as_mut_ptr(),   // vs
                n,                            // ldvs
                work.as_mut_ptr(),            // work (query)
                -1,                           // lwork = -1 -> query
                rwork.as_mut_ptr() as *mut _, // rwork (cast to match expected pointer)
                bwork,
                &mut info,
            );
        }

        if info != 0 {
            panic!("Error during workspace query (complex gees): info = {info}");
        }
        T::lwork_from_query(&work[0])
    };

    // allocate real work + bwork + rwork
    let mut work = T::allocate(lwork);
    let rwork_len = T::rwork_len_gees(n);
    // let mut rwork = T::allocate(rwork_len as i32);
    let mut rwork = vec![0.0; rwork_len];
    let mut bwork_storage = if n > 0 {
        Some(vec![0i32; n as usize])
    } else {
        None
    };
    let bwork: *mut i32 = bwork_storage
        .as_mut()
        .map_or(ptr::null_mut(), |v| v.as_mut_ptr());

    let mut info = 0i32;
    unsafe {
        T::lapack_gees(
            jobvs.try_into().unwrap(),
            b'N'.try_into().unwrap(), // no sorting
            ptr::null_mut(),          // select
            n,
            a.as_mut_ptr(),
            n,
            &mut sdim,
            eigenvalues.as_mut_ptr(),   // wr
            ptr::null_mut(),            // _wi (unused)
            schur_vectors.as_mut_ptr(), // vs
            n,                          // ldvs
            work.as_mut_ptr(),
            lwork,
            rwork.as_mut_ptr() as *mut _,
            bwork,
            &mut info,
        );
    }

    if info < 0 {
        return Err(SchurError::BackendError(-info));
    } else if info > 0 {
        return Err(SchurError::BackendDidNotConverge { iterations: info });
    }

    Ok(())
}
