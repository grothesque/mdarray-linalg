use super::scalar::{LapackScalar, Workspace};
use mdarray::{DSlice, DTensor, Layout};
use mdarray_linalg::{get_dims, into_i32, transpose_in_place};
use num_complex::ComplexFloat;

pub fn getrf<La: Layout, Ll: Layout, Lu: Layout, T: ComplexFloat + Default + LapackScalar>(
    a: &mut DSlice<T, 2, La>,
    l: &mut DSlice<T, 2, Ll>,
    u: &mut DSlice<T, 2, Lu>,
) -> Vec<i32>
where
    T::Real: Into<T>,
{
    let ((m, n), (ml, nl), (mu, nu)) = get_dims!(a, l, u);
    let min_mn = m.min(n);

    assert_eq!(ml, m, "L must have m rows");
    assert_eq!(nl, min_mn, "L must have min(m,n) columns");
    assert_eq!(mu, min_mn, "U must have min(m,n) rows");
    assert_eq!(nu, n, "U must have n columns");

    let mut ipiv = vec![0i32; min_mn as usize];
    let mut info = 0;

    let mut a_col_major = DTensor::<T, 2>::zeros([n as usize, m as usize]);
    for i in 0..(n as usize) {
        for j in 0..(m as usize) {
            a_col_major[[i, j]] = a[[j, i]];
        }
    }

    unsafe {
        T::lapack_getrf(
            m,
            n,
            a_col_major.as_mut_ptr(),
            m, // lda
            ipiv.as_mut_ptr(),
            &mut info,
        );
    }

    for i in 0_usize..(m as usize) {
        for j in 0_usize..(min_mn as usize) {
            if i > j {
                l[[i, j]] = a_col_major[[j, i]];
            } else if i == j {
                l[[i, j]] = T::one();
            } else {
                l[[i, j]] = T::zero();
            }
        }
    }

    for i in 0_usize..(min_mn as usize) {
        for j in 0_usize..(n as usize) {
            if i <= j {
                u[[i, j]] = a_col_major[[j, i]];
            } else {
                u[[i, j]] = T::zero();
            }
        }
    }

    ipiv
}

pub fn getri<La: Layout, Lai: Layout, T: ComplexFloat + Default + LapackScalar + Workspace>(
    a: &mut DSlice<T, 2, La>,
    ipiv: &mut [i32],
) -> i32
where
    T::Real: Into<T>,
{
    let (m, n) = get_dims!(a);
    assert_eq!(m, n, "Input matrix must be square");
    assert_eq!(ipiv.len(), n as usize, "ipiv length must equal n");

    transpose_in_place(a);

    let mut info = 0;

    unsafe {
        T::lapack_getrf(
            m,
            n,
            a.as_mut_ptr(),
            m, // lda
            ipiv.as_mut_ptr(),
            &mut info,
        );
    }

    assert_eq!(info, 0, "GETRF failed");

    let mut work_query = T::allocate(1);
    unsafe {
        T::lapack_getri(
            n,
            a.as_mut_ptr(),
            n, // lda
            ipiv.as_ptr(),
            work_query.as_mut_ptr() as *mut T,
            -1,
            &mut info,
        );
    }
    assert_eq!(
        info, 0,
        "LAPACK GETRI workspace query failed with info = {}",
        info
    );

    let lwork = T::lwork_from_query(work_query.first().expect("Query buffer is empty"));
    let mut work = vec![T::zero(); lwork as usize];

    unsafe {
        T::lapack_getri(
            n,
            a.as_mut_ptr(),
            n, // lda
            ipiv.as_ptr(),
            work.as_mut_ptr(),
            lwork,
            &mut info,
        );
    }

    transpose_in_place(a);

    info
}

pub fn potrf<La: Layout, Ll: Layout, T: ComplexFloat + Default + LapackScalar>(
    a: &mut DSlice<T, 2, La>,
    uplo: char,
) -> i32
where
    T::Real: Into<T>,
{
    let (m, n) = get_dims!(a);

    assert_eq!(m, n, "Matrix must be square for Cholesky decomposition");

    let mut info = 0;

    transpose_in_place(a);

    let uplo_byte = match uplo {
        'U' | 'u' => b'U' as i8,
        'L' | 'l' => b'L' as i8,
        _ => panic!("uplo must be 'U' or 'L'"),
    };

    unsafe {
        T::lapack_potrf(
            uplo_byte,
            n,
            a.as_mut_ptr(),
            m, // lda
            &mut info,
        );
    }
    info
}
