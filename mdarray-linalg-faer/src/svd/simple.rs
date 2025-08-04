use dyn_stack::{MemBuffer, MemStack};
use faer_traits::ComplexField;
use mdarray::{DSlice, Layout};
use mdarray_linalg::SVDError;
use num_complex::ComplexFloat;

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

/// Converts a `DSlice<T, 2, L>` (from `mdarray`) into a `faer::MatRef<'static, T>`.
/// This function **does not copy** any data.
fn into_faer<T, L: Layout>(mat: &DSlice<T, 2, L>) -> faer::mat::MatRef<'static, T> {
    let (nrows, ncols) = *mat.shape();
    let strides = (mat.stride(0), mat.stride(1));

    // SAFETY:
    // We are constructing a MatRef from raw parts. This requires that:
    // - `mat.as_ptr()` points to a valid matrix of size `nrows x ncols`
    // - The given strides correctly describe the memory layout
    unsafe { faer::MatRef::from_raw_parts(mat.as_ptr(), nrows, ncols, strides.0, strides.1) }
}

/// Converts a `DSlice<T, 2, L>` (from `mdarray`) into a `faer::MatMut<'static, T>`.
/// This function **does not copy** any data.
fn into_faer_mut<T, L: Layout>(mat: &mut DSlice<T, 2, L>) -> faer::mat::MatMut<'static, T> {
    let (nrows, ncols) = *mat.shape();
    let strides = (mat.stride(0), mat.stride(1));

    // SAFETY:
    // We are constructing a MatMut from raw parts. This requires that:
    // - `mat.as_mut_ptr()` points to a valid mutable matrix of size `nrows x ncols`
    // - The given strides correctly describe the memory layout
    unsafe {
        faer::MatMut::from_raw_parts_mut(
            mat.as_mut_ptr() as *mut _,
            nrows,
            ncols,
            strides.0,
            strides.1,
        )
    }
}

/// Converts a `DSlice<T, 2, L>` (from `mdarray`) into a
/// `faer::MatMut<'static, T>` and transposes data.  This function
/// **does not copy** any data.
fn into_faer_mut_transpose<T, L: Layout>(
    mat: &mut DSlice<T, 2, L>,
) -> faer::mat::MatMut<'static, T> {
    let (nrows, ncols) = *mat.shape();
    let strides = (mat.stride(0), mat.stride(1));

    // SAFETY:
    // We are constructing a MatMut from raw parts. This requires that:
    // - `mat.as_mut_ptr()` points to a valid mutable matrix of size `nrows x ncols`
    // - The given strides correctly describe the memory layout
    unsafe {
        faer::MatMut::from_raw_parts_mut(
            mat.as_mut_ptr() as *mut _,
            nrows,
            ncols,
            strides.1,
            strides.0,
        )
    }
}

/// Converts a mutable `DSlice<T, 2, L>` (from `mdarray`) into a `faer::diag::DiagMut<'static, T>`,
/// which is a mutable view over the diagonal elements of a matrix in Faer.
///
/// # Important Notes for Users:
/// - This function **does not copy** any data. It gives direct mutable access to
///   the diagonal values of the matrix represented by `mat`.
/// - The stride along the **Y-axis (i.e., column stride)** is chosen to be consistent
///   with LAPACK-style storage, where singular values are typically stored in the first row.
/// - This function is unsafe internally and assumes that `mat` contains at least `n` elements
///   in memory laid out consistently with the given stride.
fn into_faer_diag_mut<T, L: Layout>(mat: &mut DSlice<T, 2, L>) -> faer::diag::DiagMut<'static, T> {
    let (n, _) = *mat.shape();

    // SAFETY:
    // - `mat.as_mut_ptr()` must point to a buffer with at least `n` diagonal elements.
    // - `mat.stride(1)` is used as the step between diagonal elements, assuming storage
    //   along the first row for compatibility with LAPACK convention.
    unsafe { faer::diag::DiagMut::from_raw_parts_mut(mat.as_mut_ptr() as *mut _, n, mat.stride(1)) }
}
