use std::mem::ManuallyDrop;

use mdarray::{DSlice, DTensor, Layout, Strided, StridedMapping, View, tensor};
use num_complex::ComplexFloat;
use num_traits::{One, Zero};

/// Displays a numeric `mdarray` in a human-readable format (NumPy-style)
pub fn pretty_print<T: ComplexFloat + std::fmt::Display>(mat: &DTensor<T, 2>)
where
    <T as num_complex::ComplexFloat>::Real: std::fmt::Display,
{
    let shape = mat.shape();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            let v = mat[[i, j]];
            print!("{:>10.4} {:+.4}i  ", v.re(), v.im(),);
        }
        println!();
    }
    println!();
}

/// Textbook implementation of matrix multiplication, useful for
/// debugging and simple tests without relying on a backend
pub fn naive_matmul<T: ComplexFloat>(a: &DSlice<T, 2>, b: &DSlice<T, 2>, c: &mut DSlice<T, 2>) {
    for (mut ci, ai) in c.rows_mut().into_iter().zip(a.rows()) {
        for (aik, bk) in ai.expr().into_iter().zip(b.rows()) {
            for (cij, bkj) in ci.expr_mut().into_iter().zip(bk) {
                *cij = (*aik) * (*bkj) + *cij;
            }
        }
    }
}

/// Safely casts a value to `i32`
pub fn into_i32<T>(x: T) -> i32
where
    T: TryInto<i32>,
    <T as TryInto<i32>>::Error: std::fmt::Debug,
{
    x.try_into().expect("dimension must fit into i32")
}

/// Returns the dimensions of an arbitrary number of matrices (e.g.,
/// `A, B, C → (ma, na), (mb, nb), (mc, nc))`
#[macro_export]
macro_rules! get_dims {
    ( $( $matrix:expr ),+ ) => {
        (
            $(
                {
                    let shape = $matrix.shape();
                    (into_i32(shape.0), into_i32(shape.1))
                }
            ),*
        )
    };
}

/// Make sure that matrix shapes are compatible with `C = A * B`, and
/// return the dimensions `(m, n, k)` safely cast to `i32`, where `C` is `(m
/// x n)`, and `k` is the common dimension of `A` and `B`
pub fn dims3(
    a_shape: &(usize, usize),
    b_shape: &(usize, usize),
    c_shape: &(usize, usize),
) -> (i32, i32, i32) {
    let (m, k) = *a_shape;
    let (k2, n) = *b_shape;
    let (m2, n2) = *c_shape;

    assert!(m == m2, "a and c must agree in number of rows");
    assert!(n == n2, "b and c must agree in number of columns");
    assert!(
        k == k2,
        "a's number of columns must be equal to b's number of rows"
    );

    (into_i32(m), into_i32(n), into_i32(k))
}

/// Make sure that matrix shapes are compatible with `A * B`, and return
/// the dimensions `(m, n)` safely cast to `i32`
pub fn dims2(a_shape: &(usize, usize), b_shape: &(usize, usize)) -> (i32, i32) {
    let (m, k) = *a_shape;
    let (k2, n) = *b_shape;

    assert!(
        k == k2,
        "a's number of columns must be equal to b's number of rows"
    );

    (into_i32(m), into_i32(n))
}

/// Handles different stride layouts by selecting the correct memory
/// order and stride for contiguous arrays
#[macro_export]
macro_rules! trans_stride {
    ($x:expr, $same_order:expr, $other_order:expr) => {{
        if $x.stride(1) == 1 {
            ($same_order, into_i32($x.stride(0)))
        } else {
            {
                assert!($x.stride(0) == 1, stringify!($x must be contiguous in one dimension));
                ($other_order, into_i32($x.stride(1)))
            }
        }
    }};
}

/// Transposes a matrix in-place. Dimensions stay the same, only the memory ordering changes.
/// - For square matrices: swaps elements across the main diagonal.
/// - For rectangular matrices: reshuffles data in a temporary buffer so that the
///   same `(rows, cols)` slice now represents the transposed layout.
pub fn transpose_in_place<T, L>(c: &mut DSlice<T, 2, L>)
where
    T: ComplexFloat + Default,
    L: Layout,
{
    let (m, n) = *c.shape();

    if n == m {
        for i in 0..m {
            for j in (i + 1)..n {
                c.swap(i * n + j, j * n + i);
            }
        }
    } else {
        let mut result = tensor![[T::default(); m]; n];
        for j in 0..n {
            for i in 0..m {
                result[j * m + i] = c[i * n + j];
            }
        }
        for j in 0..n {
            for i in 0..m {
                c[j * m + i] = result[j * m + i];
            }
        }
    }
}

/// Convert pivot indices to permutation matrix
pub fn ipiv_to_permutation_matrix<T: ComplexFloat>(ipiv: &[i32], m: usize) -> DTensor<T, 2> {
    let mut p = tensor![[T::zero(); m]; m];

    for i in 0..m {
        p[[i, i]] = T::one();
    }

    // Apply row swaps according to LAPACK's ipiv convention
    for i in 0..ipiv.len() {
        let pivot_row = (ipiv[i] - 1) as usize; // LAPACK uses 1-based indexing
        if pivot_row != i {
            for j in 0..m {
                let temp = p[[i, j]];
                p[[i, j]] = p[[pivot_row, j]];
                p[[pivot_row, j]] = temp;
            }
        }
    }

    p
}

/// Given an input matrix of shape `(m × n)`, this function creates and returns
/// a new matrix of shape `(n × m)`, where each element at position `(i, j)` in the
/// original is moved to position `(j, i)` in the result.
pub fn to_col_major<T, L>(c: &DSlice<T, 2, L>) -> DTensor<T, 2>
where
    T: ComplexFloat + Default + Clone,
    L: Layout,
{
    let (m, n) = *c.shape();
    let mut result = DTensor::<T, 2>::zeros([n, m]);

    for i in 0..m {
        for j in 0..n {
            result[[j, i]] = c[[i, j]];
        }
    }

    result
}

/// Computes the trace of a square matrix (sum of diagonal elements).
/// # Examples
/// ```
/// use mdarray::tensor;
/// use mdarray_linalg::trace;
///
/// let a = tensor![[1., 2., 3.],
///                 [4., 5., 6.],
///                 [7., 8., 9.]];
///
/// let tr = trace(&a);
/// assert_eq!(tr, 15.0);
/// ```
pub fn trace<T, L>(a: &DSlice<T, 2, L>) -> T
where
    T: ComplexFloat + std::ops::Add<Output = T> + Copy,
    L: Layout,
{
    let (m, n) = *a.shape();
    assert_eq!(m, n, "trace is only defined for square matrices");

    let mut tr = T::zero();
    for i in 0..n {
        tr = tr + a[[i, i]];
    }
    tr
}

/// Creates an identity matrix of size `n x n`.
/// # Examples
/// ```
/// use mdarray::tensor;
/// use mdarray_linalg::eye;
///
/// let i3 = eye::<f64>(3);
/// assert_eq!(i3, tensor![[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);
/// ```
pub fn eye<T: Zero + One>(n: usize) -> DTensor<T, 2> {
    DTensor::<T, 2>::from_fn([n, n], |i| if i[0] == i[1] { T::one() } else { T::zero() })
}

/// Creates a diagonal matrix of size `n x n` with ones on a specified diagonal.
///
/// The diagonal can be shifted using `k`:  
/// - `k = 0` → main diagonal (default, standard identity)  
/// - `k > 0` → k-th diagonal above the main one  
/// - `k < 0` → k-th diagonal below the main one
/// # Examples
/// ```
/// use mdarray::tensor;
/// use mdarray_linalg::eye_k;
///
/// let i3 = eye_k::<f64>(3, 1);
/// assert_eq!(i3, tensor![[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]]);
/// ```
pub fn eye_k<T: Zero + One>(n: usize, k: isize) -> DTensor<T, 2> {
    DTensor::<T, 2>::from_fn([n, n], |i| {
        if (i[1] as isize - i[0] as isize) == k {
            T::one()
        } else {
            T::zero()
        }
    })
}

/// Computes the Kronecker product of two 2D tensors.
///
/// The Kronecker product of matrices `A (m×n)` and `B (p×q)` is defined as the
/// block matrix of size `(m*p) × (n*q)` where each element `a[i, j]` of `A`
/// multiplies the entire matrix `B`.
///
/// # Examples
/// ```
/// use mdarray::tensor;
/// use mdarray_linalg::kron;
///
/// let a = tensor![[1., 2.],
///                 [3., 4.]];
///
/// let b = tensor![[0., 5.],
///                 [6., 7.]];
///
/// let k = kron(&a, &b);
///
/// assert_eq!(k, tensor![
///     [ 0.,  5.,  0., 10.],
///     [ 6.,  7., 12., 14.],
///     [ 0., 15.,  0., 20.],
///     [18., 21., 24., 28.]
/// ]);
/// ```
pub fn kron<T, La, Lb>(a: &DSlice<T, 2, La>, b: &DSlice<T, 2, Lb>) -> DTensor<T, 2>
where
    T: ComplexFloat + std::ops::Mul<Output = T> + Copy,
    La: Layout,
    Lb: Layout,
{
    let (ma, na) = *a.shape();
    let (mb, nb) = *b.shape();

    let out_shape = [ma * mb, na * nb];

    DTensor::<T, 2>::from_fn(out_shape, |idx| {
        let i = idx[0];
        let j = idx[1];

        let ai = i / mb;
        let bi = i % mb;
        let aj = j / nb;
        let bj = j % nb;

        a[[ai, aj]] * b[[bi, bj]]
    })
}

/// Converts a `DSlice<T, 2, L>` (from `mdarray`) into a `faer::MatRef<'static, T>`.
/// This function **does not copy** any data.
pub fn into_faer<T, L: Layout>(mat: &DSlice<T, 2, L>) -> faer::mat::MatRef<'static, T> {
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
pub fn into_faer_mut<T, L: Layout>(mat: &mut DSlice<T, 2, L>) -> faer::mat::MatMut<'static, T> {
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

/// Converts a `faer::Mat<T>` into a `DTensor<T, 2>` (from `mdarray`) by constructing
/// a strided view over the matrix memory. This function **does not copy** any data.
pub fn into_mdarray<T: std::clone::Clone>(mat: faer::Mat<T>) -> DTensor<T, 2> {
    // Manually dropping to avoid a double free: DTensor will take ownership of the data,
    // so we must prevent Rust from automatically dropping the original matrix.
    let mut mat = ManuallyDrop::new(mat);

    let (nrows, ncols) = (mat.nrows(), mat.ncols());

    // faer and mdarray have different memory layouts; we need to construct a
    // strided mapping explicitly to describe the layout of `mat` to mdarray.
    let mapping = StridedMapping::new((nrows, ncols), &[mat.row_stride(), mat.col_stride()]);

    // SAFETY:
    // We use `new_unchecked` because the memory layout in faer isn't guaranteed
    // to satisfy mdarray's internal invariants automatically.
    // `from_raw_parts` isn't usable here due to layout incompatibilities.
    let view_strided: View<'_, _, (usize, usize), Strided> =
        unsafe { mdarray::View::new_unchecked(mat.as_ptr_mut(), mapping) };

    DTensor::<T, 2>::from(view_strided)
}

/// Converts a `DSlice<T, 2, L>` (from `mdarray`) into a
/// `faer::MatMut<'static, T>` and transposes data.  This function
/// **does not copy** any data.
pub fn into_faer_mut_transpose<T, L: Layout>(
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
pub fn into_faer_diag_mut<T, L: Layout>(
    mat: &mut DSlice<T, 2, L>,
) -> faer::diag::DiagMut<'static, T> {
    let (n, _) = *mat.shape();

    // SAFETY:
    // - `mat.as_mut_ptr()` must point to a buffer with at least `n` diagonal elements.
    // - `mat.stride(1)` is used as the step between diagonal elements, assuming storage
    //   along the first row for compatibility with LAPACK convention.
    unsafe { faer::diag::DiagMut::from_raw_parts_mut(mat.as_mut_ptr() as *mut _, n, mat.stride(1)) }
}
