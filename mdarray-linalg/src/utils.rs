//! Utility functions for matrix printing, shape retrieval, identity
//! generation, Kronecker product, trace, transpose operations, ...
//!
//! These functions were necessary for implementing this crate.  They are
//! exposed because they can be generally useful, but this is not meant to be
//! a complete collection of linear algebra utilities at this time.

use mdarray::{DSlice, DTensor, Dim, Layout, Shape, Slice, tensor};
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
pub fn dims3(a_shape: impl Shape, b_shape: impl Shape, c_shape: impl Shape) -> (i32, i32, i32) {
    let (m, k) = (a_shape.dim(0), a_shape.dim(1));
    let (k2, n) = (b_shape.dim(0), b_shape.dim(1));
    let (m2, n2) = (c_shape.dim(0), c_shape.dim(1));

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
pub fn dims2(a_shape: impl Shape, b_shape: impl Shape) -> (i32, i32) {
    let (m, k) = (a_shape.dim(0), a_shape.dim(1));
    let (k2, n) = (b_shape.dim(0), b_shape.dim(1));

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
pub fn ipiv_to_perm_mat<T: ComplexFloat>(ipiv: &[i32], m: usize) -> DTensor<T, 2> {
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
/// use mdarray_linalg::identity;
///
/// let i3 = identity::<f64>(3);
/// assert_eq!(i3, tensor![[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);
/// ```
pub fn identity<T: Zero + One>(n: usize) -> DTensor<T, 2> {
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
/// use mdarray_linalg::identity_k;
///
/// let i3 = identity_k::<f64>(3, 1);
/// assert_eq!(i3, tensor![[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]]);
/// ```
pub fn identity_k<T: Zero + One>(n: usize, k: isize) -> DTensor<T, 2> {
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

/// Converts a flat index to multidimensional coordinates.
///
/// # Examples
///
/// ```
/// use mdarray::DTensor;
/// use mdarray_linalg::unravel_index;
///
/// let x = DTensor::<usize, 2>::from_fn([2,3], |i| i[0] + i[1]);
///
/// assert_eq!(unravel_index(&x, 0), vec![0, 0]);
/// assert_eq!(unravel_index(&x, 4), vec![1, 1]);
/// assert_eq!(unravel_index(&x, 5), vec![1, 2]);
/// ```
///
/// # Panics
///
/// Panics if `flat` is out of bounds (>= `x.len()`).
pub fn unravel_index<T, S: Shape, L: Layout>(x: &Slice<T, S, L>, mut flat: usize) -> Vec<usize> {
    let rank = x.rank();

    assert!(
        flat < x.len(),
        "flat index out of bounds: {} >= {}",
        flat,
        x.len()
    );

    let mut coords = vec![0usize; rank];

    for i in (0..rank).rev() {
        let dim = x.shape().dim(i);
        coords[i] = flat % dim;
        flat /= dim;
    }

    coords
}
