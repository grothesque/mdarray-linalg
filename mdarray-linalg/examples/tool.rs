// This example demonstrates how tool code can use mdarray-linalg through backend-agnostic traits.
//
// This is the same computation as library.rs, but written as tool/application code.

use mdarray::{array, expr::Expression as _, Array, Dim, Layout, Slice};
use num_complex::{Complex64, ComplexFloat};
use num_traits::MulAdd;

use mdarray_linalg::{Contract, prelude::*, utils::identity};

// Pseudo trait-alias (Stable Rust does not yet have stable trait aliases.)
//
// This simplifies the trait bounds of functions, but this is a tradeoff:
//
// - Less granularity in trait requirements (if all functions demand Scalar).  This could be
//   mitigated by defining a hierarchy of such traits.
//
// - Error messages may say “T: Scalar is not satisfied” instead of pointing directly at Zero,
//   One, etc.
//
// - Adding a new supertrait to Scalar is a breaking change, because all existing callers now
//   need to satisfy the stronger bound.
trait Scalar: Clone + ComplexFloat + MulAdd<Output = Self> {}
impl<T> Scalar for T where T: Clone + ComplexFloat + MulAdd<Output = T> {}

// Same for the backend: in a realistic example this would combine multiple backend traits.
trait Backend<T: Scalar>: Contract<T> {}
impl<T, B> Backend<T> for B
where
    T: Scalar,
    B: Contract<T>,
{
}

/// Computes `a` to the power `exponent` by exponentiation by squaring.
fn matrix_power<T, B, L, D>(
    backend: &B,
    // `(D, D)` encodes that this function operates on square matrices.
    a: &Slice<T, (D, D), L>,
    mut exponent: u64,
) -> Array<T, (D, D)>
where
    T: Scalar,
    B: Backend<T>,
    L: Layout,
    D: Dim,
{
    let (rows, cols) = *a.shape();
    assert_eq!(rows.size(), cols.size(), "matrix must be square");

    let mut result = identity::<T, D, D>(rows.size());
    let mut base = a.to_array();

    while exponent > 0 {
        if exponent & 1 == 1 {
            result = backend.matmul(&result, &base).eval();
        }

        exponent >>= 1;
        if exponent > 0 {
            base = backend.matmul(&base, &base).eval();
        }
    }

    result
}

fn main() {
    // The Fibonacci Q-matrix satisfies:
    //
    //     [[1, 1], [1, 0]]^n = [[F(n+1), F(n)], [F(n), F(n-1)]].
    //
    // We use floating-point entries because most linear algebra backends are
    // intended for real or complex scalars rather than integers.
    let q = array![[1.0, 1.0], [1.0, 0.0]];
    let n = 21;

    let naive = mdarray_linalg::Naive;
    let qn_naive = matrix_power(&naive, &q, n);

    let faer = mdarray_linalg_faer::Faer::default();
    let qn_faer = matrix_power(&faer, &q, n);

    let expected = array![[17711.0, 10946.0], [10946.0, 6765.0]];

    println!("Q^{n} with Naive backend:\n{qn_naive:?}");
    println!("Q^{n} with Faer backend:\n{qn_faer:?}");
    println!("expected Fibonacci matrix:\n{expected:?}");

    assert_eq!(qn_naive, expected);
    assert_eq!(qn_faer, expected);

    // Convert the real matrix to a complex matrix using mdarray's expression API.
    let complex_q = q.expr().copied().map(Complex64::from).eval();
    let complex_qn_naive = matrix_power(&naive, &complex_q, n);

    println!("complex Q^{n} with Naive backend:\n{complex_qn_naive:?}");
}
