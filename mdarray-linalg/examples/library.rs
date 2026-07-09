// This example demonstrates how a library can use mdarray-linalg through backend-agnostic traits
// while leaving the backend choice to its users.
//
// See tool.rs for the same computation written in a more compact application style.

use mdarray::array;
use num_complex::Complex64;

mod library {
    use mdarray::{Array, Dim, Layout, Slice};
    use num_traits::{MulAdd, One, Zero};

    use mdarray_linalg::{Contract, prelude::*, utils::identity};

    /// Computes `a` to the power `exponent` by exponentiation by squaring.
    pub fn matrix_power<T, B, L, D>(
        backend: &B,
        // `(D, D)` encodes that this function operates on square matrices.
        a: &Slice<T, (D, D), L>,
        mut exponent: u64,
    ) -> Array<T, (D, D)>
    where
        T: Clone + Zero + One + MulAdd<Output = T>,
        B: Contract<T>,
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
    let qn_naive = library::matrix_power(&naive, &q, n);

    let faer = mdarray_linalg_faer::Faer::default();
    let qn_faer = library::matrix_power(&faer, &q, n);

    let expected = array![[17711.0, 10946.0], [10946.0, 6765.0]];

    println!("Q^{n} with Naive backend:\n{qn_naive:?}");
    println!("Q^{n} with Faer backend:\n{qn_faer:?}");
    println!("expected Fibonacci matrix:\n{expected:?}");

    assert_eq!(qn_naive, expected);
    assert_eq!(qn_faer, expected);

    let complex_q = array![
        [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    let complex_qn_naive = library::matrix_power(&naive, &complex_q, n);

    println!("complex Q^{n} with Naive backend:\n{complex_qn_naive:?}");
}
