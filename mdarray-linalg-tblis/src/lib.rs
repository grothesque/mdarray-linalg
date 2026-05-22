//! Backend using [TBLIS](https://crates.io/crates/tblis) for matrix multiplication and tensor contraction.
//!
//! ```no_run
//! use mdarray::tensor;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg_tblis::Tblis;
//!
//! let a = tensor![[1., 2.], [3., 4.]];
//! let b = tensor![[5., 6.], [7., 8.]];
//!
//! let c = Tblis.matmul(&a, &b).eval();
//! assert_eq!(c, tensor![[19., 22.], [43., 50.]]);
//! ```
//!
//! TBLIS only supports `f32`, `f64`, `Complex<f32>` and `Complex<f64>`.

pub mod matmul;

#[derive(Default)]
pub struct Tblis;

/// Chains an arbitrary number of matrix multiplications using the TBLIS backend.
#[macro_export]
macro_rules! matmul {
    ($a:expr, $b:expr) => {
        Tblis.matmul($a, $b).eval()
    };

    ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => {
        Tblis
            .matmul(
                $a,
                &matmul!($b, $($rest),+)
            )
            .eval()
    };
}
