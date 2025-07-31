use mdarray::{Tensor, expr::Expression as _};
use mdarray_linalg_blas::gemm;

use openblas_src as _;

use crate::common::example_matrix;

use mdarray_linalg::naive_matmul;

#[test]
fn test_gemm() {
    let a = example_matrix([2, 3]).eval();
    let b = example_matrix([3, 4]).eval();
    let c_expr = || example_matrix([2, 4]);
    let mut c = c_expr().eval();
    let ab_plus_c = {
        let mut ab = Tensor::from_elem([a.dim(0), b.dim(1)], 0.0);
        naive_matmul(&a, &b, &mut ab);
        ab + &c
    };

    // Test vanilla gemm with all matrices in column major order and Dense mapping.
    gemm(1.0, &a, &b, 1.0, &mut c);
    assert!(c == ab_plus_c);

    ////////////////
    // Test all combinations of row- and column major for the three matrices A, B, and C.  The
    // layout is always ‘Strided’ here, so we never test calling gemm with mixed layout, but we
    // know anyway statically that this must work.

    let a_cmajor = a.transpose().to_tensor();
    let a_cmajor = a_cmajor.transpose();
    let b_cmajor = b.transpose().to_tensor();
    let b_cmajor = b_cmajor.transpose();

    // Convert to a ‘Strided’ layout (still row major) so that ‘a’ has the same type as ‘a_cmajor’.
    let a = a.remap();
    let b = b.remap();
    let mut c = c.remap_mut();

    let mut c_cmajor = c.transpose().to_tensor();
    let mut c_cmajor = c_cmajor.transpose_mut();

    for a in [&a, &a_cmajor] {
        for b in [&b, &b_cmajor] {
            for c in [&mut c, &mut c_cmajor] {
                c.assign(c_expr());
                gemm(1.0, a, &b, 1.0, c);
                assert!(*c == ab_plus_c);
            }
        }
    }
}
