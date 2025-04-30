use mdarray::{tensor, view};
use mdarray_linalg_prototype::gemm;

extern crate openblas_src;

// By activating any combination of the three commented-out blocks, we obtain 8 different tests
// (with different types).  Not sure how to automatize this without writing 8 functions, since the
// types are differnt in each case.
#[test]
fn test_gemm() {
    let a = view![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ];
    // // Swap between row-major and column-major:
    // let a = a.transpose().to_tensor();
    // let a = a.transpose();

    let b = view![
        [1.0,  2.0,  3.0,  4.0],
        [5.0,  6.0,  7.0,  8.0],
        [9.0, 10.0, 11.0, 12.0],
    ];
    // // Swap between row-major and column-major:
    // let b = b.transpose().to_tensor();
    // let b = b.transpose();

    let mut c = tensor![
        [2.0, 6.0, 0.0, 4.0],
        [7.0, 2.0, 7.0, 2.0],
    ];
    // // Swap between row-major and column-major:
    // let mut c = c.transpose().to_tensor();
    // let mut c = c.transpose_mut();

    gemm(1.0, &a, &b, 1.0, &mut c);

    assert!(c == view![
        [40.0,  50.0,  50.0,  60.0],
        [90.0, 100.0, 120.0, 130.0],
    ]);
}
