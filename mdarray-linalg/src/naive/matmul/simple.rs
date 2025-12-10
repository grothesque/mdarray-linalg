use num_traits::MulAdd;

use mdarray::{Dim, Layout, Slice};
use num_complex::ComplexFloat;
use num_traits::MulAdd;

/// Textbook implementation of matrix multiplication, useful for
/// debugging and simple tests without relying on a external backend
pub fn naive_matmul<
    T: ComplexFloat + MulAdd<Output = T>,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    c: &mut Slice<T, (D0, D2), Lc>,
) {
    let ash = a.shape();
    let bsh = b.shape();
    let csh = c.shape();

    let d0 = ash.0.size();
    assert_eq!(d0, csh.0.size());

    let d1 = ash.1.size();
    assert_eq!(d1, bsh.0.size());

    let d2 = bsh.1.size();
    assert_eq!(d2, csh.1.size());

    for i in 0..d0 {
        for j in 0..d2 {
            c[[i, j]] = beta * c[[i, j]];
        }
    }

    for i in 0..d0 {
        for j in 0..d2 {
            for k in 0..d1 {
                // c[[i, j]] = c[[i, j]] + alpha * a[[i, k]] * b[[k, j]];
                c[[i, j]] = (alpha * a[[i, k]]).mul_add(b[[k, j]], c[[i, j]]);
            }
        }
    }
}
