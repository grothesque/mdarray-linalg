/// Simple function-based interface to linear algebra libraries

use mdarray::DSlice;
use cblas;

#[inline]
pub fn gemm(alpha: f64, a: &DSlice<f64, 2>, b: &DSlice<f64, 2>, beta: f64, c: &mut DSlice<f64, 2>) {
    let (m, k) = *a.shape();
    let (k2, n) = *b.shape();
    let (m2, n2) = *c.shape();
    assert!(m == m2, "a and c must agree in number of rows");
    assert!(n == n2, "b and c must agree in number of columns");
    assert!(k == k2, "a's number of columns must be equal to b's number of rows");
    let m: i32 = m.try_into().expect("dimensions must fit into i32");
    let n: i32 = n.try_into().expect("dimensions must fit into i32");
    let k: i32 = k.try_into().expect("dimensions must fit into i32");

    // SAFETY: is assured by the complete checks just above.
    unsafe {
        cblas::dgemm(
            cblas::Layout::RowMajor, cblas::Transpose::None, cblas::Transpose::None,
            m, n, k,
            alpha, &a[..], k,
            &b[..], n, beta,
            &mut c[..], n,
        );
    }
}
