use mdarray::{DSlice, Dim, Layout};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};

/// Textbook implementation of QR decomposition using Gram-Schmidt process
/// Useful for debugging and simple tests without relying on external backend
pub fn naive_qr<T, L, Lq, Lr>(
    a: &mut DSlice<T, 2, L>,
    q: &mut DSlice<T, 2, Lq>,
    r: &mut DSlice<T, 2, Lr>,
) where
    T: ComplexFloat + Zero + One + MulAdd<Output = T>,
    L: Layout,
    Lq: Layout,
    Lr: Layout,
{
    let (m, n) = *a.shape();
    let m_size = m.size();
    let n_size = n.size();

    assert_eq!(q.shape().0.size(), m_size);
    assert_eq!(q.shape().1.size(), n_size);
    assert_eq!(r.shape().0.size(), n_size);
    assert_eq!(r.shape().1.size(), n_size);

    for i in 0..n_size {
        for j in 0..n_size {
            r[[i, j]] = T::zero();
        }
    }

    // Modified Gram-Schmidt process
    for j in 0..n_size {
        // Copy column j of A to column j of Q
        for i in 0..m_size {
            q[[i, j]] = a[[i, j]];
        }

        // Orthogonalize against previous columns
        for i in 0..j {
            // r[i,j] = q[:,i]^H * q[:,j]
            let mut dot = T::zero();
            for k in 0..m_size {
                dot = q[[k, i]].conj().mul_add(q[[k, j]], dot);
            }
            r[[i, j]] = dot;

            // q[:,j] = q[:,j] - r[i,j] * q[:,i]
            for k in 0..m_size {
                q[[k, j]] = q[[k, j]] - r[[i, j]] * q[[k, i]];
            }
        }

        // Compute norm of column j
        let mut norm_sq = T::zero();
        for k in 0..m_size {
            norm_sq = q[[k, j]].conj().mul_add(q[[k, j]], norm_sq);
        }
        let norm = norm_sq.sqrt();
        r[[j, j]] = norm;

        // Normalize column j of Q
        // if norm.abs() > T::from(1e-10).unwrap() {
        let inv_norm = T::one() / norm;
        for k in 0..m_size {
            q[[k, j]] = q[[k, j]] * inv_norm;
        }
        // }
    }
}
