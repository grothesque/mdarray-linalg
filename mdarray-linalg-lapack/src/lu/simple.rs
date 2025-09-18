use super::scalar::LapackScalar;
use mdarray::{DSlice, DTensor, Layout};
use mdarray_linalg::get_dims;
use num_complex::ComplexFloat;

pub fn into_i32<T>(x: T) -> i32
where
    T: TryInto<i32>,
    <T as TryInto<i32>>::Error: std::fmt::Debug,
{
    x.try_into().expect("dimension must fit into i32")
}

pub fn getrf<La: Layout, Ll: Layout, Lu: Layout, T: ComplexFloat + Default + LapackScalar>(
    a: &mut DSlice<T, 2, La>,
    l: &mut DSlice<T, 2, Ll>,
    u: &mut DSlice<T, 2, Lu>,
) -> Vec<i32>
where
    T::Real: Into<T>,
{
    let ((m, n), (ml, nl), (mu, nu)) = get_dims!(a, l, u);
    let min_mn = m.min(n);

    assert_eq!(ml, m, "L must have m rows");
    assert_eq!(nl, min_mn, "L must have min(m,n) columns");
    assert_eq!(mu, min_mn, "U must have min(m,n) rows");
    assert_eq!(nu, n, "U must have n columns");

    let mut ipiv = vec![0i32; min_mn as usize];
    let mut info = 0;

    transpose(a); // LAPACK is column major

    unsafe {
        T::lapack_getrf(
            m,
            n,
            a.as_mut_ptr(),
            m, // lda
            ipiv.as_mut_ptr(),
            &mut info,
        );
    }

    transpose(a);

    for i in 0_usize..(m as usize) {
        for j in 0_usize..(min_mn as usize) {
            if i > j {
                l[[i, j]] = a[[i, j]];
            } else if i == j {
                l[[i, j]] = T::one();
            } else {
                l[[i, j]] = T::zero();
            }
        }
    }

    for i in 0_usize..(min_mn as usize) {
        for j in 0_usize..(n as usize) {
            if i <= j {
                u[[i, j]] = a[[i, j]];
            } else {
                u[[i, j]] = T::zero();
            }
        }
    }

    ipiv
}

pub fn transpose<T, L>(c: &mut DSlice<T, 2, L>)
where
    T: ComplexFloat,
    L: Layout,
{
    let (m, n) = *c.shape();

    assert_eq!(
        m, n,
        "Transpose in-place only implemented for square matrices."
    );

    for i in 0..m {
        for j in (i + 1)..n {
            let tmp = c[[i, j]];
            c[[i, j]] = c[[j, i]];
            c[[j, i]] = tmp;
        }
    }
}
