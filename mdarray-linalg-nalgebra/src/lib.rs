//! ```rust
//! use mdarray::{DArray, tensor};
//! use mdarray_linalg::prelude::*; // Imports traits anonymously
//!
//! use mdarray_linalg::svd::SVDDecomp;
//! use mdarray_linalg_nalgebra::svd;
//! use mdarray_linalg::{Naive, matmul, diag};
//!
//! use mdarray_linalg_nalgebra::Nalgebra;
//!
//! // Declare two matrices
//! let a = tensor![[1., 2.], [3., 4.]];
//! let b = tensor![[5., 6.], [7., 8.]];
//!
//!
//! // ----- Singular Value Decomposition (SVD) -----
//! let bd = Nalgebra::default();
//! let SVDDecomp { s, u, vt } = bd.svd(&mut a.clone()).expect("SVD failed");
//! println!("Singular values: {:?}", s);
//! println!("Left singular vectors U: {:?}", u);
//! println!("Right singular vectors V^T: {:?}", vt);
//! let (s,u,vt) = svd!(&mut a.clone()); // Convenience macro that directly unpacks the SVD.
//! let b = matmul!(&u, &diag(&s), &vt);
//! assert!(((a[[0,1]] - b[[0,1]]) as f64).abs() < 10e-10_f64);
//! ```

use mdarray::{Dim, Layout, Shape, Slice};
use num_complex::{Complex, ComplexFloat};

pub mod eig;
pub mod lu;
pub mod matmul;
pub mod matvec;
pub mod qr;
pub mod svd;

#[derive(Default, Debug, Clone, Copy)]
pub enum QRConfig {
    #[default]
    Reduced, // Q: M×K, R: K×N
    Complete, // Q: M×M, R: M×N
}

pub struct Nalgebra {
    qr_config: QRConfig,
}

impl Default for Nalgebra {
    fn default() -> Self {
        Self {
            qr_config: QRConfig::Reduced,
        }
    }
}

impl Nalgebra {
    pub fn config_qr(mut self, config: QRConfig) -> Self {
        self.qr_config = config;
        self
    }
}

/// Copy an mdarray matrix into a dense nalgebra matrix.
pub(crate) fn to_dmatrix<T, D0, D1, L>(a: &Slice<T, (D0, D1), L>) -> nalgebra::DMatrix<T>
where
    T: nalgebra::Scalar + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let rows = a.shape().dim(0);
    let cols = a.shape().dim(1);
    let mut data = Vec::with_capacity(rows * cols);

    // nalgebra stores dense matrices in column-major order.
    for j in 0..cols {
        for i in 0..rows {
            data.push(a[[i, j]]);
        }
    }

    nalgebra::DMatrix::from_vec(rows, cols, data)
}

/// Copy a dense nalgebra matrix back into an mdarray slice.
pub(crate) fn write_dmatrix<T, D0, D1, L>(src: &nalgebra::DMatrix<T>, dst: &mut Slice<T, (D0, D1), L>)
where
    T: nalgebra::Scalar + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    assert_eq!(src.nrows(), dst.shape().dim(0));
    assert_eq!(src.ncols(), dst.shape().dim(1));

    for i in 0..src.nrows() {
        for j in 0..src.ncols() {
            dst[[i, j]] = src[(i, j)];
        }
    }
}

/// Copy an mdarray matrix into a dense complex nalgebra matrix.
pub(crate) fn to_complex_dmatrix<T, D0, D1, L>(
    a: &Slice<T, (D0, D1), L>,
) -> nalgebra::DMatrix<Complex<T::Real>>
where
    T: ComplexFloat,
    T::Real: nalgebra::RealField + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    let rows = a.shape().dim(0);
    let cols = a.shape().dim(1);
    let mut data = Vec::with_capacity(rows * cols);

    // nalgebra stores dense matrices in column-major order.
    for j in 0..cols {
        for i in 0..rows {
            data.push(Complex::new(a[[i, j]].re(), a[[i, j]].im()));
        }
    }

    nalgebra::DMatrix::from_vec(rows, cols, data)
}

/// Copy an mdarray slice into a dense nalgebra vector using logical iteration order.
pub(crate) fn to_dvector<T, S, L>(x: &Slice<T, S, L>) -> nalgebra::DVector<T>
where
    T: nalgebra::Scalar + Copy,
    S: Shape,
    L: Layout,
{
    nalgebra::DVector::from_iterator(x.len(), x.iter().copied())
}

/// Copy a dense nalgebra vector back into an mdarray vector slice.
pub(crate) fn write_dvector<T, D1, L>(src: &nalgebra::DVector<T>, dst: &mut Slice<T, (D1,), L>)
where
    T: nalgebra::Scalar + Copy,
    D1: Dim,
    L: Layout,
{
    assert_eq!(src.len(), dst.len());

    for (dsti, srci) in dst.iter_mut().zip(src.iter()) {
        *dsti = *srci;
    }
}

/// Copy a dense complex nalgebra matrix back into an mdarray slice.
pub(crate) fn write_complex_dmatrix<R, D0, D1, L>(
    src: &nalgebra::DMatrix<Complex<R>>,
    dst: &mut Slice<Complex<R>, (D0, D1), L>,
) where
    R: nalgebra::RealField + Copy,
    D0: Dim,
    D1: Dim,
    L: Layout,
{
    assert_eq!(src.nrows(), dst.shape().dim(0));
    assert_eq!(src.ncols(), dst.shape().dim(1));

    for i in 0..src.nrows() {
        for j in 0..src.ncols() {
            dst[[i, j]] = src[(i, j)];
        }
    }
}

/// Copy a dense complex nalgebra vector back into an mdarray slice.
pub(crate) fn write_complex_dvector<R, D1, L>(
    src: &nalgebra::DVector<Complex<R>>,
    dst: &mut Slice<Complex<R>, (D1,), L>,
) where
    R: nalgebra::RealField + Copy,
    D1: Dim,
    L: Layout,
{
    assert_eq!(src.len(), dst.len());

    for (dsti, srci) in dst.iter_mut().zip(src.iter()) {
        *dsti = *srci;
    }
}

/// Chains an arbitrary number of matrix multiplications using the nalgebra backend.
#[macro_export]
macro_rules! matmul {
    ($a:expr, $b:expr) => {
        $crate::Nalgebra::default().matmul($a, $b).eval()
    };

    ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => {
        $crate::Nalgebra::default()
            .matmul(
                $a,
                &$crate::matmul!($b, $($rest),+)
            )
            .eval()
    };
}

/// Convenience macro for SVD decomposition that unwraps the result
/// directly. Panics if the decomposition fails.
#[macro_export]
macro_rules! svd {
    ($a:expr) => {{
        let svdr = $crate::Nalgebra::default().svd($a).expect("SVD failed");
        (svdr.s, svdr.u, svdr.vt)
    }};
    ($a:expr, full) => {{
        let svdr = $crate::Nalgebra::default().svd($a).expect("SVD failed");
        (svdr.s, svdr.u, svdr.vt)
    }};
}

/// Convenience macro for eigenvalue decomposition.
#[macro_export]
macro_rules! eig {
    ($a:expr) => {{
        let eig = $crate::Nalgebra::default()
            .eig($a)
            .expect("Eigenvalue decomposition failed");

        let vectors = eig
            .right_eigenvectors
            .expect("Eigenvectors were not computed");

        (eig.eigenvalues, vectors)
    }};
}
