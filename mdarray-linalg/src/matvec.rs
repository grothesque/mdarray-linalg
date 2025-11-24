//! Basic vector and matrix-vector operations, including `Ax`, `Ax + βy`, Givens rotations, argmax, and rank-1 updates
//!
//! # Matrix-Vector Operations
//!
//! ```rust
//! use mdarray::tensor;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg::Naive;
//!
//! // Create a 3x3 matrix and a vector
//! let a = tensor![[1., 2., 3.],
//!                 [4., 5., 6.],
//!                 [7., 8., 9.]];
//! let x = tensor![1., 1., 1.];
//!
//! // Basic matrix-vector multiplication: y = A·x
//! let y = Naive.matvec(&a, &x).eval();
//! assert_eq!(y, tensor![6., 15., 24.]);
//!
//! // Scaled operation: y = 2·A·x
//! let y_scaled = Naive.matvec(&a, &x).scale(2.).eval();
//! assert_eq!(y_scaled, tensor![12., 30., 48.]);
//!
//! // Write result to existing vector: y := α·A·x
//! let mut y_write = tensor![0., 0., 0.];
//! Naive.matvec(&a, &x).scale(2.).write(&mut y_write);
//! assert_eq!(y_write, tensor![12., 30., 48.]);
//!
//! // Add to vector: y := A·x + y
//! let mut y_add = tensor![1., 1., 1.];
//! Naive.matvec(&a, &x).add_to_vec(&mut y_add);
//! assert_eq!(y_add, tensor![7., 16., 25.]);
//!
//! // Scaled addition: y := α·A·x + β·y
//! let mut y_axpy = tensor![1., 1., 1.];
//! Naive.matvec(&a, &x).add_to_scaled_vec(&mut y_axpy, 2.);
//! assert_eq!(y_axpy, tensor![8., 17., 26.]);
//! ```
//!
//! # Outer Products and Rank-1 Updates
//!
//! ```rust
//! use mdarray::tensor;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg::Naive;
//! use mdarray_linalg::matmul::{Type, Triangle};
//!
//! // Create two vectors
//! let x = tensor![1., 2.];
//! let y = tensor![1., 10., 100.];
//!
//! // Basic outer product: A = x ⊗ y
//! let a = Naive.outer(&x, &y).eval();
//! assert_eq!(a, tensor![[1., 10., 100.],
//!                       [2., 20., 200.]]);
//!
//! // Scaled outer product: A = β·(x ⊗ y)
//! let a_scaled = Naive.outer(&x, &y).scale(2.).eval();
//! assert_eq!(a_scaled, tensor![[2., 20., 200.],
//!                              [4., 40., 400.]]);
//!
//! // Write to existing matrix: A := β·(x ⊗ y)
//! let mut a_write = tensor![[0., 0., 0.],
//!                           [0., 0., 0.]];
//! Naive.outer(&x, &y).scale(2.).write(&mut a_write);
//! assert_eq!(a_write, tensor![[2., 20., 200.],
//!                             [4., 40., 400.]]);
//!
//! // Rank-1 update: A := β·(x ⊗ y) + A
//! let mut a_update = tensor![[1., 1., 1.],
//!                            [1., 1., 1.]];
//! Naive.outer(&x, &y).scale(2.).add_to(&mut a_update);
//! assert_eq!(a_update, tensor![[3., 21., 201.],
//!                              [5., 41., 401.]]);
//!
//! // Symmetric rank-1 update: A := β·(x ⊗ xᵀ) + A (upper triangle)
//! let x_sym = tensor![1., 2., 3.];
//! let mut a_sym = tensor![[2., 1., 1.],
//!                         [1., 2., 1.],
//!                         [1., 1., 2.]];
//! Naive.outer(&x_sym, &x_sym)
//!      .scale(0.5)
//!      .add_to_special(&mut a_sym, Type::Sym, Triangle::Upper);
//! // Only upper triangle is updated
//! assert_eq!(a_sym, tensor![[2.5, 2., 2.5],
//!                           [1.,  4., 4. ],
//!                           [1.,  1., 6.5]]);
//! ```
//!
//! # Complex Number Support
//!
//! ```rust
//! use mdarray::tensor;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg::Naive;
//! use mdarray_linalg::matmul::{Type, Triangle};
//! use num_complex::Complex64;
//!
//! // Complex outer product
//! let x = tensor![Complex64::new(1., 1.), Complex64::new(2., 0.)];
//! let y = tensor![Complex64::new(1., 0.), Complex64::new(0., 1.)];
//! let a = Naive.outer(&x, &y).eval();
//! assert_eq!(a[[0, 0]], Complex64::new(1., 1.));
//! assert_eq!(a[[0, 1]], Complex64::new(-1., 1.));
//!
//! // Hermitian rank-1 update: A := β·(x ⊗ x†) + A
//! let x_her = tensor![Complex64::new(1., 0.5),
//!                     Complex64::new(2., 1.0),
//!                     Complex64::new(3., 1.5)];
//! let mut a_her = tensor![[Complex64::new(2., 0.), Complex64::new(1., 0.5), Complex64::new(1., 0.5)],
//!                         [Complex64::new(1., -0.5), Complex64::new(2., 0.), Complex64::new(1., 0.5)],
//!                         [Complex64::new(1., -0.5), Complex64::new(1., -0.5), Complex64::new(2., 0.)]];
//! Naive.outer(&x_her, &x_her)
//!      .scale(Complex64::new(0.3, 0.))
//!      .add_to_special(&mut a_her, Type::Her, Triangle::Upper);
//! // Upper triangle updated with conjugate transpose
//! ```
//! # Finding Maximum Elements
//!
//! ```rust
//! use mdarray::tensor;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg::Naive;
//!
//! // Find index of maximum value in 1D array
//! let x = tensor![1., 5., 3., 8., 2.];
//! let idx = Naive.argmax(&x).unwrap();
//! assert_eq!(idx, vec![3]);  // Maximum is at index 3
//!
//! // Find index in 2D array (returns multi-dimensional index)
//! let a = tensor![[0., 1., 2.],
//!                 [3., 4., 5.]];
//! let idx = Naive.argmax(&a.view(.., ..).into_dyn()).unwrap();
//! assert_eq!(idx, vec![1, 2]);  // Maximum is at position [1, 2]
//!
//! // Find element with largest absolute value
//! let y = tensor![1., -6., 3., -2., 5.];
//! let idx = Naive.argmax_abs(&y).unwrap();
//! assert_eq!(idx, vec![1]);  // -6 has largest absolute value
//!
//! // Write result to reusable buffer
//! let mut output = Vec::new();
//! let success = Naive.argmax_write(&x, &mut output);
//! assert!(success);
//! assert_eq!(output, vec![3]);
//! ```
//! # Vector Operations
//!
//! ```rust
//! use mdarray::tensor;
//! use mdarray_linalg::prelude::*;
//! use mdarray_linalg::Naive;
//! use num_complex::Complex64;
//!
//! // Scaled vector addition: y := α·x + y
//! let x = tensor![1., 2., 3.];
//! let mut y = tensor![1., 1., 1.];
//! Naive.add_to_scaled(2.0, &x, &mut y);
//! assert_eq!(y, tensor![3., 5., 7.]);  // y = 2·x + y
//!
//! // Dot product: ∑xᵢyᵢ
//! let x = tensor![1., 2., 3.];
//! let y = tensor![2., 4., 6.];
//! let result = Naive.dot(&x, &y);
//! assert_eq!(result, 28.0);  // 1*2 + 2*4 + 3*6 = 28
//!
//! // Conjugated dot product with complex numbers: ∑(conj(xᵢ)·yᵢ)
//! let x = tensor![Complex64::new(1., 2.), Complex64::new(2., 3.)];
//! let y = tensor![Complex64::new(3., 4.), Complex64::new(4., 5.)];
//! let result = Naive.dotc(&x, &y);
//! // conj(1+2i)*(3+4i) + conj(2+3i)*(4+5i)
//! let expected = x[[0]].conj() * y[[0]] + x[[1]].conj() * y[[1]];
//! assert_eq!(result, expected);
//!
//! // L2 norm (Euclidean): √(∑|xᵢ|²)
//! let x = tensor![3., 4.];
//! let norm = Naive.norm2(&x);
//! assert_eq!(norm, 5.0);  // √(9 + 16) = 5
//!
//! // L1 norm (Manhattan): ∑|xᵢ|
//! let x = tensor![Complex64::new(1., 2.), Complex64::new(2., 3.)];
//! let norm = Naive.norm1(&x);
//! // |1+2i| + |2+3i| = (|1|+|2|) + (|2|+|3|) = 8
//! assert_eq!(norm, 8.0);
//! ```
use mdarray::{DSlice, DTensor, Layout, Shape, Slice};
use num_complex::ComplexFloat;

use crate::matmul::{Triangle, Type};

/// Matrix-vector multiplication and transformations
pub trait MatVec<T> {
    fn matvec<'a, La, Lx>(
        &self,
        a: &'a DSlice<T, 2, La>,
        x: &'a DSlice<T, 1, Lx>,
    ) -> impl MatVecBuilder<'a, T, La, Lx>
    where
        La: Layout,
        Lx: Layout;
}

/// Builder interface for configuring matrix-vector operations
pub trait MatVecBuilder<'a, T, La, Lx>
where
    La: Layout,
    Lx: Layout,
    T: 'a,
    La: 'a,
    Lx: 'a,
{
    fn parallelize(self) -> Self;

    /// `α := α·α'`
    fn scale(self, alpha: T) -> Self;

    /// Returns `α·A·x`
    fn eval(self) -> DTensor<T, 1>;

    /// `y := α·A·x`
    fn write<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>);

    /// `y := α·A·x + y`
    fn add_to_vec<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>);

    /// `y := α·A·x + β·y`
    fn add_to_scaled_vec<Ly: Layout>(self, y: &mut DSlice<T, 1, Ly>, beta: T);
}

/// Vector operations and basic linear algebra utilities
pub trait VecOps<T: ComplexFloat> {
    /// Accumulate a scaled vector: `y := α·x + y`
    fn add_to_scaled<Lx: Layout, Ly: Layout>(
        &self,
        alpha: T,
        x: &DSlice<T, 1, Lx>,
        y: &mut DSlice<T, 1, Ly>,
    );

    /// Dot product: `∑xᵢyᵢ`
    fn dot<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T;

    /// Conjugated dot product: `∑(xᵢ * conj(yᵢ))`
    fn dotc<Lx: Layout, Ly: Layout>(&self, x: &DSlice<T, 1, Lx>, y: &DSlice<T, 1, Ly>) -> T;

    /// L2 norm: `√(∑|xᵢ|²)`
    fn norm2<Lx: Layout>(&self, x: &DSlice<T, 1, Lx>) -> T::Real;

    /// L1 norm: `∑|xᵢ|`
    fn norm1<Lx: Layout>(&self, x: &DSlice<T, 1, Lx>) -> T::Real
    where
        T: ComplexFloat;

    /// Givens rotation (**TODO**)
    fn rot<Lx: Layout, Ly: Layout>(
        &self,
        x: &mut DSlice<T, 1, Lx>,
        y: &mut DSlice<T, 1, Ly>,
        c: T::Real,
        s: T,
    ) where
        T: ComplexFloat;
}

/// Argmax for tensors, unlike other traits: it requires `T: PartialOrd` and works on tensor of any rank.
pub trait Argmax<T: ComplexFloat + std::cmp::PartialOrd> {
    fn argmax_write<Lx: Layout, S: Shape>(
        &self,
        x: &Slice<T, S, Lx>,
        output: &mut Vec<usize>,
    ) -> bool;

    fn argmax_abs_write<Lx: Layout, S: Shape>(
        &self,
        x: &Slice<T, S, Lx>,
        output: &mut Vec<usize>,
    ) -> bool;

    /// Index of max xᵢ (argmaxᵢ xᵢ)
    fn argmax<Lx: Layout, S: Shape>(&self, x: &Slice<T, S, Lx>) -> Option<Vec<usize>>;

    /// Index of max |xᵢ| (argmaxᵢ |xᵢ|)
    fn argmax_abs<Lx: Layout, S: Shape>(&self, x: &Slice<T, S, Lx>) -> Option<Vec<usize>>;
}

/// Outer product and rank-1 update
pub trait Outer<T> {
    fn outer<'a, Lx, Ly>(
        &self,
        x: &'a DSlice<T, 1, Lx>,
        y: &'a DSlice<T, 1, Ly>,
    ) -> impl OuterBuilder<'a, T, Lx, Ly>
    where
        Lx: Layout,
        Ly: Layout;
}

/// Builder interface for configuring outer product and rank-1 update
pub trait OuterBuilder<'a, T, Lx, Ly>
where
    Lx: Layout,
    Ly: Layout,
    T: 'a,
    Lx: 'a,
    Ly: 'a,
{
    /// `α := α·α'`
    fn scale(self, alpha: T) -> Self;

    /// Returns `α·xy`
    fn eval(self) -> DTensor<T, 2>;

    /// `a := α·xy`
    fn write<La: Layout>(self, a: &mut DSlice<T, 2, La>);

    /// Rank-1 update: `A := α·x·yᵀ + A`
    fn add_to<La: Layout>(self, a: &mut DSlice<T, 2, La>);

    /// Rank-1 update: ` A:= α·x·xᵀ (or x·x†) + A` on special matrix
    fn add_to_special(self, a: &mut DSlice<T, 2>, ty: Type, tr: Triangle);
}
