//! Tensor contraction (generalized dot product) for arbitrary rank tensors.
//!```rust
//!use mdarray::tensor;
//!use mdarray_linalg_blas::Blas;
//!use mdarray_linalg_naive::Naive;
//!use crate::mdarray_linalg::tensordot::{Tensordot, TensordotBuilder};
//!
//!let a = tensor![[1., 2.], [3., 4.]].into_dyn(); // requires dynamic tensor
//!let b = tensor![[5., 6.], [7., 8.]].into_dyn();
//!
//!let expected_all = tensor![[70.0]].into_dyn();
//!let result_all = Naive.tensordot(&a, &b).eval();
//!let result_contract_k = Blas.tensordot(&a, &b).contract_k(2).eval();
//!assert_eq!(result_contract_k, expected_all);
//!
//!let expected_matmul = tensor![[19., 22.], [43., 50.]].into_dyn();
//!let result_specific = Blas
//!    .tensordot(&a, &b)
//!    .specific(&[1], &[0])
//!    .eval();
//!assert_eq!(result_specific, expected_matmul);
//!```
use mdarray::{Slice, Tensor};

/// Tensor contraction operations
pub trait Tensordot<T> {
    fn tensordot<'a>(&self, a: &'a Slice<T>, b: &'a Slice<T>) -> impl TensordotBuilder<'a, T>;
}

/// Builder interface for configuring tensor contraction operations
pub trait TensordotBuilder<'a, T>
where
    T: 'a,
{
    /// Contracts the last `k` axes of the first tensor with the first `k` axes of the second tensor.
    /// # Example
    /// For two matrices (2D tensors), `contract_k(1)` performs standard matrix multiplication.
    fn contract_k(self, k: isize) -> Self;

    /// Specifies exactly which axes to contract.
    /// # Example
    /// `specific([1, 2], [3, 4])` contracts axis 1 and 2 of `a`
    /// with axes 3 and 4 of `b`.
    fn specific(self, axes_a: &[isize], axes_b: &[isize]) -> Self;

    /// Returns a new owned tensor containing the result.
    fn eval(self) -> Tensor<T>;

    /// Overwrites the provided tensor with the result.
    fn overwrite(self, c: &mut Slice<T>);
}
