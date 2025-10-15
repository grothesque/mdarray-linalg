//! Tensor contraction (generalized dot product) for arbitrary rank tensors.
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
    /// `specific(Box::new([1, 2]), Box::new([3, 4]))` contracts axis 1 and 2 of `a`
    /// with axes 3 and 4 of `b`.
    fn specific(self, axes_a: Box<[isize]>, axes_b: Box<[isize]>) -> Self;

    /// Returns a new owned tensor containing the result.
    fn eval(self) -> Tensor<T>;

    /// Overwrites the provided tensor with the result.
    fn overwrite(self, c: &mut Slice<T>);
}
