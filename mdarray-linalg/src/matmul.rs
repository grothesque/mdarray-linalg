use mdarray::{DSlice, DTensor, Layout};

pub enum Side {
    Left,
    Right,
}

pub enum Type {
    Sym,
    Her,
    Tri,
}

pub enum Triangle {
    Upper,
    Lower,
}

pub trait MatMul<T> {
    fn matmul<'a, La, Lb>(
        &self,
        a: &'a DSlice<T, 2, La>,
        b: &'a DSlice<T, 2, Lb>,
    ) -> impl MatMulBuilder<'a, T, La, Lb>
    where
        La: Layout,
        Lb: Layout;
}

pub trait MatMulBuilder<'a, T, La, Lb>
where
    La: Layout,
    Lb: Layout,
    T: 'a,
    La: 'a,
    Lb: 'a,
{
    /// Enable parallelization.
    fn parallelize(self) -> Self;

    /// Multiplies the result by a scalar factor.
    fn scale(self, factor: T) -> Self;

    /// Returns a new owned tensor containing the result.
    fn eval(self) -> DTensor<T, 2>;

    /// Overwrites the provided slice with the result.
    fn overwrite<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>);

    /// Adds the result to the provided slice.
    fn add_to<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>);

    /// Adds the result to the provided slice after scaling the slice by `beta`
    /// (i.e. C := beta * C + result).
    fn add_to_scaled<Lc: Layout>(self, c: &mut DSlice<T, 2, Lc>, beta: T);

    /// Computes matrix products where one operand is a special matrix (symmetric,
    /// Hermitian, or triangular) and the other is a general matrix. The special matrix can be
    /// positioned on either the left or right side of the operation.
    ///
    /// # Parameters
    ///
    /// * `lr` - Specifies which operand is the special matrix:
    ///   - `Side::Left`: Special matrix × General matrix
    ///   - `Side::Right`: General matrix × Special matrix
    /// * `type_of_matrix` - Type of the special matrix:
    ///   - `Type::Sym`: Symmetric matrix
    ///   - `Type::Her`: Hermitian matrix  
    ///   - `Type::Tri`: Triangular matrix
    /// * `tr` - Which triangle of the special matrix contains the stored data:
    ///   - `Triangle::Upper`: Data is stored in the upper triangle
    ///   - `Triangle::Lower`: Data is stored in the lower triangle
    ///
    /// For symmetric and Hermitian matrices, only one triangle needs to be stored since
    /// the other can be derived. For triangular matrices, this specifies whether it's
    /// upper or lower triangular.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the specialized matrix multiplication.
    fn special(self, lr: Side, type_of_matrix: Type, tr: Triangle) -> DTensor<T, 2>;
}
