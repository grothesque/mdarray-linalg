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

    /// Computes a matrix product where the first operand is a special
    /// matrix (symmetric, Hermitian, or triangular) and the other is
    /// general.
    ///
    /// The special matrix is always treated as `A`. `lr` determines the multiplication order:
    /// - `Side::Left`  : C := alpha * A * B
    /// - `Side::Right` : C := alpha * B * A
    ///
    /// # Parameters
    /// * `lr` - side of multiplication (left or right)
    /// * `type_of_matrix` - special matrix type: `Sym`, `Her`, or `Tri`
    /// * `tr` - triangle containing stored data: `Upper` or `Lower`
    ///
    /// Only the specified triangle needs to be stored for symmetric/Hermitian matrices;
    /// for triangular matrices it specifies which half is used.
    ///
    /// # Returns
    /// A new tensor with the result.
    fn special(self, lr: Side, type_of_matrix: Type, tr: Triangle) -> DTensor<T, 2>;
}
