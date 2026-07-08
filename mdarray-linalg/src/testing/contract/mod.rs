use mdarray::{array, expr, expr::Expression as _, tensor};
use num_complex::Complex64;

use super::common::*;
use crate::{contract::Contract, prelude::*};

// --- Fixtures ---

pub fn create_test_matrix_f64(
    shape: [usize; 2],
) -> expr::FromFn<(usize, usize), impl FnMut(&[usize]) -> f64> {
    expr::from_fn(shape, move |i| (shape[1] * i[0] + i[1] + 1) as f64)
}

pub fn create_test_matrix_complex(
    shape: [usize; 2],
) -> expr::FromFn<(usize, usize), impl FnMut(&[usize]) -> Complex64> {
    expr::from_fn(shape, move |i| {
        let val = (shape[1] * i[0] + i[1] + 1) as f64;
        Complex64::new(val, val * 0.5)
    })
}

// --- Matrix multiplication helpers ---

pub fn matmul_complex_with_scaling_impl(backend: &impl Contract<Complex64>) {
    let a = create_test_matrix_complex([2, 3]).eval();
    let b = create_test_matrix_complex([3, 2]).eval();
    let scale_factor = Complex64::new(2.0, 1.5);

    let result = backend.matmul(&a, &b).scale(scale_factor).eval();

    let expected = naive_matmul(&a, &b);
    let expected = (expr::fill(scale_factor) * &expected).eval();

    assert_eq!(result, expected);
}

pub fn matmul_builder_methods_impl(backend: &impl Contract<f64>) {
    let a = array![[1., 2., 3.], [4., 5., 6.]];
    let b = array![[7., 8.], [9., 10.], [11., 12.]];
    let product = array![[58., 64.], [139., 154.]];
    let initial = array![[1., 2.], [3., 4.]];

    let mut c = array![[0., 0.], [0., 0.]];
    backend.matmul(&a, &b).write(&mut c);
    assert_eq!(c, product);

    let mut c = initial.clone();
    backend.matmul(&a, &b).add_to(&mut c);
    assert_eq!(c, array![[59., 66.], [142., 158.]]);

    let mut c = initial;
    backend.matmul(&a, &b).add_to_scaled(&mut c, 2.0);
    assert_eq!(c, array![[60., 68.], [145., 162.]]);
}

// --- Builder helpers ---

pub fn contract_builder_methods_impl(backend: &impl Contract<f64>) {
    let a = array![[1., 2., 3.], [4., 5., 6.]].into_dyn();
    let b = array![[7., 8.], [9., 10.], [11., 12.]].into_dyn();
    let product = array![[58., 64.], [139., 154.]].into_dyn();
    let initial = array![[1., 2.], [3., 4.]].into_dyn();

    let mut c = array![[0., 0.], [0., 0.]].into_dyn();
    backend.contract_pairs(&a, &b, &[1], &[0]).write(&mut c);
    assert_eq!(c, product);

    let mut c = initial.clone();
    backend.contract_pairs(&a, &b, &[1], &[0]).add_to(&mut c);
    assert_eq!(c, array![[59., 66.], [142., 158.]].into_dyn());

    let mut c = initial;
    backend
        .contract_pairs(&a, &b, &[1], &[0])
        .add_to_scaled(&mut c, 2.0);
    assert_eq!(c, array![[60., 68.], [145., 162.]].into_dyn());
}

// --- Structured contraction helpers ---

pub fn contract_all_impl(backend: &impl Contract<f64>) {
    // contract_all(a, b) -> 70.0
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = 70.;
    let result = backend.contract_all(&a, &b);
    assert_eq!(result, expected);
}

pub fn contract_n_2_should_match_all_axes_impl(backend: &impl Contract<f64>) {
    // contract_n(2) is equivalent to full contraction for 2D tensors
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[70.0]].into_dyn();
    let result = backend.contract_n(&a, &b, 2).eval();
    assert_eq!(result, expected);
}

pub fn contract_pairs_matrix_multiplication_impl(backend: &impl Contract<f64>) {
    // contract_pairs(a, b, &[1], &[0]) -> matrix product
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[19., 22.], [43., 50.]].into_dyn();
    let result = backend.contract_pairs(&a, &b, &[1], &[0]).eval();
    assert_eq!(result, expected);
}

pub fn contract_n_0_should_outer_product_impl(backend: &impl Contract<f64>) {
    // contract_n(a, b, 0) -> outer product
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![
        [[[5.0, 6.0], [7.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]],
        [[[15.0, 18.0], [21.0, 24.0]], [[20.0, 24.0], [28.0, 32.0]]]
    ]
    .into_dyn();
    let result = backend.contract_n(&a, &b, 0).eval();
    assert_eq!(result, expected);
}

pub fn contract_scalar_inputs_should_multiply_impl(backend: &impl Contract<f64>) {
    let a = tensor![3.].into_dyn();
    let b = tensor![5.].into_dyn();
    let expected = 15.0;
    let result = backend.contract_all(&a, &b);
    assert_eq!(result, expected);
}

pub fn contract_increase_deep_impl(backend: &impl Contract<f64>) {
    let r = tensor![[[1.]]].into_dyn();
    let mps = tensor![[[1.], [0.]]].into_dyn();
    let expected = tensor![[[[1.0], [0.]]]].into_dyn();
    let result = backend.contract_pairs(&r, &mps, &[1], &[0]).eval();
    assert_eq!(result, expected);
}

pub fn contract_vector_dot_product_impl(backend: &impl Contract<f64>) {
    // contract_all(a, b) -> scalar inner product
    let a = tensor![1., 2., 3.].into_dyn();
    let b = tensor![4., 5., 6.].into_dyn();
    let expected = 32.0; // 1*4 + 2*5 + 3*6
    let result = backend.contract_all(&a, &b);
    assert_eq!(result, expected);
}

pub fn contract_mismatched_dimensions_should_panic_impl(
    backend: &(impl Contract<f64> + std::panic::RefUnwindSafe),
) {
    // Should panic when dimensions are not aligned
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[1., 2., 3.]].into_dyn(); // shape mismatch
    let result = std::panic::catch_unwind(|| backend.contract_all(&a, &b));
    assert!(result.is_err());
}

pub fn contract_outer_should_match_manual_kronecker_impl(backend: &impl Contract<f64>) {
    // The outer product should be equal to np.kron(a,b)
    let a = tensor![1., 2.].into_dyn();
    let b = tensor![3., 4.].into_dyn();
    let expected = tensor![[3., 4.], [6., 8.]].into_dyn();
    let result = backend.contract_n(&a, &b, 0).eval();
    assert_eq!(result, expected);
}

// --- Einsum-style contraction helpers ---

pub fn contract_einsum_matrix_multiplication_impl(backend: &impl Contract<f64>) {
    // ij,jk->ik
    let a = array![[1., 2.], [3., 4.]].into_dyn();
    let b = array![[5., 6.], [7., 8.]].into_dyn();
    let expected = array![[19., 22.], [43., 50.]].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1], &[1, 2], &[0, 2]).eval();
    assert_eq!(result, expected);
}

pub fn contract_einsum_full_contraction_impl(backend: &impl Contract<f64>) {
    // ij,ij->
    let a = array![[1., 2.], [3., 4.]].into_dyn();
    let b = array![[5., 6.], [7., 8.]].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1], &[0, 1], &[]).eval();
    assert_eq!(result.into_scalar(), 70.);
}

pub fn contract_einsum_output_permutation_impl(backend: &impl Contract<f64>) {
    // ij,jk->ki
    let a = array![[1., 2.], [3., 4.]].into_dyn();
    let b = array![[5., 6.], [7., 8.]].into_dyn();
    let expected = array![[19., 43.], [22., 50.]].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1], &[1, 2], &[2, 0]).eval();
    assert_eq!(result, expected);
}

pub fn contract_einsum_outer_product_impl(backend: &impl Contract<f64>) {
    // i,j->ij
    let a = array![1., 2.].into_dyn();
    let b = array![3., 4.].into_dyn();
    let expected = array![[3., 4.], [6., 8.]].into_dyn();
    let result = backend.contract(&a, &b, &[0], &[1], &[0, 1]).eval();
    assert_eq!(result, expected);
}

pub fn contract_einsum_trace_diagonal_impl(backend: &impl Contract<f64>) {
    // ijj,ij-> = 32
    let a = array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn();
    let b = array![[0., 1.], [2., 3.]].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1, 1], &[0, 1], &[]).eval();
    assert_eq!(result.into_scalar(), 32.);
}

pub fn contract_einsum_index_relabelling_impl(backend: &impl Contract<f64>) {
    // Same as above with permuted label assignments: result must be identical.
    let a = array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn();
    let b = array![[0., 1.], [2., 3.]].into_dyn();
    let result = backend.contract(&a, &b, &[1, 0, 0], &[1, 0], &[]).eval();
    assert_eq!(result.into_scalar(), 32.);
}

pub fn contract_einsum_partial_trace_then_contract_impl(backend: &impl Contract<f64>) {
    // ijj,ik->k  expected = [22, 36]
    let a = array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn();
    let b = array![[0., 1.], [2., 3.]].into_dyn();
    let expected = array![22., 36.].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1, 1], &[0, 2], &[2]).eval();
    assert_eq!(result, expected);
}

pub fn contract_einsum_cross_diagonal_impl(backend: &impl Contract<f64>) {
    // ijj,iij-> = 76
    let a = array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn();
    let b = array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn();
    let result = backend.contract(&a, &b, &[0, 1, 1], &[0, 0, 1], &[]).eval();
    assert_eq!(result.into_scalar(), 76.);
}

pub fn contract_einsum_vector_result_impl(backend: &impl Contract<f64>) {
    // ijjj,ijkl->l
    let a = mdarray::DArray::<f64, 4>::from_fn([2, 2, 2, 2], |i| {
        (i[0] * 8 + i[1] * 4 + i[2] * 2 + i[3]) as f64
    })
    .into_dyn();
    let b = mdarray::DArray::<f64, 4>::from_fn([2, 2, 2, 2], |i| {
        (i[0] * 8 + i[1] * 4 + i[2] * 2 + i[3]) as f64
    })
    .into_dyn();

    let mut expected = [0f64; 2];
    for (l, expected_l) in expected.iter_mut().enumerate() {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    *expected_l +=
                        (i * 8 + j * 4 + j * 2 + j) as f64 * (i * 8 + j * 4 + k * 2 + l) as f64;
                }
            }
        }
    }

    let result = backend
        .contract(&a, &b, &[0, 1, 1, 1], &[0, 1, 2, 3], &[3])
        .eval();
    assert_eq!(result, array![expected[0], expected[1]].into_dyn());
}
