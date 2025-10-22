use mdarray::tensor;
use mdarray_linalg::matmul::{MatMul, TensordotBuilder};

// --- Basic functionality ---

pub fn tensordot_all_axes_impl(backend: &impl MatMul<f64>) {
    // np.tensordot(a, b, axes=2) -> [[70.0]]
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[70.0]].into_dyn();
    let result = backend.contract_all(&a, &b).eval();
    assert_eq!(result, expected);
}

pub fn tensordot_contract_k_2_should_match_all_axes_impl(backend: &impl MatMul<f64>) {
    // contract_k(2) is equivalent to All for 2D tensors
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[70.0]].into_dyn();
    let result = backend.contract_n(&a, &b, 2).eval();
    assert_eq!(result, expected);
}

pub fn tensordot_specific_axes_matrix_multiplication_impl(backend: &impl MatMul<f64>) {
    // tensordot(a, b, axes=([1], [0])) -> matrix product
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[19., 22.], [43., 50.]].into_dyn();
    let result = backend.contract(&a, &b, vec![1], vec![0]).eval();
    assert_eq!(result, expected);
}

pub fn tensordot_specific_empty_axes_should_outer_product_impl(backend: &impl MatMul<f64>) {
    // tensordot(a, b, axes=0) -> outer product
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

// --- Edge cases ---

pub fn tensordot_scalar_inputs_should_multiply_impl(backend: &impl MatMul<f64>) {
    let a = tensor![3.].into_dyn();
    let b = tensor![5.].into_dyn();
    let expected = tensor![[15.0]].into_dyn();
    let result = backend.contract_all(&a, &b).eval();
    assert_eq!(result, expected);
}

pub fn tensordot_increase_deep_impl(backend: &impl MatMul<f64>) {
    let r = tensor![[[1.]]].into_dyn();
    let mps = tensor![[[1.], [0.]]].into_dyn();
    let expected = tensor![[[[1.0], [0.]]]].into_dyn();
    let result = backend.contract(&r, &mps, vec![1], vec![0]).eval();
    assert_eq!(result, expected);
}

pub fn tensordot_vector_dot_product_impl(backend: &impl MatMul<f64>) {
    // tensordot(a, b, axes=1) -> scalar inner product
    let a = tensor![1., 2., 3.].into_dyn();
    let b = tensor![4., 5., 6.].into_dyn();
    let expected = tensor![[32.0]].into_dyn(); // 1*4 + 2*5 + 3*6
    let result = backend.contract_all(&a, &b).eval();
    assert_eq!(result, expected);
}

pub fn tensordot_mismatched_dimensions_should_panic_impl(
    backend: &(impl MatMul<f64> + std::panic::RefUnwindSafe),
) {
    // Should panic when dimensions are not aligned
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[1., 2., 3.]].into_dyn(); // shape mismatch
    let result = std::panic::catch_unwind(|| backend.contract_all(&a, &b).eval());
    assert!(result.is_err());
}

// --- Structural and mathematical properties ---

pub fn tensordot_outer_should_match_manual_kronecker_impl(backend: &impl MatMul<f64>) {
    // The outer product should be equal to np.kron(a,b)
    let a = tensor![1., 2.].into_dyn();
    let b = tensor![3., 4.].into_dyn();
    let expected = tensor![[3., 4.], [6., 8.]].into_dyn();
    let result = backend.contract_n(&a, &b, 0).eval();
    assert_eq!(result, expected);
}

// --- Test overwrite functionality ---

// fn tensordot_overwrite_impl(backend: &impl MatMul<f64>) {
//     let a = tensor![[1., 2.], [3., 4.]].into_dyn();
//     let b = tensor![[5., 6.], [7., 8.]].into_dyn();
//     let expected = tensor![[19., 22.], [43., 50.]].into_dyn();

//     let mut c = tensor![[0., 0.], [0., 0.]].into_dyn();
//     backend
//         .contract(&a, &b, vec![1], vec![0])
//         .overwrite(&mut c);

//     assert_eq!(c, expected);
// }

// #[test]
// fn tensordot_overwrite() {
//     tensordot_overwrite_impl(&Naive);
//     tensordot_overwrite_impl(&Blas);
// }

// fn tensordot_overwrite_all_axes_impl(backend: &impl MatMul<f64>) {
//     let a = tensor![[1., 2.], [3., 4.]].into_dyn();
//     let b = tensor![[5., 6.], [7., 8.]].into_dyn();
//     let expected = tensor![[70.0]].into_dyn();

//     let mut c = tensor![[0.0]].into_dyn();
//     backend.contract_all(&a, &b).overwrite(&mut c);

//     assert_eq!(c, expected);
// }

// #[test]
// fn tensordot_overwrite_all_axes() {
//     tensordot_overwrite_all_axes_impl(&Naive);
//     tensordot_overwrite_all_axes_impl(&Blas);
// }
