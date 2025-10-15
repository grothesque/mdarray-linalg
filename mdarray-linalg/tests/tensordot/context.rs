use mdarray::tensor;
use mdarray_linalg::tensordot::{Tensordot, TensordotBuilder};
use mdarray_linalg_naive::Naive;

// --- Basic functionality ---

#[test]
fn tensordot_all_axes() {
    // np.tensordot(a, b, axes=2) -> [[70.0]]
    let backend = Naive;
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[70.0]].into_dyn();
    let result = backend.tensordot(&a, &b).eval();
    assert_eq!(result, expected);
}

#[test]
fn tensordot_contract_k_2_should_match_all_axes() {
    // contract_k(2) is equivalent to All for 2D tensors
    let backend = Naive;
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[70.0]].into_dyn();
    let result = backend.tensordot(&a, &b).contract_k(2).eval();
    assert_eq!(result, expected);
}

#[test]
fn tensordot_specific_axes_matrix_multiplication() {
    // tensordot(a, b, axes=([1], [0])) -> matrix product
    let backend = Naive;
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[19., 22.], [43., 50.]].into_dyn();
    let result = backend
        .tensordot(&a, &b)
        .specific(Box::new([1]), Box::new([0]))
        .eval();
    assert_eq!(result, expected);
}

#[test]
fn tensordot_specific_empty_axes_should_outer_product() {
    // tensordot(a, b, axes=0) -> outer product
    let backend = Naive;
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![
        [[[5.0, 6.0], [7.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]],
        [[[15.0, 18.0], [21.0, 24.0]], [[20.0, 24.0], [28.0, 32.0]]]
    ]
    .into_dyn();
    let result = backend.tensordot(&a, &b).contract_k(0).eval();
    assert_eq!(result, expected);
}

// --- Edge cases ---

#[test]
fn tensordot_scalar_inputs_should_multiply() {
    let backend = Naive;
    let a = tensor![3.].into_dyn();
    let b = tensor![5.].into_dyn();
    let expected = tensor![[15.0]].into_dyn();
    let result = backend.tensordot(&a, &b).eval();
    assert_eq!(result, expected);
}

#[test]
fn tensordot_increase_deep() {
    let backend = Naive;
    let r = tensor![[[1.]]].into_dyn();
    let mps = tensor![[[1.], [0.]]].into_dyn();
    let expected = tensor![[[[1.0], [0.]]]].into_dyn();
    let result = backend
        .tensordot(&r, &mps)
        .specific(Box::new([1]), Box::new([0]))
        .eval();
    assert_eq!(result, expected);
}

#[test]
fn tensordot_vector_dot_product() {
    // tensordot(a, b, axes=1) -> scalar inner product
    let backend = Naive;
    let a = tensor![1., 2., 3.].into_dyn();
    let b = tensor![4., 5., 6.].into_dyn();
    let expected = tensor![[32.0]].into_dyn(); // 1*4 + 2*5 + 3*6
    let result = backend.tensordot(&a, &b).eval();
    assert_eq!(result, expected);
}

#[test]
fn tensordot_specific_negative_axes() {
    // tensordot(a, b, axes=([-1], [0])) -> matrix product
    let backend = Naive;
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[19., 22.], [43., 50.]].into_dyn();
    let result = backend
        .tensordot(&a, &b)
        .specific(Box::new([-1]), Box::new([0]))
        .eval();
    assert_eq!(result, expected);
}

#[test]
fn tensordot_mismatched_dimensions_should_panic() {
    // Should panic when dimensions are not aligned
    let backend = Naive;
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[1., 2., 3.]].into_dyn(); // shape mismatch
    let result = std::panic::catch_unwind(|| backend.tensordot(&a, &b).eval());
    assert!(result.is_err());
}

// --- Structural and mathematical properties ---

#[test]
fn tensordot_outer_should_match_manual_kronecker() {
    // The outer product should be equal to np.kron(a,b)
    let backend = Naive;
    let a = tensor![1., 2.].into_dyn();
    let b = tensor![3., 4.].into_dyn();
    let expected = tensor![[3., 4.], [6., 8.]].into_dyn();
    let result = backend.tensordot(&a, &b).contract_k(0).eval();
    assert_eq!(result, expected);
}

// --- Test overwrite functionality ---

#[test]
fn tensordot_overwrite() {
    let backend = Naive;
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[19., 22.], [43., 50.]].into_dyn();

    let mut c = tensor![[0., 0.], [0., 0.]].into_dyn();
    backend
        .tensordot(&a, &b)
        .specific(Box::new([1]), Box::new([0]))
        .overwrite(&mut c);

    assert_eq!(c, expected);
}

#[test]
fn tensordot_overwrite_all_axes() {
    let backend = Naive;
    let a = tensor![[1., 2.], [3., 4.]].into_dyn();
    let b = tensor![[5., 6.], [7., 8.]].into_dyn();
    let expected = tensor![[70.0]].into_dyn();

    let mut c = tensor![[0.0]].into_dyn();
    backend.tensordot(&a, &b).overwrite(&mut c);

    assert_eq!(c, expected);
}
