extern crate openblas_src as _;
use mdarray::{Array, Shape, Strided, StridedMapping, View, array, expr::Expression, tensor};
use mdarray_linalg::{
    prelude::*,
    testing::{common::*, contract::*},
};
use mdarray_linalg_blas::Blas;

#[test]
fn matmul_complex_with_scaling() {
    matmul_complex_with_scaling_impl(&Blas);
}

#[test]
#[should_panic]
fn dimension_mismatch_panic() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([4, 2]).eval(); // Wrong inner dimension

    let _result = Blas.matmul(&a, &b).eval();
}

#[test]
fn empty_matrix_multiplication() {
    let a = Array::from_elem([0, 3], 0.0f64);
    let b = Array::from_elem([3, 0], 0.0f64);

    let result = Blas.matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn single_element_matrices() {
    let a = tensor![[3.]];
    let b = tensor![[4.]];

    let result = Blas.matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn rectangular_matrices() {
    let a = create_test_matrix_f64([3, 5]).eval();
    let b = create_test_matrix_f64([5, 4]).eval();

    let result = Blas.matmul(&a, &b).eval();

    assert_eq!(result, naive_matmul(&a, &b));
}

#[test]
fn zero_matrices() {
    let a = Array::from_elem([2, 3], 0.0f64);
    let b = Array::from_elem([3, 2], 5.0f64);

    let result = Blas.matmul(&a, &b).eval();

    assert_eq!(*result.shape(), (2, 2));

    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn chained_operations() {
    let a = create_test_matrix_f64([2, 3]).eval();
    let b = create_test_matrix_f64([3, 2]).eval();

    // Test scale then write
    let scale_factor = 2.0;
    let mut c = create_test_matrix_f64([2, 2]).eval();

    Blas.matmul(&a, &b).scale(scale_factor).write(&mut c);

    let expected = naive_matmul(&a, &b);

    for (cij, eij) in std::iter::zip(c, expected) {
        assert_eq!(cij, 2. * eij);
    }
}

#[test]
fn backend_defaults() {
    let _bd = Blas::default();
}

#[test]
fn matmul_builder_methods() {
    matmul_builder_methods_impl(&Blas);
}

#[test]
fn contract_builder_methods() {
    contract_builder_methods_impl(&Blas);
}

#[test]
fn test_matmul_builder_accumulation() {
    let a = example_matrix([2, 3]).eval();
    let b = example_matrix([3, 4]).eval();
    let c_expr = || example_matrix([2, 4]);
    let ab = naive_matmul(&a, &b);

    assert_eq!(Blas.matmul(&a, &b).eval(), ab);

    // Test all combinations of row- and column-major A, B, and C through
    // the public builder API.  This preserves the old raw-GEMM coverage
    // without exposing GEMM as public backend API.
    let a_cmajor = a.transpose().to_tensor();
    let a_cmajor = a_cmajor.transpose();
    let b_cmajor = b.transpose().to_tensor();
    let b_cmajor = b_cmajor.transpose();

    // Convert to a ‘Strided’ layout (still row major) so that ‘a’ has the same type as ‘a_cmajor’.
    let a = a.remap();
    let b = b.remap();
    let mut c = c_expr().eval();
    let mut c = c.remap_mut();

    let mut c_cmajor = c.transpose().to_tensor();
    let mut c_cmajor = c_cmajor.transpose_mut();

    for a in [&a, &a_cmajor] {
        for b in [&b, &b_cmajor] {
            for c in [&mut c, &mut c_cmajor] {
                c.assign(c_expr());
                Blas.matmul(a, b).write(c);
                assert!(*c == ab);

                c.assign(c_expr());
                let initial = c.to_array();
                Blas.matmul(a, b).add_to(c);
                for (actual, (initial, product)) in c.iter().zip(initial.iter().zip(ab.iter())) {
                    assert_eq!(*actual, *initial + *product);
                }

                c.assign(c_expr());
                let initial = c.to_array();
                Blas.matmul(a, b).add_to_scaled(c, 2.0);
                for (actual, (initial, product)) in c.iter().zip(initial.iter().zip(ab.iter())) {
                    assert_eq!(*actual, 2.0 * *initial + *product);
                }
            }
        }
    }
}

#[test]
pub fn non_contiguous_along_both_axis() {
    let bufa: Vec<f64> = vec![1., 0., 3., 0., 2., 0., 4.];

    let av: View<'_, _, (usize, usize), Strided> = unsafe {
        let sh = <(usize, usize) as Shape>::from_dims(&[2, 2]);
        let mapping = StridedMapping::new(sh, &[2, 4]);
        View::new_unchecked(bufa.as_ptr(), mapping)
    };

    let bufb: Vec<f64> = vec![5., 0., 7., 0., 6., 0., 8.];

    let bv: View<'_, _, (usize, usize), Strided> = unsafe {
        let sh = <(usize, usize) as Shape>::from_dims(&[2, 2]);
        let mapping = StridedMapping::new(sh, &[2, 4]);
        View::new_unchecked(bufb.as_ptr(), mapping)
    };

    let c = Blas.matmul(&av, &bv).eval();

    assert_eq!(c, array![[19., 22.], [43., 50.]]);
}

#[test]
fn macro_matmul() {
    let a = create_test_matrix_f64([3, 5]).eval();
    let b = create_test_matrix_f64([5, 4]).eval();
    let c = create_test_matrix_f64([4, 2]).eval();
    let d = create_test_matrix_f64([2, 6]).eval();

    let cd = Blas.matmul(&c, &d).eval();
    let bcd = Blas.matmul(&b, &cd).eval();
    let result = Blas.matmul(&a, &bcd).eval();

    let expected = naive_matmul(&a, &naive_matmul(&b, &naive_matmul(&c, &d)));

    assert_eq!(result, expected);
}

// --- Structured contractions ---

#[test]
fn contract_all() {
    contract_all_impl(&Blas);
}

#[test]
fn contract_n_2_should_match_all_axes() {
    contract_n_2_should_match_all_axes_impl(&Blas);
}

#[test]
fn contract_pairs_matrix_multiplication() {
    contract_pairs_matrix_multiplication_impl(&Blas);
}

#[test]
fn contract_n_0_should_outer_product() {
    contract_n_0_should_outer_product_impl(&Blas);
}

#[test]
fn contract_scalar_inputs_should_multiply() {
    contract_scalar_inputs_should_multiply_impl(&Blas);
}

#[test]
fn contract_increase_deep() {
    contract_increase_deep_impl(&Blas);
}

#[test]
fn contract_vector_dot_product() {
    contract_vector_dot_product_impl(&Blas);
}

#[test]
fn contract_all_invalid_shapes_should_panic() {
    contract_all_invalid_shapes_should_panic_impl(&Blas);
}

#[test]
fn contract_outer_should_match_manual_kronecker() {
    contract_outer_should_match_manual_kronecker_impl(&Blas);
}

// --- Einsum-style contractions ---

#[test]
fn contract_einsum_matrix_multiplication() {
    contract_einsum_matrix_multiplication_impl(&Blas)
}

#[test]
fn contract_einsum_full_contraction() {
    contract_einsum_full_contraction_impl(&Blas)
}

#[test]
fn contract_einsum_output_permutation() {
    contract_einsum_output_permutation_impl(&Blas)
}

#[test]
fn contract_einsum_outer_product() {
    contract_einsum_outer_product_impl(&Blas)
}

#[test]
fn contract_einsum_trace_diagonal() {
    contract_einsum_trace_diagonal_impl(&Blas)
}

#[test]
fn contract_einsum_index_relabelling() {
    contract_einsum_index_relabelling_impl(&Blas)
}

#[test]
fn contract_einsum_partial_trace_then_contract() {
    contract_einsum_partial_trace_then_contract_impl(&Blas)
}

#[test]
fn contract_einsum_cross_diagonal() {
    contract_einsum_cross_diagonal_impl(&Blas)
}

#[test]
fn contract_einsum_vector_result() {
    contract_einsum_vector_result_impl(&Blas)
}
