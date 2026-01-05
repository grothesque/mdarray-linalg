// use mdarray::{Tensor, expr::Expression, tensor};
// use mdarray_linalg::{
//     matmul::{Side, Triangle, Type},
//     prelude::*,
//     testing::{common::*, matmul::*},
// };
// use mdarray_linalg_nalgebra::Nalgebra;

// #[test]
// fn matmul_complex_with_scaling() {
//     test_matmul_complex_with_scaling_impl(&Nalgebra);
// }

// #[test]
// #[should_panic]
// fn dimension_mismatch_panic() {
//     let a = create_test_matrix_f64([2, 3]).eval();
//     let b = create_test_matrix_f64([4, 2]).eval(); // Wrong inner dimension

//     let _result = Nalgebra.matmul(&a, &b).eval();
// }

// #[test]
// fn empty_matrix_multiplication() {
//     let a = Tensor::from_elem([0, 3], 0.0f64);
//     let b = Tensor::from_elem([3, 0], 0.0f64);

//     let result = Nalgebra.matmul(&a, &b).eval();

//     assert_eq!(result, naive_matmul(&a, &b));
// }

// #[test]
// fn single_element_matrices() {
//     let a = tensor![[3.]];
//     let b = tensor![[4.]];

//     let result = Nalgebra.matmul(&a, &b).eval();

//     assert_eq!(result, naive_matmul(&a, &b));
// }

// #[test]
// fn rectangular_matrices() {
//     let a = create_test_matrix_f64([3, 5]).eval();
//     let b = create_test_matrix_f64([5, 4]).eval();

//     let result = Nalgebra.matmul(&a, &b).eval();

//     assert_eq!(result, naive_matmul(&a, &b));
// }

// #[test]
// fn zero_matrices() {
//     let a = Tensor::from_elem([2, 3], 0.0f64);
//     let b = Tensor::from_elem([3, 2], 5.0f64);

//     let result = Nalgebra.matmul(&a, &b).eval();

//     assert_eq!(*result.shape(), (2, 2));

//     assert!(result.iter().all(|&x| x == 0.0));
// }

// #[test]
// fn chained_operations() {
//     let a = create_test_matrix_f64([2, 3]).eval();
//     let b = create_test_matrix_f64([3, 2]).eval();

//     // Test scale then write
//     let scale_factor = 2.0;
//     let mut c = create_test_matrix_f64([2, 2]).eval();

//     Nalgebra.matmul(&a, &b).scale(scale_factor).write(&mut c);

//     let expected = naive_matmul(&a, &b);

//     for (cij, eij) in std::iter::zip(c, expected) {
//         assert_eq!(cij, 2. * eij);
//     }
// }

// #[test]
// fn backend_defaults() {
//     let _bd = Nalgebra::default();
// }

// #[test]
// fn special_symmetric_left_lower() {
//     let a_sym = create_symmetric_matrix_f64(3);
//     let b = create_test_matrix_f64([3, 4]).eval();

//     let result = Nalgebra
//         .matmul(&a_sym, &b)
//         .special(Side::Left, Type::Sym, Triangle::Lower);

//     assert_eq!(*result.shape(), (3, 4));

//     let result_upper = Nalgebra
//         .matmul(&a_sym, &b)
//         .special(Side::Left, Type::Sym, Triangle::Upper);

//     assert_eq!(result, result_upper);
// }

// #[test]
// fn special_triangular_upper_left() {
//     let a_tri = create_upper_triangular_f64(3);
//     let b = create_test_matrix_f64([3, 4]).eval();

//     let result = Nalgebra
//         .matmul(&a_tri, &b)
//         .special(Side::Left, Type::Tri, Triangle::Upper);

//     let result_std = Nalgebra.matmul(&a_tri, &b).eval();

//     assert_eq!(result, result_std);
// }

// #[test]
// fn special_triangular_lower_left() {
//     let a_tri = create_lower_triangular_f64(3);
//     let b = create_test_matrix_f64([3, 4]).eval();

//     let result = Nalgebra
//         .matmul(&a_tri, &b)
//         .special(Side::Left, Type::Tri, Triangle::Lower);

//     let result_std = Nalgebra.matmul(&a_tri, &b).eval();
//     assert_eq!(result, result_std);
// }

// #[test]
// fn special_triangular_upper_right() {
//     let a = create_test_matrix_f64([3, 4]).eval();
//     let b_tri = create_upper_triangular_f64(3);

//     let result = Nalgebra
//         .matmul(&b_tri, &a)
//         .special(Side::Left, Type::Tri, Triangle::Upper);

//     let result_std = Nalgebra.matmul(&b_tri, &a).eval();
//     assert_eq!(result, result_std);
// }

// #[test]
// fn special_hermitian_left_lower() {
//     let a_her = create_hermitian_matrix_complex(3);
//     let b = create_test_matrix_complex([3, 4]).eval();

//     let result = Nalgebra
//         .matmul(&a_her, &b)
//         .special(Side::Left, Type::Her, Triangle::Lower);

//     assert_eq!(*result.shape(), (3, 4));

//     let result_upper = Nalgebra.matmul(&a_her, &b).eval();

//     for (a, b) in result.iter().zip(result_upper.iter()) {
//         assert!((a - b).norm() < 1e-10);
//     }
// }

// #[test]
// fn special_with_scaling() {
//     let a_sym = create_symmetric_matrix_f64(3);
//     let b = create_test_matrix_f64([3, 4]).eval();
//     let scale_factor = 2.5;

//     let result =
//         Nalgebra.matmul(&a_sym, &b)
//             .scale(scale_factor)
//             .special(Side::Left, Type::Sym, Triangle::Upper);

//     let result_std =
//         Nalgebra.matmul(&a_sym, &b)
//             .scale(scale_factor)
//             .special(Side::Left, Type::Sym, Triangle::Upper);

//     assert_eq!(result, result_std);
// }

// #[test]
// fn special_single_element() {
//     let a = tensor![[5.0]];
//     let b = tensor![[2.0]];

//     let result = Nalgebra
//         .matmul(&a, &b)
//         .special(Side::Left, Type::Sym, Triangle::Upper);

//     assert_eq!(*result.shape(), (1, 1));
//     assert_eq!(result[[0, 0]], 10.0);
// }
