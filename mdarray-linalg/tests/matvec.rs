use mdarray_linalg::{Naive, testing::matvec::*};

// #[test]
// fn eval_and_write() {
//     test_eval_and_write(Naive)
// }

// #[test]
// fn add_to_scaled() {
//     test_add_to_scaled(Naive)
// }

// #[test]
// fn add_to() {
//     test_add_to(Naive)
// }

#[test]
fn add_outer_basic() {
    test_add_outer_basic(Naive)
}

#[test]
fn add_outer_cplx() {
    test_add_outer_cplx(Naive)
}

#[test]
fn add_outer_subview() {
    test_add_outer_subview(Naive)
}

// #[test]
// fn add_outer_sym() {
//     test_add_outer_sym(Naive)
// }

// #[test]
// fn add_outer_her() {
//     test_add_outer_her(Naive)
// }

// #[test]
// fn add_to_scaled_vecvec() {
//     test_add_to_scaled_vecvec(Naive)
// }

#[test]
fn dot_real() {
    test_dot_real(Naive)
}

#[test]
fn dot_complex() {
    test_dot_complex(Naive)
}

// #[test]
// fn dotc_complex() {
//     test_dotc_complex(Naive)
// }

// #[test]
// fn norm1_complex() {
//     test_norm1_complex(Naive)
// }

// #[test]
// fn norm2_complex() {
//     test_norm2_complex(Naive)
// }

#[test]
fn argmax_real() {
    test_argmax_real(Naive);
}

#[test]
fn argmax_abs() {
    test_argmax_abs(Naive)
}

#[test]
fn argmax_write_real() {
    test_argmax_write_real(Naive)
}
