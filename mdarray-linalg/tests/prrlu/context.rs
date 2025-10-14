use approx::assert_relative_eq;

use mdarray::DTensor;
use mdarray_linalg::{MatMul, MatMulBuilder, PRRLU, PRRLUDecomp};
use mdarray_linalg_naive::Naive;

use mdarray_linalg_naive::prrlu::simple::minus_outer_pivot;

use crate::{
    assert_matrix_eq,
    common::{random_matrix, rank_k_matrix},
};

// Matrix inversion for permutation matrices (transpose)
fn invert_permutation(p: &DTensor<f64, 2>) -> DTensor<f64, 2> {
    let n = p.shape().0;
    let mut p_inv = DTensor::<f64, 2>::zeros([n, n]);

    // For permutation matrices, inverse = transpose
    for i in 0..n {
        for j in 0..n {
            p_inv[[i, j]] = p[[j, i]];
        }
    }
    p_inv
}

/// Reconstruct matrix from PRRLU decomposition: A = P^-1 * L * U * Q^-1
fn reconstruct_from_prrlu(decomp: &PRRLUDecomp<f64>) -> DTensor<f64, 2> {
    let PRRLUDecomp { p, l, u, q, .. } = decomp;
    let (n, m) = (l.shape().0, u.shape().1);

    let p_inv = invert_permutation(p);
    let q_inv = invert_permutation(q);

    let mut temp1 = DTensor::<f64, 2>::zeros([n, m]);
    let mut temp2 = DTensor::<f64, 2>::zeros([n, m]);
    let mut reconstructed = DTensor::<f64, 2>::zeros([n, m]);

    Naive.matmul(l, u).overwrite(&mut temp1);
    Naive.matmul(&p_inv, &temp1).overwrite(&mut temp2);
    Naive.matmul(&temp2, &q_inv).overwrite(&mut reconstructed);

    reconstructed
}

#[test]
fn rank_deficient() {
    let n = 10;
    let m = 5;
    let k = 2; // rank

    // Generate rank-k matrix
    let original = rank_k_matrix(n, m, k);
    let mut a = original.clone();

    // Perform PRR-LU
    let decomp = Naive.prrlu(&mut a);
    let reconstructed = reconstruct_from_prrlu(&decomp);

    println!("{:?}", decomp.u);

    assert_eq!(decomp.rank, k);
    assert_matrix_eq!(original, reconstructed);
}

#[test]
fn full_rank() {
    let n = 10;
    let m = 10;

    // Generate a well-conditioned full rank matrix
    let original = random_matrix(m, n);
    let mut a = original.clone();

    // Perform full PRR-LU
    let decomp = Naive.prrlu(&mut a);
    let reconstructed = reconstruct_from_prrlu(&decomp);

    assert_eq!(decomp.rank, n);
    assert_matrix_eq!(original, reconstructed);
}

#[test]
fn rectangular() {
    let n = 4;
    let m = 6;
    let k = 2;

    // Test with more columns than rows
    let original = rank_k_matrix(n, m, k);
    let mut a = original.clone();

    let decomp = Naive.prrlu(&mut a);
    let reconstructed = reconstruct_from_prrlu(&decomp);

    println!("{:?}", decomp.u);

    assert_eq!(decomp.rank, k);
    assert_matrix_eq!(original, reconstructed);
}

// Generate Hilbert matrix H_ij = 1/(i+j+1)
fn gen_hilbert_matrix(n: usize) -> DTensor<f64, 2> {
    DTensor::<f64, 2>::from_fn([n, n], |idx| 1.0 / (idx[0] + idx[1] + 1) as f64)
}

#[test]
fn hilbert_matrix() {
    let n = 20;

    let original = gen_hilbert_matrix(n);
    let mut a = original.clone();

    let decomp = Naive.prrlu(&mut a);
    let reconstructed = reconstruct_from_prrlu(&decomp);

    assert_matrix_eq!(original, reconstructed);
}

// #[test]
// fn backend_prrlu_random_matrix() {
//     prrlu_random_matrix(&Naive);
// }

// fn prrlu_random_matrix(bd: &impl PRRLU<f64>) {
//     let mut rng = rand::rng();
//     let n = 10;
//     let a = DTensor::<f64, 2>::from_fn([n, n], |_| rng.random::<f64>());
//     prrlu_reconstruction(bd, &a, false);
// }
