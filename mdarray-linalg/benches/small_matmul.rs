use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box as bb;

use mdarray::{Const, DTensor, Dim, Slice, array};
use nalgebra::SMatrix;

use mdarray_linalg::{Naive, prelude::*};
use mdarray_linalg_blas::Blas;
use mdarray_linalg_faer::Faer;

const N: i32 = 100;

type Slice4x4 = Slice<f64, (Const<4>, Const<4>)>;

type SliceN = Slice<f64, (Const<{ N as usize }>, Const<{ N as usize }>)>;

// Compiles to compact and efficient machine code.
#[inline(never)]
pub fn matmul4x4_baseline(a: &Slice4x4, b: &Slice4x4, c: &mut Slice4x4) {
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                // c[[i, j]] += a[[i, k]] * b[[k, j]];
                c[[i, j]] = a[[i, k]].mul_add(b[[k, j]], c[[i, j]]);
            }
        }
    }
}

fn matmul_simple<DI: Dim, DK: Dim, DJ: Dim>(
    a: &Slice<f64, (DI, DK)>,
    b: &Slice<f64, (DK, DJ)>,
    c: &mut Slice<f64, (DI, DJ)>,
) {
    let ash = a.shape();
    let bsh = b.shape();
    let csh = c.shape();

    let di = ash.0.size();
    assert_eq!(di, csh.0.size());

    let dk = ash.1.size();
    assert_eq!(dk, bsh.0.size());

    let dj = bsh.1.size();
    assert_eq!(dj, csh.1.size());

    for i in 0..di {
        for j in 0..dj {
            for k in 0..dk {
                c[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
}

// Equivalent performance can be reached with a generic function.
#[inline(never)]
pub fn matmul4x4_simple(a: &Slice4x4, b: &Slice4x4, c: &mut Slice4x4) {
    matmul_simple(a, b, c);
}

// But not if we call it with &DSlice arguments.
#[inline(never)]
pub fn matmul4x4_view(a: &Slice4x4, b: &Slice4x4, c: &mut Slice4x4) {
    let a = a.reshape([4, 4]);
    let b = b.reshape([4, 4]);
    let mut c = c.reshape_mut([4, 4]);
    matmul_simple(&a, &b, &mut c);
}

// The same (even bigger) problem exists for our MatMul trait.
#[inline(never)]
pub fn matmul4x4_backend_naive(a: &Slice4x4, b: &Slice4x4, c: &mut Slice4x4) {
    Naive.matmul(a, b).write(c);
}

#[inline(never)]
pub fn matmul4x4_backend_faer(a: &Slice4x4, b: &Slice4x4, c: &mut Slice4x4) {
    Faer.matmul(a, b).write(c);
}

#[inline(never)]
pub fn matmul4x4_backend_blas(a: &Slice4x4, b: &Slice4x4, c: &mut Slice4x4) {
    Blas.matmul(a, b).write(c);
}

#[inline(never)]
pub fn matmuln_backend_naive(a: &SliceN, b: &SliceN, c: &mut SliceN) {
    Naive.matmul(a, b).write(c);
}

#[inline(never)]
pub fn matmuln_backend_faer(a: &SliceN, b: &SliceN, c: &mut SliceN) {
    Faer.matmul(a, b).write(c);
}

#[inline(never)]
pub fn matmuln_backend_blas(a: &SliceN, b: &SliceN, c: &mut SliceN) {
    Blas.matmul(a, b).write(c);
}

#[inline(never)]
pub fn matmuln_nalgebra<const N: usize>(
    a: &SMatrix<f64, N, N>,
    b: &SMatrix<f64, N, N>,
) -> SMatrix<f64, N, N> {
    a * b
}

fn criterion_benchmark(crit: &mut Criterion) {
    let a: DTensor<_, 1> = (0..16).map(f64::from).collect::<Vec<_>>().into();
    let a = a.reshape((Const::<4>, Const::<4>));
    let b: DTensor<_, 1> = (16..32).map(f64::from).collect::<Vec<_>>().into();
    let b = b.reshape((Const::<4>, Const::<4>));
    crit.bench_function("matmul_baseline", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_baseline(bb(&a), bb(&b), bb(&mut c)))
    });
    crit.bench_function("matmul_simple", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_simple(bb(&a), bb(&b), bb(&mut c)))
    });
    crit.bench_function("matmul_view", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_view(bb(&a), bb(&b), bb(&mut c)))
    });
    crit.bench_function("matmul_bd_naive", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_backend_naive(bb(&a), bb(&b), bb(&mut c)))
    });
    crit.bench_function("matmul_bd_blas", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_backend_blas(bb(&a), bb(&b), bb(&mut c)))
    });
    crit.bench_function("matmul_bd_faer", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_backend_faer(bb(&a), bb(&b), bb(&mut c)))
    });

    // === NxN ===
    let a_n: DTensor<_, 1> = (0..N * N).map(f64::from).collect::<Vec<_>>().into();
    let a_n = a_n.reshape((Const::<{ N as usize }>, Const::<{ N as usize }>));
    let b_n: DTensor<_, 1> = (N * N..2 * N * N).map(f64::from).collect::<Vec<_>>().into();
    let b_n = b_n.reshape((Const::<{ N as usize }>, Const::<{ N as usize }>));
    crit.bench_function("matmulN_bd_naive", |bencher| {
        let mut c = array![[0.0; N as usize]; N as usize];
        bencher.iter(|| matmuln_backend_naive(bb(&a_n), bb(&b_n), bb(&mut c)))
    });
    crit.bench_function("matmulN_bd_blas", |bencher| {
        let mut c = array![[0.0; N as usize]; N as usize];
        bencher.iter(|| matmuln_backend_blas(bb(&a_n), bb(&b_n), bb(&mut c)))
    });
    crit.bench_function("matmulN_bd_faer", |bencher| {
        let mut c = array![[0.0; N as usize]; N as usize];
        bencher.iter(|| matmuln_backend_faer(bb(&a_n), bb(&b_n), bb(&mut c)))
    });
    let a_n_nalgebra =
        SMatrix::<f64, { N as usize }, { N as usize }>::from_iterator((0..N * N).map(f64::from));
    let b_n_nalgebra = SMatrix::<f64, { N as usize }, { N as usize }>::from_iterator(
        (N * N..2 * N * N).map(f64::from),
    );

    crit.bench_function("matmulN_nalgebra", |bencher| {
        bencher.iter(|| matmuln_nalgebra(bb(&a_n_nalgebra), bb(&b_n_nalgebra)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
