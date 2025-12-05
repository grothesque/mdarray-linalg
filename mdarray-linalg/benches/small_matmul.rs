use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box as bb;

use mdarray::{Const, DTensor, Slice, array};

use mdarray_linalg::{Naive, prelude::*};

type Slice4x4 = Slice<f64, (Const<4>, Const<4>)>;

// Compiles to compact and efficient machine code.
#[inline(never)]
pub fn matmul4x4_basic(a: &Slice4x4, b: &Slice4x4, c: &mut Slice4x4) {
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                c[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
}

// We want to make sure that our backend remains comparable.
#[inline(never)]
pub fn matmul4x4_backend(a: &Slice4x4, b: &Slice4x4, c: &mut Slice4x4) {
    Naive.matmul(a, b).add_to(c);
}

// Criterion setup

fn criterion_benchmark(crit: &mut Criterion) {
    let a: DTensor<_, 1> = (0..16).map(f64::from).collect::<Vec<_>>().into();
    let a = a.reshape((Const::<4>, Const::<4>));
    let b: DTensor<_, 1> = (16..32).map(f64::from).collect::<Vec<_>>().into();
    let b = b.reshape((Const::<4>, Const::<4>));
    crit.bench_function("basic", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_basic(bb(&a), bb(&b), bb(&mut c)))
    });
    crit.bench_function("backend", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_backend(bb(&a), bb(&b), bb(&mut c)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
