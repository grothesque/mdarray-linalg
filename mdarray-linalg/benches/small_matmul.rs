use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box as bb;

use mdarray::{Const, DTensor, Dim, Slice, array};

use mdarray_linalg::{Naive, prelude::*};

type Slice4x4 = Slice<f64, (Const<4>, Const<4>)>;

// Compiles to compact and efficient machine code.
#[inline(never)]
pub fn matmul4x4_baseline(a: &Slice4x4, b: &Slice4x4, c: &mut Slice4x4) {
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                c[[i, j]] += a[[i, k]] * b[[k, j]];
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
pub fn matmul4x4_backend(a: &Slice4x4, b: &Slice4x4, c: &mut Slice4x4) {
    Naive.matmul(a, b).add_to(c);
}

// Criterion setup

fn criterion_benchmark(crit: &mut Criterion) {
    let a: DTensor<_, 1> = (0..16).map(f64::from).collect::<Vec<_>>().into();
    let a = a.reshape((Const::<4>, Const::<4>));
    let b: DTensor<_, 1> = (16..32).map(f64::from).collect::<Vec<_>>().into();
    let b = b.reshape((Const::<4>, Const::<4>));
    crit.bench_function("baseline", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_baseline(bb(&a), bb(&b), bb(&mut c)))
    });
    crit.bench_function("simple", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_simple(bb(&a), bb(&b), bb(&mut c)))
    });
    crit.bench_function("view", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_view(bb(&a), bb(&b), bb(&mut c)))
    });
    crit.bench_function("backend", |bencher| {
        let mut c = array![[0.0; 4]; 4];
        bencher.iter(|| matmul4x4_backend(bb(&a), bb(&b), bb(&mut c)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
