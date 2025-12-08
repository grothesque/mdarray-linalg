use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box as bb;

use mdarray::{Const, DTensor, Dyn, Slice, array};

use mdarray_linalg::svd::SVDDecomp;
use mdarray_linalg::{Naive, prelude::*};
use mdarray_linalg_faer::Faer;
use mdarray_linalg_lapack::Lapack;

use nalgebra::{DMatrix, Matrix4, SVD};

const N: i32 = 300;

// type Slice4x4Const = Slice<f64, (Const<4>, Const<4>)>;
// type Slice10x10Const = Slice<f64, (Const<10>, Const<10>)>;
type Slice4x4Dyn = Slice<f64, (Dyn, Dyn)>;
type SliceNDyn = Slice<f64, (Dyn, Dyn)>;

// ============================================================================
// 4x4 benchmarks nalgebra with static size
// ============================================================================

#[inline(never)]
pub fn svd_4x4_nalgebra_static(
    data: &[f64; 16],
) -> SVD<f64, nalgebra::Const<4>, nalgebra::Const<4>> {
    let a = Matrix4::from_row_slice(data);
    a.svd(true, true)
}

// ============================================================================
// 4x4 benchmarks with allocation
// ============================================================================

#[inline(never)]
pub fn svd_4x4_dyn_backend_lapack(a: &Slice4x4Dyn) -> SVDDecomp<f64> {
    let mut a_copy = a.to_owned();
    let bd = Lapack::new();
    bd.svd(&mut a_copy).expect("SVD failed")
}

#[inline(never)]
pub fn svd_4x4_dyn_backend_faer(a: &Slice4x4Dyn) -> SVDDecomp<f64> {
    let mut a_copy = a.to_owned();
    let bd = Faer;
    bd.svd(&mut a_copy).expect("SVD failed")
}

#[inline(never)]
pub fn svd_4x4_nalgebra(data: &[f64]) -> SVD<f64, nalgebra::Dyn, nalgebra::Dyn> {
    let a = DMatrix::from_row_slice(4, 4, data);
    a.svd(true, true)
}

// ============================================================================
// NxN benchmarks
// ============================================================================

#[inline(never)]
pub fn svd_n_dyn_backend_lapack(a: &SliceNDyn) -> SVDDecomp<f64> {
    let mut a_copy = a.to_owned();
    let bd = Lapack::new();
    bd.svd(&mut a_copy).expect("SVD failed")
}

#[inline(never)]
pub fn svd_n_dyn_backend_faer(a: &SliceNDyn) -> SVDDecomp<f64> {
    let mut a_copy = a.to_owned();
    let bd = Faer;
    bd.svd(&mut a_copy).expect("SVD failed")
}

#[inline(never)]
pub fn svd_n_nalgebra(data: &[f64]) -> SVD<f64, nalgebra::Dyn, nalgebra::Dyn> {
    let a = DMatrix::from_row_slice(N as usize, N as usize, data);
    a.svd(true, true)
}

// ============================================================================
// Const dimension benchmarks
// ============================================================================

// #[inline(never)]
// pub fn svd_4x4_const_backend_lapack(a: &Slice4x4Const) -> SVDDecomp<f64> {
//     let mut a_copy = a.to_owned();
//     let bd = Lapack::new();
//     bd.svd(&mut a_copy).expect("SVD failed")
// }

// #[inline(never)]
// pub fn svd_4x4_const_backend_faer(a: &Slice4x4Const) -> SVDDecomp<f64> {
//     let mut a_copy = a.to_owned();
//     let bd = Faer;
//     bd.svd(&mut a_copy).expect("SVD failed")
// }

// #[inline(never)]
// pub fn svd_10x10_const_backend_lapack(a: &Slice10x10Const) -> SVDDecomp<f64> {
//     let mut a_copy = a.to_owned();
//     let bd = Lapack::new();
//     bd.svd(&mut a_copy).expect("SVD failed")
// }

// #[inline(never)]
// pub fn svd_10x10_const_backend_faer(a: &Slice10x10Const) -> SVDDecomp<f64> {
//     let mut a_copy = a.to_owned();
//     let bd = Faer;
//     bd.svd(&mut a_copy).expect("SVD failed")
// }

fn criterion_benchmark(crit: &mut Criterion) {
    // ========================================================================
    // 4x4 benchmarks
    // ========================================================================

    // 4x4 nalgebra static
    let a_4x4_data_array: [f64; 16] = {
        let mut arr = [0.0; 16];
        for (i, val) in (0..16).map(|x| (x as f64) * 0.5).enumerate() {
            arr[i] = val;
        }
        arr
    };

    crit.bench_function("svd_4x4_nalgebra_static", |bencher| {
        bencher.iter(|| svd_4x4_nalgebra_static(bb(&a_4x4_data_array)))
    });

    // 4x4 dynamic
    let a_4x4_dyn: DTensor<_, 1> = (0..16).map(|x| (x as f64) * 0.5).collect::<Vec<_>>().into();
    let a_4x4_dyn = a_4x4_dyn.reshape((4, 4));
    let a_4x4_data: Vec<f64> = (0..16).map(|x| (x as f64) * 0.5).collect();

    // With allocation
    crit.bench_function("svd_4x4_dyn_lapack", |bencher| {
        bencher.iter(|| svd_4x4_dyn_backend_lapack(bb(&a_4x4_dyn)))
    });

    crit.bench_function("svd_4x4_dyn_faer", |bencher| {
        bencher.iter(|| svd_4x4_dyn_backend_faer(bb(&a_4x4_dyn)))
    });

    crit.bench_function("svd_4x4_nalgebra", |bencher| {
        bencher.iter(|| svd_4x4_nalgebra(bb(&a_4x4_data)))
    });

    // ========================================================================
    // 4x4 Const dimension benchmarks
    // ========================================================================

    // let a_4x4_const: DTensor<_, 1> = (0..16).map(|x| (x as f64) * 0.5).collect::<Vec<_>>().into();
    // let a_4x4_const = a_4x4_const.reshape((Const::<4>, Const::<4>));

    // crit.bench_function("svd_4x4_const_lapack", |bencher| {
    //     bencher.iter(|| svd_4x4_const_backend_lapack(bb(&a_4x4_const)))
    // });

    // crit.bench_function("svd_4x4_const_faer", |bencher| {
    //     bencher.iter(|| svd_4x4_const_backend_faer(bb(&a_4x4_const)))
    // });

    // ========================================================================
    // NxN benchmarks
    // ========================================================================

    let a_n_dyn: DTensor<_, 1> = (0..N * N)
        .map(|x| (x as f64) * 0.5)
        .collect::<Vec<_>>()
        .into();
    let a_n_dyn = a_n_dyn.reshape((N as usize, N as usize));
    let a_n_data: Vec<f64> = (0..N * N).map(|x| (x as f64) * 0.5).collect();

    // With allocation
    crit.bench_function("svd_n_dyn_lapack", |bencher| {
        bencher.iter(|| svd_n_dyn_backend_lapack(bb(&a_n_dyn)))
    });

    crit.bench_function("svd_n_dyn_faer", |bencher| {
        bencher.iter(|| svd_n_dyn_backend_faer(bb(&a_n_dyn)))
    });

    crit.bench_function("svd_n_nalgebra", |bencher| {
        bencher.iter(|| svd_n_nalgebra(bb(&a_n_data)))
    });

    // ========================================================================
    // n Const dimension benchmarks
    // ========================================================================

    // let a_n_const: DTensor<_, 1> = (0..100).map(|x| (x as f64) * 0.5).collect::<Vec<_>>().into();
    // let a_n_const = a_n_const.reshape((Const::<10>, Const::<10>));

    // crit.bench_function("svd_n_const_lapack", |bencher| {
    //     bencher.iter(|| svd_n_const_backend_lapack(bb(&a_n_const)))
    // });

    // crit.bench_function("svd_n_const_faer", |bencher| {
    //     bencher.iter(|| svd_n_const_backend_faer(bb(&a_n_const)))
    // });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
