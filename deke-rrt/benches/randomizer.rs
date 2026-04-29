//! Benchmarks for the random-number sources used by the RRT planners.
//!
//! Three groups:
//! - `rng_next_u64`: scalar `next_u64` throughput per variant.
//! - `rng_sample_unit`: cost of one full N-D unit-cube sample (the path the
//!   planners actually take).
//! - `halton_paths`: scalar (`next_u64`-driven) versus SIMD (`sample_unit`)
//!   draws of an N-D Halton point. Demonstrates the win from the trait split.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use deke_rrt::{DekeRand, DekeRng, RandomizerType};

const N: usize = 6;
const SEED: u64 = 0xDEADBEEF_CAFEBABE;
const F64_FROM_U64: f64 = 1.0 / (1u64 << 53) as f64;

#[inline]
fn u64_to_unit_f64(bits: u64) -> f64 {
    (bits >> 11) as f64 * F64_FROM_U64
}

const VARIANTS: [RandomizerType; 4] = [
    RandomizerType::Wyrand,
    RandomizerType::SplitMix,
    RandomizerType::Xorshift,
    RandomizerType::Halton,
];

fn variant_name(kind: RandomizerType) -> &'static str {
    match kind {
        RandomizerType::Wyrand => "Wyrand",
        RandomizerType::SplitMix => "SplitMix",
        RandomizerType::Xorshift => "Xorshift",
        RandomizerType::Halton => "Halton",
    }
}

fn bench_next_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng_next_u64");
    for &kind in &VARIANTS {
        group.bench_function(variant_name(kind), |b| {
            let mut rng = DekeRand::<N>::new(kind, SEED);
            b.iter(|| black_box(<DekeRand<N> as DekeRng<N>>::next_u64(&mut rng)));
        });
    }
    group.finish();
}

fn bench_sample_unit(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng_sample_unit");
    for &kind in &VARIANTS {
        group.bench_function(variant_name(kind), |b| {
            let mut rng = DekeRand::<N>::new(kind, SEED);
            b.iter(|| black_box(rng.sample_unit()));
        });
    }
    group.finish();
}

fn bench_halton_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("halton_paths");

    group.bench_function("scalar_next_u64_loop", |b| {
        let mut rng = DekeRand::<N>::new(RandomizerType::Halton, SEED);
        b.iter(|| {
            let mut sample = [0.0f64; N];
            for slot in &mut sample {
                *slot = u64_to_unit_f64(<DekeRand<N> as DekeRng<N>>::next_u64(&mut rng));
            }
            black_box(sample)
        });
    });

    group.bench_function("simd_sample_unit", |b| {
        let mut rng = DekeRand::<N>::new(RandomizerType::Halton, SEED);
        b.iter(|| black_box(rng.sample_unit()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_next_u64,
    bench_sample_unit,
    bench_halton_paths
);
criterion_main!(benches);
