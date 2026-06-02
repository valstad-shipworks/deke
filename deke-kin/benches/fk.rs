//! Benchmarks for the unified [`Kinematics`] forward-kinematics chain:
//! `fk_end`, all-link `fk`, and the `StrictFKChain` geometric Jacobian, in both
//! `f64` and `f32`, plus a comparison across the per-joint dispatch tiers
//! (canonical `Z` with fixed rotations, the identity-fixed fast path, and the
//! arbitrary-axis Rodrigues path).

use criterion::{Criterion, criterion_group, criterion_main};
use deke_types::ContinuousFKChain;
use std::hint::black_box;

use deke_kin::deke_types::{FKChain, SRobotQ};
use deke_kin::{DHJoint, JointLimits, Kinematics, URDFJoint};

fn puma_dh_f64() -> Kinematics<6, f64> {
    let pi = std::f64::consts::PI;
    let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
    let a = [0.0, 0.43180, -0.02032, 0.0, 0.0, 0.0];
    let d = [0.67183, 0.13970, 0.0, 0.43180, 0.0, 0.0565];
    Kinematics::from_dh(
        std::array::from_fn(|i| DHJoint { a: a[i], alpha: alpha[i], d: d[i], theta_offset: 0.0 }),
        JointLimits::symmetric(100.0),
        &[],
    )
}

fn puma_dh_f32() -> Kinematics<6, f32> {
    let pi = std::f32::consts::PI;
    let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
    let a = [0.0, 0.43180, -0.02032, 0.0, 0.0, 0.0];
    let d = [0.67183, 0.13970, 0.0, 0.43180, 0.0, 0.0565];
    Kinematics::from_dh(
        std::array::from_fn(|i| DHJoint { a: a[i], alpha: alpha[i], d: d[i], theta_offset: 0.0 }),
        JointLimits::symmetric(100.0f32),
        &[],
    )
}

/// 6R chain, all joints revolute about `+Z` with identity-rotation origins:
/// hits the cheapest per-joint path (no fixed rotation, 2D `Z` rotation).
fn fast_z_chain() -> Kinematics<6, f64> {
    let joints: [URDFJoint; 6] = std::array::from_fn(|i| {
        let z = 0.2 + 0.05 * i as f64;
        URDFJoint::revolute((0.1, 0.0, z), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    });
    Kinematics::from_urdf(&joints, JointLimits::symmetric(100.0), &[]).expect("6R Z chain")
}

/// 6R chain about a non-canonical axis: forces the general Rodrigues path.
fn arbitrary_axis_chain() -> Kinematics<6, f64> {
    let s = 1.0 / 3.0_f64.sqrt();
    let joints: [URDFJoint; 6] = std::array::from_fn(|i| {
        let z = 0.2 + 0.05 * i as f64;
        URDFJoint::revolute((0.1, 0.0, z), (0.0, 0.0, 0.0), (s, s, s))
    });
    Kinematics::from_urdf(&joints, JointLimits::symmetric(100.0), &[]).expect("6R arbitrary-axis chain")
}

/// 64 deterministic joint configurations spread across `[-pi, pi]`.
fn sample_qs() -> Vec<[f64; 6]> {
    let n = 64usize;
    (0..n)
        .map(|i| {
            let s = (i as f64 / n as f64 - 0.5) * std::f64::consts::TAU;
            [
                0.7 * s,
                0.5 * (s + 0.4),
                -0.3 * (s - 0.2),
                0.6 * (s + 0.1),
                -0.4 * s,
                0.8 * (s - 0.3),
            ]
        })
        .collect()
}

fn bench_fk_end(c: &mut Criterion) {
    let q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

    let mut group = c.benchmark_group("kinematics_fk_end");

    let dh64 = puma_dh_f64();
    group.bench_function("puma_dh_f64", |b| {
        let q = SRobotQ::<6, f64>::from_array(q);
        b.iter(|| dh64.fk_end(black_box(&q)).unwrap())
    });

    let dh32 = puma_dh_f32();
    group.bench_function("puma_dh_f32", |b| {
        let q = SRobotQ::<6, f32>::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        b.iter(|| dh32.fk_end(black_box(&q)).unwrap())
    });

    group.finish();
}

/// Compare the three per-joint dispatch tiers on `fk_end`.
fn bench_fk_end_by_tier(c: &mut Criterion) {
    let q = SRobotQ::<6, f64>::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);

    let mut group = c.benchmark_group("kinematics_fk_end_by_tier");

    let dh = puma_dh_f64();
    group.bench_function("canonical_z_fixed_rot", |b| {
        b.iter(|| dh.fk_end(black_box(&q)).unwrap())
    });

    let fast = fast_z_chain();
    group.bench_function("identity_fixed_fast", |b| {
        b.iter(|| fast.fk_end(black_box(&q)).unwrap())
    });

    let general = arbitrary_axis_chain();
    group.bench_function("arbitrary_axis_general", |b| {
        b.iter(|| general.fk_end(black_box(&q)).unwrap())
    });

    group.finish();
}

fn bench_fk_all_frames(c: &mut Criterion) {
    let q = SRobotQ::<6, f64>::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let dh = puma_dh_f64();

    let mut group = c.benchmark_group("kinematics_fk_all_frames");
    group.bench_function("puma_dh_f64", |b| b.iter(|| dh.fk(black_box(&q)).unwrap()));
    group.finish();
}

fn bench_jacobian(c: &mut Criterion) {
    let q = SRobotQ::<6, f64>::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let dh = puma_dh_f64();

    let mut group = c.benchmark_group("kinematics_jacobian");
    group.bench_function("puma_dh_f64", |b| {
        b.iter(|| dh.jacobian(black_box(&q)).unwrap())
    });
    group.finish();
}

fn bench_fk_end_sweep(c: &mut Criterion) {
    let qs: Vec<SRobotQ<6, f64>> = sample_qs().into_iter().map(SRobotQ::from_array).collect();
    let dh = puma_dh_f64();

    let mut group = c.benchmark_group("kinematics_fk_end_sweep_64");
    group.bench_function("puma_dh_f64", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for q in &qs {
                acc += dh.fk_end(black_box(q)).unwrap().translation.x;
            }
            acc
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_fk_end,
    bench_fk_end_by_tier,
    bench_fk_all_frames,
    bench_jacobian,
    bench_fk_end_sweep,
);
criterion_main!(benches);
