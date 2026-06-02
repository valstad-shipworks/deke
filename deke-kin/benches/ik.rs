//! Benchmarks for analytical IK on a 6-DOF spherical-wrist manipulator (Puma
//! 560) through the unified [`Kinematics`] API. The chain resolves to its
//! closed-form decomposition at construction, so `ik` exercises the analytic
//! path.

use criterion::{Criterion, criterion_group, criterion_main};
use deke_kin::{DHJoint, JointLimits, Kinematics};
use deke_kin::deke_types::SRobotQ;
use deke_kin::deke_types::{FKChain, IkSolver};
use std::hint::black_box;

fn puma() -> Kinematics<6, f64> {
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

/// 64 joint configurations spread across `[-pi, pi]`, deterministic so runs
/// are reproducible.
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

fn bench_ik_single(c: &mut Criterion) {
    let q = SRobotQ::<6, f64>::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);

    let mut group = c.benchmark_group("puma_560_ik_single_pose");
    let rr = puma();
    let target = rr.fk_end(&q).unwrap();
    group.bench_function("reaik", |b| b.iter(|| rr.ik(black_box(target)).unwrap()));
    group.finish();
}

fn bench_ik_sweep(c: &mut Criterion) {
    let rr = puma();
    let targets: Vec<_> = sample_qs()
        .into_iter()
        .map(|q| rr.fk_end(&SRobotQ::from_array(q)).unwrap())
        .collect();

    let mut group = c.benchmark_group("puma_560_ik_sweep_64");
    group.bench_function("reaik", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for target in &targets {
                if let Ok(outcome) = rr.ik(black_box(*target)) {
                    total += outcome.unwrap().len();
                }
            }
            total
        })
    });
    group.finish();
}

criterion_group!(benches, bench_ik_single, bench_ik_sweep);
criterion_main!(benches);
