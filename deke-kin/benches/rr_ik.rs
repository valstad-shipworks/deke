//! Benchmark the complete general-6R solver (`rr_ik`, Raghavan–Roth elimination
//! and Manocha–Canny eigenvalues) on a *generic* chain — one with no spherical
//! wrist and no parallel/intersecting axes, i.e. the case the analytical path
//! cannot solve. Two measurements:
//!
//! - `rr_solve_dh`      — the validated core via the DH entry point.
//! - `rr_solve_kinspec` — the KinSpec API (includes screw extraction).

use criterion::{Criterion, criterion_group, criterion_main};
use deke_kin::deke_types::{JointSpec, KinSpec};
use deke_kin::rr_ik::{DhJoint, RrConfig, fk_dh, solve_dh, solve_kinspec};
use glam::{DAffine3, DMat3, DMat4, DVec3, DVec4};
use std::hint::black_box;

/// A generic 6R DH chain: arbitrary twists, no zero/right-angle structure that
/// would make a wrist spherical or axes parallel.
fn generic_dh() -> [DhJoint; 6] {
    [
        DhJoint {
            a: 0.32,
            alpha: 0.70,
            d: 0.18,
        },
        DhJoint {
            a: 0.25,
            alpha: -0.90,
            d: 0.21,
        },
        DhJoint {
            a: 0.29,
            alpha: 0.80,
            d: 0.14,
        },
        DhJoint {
            a: 0.22,
            alpha: -1.10,
            d: 0.19,
        },
        DhJoint {
            a: 0.18,
            alpha: 0.60,
            d: 0.11,
        },
        DhJoint {
            a: 0.15,
            alpha: -0.70,
            d: 0.17,
        },
    ]
}

/// 64 deterministic joint configurations across the workspace; their FK poses
/// are the IK targets.
fn sample_qs() -> Vec<[f64; 6]> {
    (0..64)
        .map(|i| {
            let s = (i as f64 / 64.0 - 0.5) * std::f64::consts::TAU;
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

fn dh_targets(dh: &[DhJoint; 6], qs: &[[f64; 6]]) -> Vec<[[f64; 4]; 4]> {
    qs.iter().map(|q| fk_dh(dh, q)).collect()
}

fn m4_to_dmat4(m: &[[f64; 4]; 4]) -> DMat4 {
    DMat4::from_cols(
        DVec4::new(m[0][0], m[1][0], m[2][0], m[3][0]),
        DVec4::new(m[0][1], m[1][1], m[2][1], m[3][1]),
        DVec4::new(m[0][2], m[1][2], m[2][2], m[3][2]),
        DVec4::new(m[0][3], m[1][3], m[2][3], m[3][3]),
    )
}

/// DH constant block `K_i = TransZ(d)·TransX(a)·RotX(alpha)`.
fn dh_const_block(j: &DhJoint) -> DAffine3 {
    let (sa, ca) = j.alpha.sin_cos();
    DAffine3 {
        matrix3: DMat3::from_cols(
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, ca, sa),
            DVec3::new(0.0, -sa, ca),
        ),
        translation: DVec3::new(j.a, 0.0, j.d),
    }
}

/// The same generic chain expressed as a [`KinSpec`] that is *exactly*
/// FK-equivalent to the DH chain. DH composes `∏ RotZ(θ_i)·K_i`, while KinSpec
/// composes `∏ G_i·RotZ(θ_i)` followed by `end_to_ee`. Matching them shifts the
/// constant blocks by one joint: `G_1 = I`, `G_i = K_{i-1}`, `end_to_ee = K_6`.
fn generic_kinspec(dh: &[DhJoint; 6]) -> KinSpec<f64, 6> {
    let joints = std::array::from_fn(|i| {
        let g = if i == 0 {
            DAffine3::IDENTITY
        } else {
            dh_const_block(&dh[i - 1])
        };
        (
            g,
            JointSpec::Revolute {
                axis_local: DVec3::Z,
            },
        )
    });
    KinSpec::new(DAffine3::IDENTITY, joints, dh_const_block(&dh[5]))
}

fn bench_rr_ik(c: &mut Criterion) {
    let dh = generic_dh();
    let qs = sample_qs();
    let targets = dh_targets(&dh, &qs);
    let cfg = RrConfig::default();

    let mut group = c.benchmark_group("rr_ik");

    group.bench_function("rr_solve_dh", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let t = &targets[i % targets.len()];
            i += 1;
            black_box(solve_dh(black_box(&dh), black_box(t), &cfg).len())
        })
    });

    let spec = generic_kinspec(&dh);
    let poses: Vec<DMat4> = targets.iter().map(m4_to_dmat4).collect();
    group.bench_function("rr_solve_kinspec", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let pose = poses[i % poses.len()];
            i += 1;
            black_box(solve_kinspec(black_box(&spec), black_box(pose), &cfg).map(|s| s.len()))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_rr_ik);
criterion_main!(benches);
