//! Independent validation: recompute the output's finite differences here (not
//! through the retimer's own `verify_joint_fd`) and assert joint v/a/j and TCP
//! velocity stay under their limits, that every sample lies on the input chord,
//! and that the endpoints are exact. Catches a shared-stencil bug the in-crate
//! verify could not.

mod common;

use std::time::Duration;

use deke_topp3_lp::{Topp3Lp, Topp3LpConstraints, Topp3LpTcp};
use deke_types::{Retimer, SRobotPath, SRobotQ, SRobotTraj};

const TOL: f64 = 1e-3; // above the retimer's 1e-6 gate, tight enough to catch real overshoot

fn jl<const N: usize>(v: f64, a: f64, j: f64) -> Topp3LpConstraints<N> {
    Topp3LpConstraints::symmetric(v, a, j, Duration::from_millis(8))
}

/// Worst per-joint backward-FD violation ratio (value/limit) over v, a, j.
fn joint_fd_violation<const N: usize>(
    traj: &SRobotTraj<N, f64>,
    v: f64,
    a: f64,
    j: f64,
) -> (f64, &'static str) {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let n = p.len();
    let mut worst = (0.0_f64, "none");
    let consider = |val: f64, lim: f64, kind: &'static str, w: &mut (f64, &'static str)| {
        let r = val.abs() / lim;
        if r > w.0 {
            *w = (r, kind);
        }
    };
    for i in 1..n {
        for jj in 0..N {
            consider(
                (p[i].0[jj] - p[i - 1].0[jj]) / dt,
                v,
                "velocity",
                &mut worst,
            );
        }
    }
    for i in 2..n {
        for jj in 0..N {
            let acc = (p[i].0[jj] - 2.0 * p[i - 1].0[jj] + p[i - 2].0[jj]) / (dt * dt);
            consider(acc, a, "acceleration", &mut worst);
        }
    }
    for i in 3..n {
        for jj in 0..N {
            let jk = (p[i].0[jj] - 3.0 * p[i - 1].0[jj] + 3.0 * p[i - 2].0[jj] - p[i - 3].0[jj])
                / (dt * dt * dt);
            consider(jk, j, "jerk", &mut worst);
        }
    }
    worst
}

/// Worst central-difference velocity/acceleration ratio — the convention
/// `SRobotTraj::velocity_at`/`acceleration_at` use, i.e. what a consumer reading
/// those helpers would see.
fn central_va_violation<const N: usize>(traj: &SRobotTraj<N, f64>, v: f64, a: f64) -> f64 {
    let n = traj.len();
    let mut worst = 0.0f64;
    for i in 0..n {
        if let Some(vel) = traj.velocity_at(i) {
            for jj in 0..N {
                worst = worst.max(vel.0[jj].abs() / v);
            }
        }
        if let Some(acc) = traj.acceleration_at(i) {
            for jj in 0..N {
                worst = worst.max(acc.0[jj].abs() / a);
            }
        }
    }
    worst
}

fn seg_dist<const N: usize>(p: &[f64; N], a: &[f64; N], b: &[f64; N]) -> f64 {
    let ab: [f64; N] = std::array::from_fn(|i| b[i] - a[i]);
    let ab2: f64 = ab.iter().map(|x| x * x).sum();
    let t = if ab2 > 1e-18 {
        ((0..N).map(|i| (p[i] - a[i]) * ab[i]).sum::<f64>() / ab2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0..N)
        .map(|i| {
            let d = p[i] - (a[i] + t * ab[i]);
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

fn polyline_dev<const N: usize>(traj: &SRobotTraj<N, f64>, wps: &[[f64; N]]) -> f64 {
    let p = traj.path();
    (0..p.len())
        .map(|i| {
            let q = p[i].0;
            (0..wps.len() - 1)
                .map(|s| seg_dist(&q, &wps[s], &wps[s + 1]))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

fn assert_clean<const N: usize>(
    traj: &SRobotTraj<N, f64>,
    wps: &[[f64; N]],
    v: f64,
    a: f64,
    j: f64,
) {
    let (r, kind) = joint_fd_violation(traj, v, a, j);
    assert!(r <= 1.0 + TOL, "backward-FD {kind} at {:.4}x limit", r);
    let c = central_va_violation(traj, v, a);
    assert!(c <= 1.0 + TOL, "central-FD v/a at {:.4}x limit", c);
    assert!(polyline_dev(traj, wps) < 1e-9, "off-chord deviation");
    let p = traj.path();
    assert_eq!(p[0].0, wps[0], "start not on first waypoint");
    assert_eq!(
        p[p.len() - 1].0,
        wps[wps.len() - 1],
        "end not on last waypoint"
    );
}

fn fuzz<const N: usize>(seed: u64, start: [f64; N], n: usize, delta: f64) -> Vec<[f64; N]> {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let mut out = vec![start];
    let mut cur = start;
    for _ in 1..n {
        for c in cur.iter_mut() {
            *c += delta * next();
        }
        out.push(cur);
    }
    out
}

#[test]
fn joint_fd_under_limits_across_paths() {
    let v6 = common::wide_validator::<6>();
    let r6 = Topp3Lp::<6>::new();
    let (v, a, j) = (1.5, 8.0, 400.0);

    let cases: Vec<Vec<[f64; 6]>> = vec![
        vec![
            [0.0, -1.2, 1.5, -0.3, 0.5, 0.0],
            [0.6, -0.6, 0.9, 0.3, -0.2, 0.8],
        ],
        vec![
            [0.0, -1.3, 1.5, 0.0, 0.0, 0.0],
            [0.2, -1.1, 1.3, -0.1, 0.1, 0.1],
            [0.4, -0.9, 1.1, -0.2, 0.2, 0.2],
            [0.6, -0.7, 0.9, -0.3, 0.1, 0.3],
            [0.8, -0.5, 0.7, -0.4, 0.0, 0.4],
        ],
        vec![
            [0.0, -1.0, 1.2, 0.0, 0.0, 0.0],
            [0.3, -1.0, 1.2, 0.0, 0.0, 0.0],
            [0.3, -1.0, 1.2, 0.3, 0.0, 0.0],
            [0.3, -1.0, 1.2, 0.3, 0.3, 0.0],
        ],
    ];
    for (idx, wps) in cases.iter().enumerate() {
        let path =
            SRobotPath::<6, f64>::try_new(wps.iter().map(|w| SRobotQ::from_array(*w)).collect())
                .unwrap();
        let traj = r6
            .retime(&jl::<6>(v, a, j), &path, &v6, &())
            .0
            .unwrap_or_else(|e| panic!("case {idx}: {e}"));
        assert_clean(&traj, wps, v, a, j);
    }
}

#[test]
fn joint_fd_under_limits_fuzz() {
    let v6 = common::wide_validator::<6>();
    let r6 = Topp3Lp::<6>::new();
    let (v, a, j) = (1.5, 6.0, 250.0);
    let seeds = [
        0xDEADu64, 0xCAFE, 0xBADD, 0x1234, 0xFEED, 0x0F0F, 0xABCD, 0x9999,
    ];
    for &seed in &seeds {
        for delta in [0.3, 0.5] {
            let wps = fuzz::<6>(seed, [0.0, -0.8, 1.0, 0.0, 0.2, 0.5], 6, delta);
            let path = SRobotPath::<6, f64>::try_new(
                wps.iter().map(|w| SRobotQ::from_array(*w)).collect(),
            )
            .unwrap();
            let traj = r6
                .retime(&jl::<6>(v, a, j), &path, &v6, &())
                .0
                .unwrap_or_else(|e| panic!("seed {seed:#x} delta {delta}: {e}"));
            let (r, kind) = joint_fd_violation(&traj, v, a, j);
            assert!(
                r <= 1.0 + TOL,
                "seed {seed:#x} delta {delta}: {kind} {:.4}x",
                r
            );
            assert!(
                polyline_dev(&traj, &wps) < 1e-9,
                "seed {seed:#x}: off-chord"
            );
        }
    }
}

#[test]
fn tcp_velocity_under_cap_and_joints_clean() {
    let fk = common::dh_6dof();
    let rt = Topp3LpTcp::new(&fk);
    let v6 = common::wide_validator::<6>();
    let (v, a, j) = (5.0, 30.0, 3000.0);
    let cap = 0.25;
    let wps = vec![
        [0.0, -1.2, 1.5, -0.3, 0.5, 0.0],
        [0.6, -0.6, 0.9, 0.3, -0.2, 0.8],
    ];
    let path = SRobotPath::<6, f64>::try_new(wps.iter().map(|w| SRobotQ::from_array(*w)).collect())
        .unwrap();
    let cfg = jl::<6>(v, a, j).with_tcp_speed(cap);
    let traj = rt.retime(&cfg, &path, &v6, &()).0.expect("retime");

    // independent joint FD
    let (r, kind) = joint_fd_violation(&traj, v, a, j);
    assert!(r <= 1.0 + TOL, "joint {kind} {:.4}x", r);
    // independent TCP linear-speed backward FD via FK
    use deke_types::FKChain;
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let mut peak = 0.0f64;
    for i in 1..p.len() {
        let a = fk.fk_end(&p[i - 1]).unwrap().translation;
        let b = fk.fk_end(&p[i]).unwrap().translation;
        peak = peak.max(a.distance(b) / dt);
    }
    assert!(peak <= cap * (1.0 + TOL), "TCP speed {peak:.4} > cap {cap}");
    assert!(polyline_dev(&traj, &wps) < 1e-9, "off-chord");
}

#[test]
fn one_dof_fd_under_limits() {
    let r1 = Topp3Lp::<1>::new();
    let v1 = common::wide_validator::<1>();
    let wps = vec![[0.0], [1.0]];
    let path = SRobotPath::<1, f64>::try_new(wps.iter().map(|w| SRobotQ::from_array(*w)).collect())
        .unwrap();
    let traj = r1
        .retime(&jl::<1>(1.0, 2.0, 200.0), &path, &v1, &())
        .0
        .expect("retime");
    assert_clean(&traj, &wps, 1.0, 2.0, 200.0);
}
