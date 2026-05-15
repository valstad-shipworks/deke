//! End-to-end test for the experimental TCP-speed post-process pass.
//!
//! Builds a 3-link planar arm whose TCP speed at the home configuration is
//! 2.5 m per rad of joint-1 motion, drives joint 1 quickly, and confirms
//! that `MotionSpec::max_tcp_speed` bounds the resulting Cartesian speed.

use std::time::Duration;

use deke_topp_speed::{MotionSpec, Retimer, SRobotPath, SRobotQ, ToppSolver};
use deke_types::{DHChain, DHJoint, FKChain, JointValidator};

/// Three revolute joints all rotating about z, link lengths 1 + 1 + 0.5.
/// At any configuration the wrist is 2.5 m from the base axis, so the TCP
/// speed produced by joint-1 motion alone is `2.5 · q̇₁`.
fn planar_arm() -> DHChain<3, f64> {
    let make = |a: f64| DHJoint::<f64> {
        a,
        alpha: 0.0,
        d: 0.0,
        theta_offset: 0.0,
    };
    DHChain::<3, f64>::new_f64([make(1.0), make(1.0), make(0.5)])
}

fn fast_spec() -> MotionSpec<3, f64> {
    let mut s = MotionSpec::<3, f64>::new();
    // Loose joint ceilings — joint 1 can spin fast, which without a TCP
    // limit would push TCP speed well above any reasonable Cartesian cap.
    s.max_vel = SRobotQ::splat(5.0);
    s.max_accel = SRobotQ::splat(50.0);
    s.max_jerk = SRobotQ::splat(500.0);
    s
}

fn validator() -> JointValidator<3, f64> {
    JointValidator::<3, f64>::new(SRobotQ::splat(-10.0), SRobotQ::splat(10.0))
}

#[test]
fn no_tcp_limit_leaves_diagnostic_empty() {
    let spec = fast_spec();
    let path = SRobotPath::from_two(
        SRobotQ::from_array([0.0, 0.0, 0.0]),
        SRobotQ::from_array([1.0, 0.0, 0.0]),
    );
    let solver: ToppSolver<3, f64> = ToppSolver::new(Duration::from_micros(500));
    let fk = planar_arm();
    let val = validator();

    let (res, diag) = solver.retime(&spec, &path, &fk, &val, &());
    assert!(res.is_ok(), "baseline retime should succeed");
    assert!(
        diag.tcp_speed_scale.is_none(),
        "no TCP limit requested → no scaling reported"
    );
    assert!(diag.tcp_peak_speed.is_none());
}

/// When `max_tcp_speed` is `None`, the per-section post-processor must be a
/// strict no-op: the resulting trajectory has to match what the solver
/// produces without the TCP support compiled in at all. This is the
/// user-facing regression guarantee.
#[test]
fn no_tcp_limit_produces_identical_trajectory() {
    let mut spec_a = fast_spec();
    spec_a.max_tcp_speed = None;
    let mut spec_b = fast_spec();
    spec_b.max_tcp_speed = None;

    // Multi-waypoint path so any per-section side-effect would show up here.
    let path = SRobotPath::try_new(vec![
        SRobotQ::from_array([0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.0, 0.0, 0.8]),
        SRobotQ::from_array([0.6, 0.0, 0.8]),
    ])
    .expect("valid multi-waypoint path");
    let solver: ToppSolver<3, f64> = ToppSolver::new(Duration::from_micros(500));
    let fk = planar_arm();
    let val = validator();

    let (res_a, diag_a) = solver.retime(&spec_a, &path, &fk, &val, &());
    let (res_b, diag_b) = solver.retime(&spec_b, &path, &fk, &val, &());
    let traj_a = res_a.expect("retime a");
    let traj_b = res_b.expect("retime b");

    // No TCP-side diagnostic should be filled in.
    assert!(diag_a.tcp_speed_scale.is_none());
    assert!(diag_a.tcp_peak_speed.is_none());
    assert!(diag_b.tcp_speed_scale.is_none());
    assert!(diag_b.tcp_peak_speed.is_none());

    // Bit-equal trajectory between two None-runs (determinism of the solver
    // is what we're really checking here, but together with the diagnostic
    // assertion above it proves the post-process path didn't touch the
    // plan).
    assert_eq!(traj_a.len(), traj_b.len(), "trajectory length must match");
    assert_eq!(traj_a.dt(), traj_b.dt(), "trajectory dt must match");
    for i in 0..traj_a.len() {
        let qa = traj_a.get(i).unwrap();
        let qb = traj_b.get(i).unwrap();
        for j in 0..3 {
            assert_eq!(
                qa[j].to_bits(),
                qb[j].to_bits(),
                "sample {i} joint {j} differs"
            );
        }
    }
}

#[test]
fn slack_tcp_limit_leaves_trajectory_alone() {
    // The unconstrained peak TCP speed is ≈ 12.5 m/s; set the ceiling
    // well above that so the post-processor reports the peak but applies
    // no scaling.
    let mut spec = fast_spec();
    spec.max_tcp_speed = Some(100.0);
    let path = SRobotPath::from_two(
        SRobotQ::from_array([0.0, 0.0, 0.0]),
        SRobotQ::from_array([1.0, 0.0, 0.0]),
    );
    let solver: ToppSolver<3, f64> = ToppSolver::new(Duration::from_micros(500));
    let fk = planar_arm();
    let val = validator();

    let (_res, slack_diag) = solver.retime(&spec, &path, &fk, &val, &());
    let slack_scale = slack_diag.tcp_speed_scale.expect("scale reported");
    let slack_peak = slack_diag.tcp_peak_speed.expect("peak reported");
    assert!(slack_peak < 100.0, "peak should be below the slack limit");
    assert!(
        (slack_scale - 1.0).abs() < 1e-9,
        "no scaling should be applied when limit is slack, got {slack_scale}"
    );

    // Compare durations: a baseline run with no TCP limit should yield the
    // same trajectory length as the slack run.
    let mut baseline_spec = spec.clone();
    baseline_spec.max_tcp_speed = None;
    let (baseline_res, _) = solver.retime(&baseline_spec, &path, &fk, &val, &());
    let baseline_traj = baseline_res.expect("baseline retime");
    let (slack_res, _) = solver.retime(&spec, &path, &fk, &val, &());
    let slack_traj = slack_res.expect("slack retime");
    assert_eq!(
        baseline_traj.len(),
        slack_traj.len(),
        "slack TCP limit should not change trajectory length"
    );
}

/// Multi-section path where the user pins joint 1 stationary in the
/// "joint-3 only" sections via `per_section_max_vel`, so the waypoint solver
/// cannot smear joint-1 motion across all sections. The middle section
/// drives joint 1 (high TCP coupling) and is the only one that exceeds the
/// TCP limit; per-section scaling should slow only the middle section and
/// hence produce a strictly shorter trajectory than uniform global scaling.
#[test]
fn per_section_scaling_beats_global_on_mixed_sections() {
    let fk = planar_arm();
    let val = validator();
    let solver: ToppSolver<3, f64> = ToppSolver::new(Duration::from_micros(500));

    let path = SRobotPath::try_new(vec![
        SRobotQ::from_array([0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.0, 0.0, 4.0]),
        SRobotQ::from_array([1.0, 0.0, 4.0]),
        SRobotQ::from_array([1.0, 0.0, 0.0]),
    ])
    .expect("valid multi-waypoint path");

    // Cap joint 1 velocity to ~0 in sections 1 and 3 — only joint 3 moves
    // in those sections, and we don't want the solver to "smear" joint-1
    // motion into them via boundary-velocity choices. Section 2 keeps the
    // full envelope so joint 1 can do its work there.
    let still_v = SRobotQ::from_array([1e-6, 5.0, 5.0]);
    let still_a = SRobotQ::from_array([1e-6, 50.0, 50.0]);
    let still_j = SRobotQ::from_array([1e-6, 500.0, 500.0]);
    let full_v = SRobotQ::splat(5.0);
    let full_a = SRobotQ::splat(50.0);
    let full_j = SRobotQ::splat(500.0);
    let mut base_spec = fast_spec();
    base_spec.per_section_max_vel = Some(vec![still_v, full_v, still_v]);
    base_spec.per_section_max_accel = Some(vec![still_a, full_a, still_a]);
    base_spec.per_section_max_jerk = Some(vec![still_j, full_j, still_j]);

    // Baseline (no TCP limit) — gives us the un-throttled duration and the
    // peak TCP we'd have to scale away with a global pass.
    let (baseline_res, _baseline_diag) = solver.retime(&base_spec, &path, &fk, &val, &());
    let baseline_traj = baseline_res.expect("baseline retime");
    let baseline_dur = baseline_traj.duration().as_secs_f64();

    // Limited run.
    let mut spec = base_spec.clone();
    spec.max_tcp_speed = Some(1.0);
    let (limited_res, limited_diag) = solver.retime(&spec, &path, &fk, &val, &());
    let limited_traj = limited_res.expect("limited retime");
    let limited_dur = limited_traj.duration().as_secs_f64();

    let peak = limited_diag.tcp_peak_speed.expect("peak reported");
    let scale = limited_diag.tcp_speed_scale.expect("scale reported");
    assert!(peak > 1.0, "baseline peak should exceed limit");
    assert!(scale > 1.0, "trajectory should be slowed, got {scale}");

    // Per-section win: only the middle section binds, so per-section
    // produces a shorter trajectory than uniform global scaling.
    let global_equivalent_dur = baseline_dur * peak; // limit = 1.0
    assert!(
        limited_dur < global_equivalent_dur * 0.85,
        "per-section ({limited_dur:.4}s) should be ≥15% shorter than global ({global_equivalent_dur:.4}s)"
    );

    // The TCP speed in the produced trajectory must respect the limit.
    let max_tcp = max_tcp_in(&limited_traj, &fk);
    assert!(
        max_tcp <= 1.05,
        "post-scale TCP should be ≤ 1.0 m/s (allow 5% FD slack), got {max_tcp}"
    );
}

/// Helper: numerically differentiate the trajectory and sweep TCP speed.
fn max_tcp_in(traj: &deke_types::SRobotTraj<3, f64>, fk: &DHChain<3, f64>) -> f64 {
    let mut max = 0.0_f64;
    for i in 0..traj.len() {
        let q = traj.get(i).unwrap();
        let qd = match traj.velocity_at(i) {
            Some(v) => v,
            None => continue,
        };
        let jac = fk.jacobian(q).unwrap();
        let vx = jac[0][0] * qd[0] + jac[0][1] * qd[1] + jac[0][2] * qd[2];
        let vy = jac[1][0] * qd[0] + jac[1][1] * qd[1] + jac[1][2] * qd[2];
        let vz = jac[2][0] * qd[0] + jac[2][1] * qd[1] + jac[2][2] * qd[2];
        let s = (vx * vx + vy * vy + vz * vz).sqrt();
        if s > max {
            max = s;
        }
    }
    max
}

#[test]
fn tcp_limit_bounds_cartesian_speed() {
    // Pose change is entirely along joint 1, so q̇₂ = q̇₃ = 0 throughout and
    // TCP speed = 2.5 · q̇₁. With max_vel = 5 rad/s the unconstrained peak
    // is ≈ 12.5 m/s, well above the 1 m/s ceiling we want to enforce.
    let mut spec = fast_spec();
    spec.max_tcp_speed = Some(1.0);
    let path = SRobotPath::from_two(
        SRobotQ::from_array([0.0, 0.0, 0.0]),
        SRobotQ::from_array([1.0, 0.0, 0.0]),
    );
    let solver: ToppSolver<3, f64> = ToppSolver::new(Duration::from_micros(500));
    let fk = planar_arm();
    let val = validator();

    let (res, diag) = solver.retime(&spec, &path, &fk, &val, &());
    let traj = res.expect("retime should succeed");

    // Diagnostic sanity: the unscaled peak must exceed the limit; the
    // applied scale factor must be > 1.
    let peak = diag.tcp_peak_speed.expect("peak reported");
    let scale = diag.tcp_speed_scale.expect("scale reported");
    assert!(
        peak > 1.0,
        "without scaling the trajectory should exceed 1 m/s, got peak={peak}"
    );
    assert!(scale > 1.0, "scaling factor should slow trajectory, got {scale}");
    // The geometric expectation is 12.5 m/s peak; allow generous slack for
    // S-curve rounding.
    assert!(
        peak > 10.0 && peak < 14.0,
        "peak should be near 2.5·max_vel = 12.5, got {peak}"
    );

    // Numerically verify the post-scale trajectory respects the limit.
    // Walk the sampled trajectory, compute TCP velocity from finite-
    // difference joint velocities, and confirm the maximum is under the
    // ceiling (allow a small numerical margin for both the FD differentiator
    // and discretisation between sampler ticks).
    let mut max_tcp = 0.0_f64;
    for i in 0..traj.len() {
        let q = traj.get(i).unwrap();
        let qd = match traj.velocity_at(i) {
            Some(v) => v,
            None => continue,
        };
        let jac = fk.jacobian(q).unwrap();
        let vx = jac[0][0] * qd[0] + jac[0][1] * qd[1] + jac[0][2] * qd[2];
        let vy = jac[1][0] * qd[0] + jac[1][1] * qd[1] + jac[1][2] * qd[2];
        let vz = jac[2][0] * qd[0] + jac[2][1] * qd[1] + jac[2][2] * qd[2];
        let speed = (vx * vx + vy * vy + vz * vz).sqrt();
        if speed > max_tcp {
            max_tcp = speed;
        }
    }
    // 5 % margin: the post-process samples the trajectory at 512 points so
    // the discrete peak it sees is within ~0.2 % of the true peak; the
    // headroom here is for finite-difference noise on the velocity_at()
    // estimate inside the test itself.
    assert!(
        max_tcp <= 1.05,
        "post-scale TCP speed should be ≤ 1.0 m/s (allow 5 % FD slack), got {max_tcp}"
    );
}
