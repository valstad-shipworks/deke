//! Property-based fuzzing of the three output criteria that matter for welding:
//! minimal TCP Cartesian deviation from the commanded path, joint v/a/j limits
//! that are never exceeded, and constant TCP speed outside the accel/decel
//! ramps. Each case is generated from a reproducible seed; a failure prints the
//! exact geometry and limits so it can be replayed.

mod common;

use deke_linear::JointLimits;
use deke_types::glam::DVec3;

/// A captured near-singular case (fuzz seed `0x5EED_0004`) whose joint curvature
/// demands more jerk than the limit allows. The solver must surface the
/// jerk-limit error rather than return an inexecutable trajectory.
#[test]
fn jerk_overrun_is_reported_not_emitted() {
    let robot = common::ur();
    let dir = DVec3::new(
        -0.204_193_764_827_666_16,
        -0.926_076_600_752_295_5,
        0.317_312_205_791_988_1,
    );
    let poses = common::straight(&robot, dir, 0.118_001_859_153_488, 7);
    let mut cfg = common::config(0.119_636_760_917_933_6);
    cfg.constraints.joint = JointLimits::symmetric(
        0.364_662_007_797_896_65,
        2.343_016_777_374_244_5,
        16.696_939_610_631_677,
    );
    let err = common::follow(&robot, &poses, &cfg, &common::noop(), &()).unwrap_err();
    assert!(
        format!("{err}").contains("jerk limit"),
        "expected a jerk-limit error, got: {err}",
    );
}

/// Criterion 1 — the executed tool path must stay on the commanded line. Corner
/// smoothing may only round sub-millimetre sampling artefacts, never bow the
/// weld off its seam (this is the regression guard for the spline blow-up).
#[test]
fn fuzz_tcp_stays_on_commanded_line() {
    let robot = common::ur();
    let mut rng = common::Rng::new(0x5EED_0001);
    let mut cases = 0;
    for _ in 0..600 {
        let dir = rng.unit_dir();
        let len = rng.range(0.05, 0.20);
        let n = rng.int(2, 8);
        let speed = rng.range(0.01, 0.06);
        let poses = common::straight(&robot, dir, len, n);
        let Ok((traj, _)) =
            common::follow(&robot, &poses, &common::config(speed), &common::noop(), &())
        else {
            continue;
        };
        cases += 1;
        let dev = common::tcp_polyline_deviation(&robot, &traj, &poses);
        assert!(
            dev < 1e-3,
            "TCP deviation {dev:.6} m — dir={dir:?} len={len:.3} n={n} speed={speed:.4}",
        );
    }
    assert!(cases > 400, "too few reachable cases ({cases}) — generator drifted");
}

/// Criterion 2 — never exceed joint limits. With tight, randomised limits and
/// `forbid_interior_dips` (so only feasible, well-conditioned motions are timed),
/// the continuous joint accel/jerk the integrator bounds stay under the true
/// limits — `LIMIT_MARGIN` absorbs the discrete-sampling overshoot — and the
/// sampled joint velocity stays under its ceiling.
#[test]
fn fuzz_feasible_motion_never_exceeds_joint_limits() {
    let robot = common::ur();
    let mut rng = common::Rng::new(0x5EED_0002);
    let mut cases = 0;
    for _ in 0..900 {
        let (vl, al, jl) = (rng.range(0.4, 1.6), rng.range(1.5, 6.0), rng.range(15.0, 60.0));
        let dir = rng.unit_dir();
        let len = rng.range(0.05, 0.20);
        let n = rng.int(2, 8);
        let speed = rng.range(0.02, 0.10);
        let poses = common::straight(&robot, dir, len, n);
        let mut cfg = common::config_flag(speed, true);
        cfg.constraints.joint = JointLimits::symmetric(vl, al, jl);
        let Ok((traj, out)) = common::follow(&robot, &poses, &cfg, &common::noop(), &()) else {
            continue;
        };
        cases += 1;
        let fd_v = common::joint_vel_peak(&traj);
        assert!(
            fd_v <= vl * 1.03,
            "joint velocity {fd_v:.4} > limit {vl:.4} — dir={dir:?} len={len:.3} speed={speed:.4}",
        );
        for r in &out.retimer {
            assert!(
                r.peak_joint_accel <= al,
                "continuous joint accel {:.4} > limit {al:.4} — dir={dir:?} len={len:.3}",
                r.peak_joint_accel,
            );
            assert!(
                r.peak_joint_jerk <= jl,
                "continuous joint jerk {:.4} > limit {jl:.4} — dir={dir:?} len={len:.3}",
                r.peak_joint_jerk,
            );
        }
    }
    assert!(cases > 300, "too few feasible cases ({cases})");
}

/// Criterion 2, unconditional form — a *returned* trajectory is never over any
/// joint limit. Without `forbid_interior_dips`, near-singular curvature can
/// demand more joint accel/jerk than the limits allow; the solver must then fail
/// rather than emit an inexecutable path. So for every case: either it errors, or
/// the continuous joint accel and jerk are within their limits (the integrator
/// enforces velocity directly). The sampled FD velocity must also stay bounded.
#[test]
fn fuzz_returned_trajectory_never_exceeds_a_joint_limit() {
    let robot = common::ur();
    let mut rng = common::Rng::new(0x5EED_0004);
    let mut ok = 0;
    for _ in 0..1200 {
        let (vl, al, jl) = (rng.range(0.3, 1.5), rng.range(1.0, 5.0), rng.range(10.0, 50.0));
        let dir = rng.unit_dir();
        let len = rng.range(0.05, 0.20);
        let n = rng.int(2, 8);
        let speed = rng.range(0.02, 0.12);
        let poses = common::straight(&robot, dir, len, n);
        let mut cfg = common::config(speed);
        cfg.constraints.joint = JointLimits::symmetric(vl, al, jl);
        let Ok((traj, out)) = common::follow(&robot, &poses, &cfg, &common::noop(), &()) else {
            continue;
        };
        ok += 1;
        let fd_v = common::joint_vel_peak(&traj);
        assert!(fd_v <= vl * 1.03, "returned trajectory over velocity limit: {fd_v:.4} > {vl:.4}");
        for r in &out.retimer {
            assert!(
                r.peak_joint_accel <= al,
                "returned trajectory over accel limit: {:.3} > {al:.3} — dir={dir:?} len={len:.3} speed={speed:.4}",
                r.peak_joint_accel,
            );
            assert!(
                r.peak_joint_jerk <= jl,
                "returned trajectory over jerk limit: {:.2} > {jl:.2} — dir={dir:?} len={len:.3} speed={speed:.4}",
                r.peak_joint_jerk,
            );
        }
    }
    assert!(ok > 500, "suspiciously few successes ({ok})");
}

/// Criterion 3 — constant TCP speed outside the ramps. When the command is
/// feasible everywhere (`forbid_interior_dips` succeeds), the cruise plateau
/// holds the command flat: no interior notch and no overshoot beyond sampling
/// noise. This is the regression guard for the single-sample speed glitch.
#[test]
fn fuzz_constant_speed_during_cruise() {
    let robot = common::ur();
    let mut rng = common::Rng::new(0x5EED_0003);
    let mut cases = 0;
    for _ in 0..900 {
        let (vl, al, jl) = (rng.range(0.5, 1.6), rng.range(2.0, 6.0), rng.range(20.0, 60.0));
        let dir = rng.unit_dir();
        let len = rng.range(0.06, 0.20);
        let n = rng.int(2, 6);
        let speed = rng.range(0.02, 0.08);
        let poses = common::straight(&robot, dir, len, n);
        let mut cfg = common::config_flag(speed, true);
        cfg.constraints.joint = JointLimits::symmetric(vl, al, jl);
        let Ok((traj, _)) = common::follow(&robot, &poses, &cfg, &common::noop(), &()) else {
            continue;
        };
        let speeds = common::tcp_speeds(&robot, &traj);
        let peak = speeds.iter().cloned().fold(0.0, f64::max);
        if peak < speed * 0.99 {
            continue; // too short to reach cruise — nothing to hold flat
        }
        cases += 1;
        assert!(
            peak <= speed * 1.02,
            "TCP speed overshoot: peak {peak:.5} vs commanded {speed:.5}",
        );
        let thr = speed * 0.98;
        let first = speeds.iter().position(|&v| v >= thr).unwrap();
        let last = speeds.iter().rposition(|&v| v >= thr).unwrap();
        let notch = speeds[first..=last].iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            notch >= speed * 0.95,
            "cruise dipped to {notch:.5} (commanded {speed:.5}) — dir={dir:?} len={len:.3} n={n}",
        );
    }
    assert!(cases > 300, "too few cruising cases ({cases})");
}
