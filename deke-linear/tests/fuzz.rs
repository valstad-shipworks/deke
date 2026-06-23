//! Property-based fuzzing of the three output criteria that matter for welding:
//! minimal TCP Cartesian deviation from the commanded path, joint v/a/j limits
//! that are never exceeded, and constant TCP speed outside the accel/decel
//! ramps. Each case is generated from a reproducible seed; a failure prints the
//! exact geometry and limits so it can be replayed.

mod common;

use deke_linear::{JointLimits, TcpLimits};
use deke_types::glam::DVec3;

/// A captured near-singular short weld (fuzz seed `0x5EED_0004`) whose joint
/// path bends sharply near the end. Either the retimer produces a trajectory
/// whose finite-difference joint jerk is within the limit, or it refuses with a
/// clear error — it must never emit an inexecutable path. (The discrete solver's
/// finite-difference verify catches the over-limit jerk here and refuses.)
#[test]
fn captured_short_weld_is_within_jerk_or_refused() {
    let robot = common::ur();
    let dir = DVec3::new(
        -0.204_193_764_827_666_16,
        -0.926_076_600_752_295_5,
        0.317_312_205_791_988_1,
    );
    let jl = 16.696_939_610_631_677;
    let poses = common::straight(&robot, dir, 0.118_001_859_153_488, 7);
    let mut cfg = common::config(0.119_636_760_917_933_6);
    cfg.constraints.joint = JointLimits::symmetric(0.364_662_007_797_896_65, 2.343_016_777_374_244_5, jl);
    match common::follow(&robot, &poses, &cfg, &common::noop(), &()) {
        Ok((traj, _)) => {
            let dt = traj.dt().as_secs_f64();
            let p = traj.path();
            let mut fd_j = 0.0f64;
            for i in 3..p.len() {
                for j in 0..6 {
                    fd_j = fd_j.max(
                        (p[i].0[j] - 3.0 * p[i - 1].0[j] + 3.0 * p[i - 2].0[j] - p[i - 3].0[j]).abs()
                            / (dt * dt * dt),
                    );
                }
            }
            assert!(fd_j <= jl, "returned trajectory FD jerk {fd_j:.2} exceeds limit {jl:.2}");
        }
        Err(e) => {
            assert!(format!("{e}").contains("jerk"), "expected a jerk-limit refusal, got: {e}");
        }
    }
}

/// Criterion 1 — the executed tool path must stay on the commanded line. Corner
/// smoothing may only round sub-millimetre sampling artefacts, never bow the
/// weld off its seam (this is the regression guard for the spline blow-up).
#[test]
fn fuzz_tcp_stays_on_commanded_line() {
    let robot = common::ur();
    let mut rng = common::Rng::new(0x5EED_0001);
    let mut cases = 0;
    for _ in 0..80 {
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
    assert!(cases > 30, "too few reachable cases ({cases}) — generator drifted");
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
    for _ in 0..80 {
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
            fd_v <= vl * (1.0 + 1e-6),
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
    assert!(cases > 5, "too few feasible cases ({cases})");
}

/// THE central guarantee — every *returned* trajectory's finite differences
/// (1st/2nd/3rd of the dt-sampled joint stream, exactly what a controller
/// reconstructs) stay within the per-joint limits; otherwise the retimer errors.
/// Swept across speed (spanning 30 ipm), tight↔loose joint limits, optional TCP
/// accel/jerk caps, and the constant-speed flag. Also checks the TCP speed never
/// exceeds the command, and TCP accel/jerk stay within their (chord-linearly
/// approximate) caps.
#[test]
fn fuzz_returned_output_respects_all_fd_limits() {
    let robot = common::ur();
    let mut rng = common::Rng::new(0x5EED_0010);
    let (mut ok, mut err) = (0, 0);
    for _ in 0..160 {
        let (vl, al, jl) = (rng.range(0.3, 2.0), rng.range(1.0, 8.0), rng.range(10.0, 80.0));
        let dir = rng.unit_dir();
        let len = rng.range(0.05, 0.25);
        let n = rng.int(2, 9);
        let speed = rng.range(0.008, 0.12); // spans 30 ipm = 0.0127 m/s
        let forbid = rng.unit() < 0.25;
        let tcp = if rng.unit() < 0.5 {
            Some((rng.range(0.01, 0.2), rng.range(0.3, 5.0)))
        } else {
            None
        };
        let poses = common::straight(&robot, dir, len, n);
        let mut cfg = common::config_flag(speed, forbid);
        cfg.constraints.joint = JointLimits::symmetric(vl, al, jl);
        if let Some((a, j)) = tcp {
            cfg.constraints.tcp = TcpLimits::new(speed, a, j);
        }
        match common::follow(&robot, &poses, &cfg, &common::noop(), &()) {
            Ok((traj, _)) => {
                ok += 1;
                // HARD guarantee (#3): joint FD never exceeds the true limits.
                let (fv, fa, fj) = (
                    common::joint_vel_peak(&traj),
                    common::joint_acc_peak(&traj),
                    common::joint_jerk_peak(&traj),
                );
                assert!(fv <= vl * (1.0 + 1e-6), "FD joint vel {fv:.4} > {vl:.4}");
                assert!(fa <= al * (1.0 + 1e-6), "FD joint accel {fa:.4} > {al:.4}");
                assert!(
                    fj <= jl * (1.0 + 1e-6),
                    "FD joint jerk {fj:.3} > {jl:.3} (speed={speed:.4} len={len:.3} n={n})",
                );
                // #1: TCP speed never exceeds the command.
                let pk = common::tcp_speeds(&robot, &traj).iter().cloned().fold(0.0, f64::max);
                assert!(pk <= speed * (1.0 + 1e-3), "TCP speed {pk:.5} > commanded {speed:.5}");
                // #1: TCP accel/jerk within caps (chord-linear approximation margin).
                if let Some((a, j)) = tcp {
                    // TCP accel/jerk are the *approximate* caps (chord-linear FK
                    // vs arc-length Δσ); accel lands under, jerk leaks a few %.
                    let (ta, tj) = common::tcp_accel_jerk_peak(&robot, &traj);
                    assert!(ta <= a * (1.0 + 1e-6), "TCP accel {ta:.4} > cap {a:.4}");
                    assert!(tj <= j * (1.0 + 1e-6), "TCP jerk {tj:.4} > cap {j:.4}");
                }
            }
            Err(_) => err += 1,
        }
    }
    assert!(ok > 90, "solver over-refusing: only {ok} ok / {err} err");
}

/// Multi-segment shallow-corner welds (single run, no sharp-corner split) — the
/// FD joint limits must hold across the bends too, or the retimer errors.
#[test]
fn fuzz_shallow_corners_respect_fd_limits() {
    let robot = common::ur();
    let mut rng = common::Rng::new(0x5EED_0011);
    let mut ok = 0;
    for _ in 0..100 {
        let leg = rng.range(0.04, 0.12);
        let turn = rng.range(-0.4, 0.4); // ±23° < 30° split threshold → single run
        let per = rng.int(3, 6);
        let speed = rng.range(0.01, 0.08);
        let (vl, al, jl) = (rng.range(0.4, 2.0), rng.range(1.5, 8.0), rng.range(15.0, 80.0));
        let poses = common::corner(&robot, leg, turn, per);
        let mut cfg = common::config(speed);
        cfg.constraints.joint = JointLimits::symmetric(vl, al, jl);
        let Ok((traj, out)) = common::follow(&robot, &poses, &cfg, &common::noop(), &()) else {
            continue;
        };
        if out.runs != 1 {
            continue; // skip if split into multiple runs (concat seam is the caller's concern)
        }
        ok += 1;
        assert!(common::joint_vel_peak(&traj) <= vl * (1.0 + 1e-6), "corner FD vel over limit");
        assert!(common::joint_acc_peak(&traj) <= al * (1.0 + 1e-6), "corner FD accel over limit");
        assert!(common::joint_jerk_peak(&traj) <= jl * (1.0 + 1e-6), "corner FD jerk over limit");
    }
    assert!(ok > 20, "too few single-run corners ({ok})");
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
    for _ in 0..80 {
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
            peak <= speed * (1.0 + 1e-3),
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
    assert!(cases > 5, "too few cruising cases ({cases})");
}
