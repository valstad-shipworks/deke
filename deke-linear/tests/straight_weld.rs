mod common;

use deke_linear::TcpLimits;
use deke_types::glam::DVec3;

#[test]
fn ik_is_analytic() {
    let robot = common::ur();
    assert!(
        robot.ik_diagnostic().viable,
        "test chain must have viable analytic IK"
    );
}

#[test]
fn straight_weld_holds_constant_speed_within_limits() {
    let robot = common::ur();
    let poses = common::straight(&robot, DVec3::X, 0.12, 4);
    let cfg = common::config(0.05);

    let (traj, diag) = common::follow(&robot, &poses, &cfg, &common::noop(), &())
        .expect("follow failed");

    assert_eq!(diag.runs, 1, "a straight line is one run");
    assert!(
        traj.path().len() > 10,
        "expected a densely sampled trajectory"
    );

    let speeds = common::tcp_speeds(&robot, &traj);
    assert!(!speeds.is_empty());

    // Interior (cruise) speeds should sit flat at the commanded speed.
    let lo = speeds.len() / 4;
    let hi = speeds.len() * 3 / 4;
    for &v in &speeds[lo..hi] {
        assert!(
            (v - 0.05).abs() < 0.05 * 0.1,
            "cruise speed {v} should be within 10% of commanded 0.05 m/s"
        );
    }

    // Joint velocity / acceleration stay within limits (small FD tolerance).
    assert!(
        common::joint_vel_peak(&traj) <= 2.0 * 1.05,
        "joint velocity exceeded limit"
    );
    assert!(
        common::joint_acc_peak(&traj) <= 8.0 * 1.20,
        "joint acceleration exceeded limit"
    );
    // Continuous chain-rule joint accel/jerk (the quantities the retimer bounds)
    // stay within the joint limits. The FD third difference of the dt-sampled
    // output can read higher at the unavoidable jerk steps of a jerk-limited
    // (not snap-limited) profile, so it is not the right gauge here.
    let r = &diag.retimer[0];
    assert!(
        r.peak_joint_accel <= 8.0 * 1.001,
        "continuous joint accel {:.3} exceeds limit 8",
        r.peak_joint_accel,
    );
    assert!(
        r.peak_joint_jerk <= 80.0 * 1.001,
        "continuous joint jerk {:.3} exceeds limit 80",
        r.peak_joint_jerk,
    );
}

#[test]
fn starts_and_ends_at_rest() {
    let robot = common::ur();
    // A short, coarse weld (the old finite-difference q''' spiked its ramp jerk
    // ~8× and forced a failure here; analytic derivatives make it feasible).
    let poses = common::straight(&robot, DVec3::X, 0.10, 3);
    let cfg = common::config(0.04);
    let (traj, _) = common::follow(&robot, &poses, &cfg, &common::noop(), &()).unwrap();

    let speeds = common::tcp_speeds(&robot, &traj);
    assert!(
        speeds.first().copied().unwrap() < 0.04 * 0.5,
        "should ramp up from rest"
    );
    assert!(
        speeds.last().copied().unwrap() < 0.04 * 0.5,
        "should ramp down to rest"
    );
}

/// Coarse input waypoints at a slow feedrate must not inflate the trajectory.
/// The terminal deceleration has to take the physical jerk-limited stopping
/// distance (a few mm), not the whole final segment: pinning the end to rest
/// and interpolating the velocity ceiling linearly in arc drags `v` to zero
/// across the entire segment, whose traversal time diverges. The fixed retimer
/// holds within ~0.5% of the cruise time here; the crawl inflated it past 30%.
#[test]
fn coarse_knots_do_not_inflate_duration() {
    let robot = common::ur();
    // 0.05 m over 6 vertices (10 mm segments) at a slow 0.01 m/s feedrate —
    // segments far longer than the millimetre-scale jerk stopping distance.
    let poses = common::straight(&robot, DVec3::X, 0.05, 6);
    let cfg = common::config(0.01);
    let (_, out) = common::follow(&robot, &poses, &cfg, &common::noop(), &()).unwrap();

    let r = &out.retimer[0];
    let cruise_time = r.arc_length / r.commanded_speed;
    assert!(
        r.duration.as_secs_f64() < 1.15 * cruise_time,
        "coarse-knot weld inflated to {:.2}s vs cruise {:.2}s — terminal decel crawl?",
        r.duration.as_secs_f64(),
        cruise_time,
    );
}

/// Corner smoothing must not bow the executed path: a coincident knot in the
/// planned path makes the natural-cubic-spline system singular, which inflates
/// the Cartesian arc length. A straight 0.2 m move must stay ~0.2 m.
#[test]
fn smoothing_preserves_arc_length() {
    let robot = common::ur();
    let poses = common::straight(&robot, DVec3::X, 0.2, 9);
    let cfg = common::config(0.05);
    let (_, out) = common::follow(&robot, &poses, &cfg, &common::noop(), &()).unwrap();

    let arc = out.retimer[0].arc_length;
    assert!(
        arc < 0.2 * 1.02,
        "smoothing inflated arc length to {arc:.4} m (straight move is 0.2 m)",
    );
}

/// An explicit Cartesian TCP acceleration cap, far below what the joints would
/// permit at this feedrate, must bound the realized tangential TCP acceleration.
/// The integrator clamps the continuous path accel `s̈` to the cap; the second
/// difference of the dt-sampled output reads a little higher (same discrete-FD
/// overshoot the joint-accel check tolerates), so the bound carries a margin.
#[test]
fn respects_tcp_accel_cap() {
    let robot = common::ur();
    let poses = common::straight(&robot, DVec3::X, 0.2, 9);
    let mut cfg = common::config(0.05);
    let accel = 0.02;
    cfg.constraints.tcp = TcpLimits::new(cfg.constraints.tcp.speed, accel, 0.5);

    let (traj, _) = common::follow(&robot, &poses, &cfg, &common::noop(), &()).unwrap();

    let speeds = common::tcp_speeds(&robot, &traj);
    let dt = cfg.constraints.output_dt.as_secs_f64();
    let peak_accel = speeds
        .windows(2)
        .map(|w| ((w[1] - w[0]) / dt).abs())
        .fold(0.0f64, f64::max);
    assert!(
        peak_accel <= accel * 1.30,
        "TCP tangential accel {peak_accel:.4} exceeds cap {accel} m/s²",
    );
}
