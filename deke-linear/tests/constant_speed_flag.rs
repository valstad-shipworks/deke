mod common;

use deke_types::DekeError;
use deke_types::glam::DVec3;

/// A straight line is dip-free, so the flag changes nothing.
#[test]
fn straight_line_passes_with_flag() {
    let robot = common::ur();
    let poses = common::straight(&robot, DVec3::X, 0.12, 4);

    let (traj, _) = common::follow(
        &robot,
        &poses,
        &common::config_flag(0.05, true),
        &common::noop(),
        &(),
    )
    .expect("a straight line never dips in the interior");
    let speeds = common::tcp_speeds(&robot, &traj);
    let lo = speeds.len() / 4;
    let hi = speeds.len() * 3 / 4;
    for &v in &speeds[lo..hi] {
        assert!((v - 0.05).abs() < 0.05 * 0.1);
    }
}

/// A shallow corner under a command the joints can't sustain forces an interior
/// dip: the flag turns the (otherwise graceful) slowdown into a hard, located
/// error, while leaving it off slows through.
#[test]
fn interior_dip_is_forbidden_when_flag_set() {
    let robot = common::ur();
    let poses = common::corner(&robot, 0.05, 25f64.to_radians(), 4);
    let speed = 1.5; // above what the joints can sustain through the corner

    // Flag off: succeeds by slowing through the infeasible region.
    let (traj, diag_off) = common::follow(
        &robot,
        &poses,
        &common::config_flag(speed, false),
        &common::noop(),
        &(),
    )
    .expect("flag-off should slow through");
    assert_eq!(diag_off.runs, 1, "25° corner is one run");
    let speeds = common::tcp_speeds(&robot, &traj);
    let peak = speeds.iter().cloned().fold(0.0, f64::max);
    assert!(
        peak < speed,
        "the path cannot actually reach the command anywhere"
    );

    // Flag on: refuses, naming where and how fast it could actually go. The
    // structured `SpeedDipRequired` is collapsed to `DekeError` across the
    // `Retimer` trait boundary; its descriptive message survives.
    let err = common::follow(
        &robot,
        &poses,
        &common::config_flag(speed, true),
        &common::noop(),
        &(),
    )
    .unwrap_err();
    assert!(
        matches!(&err, DekeError::RetimerFailed(s) if s.contains("interior dip")),
        "expected SpeedDipRequired, got {err:?}"
    );
}
