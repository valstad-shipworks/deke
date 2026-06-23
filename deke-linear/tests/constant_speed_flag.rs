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

/// A shallow corner under a command the joints can't sustain through it. Neither
/// mode may emit an inexecutable trajectory: with the flag set it refuses up
/// front because it cannot hold constant speed; without it, the slowdown still
/// cannot round the corner under the jerk limit at this speed, so it refuses
/// there instead. Both errors are collapsed to `DekeError` across the `Retimer`
/// trait boundary; their descriptive messages survive.
#[test]
fn corner_too_fast_is_refused_not_emitted() {
    let robot = common::ur();
    let poses = common::corner(&robot, 0.05, 25f64.to_radians(), 4);
    let speed = 1.5; // above what the joints can sustain through the corner

    // Flag on: the constant-speed contract can't hold, so it names where and how
    // fast it could actually go.
    let err_on = common::follow(
        &robot,
        &poses,
        &common::config_flag(speed, true),
        &common::noop(),
        &(),
    )
    .unwrap_err();
    assert!(
        matches!(&err_on, DekeError::RetimerFailed(s) if s.contains("interior dip")),
        "expected SpeedDipRequired, got {err_on:?}"
    );

    // Flag off: it would slow through, but the corner is too sharp to round under
    // the jerk limit at this speed — so it still refuses rather than emit a
    // trajectory the arm cannot execute.
    let err_off = common::follow(
        &robot,
        &poses,
        &common::config_flag(speed, false),
        &common::noop(),
        &(),
    )
    .unwrap_err();
    assert!(
        matches!(&err_off, DekeError::RetimerFailed(s) if s.contains("jerk limit")),
        "expected a jerk-limit error, got {err_off:?}"
    );
}
