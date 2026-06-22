mod common;

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
}

#[test]
fn starts_and_ends_at_rest() {
    let robot = common::ur();
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
