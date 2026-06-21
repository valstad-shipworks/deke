mod common;

use deke_topp3tcp_nlp::continuous::{SolveStatus, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{DekeError, Retimer, SRobotPath, SRobotQ};

#[test]
fn locks_first_two_joints_across_trajectory() {
    let fk = common::dh_6dof();
    let a = SRobotQ::from_array([0.2, -0.9, 1.3, 0.0, 0.0, 0.0]);
    let b = SRobotQ::from_array([0.2, -0.9, 0.9, 0.3, 0.2, 0.4]);
    let path = SRobotPath::<6, f64>::try_new(vec![a, b]).unwrap();

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 4.0, 400.0);
    cfg.locked_prefix = 2;

    let validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6::new(&fk).retime(&cfg, &path, &validator, &());
    eprintln!("{}", diag);
    let traj = result.expect("retime failed");
    assert_eq!(diag.status, SolveStatus::Success);

    // Every sample must preserve joints 0 and 1 at their starting values.
    for q in traj.iter() {
        assert!((q.0[0] - 0.2).abs() < 1e-4, "joint 0 drifted: {}", q.0[0]);
        assert!((q.0[1] - (-0.9)).abs() < 1e-4, "joint 1 drifted: {}", q.0[1]);
    }
}

#[test]
fn mismatched_locked_prefix_errors() {
    let fk = common::dh_6dof();
    let a = SRobotQ::from_array([0.2, -0.9, 1.3, 0.0, 0.0, 0.0]);
    let b = SRobotQ::from_array([0.3, -0.9, 0.9, 0.3, 0.2, 0.4]); // joint 0 moves
    let path = SRobotPath::<6, f64>::try_new(vec![a, b]).unwrap();

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 4.0, 400.0);
    cfg.locked_prefix = 2;

    let validator = common::wide_validator::<6>();
    let (result, _diag) = Topp3Tcp6::new(&fk).retime(&cfg, &path, &validator, &());
    match result {
        Err(DekeError::LockedPrefixViolation { waypoint: _, joint: 0 }) => {}
        other => panic!("expected LockedPrefixViolation on joint 0, got {:?}", other),
    }
}
