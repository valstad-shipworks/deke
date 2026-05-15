mod common;

use deke_topp3tcp6_discrete::{SolveStatus, Topp3Tcp6Discrete, Topp3Tcp6DiscreteConstraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
fn multi_waypoint_curved_path_solves_and_is_feasible() {
    let fk = common::dh_6dof();
    let waypoints = vec![
        SRobotQ::from_array([0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.2, -1.1, 1.3, -0.1, 0.1, 0.1]),
        SRobotQ::from_array([0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::from_array([0.6, -0.7, 0.9, -0.3, 0.1, 0.3]),
        SRobotQ::from_array([0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<6, f64>::try_new(waypoints).unwrap();

    let cfg = Topp3Tcp6DiscreteConstraints::<6>::symmetric(1.5, 8.0, 400.0);
    let validator = common::wide_validator::<6>();

    let (result, diag) = Topp3Tcp6Discrete.retime(&cfg, &path, &fk, &validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "retime failed: {}", diag);
    assert_eq!(diag.status, SolveStatus::Success);

    // Strict FD residual ≈ 0.
    assert!(
        diag.output_fd_residual.joint_v <= 1e-3,
        "joint_v residual {:+e}",
        diag.output_fd_residual.joint_v
    );
}
