mod common;

use deke_topp3tcp_nlp::continuous::{BoundaryConditions, SolveStatus, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{DekeError, Retimer, SRobotPath, SRobotQ};

#[test]
fn aligned_non_zero_velocity_is_feasible() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([1.0]),
    ])
    .unwrap();

    let mut cfg = Topp3Tcp6Constraints::<1>::symmetric(1.0, 2.0, 200.0);
    cfg.boundary = BoundaryConditions {
        v_start: SRobotQ::from_array([0.4]),
        a_start: SRobotQ::from_array([0.0]),
        v_end: SRobotQ::from_array([0.4]),
        a_end: SRobotQ::from_array([0.0]),
        projection_tolerance: 1e-3,
    };

    let validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6::new(&fk).retime(&cfg, &path, &validator, &());
    eprintln!("{}", diag);
    let traj = result.expect("retime failed");
    assert_eq!(diag.status, SolveStatus::Success);

    // First and last velocity samples of the output trajectory should match the requested
    // boundary. Use the trajectory's own finite-difference velocity.
    let v0 = traj.velocity_at(0).unwrap().0[0];
    let vn = traj.velocity_at(traj.len() - 1).unwrap().0[0];
    assert!((v0 - 0.4).abs() < 0.15, "start velocity mismatch: got {}", v0);
    assert!((vn - 0.4).abs() < 0.15, "end velocity mismatch: got {}", vn);
}

#[test]
fn perpendicular_velocity_rejects() {
    let fk = common::dh_6dof();
    let a = SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]);
    let b = SRobotQ::from_array([1.0, -1.0, 1.2, 0.0, 0.0, 0.0]);
    let path = SRobotPath::<6, f64>::try_new(vec![a, b]).unwrap();

    // Path tangent is along joint 0. v_start is along joint 2 — perpendicular.
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.0, 3.0, 200.0);
    cfg.boundary = BoundaryConditions {
        v_start: SRobotQ::from_array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
        a_start: SRobotQ::zeros(),
        v_end: SRobotQ::zeros(),
        a_end: SRobotQ::zeros(),
        projection_tolerance: 1e-3,
    };

    let validator = common::wide_validator::<6>();
    let (result, _diag) = Topp3Tcp6::new(&fk).retime(&cfg, &path, &validator, &());
    match result {
        Err(DekeError::BoundaryInfeasible(_)) => {}
        other => panic!("expected BoundaryInfeasible, got {:?}", other),
    }
}
