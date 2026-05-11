mod common;

use deke_topp3tcp6::{SolveStatus, TcpLimits, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
fn seven_dof_curved_with_tcp_solves_and_is_feasible() {
    let fk = common::dh_7dof_prismatic();
    // q[0] is the prismatic rail along +X. Joints 1..7 are the UR5-ish arm.
    // Rail moves 0 → 0.4 m while the arm sweeps through a curved pose.
    let waypoints = vec![
        SRobotQ::from_array([0.0, 0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.1, 0.2, -1.1, 1.3, -0.1, 0.1, 0.1]),
        SRobotQ::from_array([0.2, 0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::from_array([0.3, 0.6, -0.7, 0.9, -0.3, 0.1, 0.3]),
        SRobotQ::from_array([0.4, 0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<7, f64>::try_new(waypoints).unwrap();

    let mut cfg = Topp3Tcp6Constraints::<7>::symmetric(1.5, 8.0, 400.0);
    cfg.tcp = TcpLimits {
        v_max: 1.0,
        a_max: 10.0,
        j_max: 500.0,
    };
    let mut validator = common::wide_validator::<7>();

    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "7-DOF retime failed: {}", diag);
    assert_eq!(diag.status, SolveStatus::Success);
    assert!(diag.peak_joint_velocity <= 1.5 * 1.2);
    assert!(diag.peak_tcp_velocity <= 1.0 * 1.1);
}

#[test]
fn seven_dof_locked_rail_acts_like_6dof() {
    // Rail (q[0]) is locked at 0 across every waypoint. With locked_prefix=1 the retimer
    // ignores the prismatic joint entirely; the 6-DOF arm beneath should retime as if it
    // were a standalone 6-DOF problem.
    let fk = common::dh_7dof_prismatic();
    let waypoints = vec![
        SRobotQ::from_array([0.0, 0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.0, 0.2, -1.1, 1.3, -0.1, 0.1, 0.1]),
        SRobotQ::from_array([0.0, 0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::from_array([0.0, 0.6, -0.7, 0.9, -0.3, 0.1, 0.3]),
        SRobotQ::from_array([0.0, 0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<7, f64>::try_new(waypoints).unwrap();

    let mut cfg = Topp3Tcp6Constraints::<7>::symmetric(1.5, 8.0, 400.0);
    cfg.locked_prefix = 1;
    let mut validator = common::wide_validator::<7>();

    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("{}", diag);
    let traj = result.expect("locked-rail retime failed");
    assert_eq!(diag.status, SolveStatus::Success);

    // Rail must stay at zero across the resampled trajectory.
    for q in traj.iter() {
        assert!(q.0[0].abs() < 1e-4, "rail drifted to {}", q.0[0]);
    }
}

#[test]
fn seven_dof_rail_dominant_motion() {
    // Mostly-rail motion: the prismatic q[0] sweeps 1 m while the arm barely moves.
    // The TCP velocity bound is tight; the solver has to use the rail to push the arm,
    // which exercises the prismatic Jacobian column in the FK chain (the same chain
    // joint_axes_positions feeds back through the retimer's FK calls).
    let fk = common::dh_7dof_prismatic();
    let waypoints = vec![
        SRobotQ::from_array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.25, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.5, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.75, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
    ];
    let path = SRobotPath::<7, f64>::try_new(waypoints).unwrap();

    let mut cfg = Topp3Tcp6Constraints::<7>::symmetric(2.0, 10.0, 500.0);
    cfg.tcp = TcpLimits {
        v_max: 0.5,
        a_max: 5.0,
        j_max: 250.0,
    };
    cfg.solver.max_iterations = 3_000;
    let mut validator = common::wide_validator::<7>();

    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "rail-dominant retime failed: {}", diag);
    assert_eq!(diag.status, SolveStatus::Success);
    // TCP velocity must be the binding constraint since the arm is static and rail
    // straight-line motion means peak |pp| == 1.
    assert!(
        diag.peak_tcp_velocity <= 0.5 * 1.05,
        "peak TCP velocity {} exceeded 1.05x v_max",
        diag.peak_tcp_velocity
    );
}
