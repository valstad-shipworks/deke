mod common;

use deke_topp3tcp_nlp::continuous::{SolveStatus, Topp3Tcp6, Topp3Tcp6Constraints};
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

    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 8.0, 400.0);
    let mut validator = common::wide_validator::<6>();

    let (result, diag) = Topp3Tcp6::new(&fk).retime(&cfg, &path, &mut validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "retime failed: {}", diag);
    assert_eq!(diag.status, SolveStatus::Success);

    // Feasibility (with slack to absorb resampling noise).
    assert!(diag.peak_joint_velocity <= 1.5 * 1.2);
    assert!(diag.peak_joint_acceleration <= 8.0 * 1.5);
}
