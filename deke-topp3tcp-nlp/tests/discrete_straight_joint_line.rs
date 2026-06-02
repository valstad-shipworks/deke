mod common;

use deke_topp3tcp_nlp::discrete::{SolveStatus, Topp3Tcp6Discrete, Topp3Tcp6DiscreteConstraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
fn straight_line_six_dof_joint_limits_dominant() {
    let fk = common::dh_6dof();
    let a = SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]);
    let b = SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]);
    let path = SRobotPath::<6, f64>::try_new(vec![a, b]).unwrap();

    let cfg = Topp3Tcp6DiscreteConstraints::<6>::symmetric(1.0, 3.0, 300.0);

    let validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6Discrete::new(&fk).retime(&cfg, &path, &validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "retime failed: {}", diag);
    assert_eq!(diag.status, SolveStatus::Success);

    // Strict FD residual — should be ~0 since the discrete formulation enforces
    // exactly what the consumer measures.
    assert!(
        diag.output_fd_residual.joint_v <= 1e-3,
        "joint_v FD residual {:+e}",
        diag.output_fd_residual.joint_v,
    );
    assert!(
        diag.output_fd_residual.joint_a <= 1e-3,
        "joint_a FD residual {:+e}",
        diag.output_fd_residual.joint_a,
    );
    assert!(
        diag.output_fd_residual.joint_j <= 1e-3,
        "joint_j FD residual {:+e}",
        diag.output_fd_residual.joint_j,
    );
}
