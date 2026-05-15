mod common;

use deke_topp3tcp6_discrete::{
    SolveStatus, TcpLimits, Topp3Tcp6Discrete, Topp3Tcp6DiscreteConstraints,
};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
fn tcp_velocity_is_limiting() {
    let fk = common::dh_6dof();
    let a = SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]);
    let b = SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]);
    let path = SRobotPath::<6, f64>::try_new(vec![a, b]).unwrap();

    let mut cfg = Topp3Tcp6DiscreteConstraints::<6>::symmetric(5.0, 30.0, 3_000.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 0.25,
        a_max: f64::INFINITY,
        j_max: f64::INFINITY,
    });
    cfg.solver.max_iterations = 2_000;

    let validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6Discrete.retime(&cfg, &path, &fk, &validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "retime failed: {}", diag);
    assert_eq!(diag.status, SolveStatus::Success);

    // TCP v should be at or below the limit (within strict tol).
    assert!(
        diag.peak_tcp_velocity <= 0.25 * 1.01,
        "peak tcp v {} exceeded 1.01x v_max",
        diag.peak_tcp_velocity
    );
    // Joints should not be at their (loose) limits — TCP is binding.
    assert!(
        diag.peak_joint_velocity <= 5.0,
        "peak joint v {} exceeded v_max=5",
        diag.peak_joint_velocity
    );
}
