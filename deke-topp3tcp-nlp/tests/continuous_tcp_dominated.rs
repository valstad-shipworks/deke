mod common;

use deke_topp3tcp_nlp::continuous::{SolveStatus, TcpLimits, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
fn tcp_velocity_is_limiting() {
    let fk = common::dh_6dof();
    let a = SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]);
    let b = SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]);
    let path = SRobotPath::<6, f64>::try_new(vec![a, b]).unwrap();

    // Moderate joint limits, TCP velocity is the tight limit.
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(5.0, 30.0, 3_000.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 0.25,
        a_max: f64::INFINITY,
        j_max: f64::INFINITY,
    });
    cfg.solver.max_iterations = 2_000;

    let validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6::new(&fk).retime(&cfg, &path, &validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "retime failed: {}", diag);
    assert_eq!(diag.status, SolveStatus::Success);

    // The TCP-velocity bound should force the peak TCP speed to be near v_max.
    assert!(
        diag.peak_tcp_velocity <= 0.25 * 1.05,
        "peak tcp v {} exceeded 1.05x v_max",
        diag.peak_tcp_velocity
    );
    // And the joints shouldn't be near their own (loose) velocity limits.
    assert!(
        diag.peak_joint_velocity <= 5.0,
        "joints exceeded expected low speed (peak {} rad/s, bound 20)",
        diag.peak_joint_velocity
    );
}
