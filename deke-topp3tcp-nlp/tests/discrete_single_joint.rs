mod common;

use deke_topp3tcp_nlp::discrete::{SolveStatus, Topp3Tcp6Discrete, Topp3Tcp6DiscreteConstraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
fn single_joint_rest_to_rest() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([1.0]),
    ])
    .unwrap();

    let cfg = Topp3Tcp6DiscreteConstraints::<1>::symmetric(1.0, 2.0, 200.0);
    let validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6Discrete::new(&fk).retime(&cfg, &path, &validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "retime failed: {}", diag);
    let traj = result.unwrap();
    assert_eq!(diag.status, SolveStatus::Success);

    // Analytic trapezoidal time = v/a + Δ/v = 0.5 + 1.0 = 1.5 s.
    // With finite jerk + discrete sampling the optimum is somewhat larger.
    let total = traj.duration().as_secs_f64();
    assert!(
        (1.5..=2.5).contains(&total),
        "total time {} outside [1.5, 2.5]",
        total
    );

    // Strict FD residual should be ≤ tolerance.
    assert!(
        diag.output_fd_residual.joint_v <= cfg.solver.tolerance.max(1e-6),
        "joint_v FD residual {:+e} exceeds tol {:+e}",
        diag.output_fd_residual.joint_v,
        cfg.solver.tolerance,
    );
    assert!(
        diag.output_fd_residual.joint_a <= cfg.solver.tolerance.max(1e-6),
        "joint_a FD residual {:+e} exceeds tol {:+e}",
        diag.output_fd_residual.joint_a,
        cfg.solver.tolerance,
    );
}
