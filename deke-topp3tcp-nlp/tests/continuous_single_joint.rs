mod common;

use deke_topp3tcp_nlp::continuous::{SolveStatus, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
fn single_joint_rest_to_rest_matches_trapezoidal() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([1.0]),
    ])
    .unwrap();

    // v_max = 1 rad/s, a_max = 2 rad/s², loose jerk.
    let cfg = Topp3Tcp6Constraints::<1>::symmetric(1.0, 2.0, 200.0);

    let validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6::new(&fk).retime(&cfg, &path, &validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "retime failed: {}", diag);
    let traj = result.unwrap();

    assert_eq!(diag.status, SolveStatus::Success);

    // Analytic trapezoidal time = v/a + Δ/v = 0.5 + 1.0 = 1.5 s.
    // With finite jerk and interior-point convergence the optimum is slightly larger.
    let total = traj.duration().as_secs_f64();
    assert!(
        (1.5..=1.9).contains(&total),
        "total time {} outside [1.5, 1.9]",
        total
    );
}
