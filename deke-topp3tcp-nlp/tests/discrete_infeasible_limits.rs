mod common;

use deke_topp3tcp_nlp::discrete::{SolveStatus, Topp3Tcp6Discrete, Topp3Tcp6DiscreteConstraints};
use deke_types::{DekeError, Retimer, SRobotPath, SRobotQ};

#[test]
fn impossible_bounds_return_error_not_panic() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([1.0]),
    ])
    .unwrap();

    // v_max nearly zero but jerk and accel still finite — the solver cannot traverse the path.
    let cfg = Topp3Tcp6DiscreteConstraints::<1>::symmetric(1e-4, 1e-3, 1e-2);

    let validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6Discrete::new(&fk).retime(&cfg, &path, &validator, &());
    eprintln!("{}", diag);
    match result {
        Err(DekeError::RetimerFailed(_)) => {}
        other => panic!("expected RetimerFailed, got {:?}", other),
    }
    assert!(matches!(
        diag.status,
        SolveStatus::LocallyInfeasible
            | SolveStatus::GloballyInfeasible
            | SolveStatus::MaxIterationsExceeded
            | SolveStatus::DivergingIterates
            | SolveStatus::FeasibilityRestorationFailed
    ));
}
