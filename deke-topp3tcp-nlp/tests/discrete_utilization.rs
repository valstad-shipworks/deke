mod common;

use deke_topp3tcp_nlp::discrete::{SolveStatus, Topp3Tcp6Discrete, Topp3Tcp6DiscreteConstraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

/// A time-optimal solution should be pressing against SOME limit on every step (otherwise the
/// solver could shorten dt). We check that the mean per-step max-utilization is high.
#[test]
fn time_optimal_solution_saturates_some_limit_on_average() {
    let fk = common::dh_6dof();
    let waypoints = vec![
        SRobotQ::from_array([0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.2, -1.1, 1.3, -0.1, 0.1, 0.1]),
        SRobotQ::from_array([0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::from_array([0.6, -0.7, 0.9, -0.3, 0.1, 0.3]),
        SRobotQ::from_array([0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<6, f64>::try_new(waypoints).unwrap();

    let cfg = Topp3Tcp6DiscreteConstraints::<6>::symmetric(1.5, 4.0, 200.0);
    let validator = common::wide_validator::<6>();

    let (result, diag) = Topp3Tcp6Discrete::new(&fk).retime(&cfg, &path, &validator, &());
    eprintln!("{}", diag);
    result.expect("retime failed");
    assert_eq!(diag.status, SolveStatus::Success);

    // No individual limit may exceed 100% utilization (with small solver-tolerance slack).
    let peaks = [
        diag.peak_joint_velocity / 1.5,
        diag.peak_joint_acceleration / 4.0,
        diag.peak_joint_jerk / 200.0,
    ];
    for p in peaks {
        assert!(p <= 1.01, "an individual peak exceeded the limit: {}", p);
    }

    // And on average the trajectory should be driving SOMETHING hard. If the solver left most
    // of the path well under all limits, the NLP didn't find a time-optimal solution.
    assert!(
        diag.average_utilization >= 0.80,
        "average limit utilization only {:.1}% — solver is not time-optimal",
        diag.average_utilization * 100.0
    );
}

#[test]
fn single_joint_rest_to_rest_utilization_is_high() {
    let fk = common::dh_1dof();
    let path =
        SRobotPath::<1, f64>::try_new(vec![SRobotQ::from_array([0.0]), SRobotQ::from_array([1.0])])
            .unwrap();

    let cfg = Topp3Tcp6DiscreteConstraints::<1>::symmetric(1.0, 2.0, 200.0);
    let validator = common::wide_validator::<1>();

    let (result, diag) = Topp3Tcp6Discrete::new(&fk).retime(&cfg, &path, &validator, &());
    eprintln!("{}", diag);
    result.expect("retime failed");

    // 1D rest-to-rest has a trapezoidal/S-curve profile; the accel phase saturates a_max and
    // the cruise phase saturates v_max. The only "under-utilized" portion is the smooth-jerk
    // transitions, which are a small fraction of the total time.
    assert!(
        diag.average_utilization >= 0.80,
        "average limit utilization only {:.1}%",
        diag.average_utilization * 100.0
    );
}
