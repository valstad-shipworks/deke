mod common;

use deke_topp3tcp6::{SolveStatus, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[test]
fn straight_line_six_dof_joint_limits_dominant() {
    let fk = common::dh_6dof();
    let a = SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]);
    let b = SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]);
    let path = SRobotPath::<6, f64>::try_new(vec![a, b]).unwrap();

    // Tight joint limits, loose TCP limits.
    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.0, 3.0, 300.0);

    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "retime failed: {}", diag);
    assert_eq!(diag.status, SolveStatus::Success);

    // Peak quantities reported by the retimer's post-hoc finite-difference check should
    // respect the constraints within a resampling tolerance. We use 1.2x to absorb both the
    // solver tolerance and the finite-difference noise at the endpoints of the trajectory.
    assert!(
        diag.peak_joint_velocity <= 1.0 * 1.2,
        "peak joint velocity {} exceeded 1.2x v_max",
        diag.peak_joint_velocity
    );
    assert!(
        diag.peak_joint_acceleration <= 3.0 * 1.4,
        "peak joint acceleration {} exceeded 1.4x a_max",
        diag.peak_joint_acceleration
    );
}
