mod common;

use deke_kin::{JointLimits, Kinematics};
use deke_topp3tcp_nlp::discrete::{SolveStatus, Topp3Tcp6Discrete, Topp3Tcp6DiscreteConstraints};
use deke_types::FKChain;
use deke_types::{Retimer, SRobotPath, SRobotQ};

/// A chain authored at f32 and lifted to f64 via [`Kinematics::to_f64`] should
/// plug straight into `Topp3Tcp6Discrete` (which demands an `FKChain<N, f64>`).
/// Replacement for the removed `FPDispatch` "author once, derive the sibling
/// precision" pattern.
#[test]
fn f32_built_chain_drives_f64_discrete_retimer() {
    let dh = [
        deke_kin::DHJoint::<f32> { a: 0.0, alpha: std::f32::consts::FRAC_PI_2, d: 0.089, theta_offset: 0.0 },
        deke_kin::DHJoint::<f32> { a: -0.425, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
        deke_kin::DHJoint::<f32> { a: -0.392, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
        deke_kin::DHJoint::<f32> { a: 0.0, alpha: std::f32::consts::FRAC_PI_2, d: 0.109, theta_offset: 0.0 },
        deke_kin::DHJoint::<f32> { a: 0.0, alpha: -std::f32::consts::FRAC_PI_2, d: 0.094, theta_offset: 0.0 },
        deke_kin::DHJoint::<f32> { a: 0.0, alpha: 0.0, d: 0.082, theta_offset: 0.0 },
    ];
    let chain_f32: Kinematics<6, f32> = Kinematics::from_dh(dh, JointLimits::symmetric(10.0), &[]);
    let chain_f64: Kinematics<6, f64> = chain_f32.to_f64();

    let _ = chain_f32.fk_end(&SRobotQ::<6, f32>::zeros()).unwrap();
    let _ = chain_f64.fk_end(&SRobotQ::<6, f64>::zeros()).unwrap();

    let waypoints = vec![
        SRobotQ::<6, f64>::from_array([0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::<6, f64>::from_array([0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::<6, f64>::from_array([0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<6, f64>::try_new(waypoints).unwrap();

    let cfg = Topp3Tcp6DiscreteConstraints::<6>::symmetric(1.5, 4.0, 200.0);

    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6Discrete::new(&chain_f64).retime(&cfg, &path, &mut validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "retime failed: {}", diag);
    assert_eq!(diag.status, SolveStatus::Success);
}
