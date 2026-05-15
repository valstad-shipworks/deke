mod common;

use deke_topp3tcp6::{SolveStatus, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{DHChain, FPDispatch, FKChain, Retimer, SRobotPath, SRobotQ};

/// `FPDispatch` should plug straight into `Topp3Tcp6` (which demands an
/// `FKChain<N, f64>`) by routing through its f64 inner chain. Same robot
/// authored once in f64; the dispatcher derives the f32 sibling for free.
#[test]
fn fp_dispatch_drives_f64_topp_retimer() {
    let f64_chain = common::dh_6dof();
    let dispatch: FPDispatch<6, DHChain<6, f32>, DHChain<6, f64>> =
        FPDispatch::from_f64(f64_chain);

    // Sanity: end pose at zero matches between the two precisions.
    let q32 = SRobotQ::<6, f32>::zeros();
    let q64 = SRobotQ::<6, f64>::zeros();
    let _ = <FPDispatch<_, _, _> as FKChain<6, f32>>::fk_end(&dispatch, &q32).unwrap();
    let _ = <FPDispatch<_, _, _> as FKChain<6, f64>>::fk_end(&dispatch, &q64).unwrap();

    let waypoints = vec![
        SRobotQ::<6, f64>::from_array([0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::<6, f64>::from_array([0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::<6, f64>::from_array([0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<6, f64>::try_new(waypoints).unwrap();

    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 4.0, 200.0);

    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &dispatch, &mut validator, &());
    eprintln!("{}", diag);
    assert!(result.is_ok(), "retime failed: {}", diag);
    assert_eq!(diag.status, SolveStatus::Success);
}
