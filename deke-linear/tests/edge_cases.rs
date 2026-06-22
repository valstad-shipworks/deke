mod common;

use deke_types::DekeError;
use deke_types::glam::{DAffine3, DVec3};

#[test]
fn unreachable_pose_reports_clearly() {
    let robot = common::ur();
    // Translate far outside the workspace.
    let base = {
        use deke_types::FKChain;
        robot.fk_end(&common::anchor()).unwrap()
    };
    let poses: Vec<DAffine3> = (0..3)
        .map(|i| {
            DAffine3::from_mat3_translation(
                base.matrix3,
                base.translation + DVec3::X * (5.0 + i as f64),
            )
        })
        .collect();

    let err = common::follow(&robot, &poses, &common::config(0.05), &common::noop(), &())
        .unwrap_err();
    // The structured `LinearError::Unreachable` is collapsed to `DekeError` as it
    // crosses the `Planner` trait boundary; the descriptive message survives.
    assert!(
        matches!(&err, DekeError::RetimerFailed(s) if s.contains("reachable")),
        "expected unreachable, got {err:?}"
    );
}

#[test]
fn too_few_poses_is_rejected() {
    let robot = common::ur();
    let base = {
        use deke_types::FKChain;
        robot.fk_end(&common::anchor()).unwrap()
    };
    let err = common::follow(&robot, &[base], &common::config(0.05), &common::noop(), &())
        .unwrap_err();
    assert!(
        matches!(&err, DekeError::RetimerFailed(s) if s.contains("at least 2")),
        "expected too-few-poses, got {err:?}"
    );
}
