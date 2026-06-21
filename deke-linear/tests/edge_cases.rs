mod common;

use deke_linear::{LinearError, LinearFollower};
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

    let follower = LinearFollower::new(&robot);
    let err = follower.follow(&poses, &common::config(0.05), &common::noop(), &()).unwrap_err();
    assert!(
        matches!(err, LinearError::Unreachable { .. }),
        "expected Unreachable, got {err:?}"
    );
}

#[test]
fn too_few_poses_is_rejected() {
    let robot = common::ur();
    let base = {
        use deke_types::FKChain;
        robot.fk_end(&common::anchor()).unwrap()
    };
    let follower = LinearFollower::new(&robot);
    let err = follower.follow(&[base], &common::config(0.05), &common::noop(), &()).unwrap_err();
    assert!(matches!(err, LinearError::TooFewPoses(1)));
}
