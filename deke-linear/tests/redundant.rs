mod common;

use std::time::Duration;

use deke_linear::{FollowConfig, JointLimits, LinearFollower, RedundantAxis, RedundantOptions};
use deke_types::FKChain;
use deke_types::glam::DVec3;

/// The redundant planner produces a usable trajectory and reports the yaw it
/// resolved about the configured tool axis.
#[test]
fn redundant_follow_resolves_yaw_and_runs() {
    let robot = common::ur();
    let poses = common::straight(&robot, DVec3::X, 0.10, 4);

    let cfg = FollowConfig::weld(
        35.0,
        JointLimits::symmetric(2.0, 8.0, 80.0),
        Duration::from_millis(8),
    )
    .with_redundancy(RedundantOptions {
        axis: RedundantAxis::PosZ,
        yaw_window: (-45f64.to_radians(), 45f64.to_radians()),
        yaw_samples: 16,
        ..RedundantOptions::default()
    });

    let follower = LinearFollower::new(&robot);
    let (traj, diag) = follower
        .follow(&poses, &cfg, &common::noop(), &())
        .expect("redundant follow failed");

    assert_eq!(diag.redundant.len(), 1, "one run, one redundant report");
    assert!(diag.planner.is_empty(), "fixed planner should not be used");
    assert!(traj.path().len() > 10);

    let rd = &diag.redundant[0];
    let (lo, hi) = rd.yaw_range;
    assert!(
        lo >= -45f64.to_radians() - 1e-6 && hi <= 45f64.to_radians() + 1e-6,
        "resolved yaw must stay inside the window: [{lo}, {hi}]"
    );
    assert!(
        rd.min_manipulability > 0.0,
        "track should stay off singularities"
    );

    // 35 IPM ≈ 14.8 mm/s; the TCP should hold roughly that in the interior.
    let speeds = common::tcp_speeds(&robot, &traj);
    let expected = 35.0 * 0.0254 / 60.0;
    let lo_i = speeds.len() / 4;
    let hi_i = speeds.len() * 3 / 4;
    for &v in &speeds[lo_i..hi_i] {
        assert!(
            (v - expected).abs() < expected * 0.15,
            "cruise {v} vs {expected}"
        );
    }
}

/// The configured axis actually changes the resolved IK: planning about Z vs X
/// produces different joint tracks (the free DOF is a different physical axis).
#[test]
fn axis_choice_changes_the_solution() {
    let robot = common::ur();
    let poses = common::straight(&robot, DVec3::Y, 0.08, 3);
    let joint = JointLimits::symmetric(2.0, 8.0, 80.0);
    let follower = LinearFollower::new(&robot);

    let base = |axis| {
        FollowConfig::weld(30.0, joint.clone(), Duration::from_millis(8)).with_redundancy(
            RedundantOptions {
                axis,
                yaw_samples: 16,
                ..RedundantOptions::default()
            },
        )
    };

    let (tz, _) = follower
        .follow(&poses, &base(RedundantAxis::PosZ), &common::noop(), &())
        .unwrap();
    let (tx, _) = follower
        .follow(&poses, &base(RedundantAxis::PosX), &common::noop(), &())
        .unwrap();

    // Same first TCP position regardless of which axis is free.
    let pz = robot.fk_end(&tz.path()[0]).unwrap().translation;
    let px = robot.fk_end(&tx.path()[0]).unwrap().translation;
    assert!(pz.distance(px) < 1e-6, "start position should match");

    // But the joint configurations differ — a different DOF was freed.
    let mut max_diff = 0.0f64;
    for j in 0..6 {
        max_diff = max_diff.max((tz.path()[0].0[j] - tx.path()[0].0[j]).abs());
    }
    assert!(
        max_diff > 1e-3,
        "freeing X vs Z should change the joint solution"
    );
}
