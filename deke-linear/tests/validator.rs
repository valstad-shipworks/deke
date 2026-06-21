mod common;

use std::time::Duration;

use deke_linear::{
    FollowConfig, JointLimits, LinearError, LinearFollower, RedundantAxis, RedundantOptions,
};
use deke_types::glam::DVec3;
use deke_types::{DekeError, DekeResult, SRobotQ, SRobotQLike, Validator};

/// Rejects every configuration.
#[derive(Clone, Debug)]
struct RejectAll;

impl Validator<6, (), f64> for RejectAll {
    type Context<'ctx> = ();
    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<6, E, f64>>(
        &self,
        _q: A,
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        Err(DekeError::SelfCollision(0, 0))
    }
    fn validate_motion<'ctx>(
        &self,
        _qs: &[SRobotQ<6, f64>],
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        Err(DekeError::SelfCollision(0, 0))
    }
}

/// A virtual obstacle in configuration space: rejects configs whose `joint` lies
/// within `radius` of `center`. Because each tool yaw maps to a different wrist
/// configuration, forbidding a band of joint 5 is effectively an obstacle the
/// free yaw must rotate around.
#[derive(Clone, Debug)]
struct RejectBand {
    joint: usize,
    center: f64,
    radius: f64,
}

impl Validator<6, (), f64> for RejectBand {
    type Context<'ctx> = ();
    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<6, E, f64>>(
        &self,
        q: A,
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        let q = q.to_srobotq().map_err(Into::into)?;
        if (q.0[self.joint] - self.center).abs() < self.radius {
            Err(DekeError::EnvironmentCollision(self.joint as i16, 0))
        } else {
            Ok(())
        }
    }
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<6, f64>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        for q in qs {
            self.validate(*q, ctx)?;
        }
        Ok(())
    }
}

#[test]
fn reject_all_obstructs_both_planners() {
    let robot = common::ur();
    let poses = common::straight(&robot, DVec3::X, 0.05, 3);
    let follower = LinearFollower::new(&robot);
    let cfg = common::config(0.04);
    let cfg_red = cfg.clone().with_redundancy(RedundantOptions::default());

    let e_fixed = follower.follow(&poses, &cfg, &RejectAll, &()).unwrap_err();
    assert!(matches!(e_fixed, LinearError::Obstructed { .. }), "fixed: {e_fixed:?}");
    let e_red = follower.follow(&poses, &cfg_red, &RejectAll, &()).unwrap_err();
    assert!(matches!(e_red, LinearError::Obstructed { .. }), "redundant: {e_red:?}");
}

#[test]
fn redundant_planner_rotates_yaw_around_an_obstacle() {
    let robot = common::ur();
    let poses = common::straight(&robot, DVec3::X, 0.05, 3);
    let follower = LinearFollower::new(&robot);
    let cfg = FollowConfig::weld(30.0, JointLimits::symmetric(2.0, 8.0, 80.0), Duration::from_millis(8))
        .with_redundancy(RedundantOptions {
            axis: RedundantAxis::PosZ,
            yaw_samples: 36,
            ..RedundantOptions::default()
        });

    // Baseline (no obstacle): record the wrist-roll value the planner settles on.
    let (traj0, _) = follower.follow(&poses, &cfg, &common::noop(), &()).expect("noop follow");
    let mid = traj0.path().len() / 2;
    let center = traj0.path()[mid].0[5];

    let radius = 0.4;
    let obstacle = RejectBand { joint: 5, center, radius };
    // The unobstructed track really does pass through the forbidden band.
    let baseline: Vec<_> = traj0.path().iter().copied().collect();
    assert!(obstacle.validate_motion(&baseline, &()).is_err());

    // With the obstacle, the free yaw must rotate the arm to a config outside it.
    let (traj1, diag) = follower
        .follow(&poses, &cfg, &obstacle, &())
        .expect("free yaw should route around the obstacle");
    for q in traj1.path().iter() {
        assert!(
            (q.0[5] - center).abs() >= radius - 1e-6,
            "every output config must avoid the forbidden joint-5 band"
        );
    }
    println!(
        "obstacle-avoiding yaw ∈ [{:.0}°, {:.0}°]",
        diag.redundant[0].yaw_range.0.to_degrees(),
        diag.redundant[0].yaw_range.1.to_degrees()
    );
}
