//! Constant-TCP-speed Cartesian polyline following for serial manipulators.
//!
//! Where the `deke-topp*` retimers are time-optimal (maximise speed under caps),
//! `deke-linear` holds a *constant* TCP travel speed — the requirement for
//! welding and similar process motions — and degrades gracefully near
//! singularities. It is a CNC-style constant-feedrate interpolator in three
//! stages:
//!
//! - **Stage A** ([`path`]) conditions the raw polyline into smooth, arc-length
//!   parameterised [`CartesianRun`]s, splitting at sharp corners.
//! - **Stage B** ([`planner`]) resolves each run to a continuous joint path by
//!   analytic-IK branch tracking, steering away from singularities
//!   ([`CartesianLinearPlanner`], implements [`deke_types::Planner`]).
//! - **Stage C** ([`retimer`]) holds the commanded speed wherever the joint
//!   v/a/j limits allow and dips smoothly where they don't
//!   ([`ConstantSpeedRetimer`], implements [`deke_types::Retimer`]).
//!
//! [`LinearFollower`] runs all three and stitches the per-run trajectories.
//!
//! ```no_run
//! # use deke_linear::*;
//! # use deke_types::glam::DAffine3;
//! # fn run<FK: deke_types::ContinuousFKChain<6, f64> + deke_types::IkSolver<6, f64>>(
//! #     fk: &FK, poses: &[DAffine3], cfg: &FollowConfig<6>) {
//! let follower = LinearFollower::new(fk);
//! // `NoopValidator` plans without obstacle checks; pass a real `Validator`
//! // (and its context) to route the arm around obstacles inside the planner.
//! let (traj, diag) = follower.follow(poses, cfg, &NoopValidator::<6>, &()).unwrap();
//! # let _ = (traj, diag);
//! # }
//! ```
//!
//! For a symmetric welding torch, declare the free tool-axis on
//! [`FollowConfig::redundant`] and the [`RedundantLinearPlanner`] resolves the yaw
//! globally to dodge singularities — and, with a real `Validator`, obstacles.

pub mod constraints;
pub mod diagnostic;
pub mod error;
pub mod follower;
pub mod path;
pub mod planner;
pub mod redundant;
pub mod retimer;
mod util;

pub use constraints::{FollowConfig, JointLimits, LinearConstraints, PathConditioning, PlannerOptions};
pub use diagnostic::{
    LinearFollowDiagnostic, LinearPlannerDiagnostic, LinearRetimerDiagnostic, RedundantDiagnostic,
};
pub use error::LinearError;
pub use follower::{LinearFollower, NoopValidator};
pub use path::{condition, CartesianRun};
pub use planner::CartesianLinearPlanner;
pub use redundant::{RedundantAxis, RedundantLinearPlanner, RedundantOptions};
pub use retimer::ConstantSpeedRetimer;
