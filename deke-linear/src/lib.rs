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
//! Each stage is a standalone, trait-conforming component; the caller drives
//! them. [`condition`] splits the polyline into runs, then for each run the
//! [`CartesianLinearPlanner`] resolves a joint path and the
//! [`ConstantSpeedRetimer`] times it:
//!
//! ```no_run
//! # use deke_linear::*;
//! # use deke_types::{Planner, Retimer};
//! # use deke_types::glam::DAffine3;
//! # fn run<FK: deke_types::ContinuousFKChain<6, f64> + deke_types::IkSolver<6, f64>>(
//! #     fk: &FK, poses: &[DAffine3], cond: &PathConditioning,
//! #     opts: &PlannerOptions<6>, cons: &LinearConstraints<6>) {
//! let planner = CartesianLinearPlanner::new(fk);
//! let retimer = ConstantSpeedRetimer::new(fk);
//! // `NoopValidator` plans without obstacle checks; pass a real `Validator`
//! // (and its context) to route the arm around obstacles inside the planner.
//! for run in condition(poses, cond).unwrap() {
//!     let (path, _) = planner.plan::<deke_types::DekeError, _>(opts, &run, &NoopValidator::<6>, &());
//!     let (traj, _) = retimer.retime(cons, &path.unwrap(), &NoopValidator::<6>, &());
//!     let _ = traj;
//! }
//! # }
//! ```
//!
//! For a symmetric welding torch, the [`RedundantLinearPlanner`] treats the free
//! tool-axis yaw as a DOF and resolves it globally to dodge singularities — and,
//! with a real `Validator`, obstacles.

pub mod constraints;
pub mod diagnostic;
pub mod error;
pub mod path;
pub mod planner;
pub mod redundant;
pub mod retimer;
mod util;
mod validator;

pub use constraints::{JointLimits, LinearConstraints, PathConditioning, PlannerOptions};
pub use diagnostic::{LinearPlannerDiagnostic, LinearRetimerDiagnostic, RedundantDiagnostic};
pub use error::LinearError;
pub use path::{CartesianRun, condition};
pub use planner::CartesianLinearPlanner;
pub use redundant::{RedundantAxis, RedundantConfig, RedundantLinearPlanner, RedundantOptions};
pub use retimer::ConstantSpeedRetimer;
pub use validator::NoopValidator;
