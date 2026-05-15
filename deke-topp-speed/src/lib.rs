//! Real-time, jerk-limited trajectory shaping for joint paths.
//!
//! This crate produces time-optimal joint trajectories that respect velocity,
//! acceleration and jerk ceilings, optionally passing through intermediate
//! waypoints, and provides a real-time follower for moving goal states.
//!
//! The number of joints is fixed at compile time via the const generic `N`.
//! All numerics are generic over the scalar `F`, which must implement
//! [`deke_types::FKScalar`] (so this works with `f32` or `f64`).
//!
//! # Public surface
//!
//! - [`ToppSolver`]: offline, jerk-limited path-to-trajectory solver. Implements
//!   the [`deke_types::Retimer`] trait so it slots into the broader planning
//!   ecosystem.
//! - [`Pursuer`]: real-time tracker that adapts the goal each control cycle to
//!   follow a moving [`PursuitTarget`].
//! - [`MotionSpec`]: kinematic limits and goal description for a solve.
//! - [`MotionSample`]: a single sample of the produced motion at a wall-clock
//!   instant.
//! - [`StepStatus`]: status enum used by the live pursuer.
//! - Mode enums: [`ControlMode`], [`Coordination`], [`DurationGrid`],
//!   [`GoalOutOfBounds`], [`FollowMode`].
//! - [`Extent`]: per-axis value range with the time at which the extrema are
//!   reached.

pub use deke_types::{
    DekeError, DekeResult, FKChain, FKScalar, Retimer, SRobotPath, SRobotQ, SRobotTraj, Validator,
};

mod check;
mod extent;
mod feasible;
mod halt_segment;
mod jacobian;
mod kin_state;
mod modes;
mod plan;
mod pose_math;
mod pursuer;
mod roots;
mod sample;
mod segment;
mod shaper;
mod solver;
mod spec;
mod status;
mod target_solver;
mod vel_math;
mod waypoint_solver;

pub use extent::Extent;
pub use modes::{ControlMode, Coordination, DurationGrid, FollowMode, GoalOutOfBounds};
pub use pursuer::{PredictionModel, Pursuer};
pub use sample::{MotionSample, PursuitTarget};
pub use shaper::{SolveDiagnostic, ToppSolver};
pub use spec::MotionSpec;
pub use status::StepStatus;
