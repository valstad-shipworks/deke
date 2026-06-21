//! Continuous-time TOPP-3TCP retimer.
//!
//! Implements [`deke_types::Retimer`] for a path-parameterized triple-integrator model and the
//! `hafgufa` (sleipnir) interior-point NLP solver. The retimer takes a fixed geometric path of
//! joint-space waypoints and emits a uniformly-time-sampled [`deke_types::SRobotTraj`] that
//! minimises total traversal time subject to:
//!
//! - per-joint position, velocity, acceleration, and jerk bounds,
//! - per-TCP translational velocity, acceleration, and jerk bounds, and
//! - boundary conditions on start/end joint-space velocity and acceleration.
//!
//! The first `locked_prefix` joints of the kinematic chain can optionally be held at their
//! starting value (useful for mobile bases or locked base rails). See [`Topp3Tcp6Constraints`].

pub mod constraints;
pub mod diagnostic;
pub mod nlp;
pub mod resample;
pub mod retimer;

#[cfg(feature = "valuable")]
mod valuable_impls;

pub use crate::common::path_derivatives::PathDerivatives;
pub use constraints::{
    BoundaryConditions, DensificationOptions, JointLimits, SolverOptions, TcpLimits,
    Topp3Tcp6Constraints,
};
pub use diagnostic::{
    BoundarySlackUsage, ConstraintCounts, DerivativeStats, InitialGuessStats, LimitingGroup,
    PathStats, PeakLocation, PhaseTiming, SolveStatus, TcpStats, Topp3Tcp6Diagnostic,
};
pub use retimer::Topp3Tcp6;
