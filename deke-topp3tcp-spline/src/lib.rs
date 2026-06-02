//! Discrete TOPP-3TCP retimer using a B-spline path representation and a
//! depth-first search over discrete jerk candidates.
//!
//! Implements [`deke_types::Retimer<N, f64>`] for [`Topp3TcpSpline`].
//! Variables of the search are the path-parameter state `(s, ṡ, s̈, s⃛)` at
//! uniform time-steps `dt`. At each step the optimizer enumerates a small
//! cosine-spaced fan of jerk candidates, prunes any that violate per-axis
//! joint v/a/j limits or Cartesian TCP v/a/j limits (via the position rows
//! of the geometric Jacobian), and closes out with a three-segment
//! boundary-condition solve when `s >= 0.7`.
//!
//! The path is represented as a clamped B-spline interpolated through the
//! input joint-space waypoints, with support-point density refined until
//! the spline lies within a configurable deviation tube around the original
//! polyline.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod bspline;
pub mod constraints;
pub mod diagnostic;
pub mod path;
pub mod retimer;
pub mod trajectory;

pub use bspline::BSpline;
pub use constraints::{
    JointLimits, SearchOptions, SplinePathOptions, TcpLimits, Topp3TcpSplineConstraints,
};
pub use diagnostic::{SolveStatus, Topp3TcpSplineDiagnostic};
pub use path::SplineInterpolatedRobotPath;
pub use retimer::Topp3TcpSpline;
pub use trajectory::{TrajPCS, Trajectory};
