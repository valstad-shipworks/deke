//! Time-optimal path-parameterization (TOPP-3TCP) retimers for N-DOF robot arms.
//!
//! Both retimers implement [`deke_types::Retimer`] over a path-parameterized triple-integrator
//! model solved with the `hafgufa` (sleipnir) interior-point NLP. They take a fixed geometric
//! path of joint-space waypoints and emit a uniformly-time-sampled [`deke_types::SRobotTraj`]
//! that minimises total traversal time subject to per-joint and per-TCP velocity, acceleration,
//! and jerk bounds plus start/end boundary conditions.
//!
//! - The [`continuous`] formulation optimises the continuous `(sd, sdd, sddd, dt)` profile per
//!   densified knot and resamples the result onto the output grid.
//! - The [`discrete`] formulation optimises the per-sample arc-length values `σ[i]` directly, so
//!   the constraint the IPM enforces is the exact backward-difference the consumer measures.
//!
//! Entry types [`Topp3Tcp6`] and [`Topp3Tcp6Discrete`] are re-exported at the crate root. Types
//! whose names collide between the two formulations (`JointLimits`, `TcpLimits`, `SolveStatus`,
//! `PathStats`, and the other diagnostic sub-types) are reached through the [`continuous`] and
//! [`discrete`] modules.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod common;
pub mod continuous;
pub mod discrete;

pub use continuous::Topp3Tcp6;
pub use discrete::Topp3Tcp6Discrete;
