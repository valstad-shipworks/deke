//! Joint-space, path-exact, jerk-limited path parameterisation via a discrete
//! convex LP.
//!
//! Given an untimed joint-space polyline (`SRobotPath`) and per-axis
//! velocity/acceleration/jerk limits, [`Topp3Lp`] emits a uniform-`dt`
//! trajectory whose samples lie *exactly* on the input chord and whose
//! finite-difference v/a/j — the quantities a downstream controller
//! reconstructs — are bounded by the limits.
//!
//! The timing is a discrete convex program. The single scalar decision variable
//! `σ[k]` is the arc length reached at output tick `k·dt`. Because the path is
//! chord-linear, the `m`-th finite difference of every joint is exactly
//! `secantᵦ · Δᵐσ` within a segment, so each per-joint v/a/j limit becomes a
//! *linear* bound on a difference of the `σ`s — including jerk, which is what
//! breaks convexity in free-time TOPP. The program maximises progress (so it
//! runs at the limits wherever it can) and is solved with a small banded LP
//! (Clarabel). A final finite-difference check against the *true* limits is the
//! airtight backstop: a path that cannot be timed under the limits fails rather
//! than emitting an over-limit trajectory.
//!
//! ```ignore
//! let retimer = Topp3Lp::<6>::new();
//! let constraints =
//!     Topp3LpConstraints::symmetric(2.0, 8.0, 80.0, std::time::Duration::from_millis(8));
//! let (traj, _diag) = retimer.retime(&constraints, &path, &validator, &());
//! ```
//!
//! [`Topp3LpTcp`] additionally caps the tool-centre-point *linear speed*: it
//! takes an FK chain, turns the cap into a per-segment ceiling on `σ̇` from the
//! Jacobian, and verifies the realised FK speed against the cap. The pure
//! joint-space core ([`Topp3Lp`]) needs no FK at all.

mod chord;
pub mod constraints;
pub mod diagnostic;
pub mod error;
pub mod retimer;
mod solve;

pub use constraints::{Conditioning, JointLimits, TcpLimits, Topp3LpConstraints};
pub use diagnostic::Topp3LpDiagnostic;
pub use error::Topp3LpError;
pub use retimer::{Topp3Lp, Topp3LpTcp};
