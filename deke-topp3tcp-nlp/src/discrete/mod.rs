//! Discrete-time TOPP-3TCP retimer.
//!
//! Variables of the underlying NLP are the path-arc-length values `σ[i]` at each output sample,
//! not the continuous `(sd, sdd, sddd, dt)` quadruple per densified knot. Each kinematic-bound row
//! is a backward-difference of chord-linear joint positions over the σ chain — the exact same FD
//! that downstream consumers compute on the output trajectory — so the IPM enforces what the
//! consumer measures. The strict verifier in [`verify`] returns `Ok` to within IPM tolerance on
//! every retime; no `resampled_check_slack` is needed.

pub mod constraints;
pub mod diagnostic;
pub mod nlp;
pub mod retimer;
pub mod verify;

pub use constraints::{
    BoundaryConditions, DensificationOptions, DiscreteSolverOptions, JointLimits, TcpLimits,
    Topp3Tcp6DiscreteConstraints,
};
pub use diagnostic::{
    BisectionStep, ConstraintCounts, DerivativeStats, LimitingGroup, PathStats, PeakLocation,
    PerLimitResidual, PhaseTiming, SolveStatus, TcpStats, Topp3Tcp6DiscreteDiagnostic,
};
pub use crate::common::path_derivatives::PathDerivatives;
pub use retimer::Topp3Tcp6Discrete;
