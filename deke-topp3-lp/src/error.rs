use deke_types::DekeError;

/// Errors surfaced by the joint-space convex-LP retimer.
#[derive(Debug, thiserror::Error)]
pub enum Topp3LpError {
    #[error("path needs at least 2 waypoints, got {0}")]
    TooShort(usize),
    #[error("path collapses to a single point (zero arc length) — nothing to time")]
    Degenerate,
    #[error(
        "the discrete program is infeasible even at the grown horizon — the path is likely too curved to time under the limits"
    )]
    Infeasible,
    #[error(
        "cannot keep joint {joint} under its {kind} limit ({value:.4} > {limit:.4}) — the joint path is too curved here for these limits; densify the path or relax the limits"
    )]
    JointLimitExceeded {
        joint: usize,
        /// `"velocity"`, `"acceleration"`, or `"jerk"`.
        kind: &'static str,
        value: f64,
        limit: f64,
    },
    #[error(
        "cannot keep the TCP linear speed under its cap ({value:.4} > {limit:.4} m/s) — the path is too curved in task space for this cap"
    )]
    TcpLimitExceeded { value: f64, limit: f64 },
    #[error("a TCP-velocity cap requires an FK chain — use `Topp3LpTcp`, not `Topp3Lp`")]
    TcpNeedsFk,
    #[error("output_dt must be finite and at least 1 microsecond")]
    InvalidOutputDt,
    #[error("the path contains non-finite (NaN/inf) joint values")]
    NonFinitePath,
    #[error("joint and TCP limits must be finite and strictly positive")]
    InvalidLimits,
    #[error(
        "the path needs more than the {max}-tick grid budget to time under these limits at this output_dt — raise output_dt or relax the limits"
    )]
    TickBudgetExceeded { max: usize },
    #[error(transparent)]
    Deke(#[from] DekeError),
}

impl From<Topp3LpError> for DekeError {
    fn from(e: Topp3LpError) -> Self {
        match e {
            Topp3LpError::Deke(d) => d,
            other => DekeError::RetimerFailed(other.to_string()),
        }
    }
}
