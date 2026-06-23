use deke_types::DekeError;

/// Errors surfaced by the constant-speed Cartesian follower.
#[derive(Debug, thiserror::Error)]
pub enum LinearError {
    #[error("need at least 2 poses to define a path, got {0}")]
    TooFewPoses(usize),
    #[error("run {run}: every vertex coincides — nothing to follow")]
    DegenerateRun { run: usize },
    #[error("run {run}: Cartesian pose at arc length {s:.4} m has no reachable IK solution")]
    Unreachable { run: usize, s: f64 },
    #[error(
        "run {run}: every reachable configuration at arc length {s:.4} m is rejected by the validator (obstructed)"
    )]
    Obstructed { run: usize, s: f64 },
    #[error("run {run}: could not route a continuous joint track through the IK candidates")]
    NoContinuousTrack { run: usize },
    #[error(
        "retimer stalled on run {run} near arc length {s:.4} m (path likely passes through a singularity)"
    )]
    Stalled { run: usize, s: f64 },
    #[error(
        "run {run}: holding constant TCP speed forbids the interior dip at arc length {s:.4} m (max feasible {feasible_speed:.4} m/s < commanded {commanded:.4} m/s)"
    )]
    SpeedDipRequired {
        run: usize,
        s: f64,
        feasible_speed: f64,
        commanded: f64,
    },
    #[error(
        "run {run}: cannot keep joint {joint} under its {kind} limit at arc length {s:.4} m ({value:.3} > {limit:.3}) — the joint path is too curved here for the commanded speed; smooth the path or lower the speed"
    )]
    LimitExceeded {
        run: usize,
        s: f64,
        joint: usize,
        /// `"velocity"`, `"acceleration"`, or `"jerk"`.
        kind: &'static str,
        value: f64,
        limit: f64,
    },
    #[error(
        "run {run}: cannot keep the TCP {kind} under its limit ({value:.4} > {limit:.4}) — the commanded speed is too high for the path here"
    )]
    TcpLimitExceeded {
        run: usize,
        /// `"acceleration"` or `"jerk"`.
        kind: &'static str,
        value: f64,
        limit: f64,
    },
    #[error(transparent)]
    Deke(#[from] DekeError),
}

impl From<LinearError> for DekeError {
    fn from(e: LinearError) -> Self {
        match e {
            LinearError::Deke(d) => d,
            other => DekeError::RetimerFailed(other.to_string()),
        }
    }
}
