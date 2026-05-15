//! Status returned by per-tick operations on the live pursuer.

/// Outcome of a single tick or solve step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepStatus {
    /// The motion is still being executed.
    InProgress,
    /// The motion has reached the goal state.
    Done,
    /// The motion is held in place (e.g. paused).
    Held,
    /// Generic failure.
    Failure,
    /// The supplied [`crate::MotionSpec`] could not be validated.
    BadInput,
    /// The requested trajectory duration is infeasible under the limits.
    DurationInfeasible,
    /// One or more axes would exceed their pose ceiling/floor.
    PoseOverrun,
    /// At least one kinematic ceiling was set to zero.
    ZeroLimit,
    /// The per-axis time-bound (Step-A) computation failed.
    StepOneFailed,
    /// The cross-axis time-synchronisation (Step-B) computation failed.
    StepTwoFailed,
    /// Phase-locked coordination was requested but no consistent phase exists.
    NoPhaseSync,
}

impl StepStatus {
    /// `true` for the three nominal status values (`InProgress`, `Done`, `Held`).
    pub fn is_ok(self) -> bool {
        matches!(self, Self::InProgress | Self::Done | Self::Held)
    }

    /// `true` for any non-OK status.
    pub fn is_err(self) -> bool {
        !self.is_ok()
    }
}
