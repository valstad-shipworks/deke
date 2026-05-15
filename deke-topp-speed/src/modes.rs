//! Mode enums controlling how the solver shapes a trajectory.

/// Selects whether the solver tracks a goal pose or a goal velocity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ControlMode {
    /// Drive the system to a goal position with optional terminal velocity / accel.
    #[default]
    Position,
    /// Drive the system to a goal velocity (terminal pose is unconstrained).
    Velocity,
}

/// Selects how the per-axis timings are coordinated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Coordination {
    /// All axes finish at exactly the same time (strict).
    #[default]
    TimeLocked,
    /// All axes finish at the same time only if needed to stay feasible.
    TimeLockedSoft,
    /// Per-axis profiles share a single ramp shape (phase coupling).
    PhaseLocked,
    /// Each axis runs independently.
    Independent,
}

/// Whether the solver may pick any real-valued duration or must snap to a
/// control-cycle multiple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DurationGrid {
    /// Any real-valued duration is allowed.
    #[default]
    Smooth,
    /// Duration is snapped to the nearest control-cycle multiple.
    Quantized,
}

/// Behaviour when the requested goal state falls outside the kinematic limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GoalOutOfBounds {
    /// Refuse the solve and return an error.
    #[default]
    Reject,
    /// Clip the goal to the limits and continue.
    Clip,
}

/// Pursuit strategy used by the live tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FollowMode {
    /// Iterate on lookahead time to minimise jerk-limited overshoot.
    #[default]
    Tuned,
    /// Single-pass analytic correction; lower CPU cost.
    Quick,
}
