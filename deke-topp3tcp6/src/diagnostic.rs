use std::fmt;
use std::time::Duration;

/// Classifies which bound group became active (or infeasible) during the solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitingGroup {
    JointVelocity,
    JointAcceleration,
    JointJerk,
    TcpVelocity,
    TcpAcceleration,
    TcpJerk,
    BoundaryCondition,
    TimestepLowerBound,
}

impl fmt::Display for LimitingGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::JointVelocity => "joint velocity",
            Self::JointAcceleration => "joint acceleration",
            Self::JointJerk => "joint jerk",
            Self::TcpVelocity => "TCP velocity",
            Self::TcpAcceleration => "TCP acceleration",
            Self::TcpJerk => "TCP jerk",
            Self::BoundaryCondition => "boundary condition",
            Self::TimestepLowerBound => "timestep lower bound",
        };
        f.write_str(s)
    }
}

/// Status of the underlying NLP solve. Mirrors `hafgufa::ExitStatus` but stays crate-private so
/// callers do not need to depend on sleipnir directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    Success,
    CallbackRequestedStop,
    TooFewDofs,
    LocallyInfeasible,
    GloballyInfeasible,
    FactorizationFailed,
    LineSearchFailed,
    FeasibilityRestorationFailed,
    NonfiniteInitialGuess,
    DivergingIterates,
    MaxIterationsExceeded,
    Timeout,
    NotAttempted,
}

impl fmt::Display for SolveStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Success => "success",
            Self::CallbackRequestedStop => "callback requested stop",
            Self::TooFewDofs => "too few degrees of freedom",
            Self::LocallyInfeasible => "locally infeasible",
            Self::GloballyInfeasible => "globally infeasible",
            Self::FactorizationFailed => "factorization failed",
            Self::LineSearchFailed => "line search failed",
            Self::FeasibilityRestorationFailed => "feasibility restoration failed",
            Self::NonfiniteInitialGuess => "non-finite initial guess",
            Self::DivergingIterates => "diverging iterates",
            Self::MaxIterationsExceeded => "max iterations exceeded",
            Self::Timeout => "timeout",
            Self::NotAttempted => "not attempted",
        };
        f.write_str(s)
    }
}

impl From<hafgufa::ExitStatus> for SolveStatus {
    fn from(value: hafgufa::ExitStatus) -> Self {
        match value {
            hafgufa::ExitStatus::Success => Self::Success,
            hafgufa::ExitStatus::CallbackRequestedStop => Self::CallbackRequestedStop,
            hafgufa::ExitStatus::TooFewDofs => Self::TooFewDofs,
            hafgufa::ExitStatus::LocallyInfeasible => Self::LocallyInfeasible,
            hafgufa::ExitStatus::GloballyInfeasible => Self::GloballyInfeasible,
            hafgufa::ExitStatus::FactorizationFailed => Self::FactorizationFailed,
            hafgufa::ExitStatus::LineSearchFailed => Self::LineSearchFailed,
            hafgufa::ExitStatus::FeasibilityRestorationFailed => Self::FeasibilityRestorationFailed,
            hafgufa::ExitStatus::NonfiniteInitialGuess => Self::NonfiniteInitialGuess,
            hafgufa::ExitStatus::DivergingIterates => Self::DivergingIterates,
            hafgufa::ExitStatus::MaxIterationsExceeded => Self::MaxIterationsExceeded,
            hafgufa::ExitStatus::Timeout => Self::Timeout,
        }
    }
}

/// Report produced by every [`crate::Topp3Tcp6::retime`] invocation, regardless of success.
#[derive(Debug, Clone)]
pub struct Topp3Tcp6Diagnostic {
    pub status: SolveStatus,
    pub iterations: i32,
    pub solve_time: Duration,
    pub densified_samples: usize,
    pub output_samples: usize,
    pub total_time: Duration,
    pub peak_joint_velocity: f32,
    pub peak_joint_acceleration: f32,
    pub peak_joint_jerk: f32,
    pub peak_tcp_velocity: f32,
    pub peak_tcp_acceleration: f32,
    pub peak_tcp_jerk: f32,
    /// Mean across all waypoints of the per-step max limit utilization.
    /// Each joint (v/a/j) and TCP (v/a/j) limit is treated independently; at every waypoint the
    /// utilization is `max_i(|q_i| / limit_i)` over every finite-bounded limit. A value near 1.0
    /// means the solver is driving against some limit on every step.
    pub average_utilization: f32,
    pub boundary_projection_residual: f32,
    pub limiting_constraint: Option<LimitingGroup>,
    pub message: Option<String>,
}

impl Default for Topp3Tcp6Diagnostic {
    fn default() -> Self {
        Self {
            status: SolveStatus::NotAttempted,
            iterations: 0,
            solve_time: Duration::ZERO,
            densified_samples: 0,
            output_samples: 0,
            total_time: Duration::ZERO,
            peak_joint_velocity: 0.0,
            peak_joint_acceleration: 0.0,
            peak_joint_jerk: 0.0,
            peak_tcp_velocity: 0.0,
            peak_tcp_acceleration: 0.0,
            peak_tcp_jerk: 0.0,
            average_utilization: 0.0,
            boundary_projection_residual: 0.0,
            limiting_constraint: None,
            message: None,
        }
    }
}

impl fmt::Display for Topp3Tcp6Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TOPP3TCP6 retimer diagnostic")?;
        writeln!(
            f,
            "  status          : {} ({} iter, {:.3}s)",
            self.status,
            self.iterations,
            self.solve_time.as_secs_f64()
        )?;
        writeln!(
            f,
            "  samples         : {} densified → {} output",
            self.densified_samples, self.output_samples
        )?;
        writeln!(f, "  total time      : {:.4}s", self.total_time.as_secs_f64())?;
        writeln!(
            f,
            "  peak joint v/a/j: {:.3} / {:.3} / {:.3}",
            self.peak_joint_velocity, self.peak_joint_acceleration, self.peak_joint_jerk
        )?;
        writeln!(
            f,
            "  peak TCP   v/a/j: {:.3} / {:.3} / {:.3}",
            self.peak_tcp_velocity, self.peak_tcp_acceleration, self.peak_tcp_jerk
        )?;
        writeln!(
            f,
            "  avg utilization : {:.1}%",
            self.average_utilization * 100.0
        )?;
        writeln!(
            f,
            "  boundary residual: {:.3e}",
            self.boundary_projection_residual
        )?;
        if let Some(group) = self.limiting_constraint {
            writeln!(f, "  limiting        : {}", group)?;
        }
        if let Some(msg) = &self.message {
            writeln!(f, "  message         : {}", msg)?;
        }
        Ok(())
    }
}
