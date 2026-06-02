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
    KSampleCountAtCeiling,
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
            Self::KSampleCountAtCeiling => "bisection K ceiling reached",
        };
        f.write_str(s)
    }
}

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

#[derive(Debug, Clone, Copy, Default)]
pub struct PeakLocation {
    pub value: f64,
    pub sample: usize,
    pub joint: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PathStats {
    pub input_waypoints: usize,
    pub merged_waypoints: usize,
    pub chord_length: f64,
    pub min_segment_length: f64,
    pub max_segment_length: f64,
    pub segment_length_ratio: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DerivativeStats {
    pub max_abs_qpp: f64,
    pub max_abs_qpp_sample: usize,
    pub max_abs_qpp_joint: usize,
    pub max_abs_qppp: f64,
    pub max_abs_qppp_sample: usize,
    pub max_abs_qppp_joint: usize,
    pub min_qp_norm_relative_sq: f64,
    pub min_qp_norm_sample: usize,
    pub degenerate_qp_samples: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TcpStats {
    pub max_abs_pp: f64,
    pub max_abs_ppp: f64,
    pub max_abs_pppp: f64,
    pub min_abs_pp_per_axis: [f64; 3],
    pub max_abs_pp_per_axis: [f64; 3],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintCounts {
    pub joint_v: usize,
    pub joint_a: usize,
    pub joint_j: usize,
    pub tcp_v: usize,
    pub tcp_a: usize,
    pub tcp_j: usize,
    pub sigma_chain: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTiming {
    pub densify: Duration,
    pub derivatives: Duration,
    pub nlp_build: Duration,
    pub nlp_solve: Duration,
    pub verify: Duration,
}

/// Per-limit-type peak FD residual against the configured bound. All values are
/// `observed / limit − 1` (so 0.0 means exactly on the limit, positive means
/// over-limit). The strict verifier returns an error when any of these exceed
/// the IPM tolerance.
#[derive(Debug, Clone, Copy, Default)]
pub struct PerLimitResidual {
    pub joint_v: f64,
    pub joint_a: f64,
    pub joint_j: f64,
    pub tcp_v: f64,
    pub tcp_a: f64,
    pub tcp_j: f64,
}

/// One step of the bisection driver over the output sample count `K`.
#[derive(Debug, Clone, Copy)]
pub struct BisectionStep {
    pub k: usize,
    pub status: SolveStatus,
    /// Sum of the slack-variable values at convergence — when this is below
    /// `solver.tolerance` the IPM treated `K` as feasible, otherwise it is treated
    /// as infeasible and the lower bound is tightened.
    pub slack_sum: f64,
    pub iter: i32,
    pub solve_time: Duration,
}

#[derive(Debug, Clone)]
pub struct Topp3Tcp6DiscreteDiagnostic {
    pub status: SolveStatus,
    pub iterations: i32,
    pub solve_time: Duration,
    pub solver_tolerance_used: f64,
    pub densified_samples: usize,
    pub output_samples: usize,
    pub total_time: Duration,
    pub peak_joint_velocity: f64,
    pub peak_joint_acceleration: f64,
    pub peak_joint_jerk: f64,
    pub peak_tcp_velocity: f64,
    pub peak_tcp_acceleration: f64,
    pub peak_tcp_jerk: f64,
    pub average_utilization: f64,
    pub boundary_projection_residual: f64,
    pub limiting_constraint: Option<LimitingGroup>,
    pub limiting_sample: Option<usize>,
    pub message: Option<String>,

    pub path_stats: PathStats,
    pub derivative_stats: DerivativeStats,
    pub tcp_stats: TcpStats,
    pub peak_joint_velocity_at: PeakLocation,
    pub peak_joint_acceleration_at: PeakLocation,
    pub peak_joint_jerk_at: PeakLocation,
    pub peak_tcp_velocity_at: PeakLocation,
    pub peak_tcp_acceleration_at: PeakLocation,
    pub peak_tcp_jerk_at: PeakLocation,
    pub constraint_counts: ConstraintCounts,
    pub phase_timing: PhaseTiming,

    // ── discrete-specific ────────────────────────────────────────────────
    pub bisection_steps: Vec<BisectionStep>,
    pub final_k: usize,
    /// Strict-FD peak overshoot per limit type, computed by [`super::verify`] on the
    /// output sample sequence. Each value is `(observed − limit) / limit` clipped
    /// at 0 from below.
    pub output_fd_residual: PerLimitResidual,
}

impl Default for Topp3Tcp6DiscreteDiagnostic {
    fn default() -> Self {
        Self {
            status: SolveStatus::NotAttempted,
            iterations: 0,
            solve_time: Duration::ZERO,
            solver_tolerance_used: 0.0,
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
            limiting_sample: None,
            message: None,
            path_stats: PathStats::default(),
            derivative_stats: DerivativeStats::default(),
            tcp_stats: TcpStats::default(),
            peak_joint_velocity_at: PeakLocation::default(),
            peak_joint_acceleration_at: PeakLocation::default(),
            peak_joint_jerk_at: PeakLocation::default(),
            peak_tcp_velocity_at: PeakLocation::default(),
            peak_tcp_acceleration_at: PeakLocation::default(),
            peak_tcp_jerk_at: PeakLocation::default(),
            constraint_counts: ConstraintCounts::default(),
            phase_timing: PhaseTiming::default(),
            bisection_steps: Vec::new(),
            final_k: 0,
            output_fd_residual: PerLimitResidual::default(),
        }
    }
}

impl fmt::Display for Topp3Tcp6DiscreteDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TOPP3TCP6-discrete retimer diagnostic")?;
        writeln!(
            f,
            "  status          : {} ({} iter, {:.3}s, tol={:.0e})",
            self.status,
            self.iterations,
            self.solve_time.as_secs_f64(),
            self.solver_tolerance_used,
        )?;
        writeln!(
            f,
            "  samples         : {} input → {} merged → {} densified → {} output (K={})",
            self.path_stats.input_waypoints,
            self.path_stats.merged_waypoints,
            self.densified_samples,
            self.output_samples,
            self.final_k,
        )?;
        writeln!(
            f,
            "  path geometry   : len={:.4} m, segs∈[{:.3e}, {:.3e}] (ratio {:.2}×)",
            self.path_stats.chord_length,
            self.path_stats.min_segment_length,
            self.path_stats.max_segment_length,
            self.path_stats.segment_length_ratio,
        )?;
        writeln!(f, "  total time      : {:.4}s", self.total_time.as_secs_f64())?;
        writeln!(
            f,
            "  peak joint v/a/j: {:.3} / {:.3} / {:.3}",
            self.peak_joint_velocity,
            self.peak_joint_acceleration,
            self.peak_joint_jerk,
        )?;
        writeln!(
            f,
            "  peak TCP   v/a/j: {:.3} / {:.3} / {:.3}",
            self.peak_tcp_velocity, self.peak_tcp_acceleration, self.peak_tcp_jerk,
        )?;
        writeln!(
            f,
            "  FD residual     : jv={:+.2e} ja={:+.2e} jj={:+.2e} | tv={:+.2e} ta={:+.2e} tj={:+.2e}",
            self.output_fd_residual.joint_v,
            self.output_fd_residual.joint_a,
            self.output_fd_residual.joint_j,
            self.output_fd_residual.tcp_v,
            self.output_fd_residual.tcp_a,
            self.output_fd_residual.tcp_j,
        )?;
        writeln!(
            f,
            "  bisection       : {} step(s), final K={}",
            self.bisection_steps.len(),
            self.final_k,
        )?;
        if let Some(group) = self.limiting_constraint {
            match self.limiting_sample {
                Some(s) => writeln!(f, "  limiting        : {} at s={}", group, s)?,
                None => writeln!(f, "  limiting        : {}", group)?,
            }
        }
        if let Some(msg) = &self.message {
            writeln!(f, "  message         : {}", msg)?;
        }
        Ok(())
    }
}
