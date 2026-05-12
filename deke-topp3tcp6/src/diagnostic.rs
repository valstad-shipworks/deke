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

/// A peak value measured over the trajectory together with where on the path it occurred.
/// `joint` is `None` when the peak is a TCP-side scalar (no joint axis applies).
#[derive(Debug, Clone, Copy, Default)]
pub struct PeakLocation {
    pub value: f64,
    pub sample: usize,
    pub joint: Option<usize>,
}

/// Scalar summary of the densified path's geometry. Computed before the NLP runs, so it
/// is populated even on solver failures.
#[derive(Debug, Clone, Copy, Default)]
pub struct PathStats {
    /// Number of waypoints in the user-supplied input.
    pub input_waypoints: usize,
    /// Waypoints remaining after the near-duplicate merge step (and before densification).
    pub merged_waypoints: usize,
    /// Total chord length of the densified path, in joint-space units.
    pub chord_length: f64,
    pub min_segment_length: f64,
    pub max_segment_length: f64,
    /// `max_segment_length / min_segment_length`. Values much greater than 1 mean the
    /// integrator equalities will see wildly different `ds[k]` factors across segments,
    /// which the IPM cannot scale away.
    pub segment_length_ratio: f64,
}

/// Stats over the path's joint-space derivatives (PCHIP `qp`, `qpp`, `qppp`). High
/// magnitudes in `qpp`/`qppp` typically mean PCHIP overshoot — a corner sample where the
/// 2nd-derivative averaging across two dissimilar segments produces a large value.
#[derive(Debug, Clone, Copy, Default)]
pub struct DerivativeStats {
    pub max_abs_qpp: f64,
    pub max_abs_qpp_sample: usize,
    pub max_abs_qpp_joint: usize,
    pub max_abs_qppp: f64,
    pub max_abs_qppp_sample: usize,
    pub max_abs_qppp_joint: usize,
    /// `min_k ‖qp[k]‖² / max_k ‖qp[k]‖²` — how close a sample gets to PCHIP-degenerate.
    /// Values below ~1e-6 mean at least one sample's `qp` is effectively zero and was
    /// either repaired by centered-FD fallback or skipped entirely from joint constraints.
    pub min_qp_norm_relative_sq: f64,
    pub min_qp_norm_sample: usize,
    /// Samples whose joint-side constraints were fully skipped because every joint's `qp_j`
    /// fell below the per-joint relative cutoff.
    pub degenerate_qp_samples: usize,
}

/// Stats over the path's TCP-space derivatives (`pp`, `ppp`, `pppp`). Per-axis `pp`
/// magnitudes flag the TCP-scaling-collapse failure mode: when one axis is near-zero
/// everywhere along the path, the squared-norm TCP constraint has a near-zero row in
/// its gradient and Sleipnir's KKT factorization can fall over.
#[derive(Debug, Clone, Copy, Default)]
pub struct TcpStats {
    pub max_abs_pp: f64,
    pub max_abs_ppp: f64,
    pub max_abs_pppp: f64,
    pub min_abs_pp_per_axis: [f64; 3],
    pub max_abs_pp_per_axis: [f64; 3],
}

/// How many inequalities of each group were actually added to the NLP. Useful as a sanity
/// check against the configuration: a TCP-bounded retime that reports `tcp_a = 0` means
/// every sample was at the boundary or had `qp` degenerate, which is rarely intended.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintCounts {
    pub joint_v: usize,
    pub joint_a: usize,
    pub joint_j: usize,
    pub tcp_v: usize,
    pub tcp_a: usize,
    pub tcp_j: usize,
}

/// Diagnostics on the integrator-consistent forward-propagation initial guess. The start
/// boundary on `sd`/`sdd` is satisfied by construction; the end boundary may have a
/// residual that the boundary slack box absorbs. Large residuals here predict slow
/// convergence or restoration-phase entry.
#[derive(Debug, Clone, Copy, Default)]
pub struct InitialGuessStats {
    pub end_sd_residual: f64,
    pub end_sdd_residual: f64,
    pub max_sddd: f64,
    pub max_sddd_segment: usize,
}

/// How much of [`crate::SolverOptions::boundary_slack`] each side ate after the solve.
/// Values close to the configured slack mean the IPM was *constrained by the slack* and
/// would have been infeasible with hard equalities.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundarySlackUsage {
    pub start_sd: f64,
    pub start_sdd: f64,
    pub end_sd: f64,
    pub end_sdd: f64,
}

/// Wall-time spent in each phase of [`crate::Topp3Tcp6::retime`]. Intended for spotting
/// performance regressions; does not include time spent inside the user's `Validator`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTiming {
    pub densify: Duration,
    pub derivatives: Duration,
    pub nlp_build: Duration,
    pub nlp_solve: Duration,
    pub resample: Duration,
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
    pub peak_joint_velocity: f64,
    pub peak_joint_acceleration: f64,
    pub peak_joint_jerk: f64,
    pub peak_tcp_velocity: f64,
    pub peak_tcp_acceleration: f64,
    pub peak_tcp_jerk: f64,
    /// Mean across all waypoints of the per-step max limit utilization.
    /// Each joint (v/a/j) and TCP (v/a/j) limit is treated independently; at every waypoint the
    /// utilization is `max_i(|q_i| / limit_i)` over every finite-bounded limit. A value near 1.0
    /// means the solver is driving against some limit on every step.
    pub average_utilization: f64,
    pub boundary_projection_residual: f64,
    pub limiting_constraint: Option<LimitingGroup>,
    /// Densified-path sample index where the limiting constraint was tightest. Populated
    /// alongside `limiting_constraint`; `None` on success or when the failure has no
    /// well-defined sample location (e.g. boundary projection rejection).
    pub limiting_sample: Option<usize>,
    pub message: Option<String>,

    // ── extended diagnostics for failure triage ────────────────────────────────
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
    pub initial_guess: InitialGuessStats,
    pub boundary_slack_usage: BoundarySlackUsage,
    pub phase_timing: PhaseTiming,
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
            initial_guess: InitialGuessStats::default(),
            boundary_slack_usage: BoundarySlackUsage::default(),
            phase_timing: PhaseTiming::default(),
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
            "  samples         : {} input → {} merged → {} densified → {} output",
            self.path_stats.input_waypoints,
            self.path_stats.merged_waypoints,
            self.densified_samples,
            self.output_samples
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
            "  peak joint v/a/j: {:.3} (s={}, j={}) / {:.3} (s={}, j={}) / {:.3} (s={}, j={})",
            self.peak_joint_velocity,
            self.peak_joint_velocity_at.sample,
            self.peak_joint_velocity_at.joint.unwrap_or(usize::MAX),
            self.peak_joint_acceleration,
            self.peak_joint_acceleration_at.sample,
            self.peak_joint_acceleration_at.joint.unwrap_or(usize::MAX),
            self.peak_joint_jerk,
            self.peak_joint_jerk_at.sample,
            self.peak_joint_jerk_at.joint.unwrap_or(usize::MAX),
        )?;
        writeln!(
            f,
            "  peak TCP   v/a/j: {:.3} (s={}) / {:.3} (s={}) / {:.3} (s={})",
            self.peak_tcp_velocity,
            self.peak_tcp_velocity_at.sample,
            self.peak_tcp_acceleration,
            self.peak_tcp_acceleration_at.sample,
            self.peak_tcp_jerk,
            self.peak_tcp_jerk_at.sample,
        )?;
        writeln!(
            f,
            "  derivatives     : max|qpp|={:.3e} (s={}, j={}), max|qppp|={:.3e} (s={}, j={})",
            self.derivative_stats.max_abs_qpp,
            self.derivative_stats.max_abs_qpp_sample,
            self.derivative_stats.max_abs_qpp_joint,
            self.derivative_stats.max_abs_qppp,
            self.derivative_stats.max_abs_qppp_sample,
            self.derivative_stats.max_abs_qppp_joint,
        )?;
        writeln!(
            f,
            "  qp degeneracy   : min‖qp‖²(rel)={:.3e} at s={}, {} samples skipped",
            self.derivative_stats.min_qp_norm_relative_sq,
            self.derivative_stats.min_qp_norm_sample,
            self.derivative_stats.degenerate_qp_samples,
        )?;
        writeln!(
            f,
            "  TCP geometry    : max|pp|={:.3e}, max|ppp|={:.3e}, max|pppp|={:.3e}",
            self.tcp_stats.max_abs_pp,
            self.tcp_stats.max_abs_ppp,
            self.tcp_stats.max_abs_pppp,
        )?;
        writeln!(
            f,
            "  TCP per-axis |pp| min/max: x=[{:.3e},{:.3e}] y=[{:.3e},{:.3e}] z=[{:.3e},{:.3e}]",
            self.tcp_stats.min_abs_pp_per_axis[0],
            self.tcp_stats.max_abs_pp_per_axis[0],
            self.tcp_stats.min_abs_pp_per_axis[1],
            self.tcp_stats.max_abs_pp_per_axis[1],
            self.tcp_stats.min_abs_pp_per_axis[2],
            self.tcp_stats.max_abs_pp_per_axis[2],
        )?;
        writeln!(
            f,
            "  constraints     : joint v/a/j = {}/{}/{}, TCP v/a/j = {}/{}/{}",
            self.constraint_counts.joint_v,
            self.constraint_counts.joint_a,
            self.constraint_counts.joint_j,
            self.constraint_counts.tcp_v,
            self.constraint_counts.tcp_a,
            self.constraint_counts.tcp_j,
        )?;
        writeln!(
            f,
            "  initial guess   : end residuals sd={:.3e}, sdd={:.3e}; max|sddd|={:.3e} (seg={})",
            self.initial_guess.end_sd_residual,
            self.initial_guess.end_sdd_residual,
            self.initial_guess.max_sddd,
            self.initial_guess.max_sddd_segment,
        )?;
        writeln!(
            f,
            "  slack used      : sd[0]={:.3e}, sdd[0]={:.3e}, sd[end]={:.3e}, sdd[end]={:.3e}",
            self.boundary_slack_usage.start_sd,
            self.boundary_slack_usage.start_sdd,
            self.boundary_slack_usage.end_sd,
            self.boundary_slack_usage.end_sdd,
        )?;
        writeln!(
            f,
            "  phase timing    : densify={:.3}s, deriv={:.3}s, build={:.3}s, solve={:.3}s, resample={:.3}s",
            self.phase_timing.densify.as_secs_f64(),
            self.phase_timing.derivatives.as_secs_f64(),
            self.phase_timing.nlp_build.as_secs_f64(),
            self.phase_timing.nlp_solve.as_secs_f64(),
            self.phase_timing.resample.as_secs_f64(),
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
