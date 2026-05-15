use std::time::Duration;

use deke_types::SRobotQ;

/// Per-joint kinematic limits expressed in joint-space units (radians or meters / second^k).
#[derive(Debug, Clone)]
pub struct JointLimits<const N: usize> {
    pub q_min: SRobotQ<N, f64>,
    pub q_max: SRobotQ<N, f64>,
    pub v_max: SRobotQ<N, f64>,
    pub a_max: SRobotQ<N, f64>,
    pub j_max: SRobotQ<N, f64>,
}

impl<const N: usize> JointLimits<N> {
    pub fn symmetric(v_max: f64, a_max: f64, j_max: f64) -> Self {
        Self {
            q_min: SRobotQ::from_array([f64::NEG_INFINITY; N]),
            q_max: SRobotQ::from_array([f64::INFINITY; N]),
            v_max: SRobotQ::from_array([v_max; N]),
            a_max: SRobotQ::from_array([a_max; N]),
            j_max: SRobotQ::from_array([j_max; N]),
        }
    }
}

/// Scalar bounds on the translational TCP trajectory. `None` on `Topp3Tcp6DiscreteConstraints::tcp`
/// skips FK and every TCP row entirely.
#[derive(Debug, Clone, Copy)]
pub struct TcpLimits {
    pub v_max: f64,
    pub a_max: f64,
    pub j_max: f64,
}

/// Boundary conditions at the start and end of the trajectory.
#[derive(Debug, Clone)]
pub struct BoundaryConditions<const N: usize> {
    pub v_start: SRobotQ<N, f64>,
    pub a_start: SRobotQ<N, f64>,
    pub v_end: SRobotQ<N, f64>,
    pub a_end: SRobotQ<N, f64>,
    pub projection_tolerance: f64,
}

impl<const N: usize> BoundaryConditions<N> {
    pub fn rest_to_rest() -> Self {
        Self {
            v_start: SRobotQ::zeros(),
            a_start: SRobotQ::zeros(),
            v_end: SRobotQ::zeros(),
            a_end: SRobotQ::zeros(),
            projection_tolerance: 1e-4,
        }
    }
}

impl<const N: usize> Default for BoundaryConditions<N> {
    fn default() -> Self {
        Self::rest_to_rest()
    }
}

/// Controls how the input path is densified before retiming.
#[derive(Debug, Clone, Copy)]
pub struct DensificationOptions {
    pub max_segment_step: Option<f64>,
    pub max_samples: usize,
    pub min_samples: usize,
    pub min_segment_fraction: f64,
}

impl Default for DensificationOptions {
    fn default() -> Self {
        Self {
            max_segment_step: Some(0.05),
            max_samples: 200,
            min_samples: 10,
            min_segment_fraction: 5e-3,
        }
    }
}

/// Numerical options for the discrete-time NLP. Differs from the continuous-time
/// crate's `SolverOptions` by:
///
/// - dropping `resampled_check_slack` (the discrete formulation makes the FD check
///   strict — what the IPM enforces *is* what the consumer differences),
/// - dropping `discrete_dt` (the output sample grid is always integer multiples of
///   `1/sample_rate_hz` by construction),
/// - dropping `two_stage_warm_start` (the bisection driver supplies its own
///   per-`K` warm start by linear interpolation of `Δσ`).
///
/// New fields gate the K-bisection loop.
#[derive(Debug, Clone, Copy)]
pub struct DiscreteSolverOptions {
    pub tolerance: f64,
    pub max_iterations: i32,
    pub timeout: Option<Duration>,
    pub diagnostics: bool,
    /// Half-width of the slack box on start/end boundary FD-V/FD-A equalities. Defaults
    /// to 1e-4. The discrete formulation pins `σ[0]=0` and `σ[K-1]=S` hard (no slack);
    /// boundary V/A row equalities use this slack to keep the IPM line search away from
    /// the cone tip when `v_start`/`a_start` are zero (rest-to-rest).
    pub boundary_slack: f64,
    /// Maximum bisection iterations searching for the smallest feasible `K`. Each
    /// iteration is one NLP solve. Defaults to 12.
    pub max_bisection_iterations: usize,
    /// Penalty weight on the slack-variable sum in the bisection-mode objective.
    /// Defaults to 1e6. The objective during bisection becomes
    /// `Σ Δσ_dev² + bisection_slack_penalty · Σ s_row`. Larger values make
    /// `slack_sum < tolerance` a sharper feasibility test; too large causes the IPM
    /// to spend iterations driving slacks numerically toward 0 even at infeasible K.
    pub bisection_slack_penalty: f64,
    /// When true, the retimer pre-solves the path with the
    /// [`deke_topp_speed`] jerk-limited shaper to produce an initial `σ`
    /// profile and `K` guess. Topp-speed is fast and respects joint V/A/J
    /// plus TCP V (it does not constrain TCP A/J — those are handled by the
    /// discrete crate's TCP rows). Using its output as a warm start
    /// typically cuts the bisection's probe phase to a single iteration and
    /// substantially shortens the IPM's convergence on jerk-tight paths.
    /// Defaults to `true`. Set false to fall back to the uniform-σ initial
    /// guess (useful for benchmarking the seed's impact).
    pub seed_from_topp_speed: bool,
}

impl Default for DiscreteSolverOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_iterations: 2500,
            timeout: None,
            diagnostics: false,
            boundary_slack: 1e-4,
            max_bisection_iterations: 12,
            bisection_slack_penalty: 1e6,
            seed_from_topp_speed: true,
        }
    }
}

/// Full constraint bundle consumed by [`crate::Topp3Tcp6Discrete::retime`].
#[derive(Debug, Clone)]
pub struct Topp3Tcp6DiscreteConstraints<const N: usize> {
    pub joint: JointLimits<N>,
    pub tcp: Option<TcpLimits>,
    pub boundary: BoundaryConditions<N>,
    pub densification: DensificationOptions,
    pub solver: DiscreteSolverOptions,
    /// Output sample rate in Hz. Output `dt = 1/sample_rate_hz` exactly.
    pub sample_rate_hz: f64,
    pub locked_prefix: usize,
    pub post_validation: bool,
    /// When `true`, post-solve verification asserts the strict FD check on the output
    /// samples (joint and TCP V/A/J) lies under the configured limits within
    /// `solver.tolerance` — no extra slack. A violation returns
    /// [`deke_types::DekeError::ExceedsDynamicsLimits`] and is treated as a bug.
    pub check_output_dynamics: bool,
}

impl<const N: usize> Topp3Tcp6DiscreteConstraints<N> {
    /// Convenience constructor: no TCP, symmetric joint limits, rest-to-rest boundary,
    /// no locked joints, 125 Hz output.
    pub fn symmetric(v_max: f64, a_max: f64, j_max: f64) -> Self {
        Self {
            joint: JointLimits::symmetric(v_max, a_max, j_max),
            tcp: None,
            boundary: BoundaryConditions::rest_to_rest(),
            densification: DensificationOptions::default(),
            solver: DiscreteSolverOptions::default(),
            sample_rate_hz: 125.0,
            locked_prefix: 0,
            post_validation: true,
            check_output_dynamics: true,
        }
    }
}
