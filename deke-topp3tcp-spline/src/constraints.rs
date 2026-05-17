//! Per-joint and Cartesian TCP limits, plus the bundle consumed by the
//! [`Retimer`](deke_types::Retimer) implementation.

use deke_types::SRobotQ;

/// Per-joint kinematic limits expressed in joint-space units (radians or
/// meters per second^k).
#[derive(Debug, Clone)]
pub struct JointLimits<const N: usize> {
    pub v_max: SRobotQ<N, f64>,
    pub a_max: SRobotQ<N, f64>,
    pub j_max: SRobotQ<N, f64>,
}

impl<const N: usize> JointLimits<N> {
    pub fn symmetric(v_max: f64, a_max: f64, j_max: f64) -> Self {
        Self {
            v_max: SRobotQ::from_array([v_max; N]),
            a_max: SRobotQ::from_array([a_max; N]),
            j_max: SRobotQ::from_array([j_max; N]),
        }
    }
}

/// Scalar bounds on the translational TCP trajectory.
#[derive(Debug, Clone, Copy)]
pub struct TcpLimits {
    pub v_max: f64,
    pub a_max: f64,
    pub j_max: f64,
}

impl TcpLimits {
    pub fn new(v_max: f64, a_max: f64, j_max: f64) -> Self {
        Self {
            v_max,
            a_max,
            j_max,
        }
    }
}

/// Spline path-construction options.
#[derive(Debug, Clone)]
pub struct SplinePathOptions<const N: usize> {
    /// Maximum joint-space deviation between the spline and the original
    /// polyline.
    pub max_deviation: f64,
    /// Maximum support-point refinement iterations.
    pub max_refine_iters: usize,
    /// Optional joint-space start tangent direction.
    pub start_direction: Option<SRobotQ<N, f64>>,
    /// Optional joint-space end tangent direction.
    pub end_direction: Option<SRobotQ<N, f64>>,
}

impl<const N: usize> Default for SplinePathOptions<N> {
    fn default() -> Self {
        Self {
            max_deviation: 1e-3,
            max_refine_iters: 4,
            start_direction: None,
            end_direction: None,
        }
    }
}

/// Numerical/search options for the depth-first jerk search.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    /// DFS internal time-step. The depth-first jerk search advances state
    /// by this step; smaller values increase search depth (worse-case
    /// `branch^(time/dt)`), so this is held coarse (~0.05–0.12 s) for
    /// tractability. Use [`output_dt`](Self::output_dt) to decouple the
    /// consumer-visible sample rate.
    pub dt: f64,
    /// Sub-step size used to verify constraint satisfaction within each
    /// `dt`. Must satisfy `0 < verify_dt <= dt`.
    pub verify_dt: f64,
    /// Output sample step seen by the consumer. `None` emits one sample
    /// per DFS state at [`dt`](Self::dt). `Some(h)` analytically integrates
    /// the converged `(s, sdot, sddot, sdddot)[k]` schedule within each
    /// DFS segment to produce dense samples at `h` — gives the consumer
    /// a high-rate trajectory without paying the DFS-depth cost.
    pub output_dt: Option<f64>,
    /// Number of smoothing passes applied to the per-segment jerk schedule
    /// after the DFS converges. Each pass replaces `sdddot[k]` with a
    /// binomial-weighted average of its left/right neighbours
    /// (`(sdddot[k-1] + 2·sdddot[k] + sdddot[k+1]) / 4`), with the schedule
    /// renormalized by a uniform time-scale so `s_final` lands at 1.
    /// Reduces the FD-jerk spike that the backward-FD stencil reads when
    /// it straddles a DFS segment boundary (those spikes are
    /// proportional to `|sdddot[k+1] - sdddot[k]|`, and smoothing
    /// halves the jump per pass).
    /// `0` disables smoothing.
    ///
    /// Defaults to `0`. The smoothing pass currently changes integrated
    /// arc length and forces a renormalization that can push state
    /// values past the DFS-validated analytical bounds — leaving it off
    /// for now until a richer post-DFS optimizer (full NLP, planned
    /// Phase B) lands.
    pub jerk_smoothing_passes: u32,
    /// Per-sample backward-FD readout slack used by the post-output
    /// safety pass. Any output sample whose `max_j(|reading_j|/limit_j)`
    /// exceeds `1.0 + fd_safety_slack` triggers a uniform time-rescale
    /// of the trajectory. Defaults to `0.05` (the slicer spec ceiling).
    pub fd_safety_slack: f64,
    /// Optional cap on `|sdddot[k+1] − sdddot[k]|` between consecutive
    /// DFS segments. Bounds the FD-jerk spike at segment boundaries
    /// (which is roughly proportional to `|qp| × |Δsdddot|`). When
    /// `Some(jump)`, the DFS restricts each next-segment jerk candidate
    /// to within `jump` of the current segment's jerk. Smaller values
    /// reduce the spike but shrink the DFS branching factor; too small
    /// can make the search infeasible.
    /// `None` disables the cap (default — preserve existing behaviour
    /// until benchmarks justify a value).
    pub max_jerk_jump: Option<f64>,
    /// Initial path-parameter velocity at `s = 0`.
    pub start_sdot: f64,
    /// Target path-parameter velocity at `s = 1`.
    pub end_sdot: f64,
    /// Upper bound on `sdot` used by the DFS (currently informational).
    pub max_sdot: f64,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            dt: 0.001,
            verify_dt: 0.001,
            output_dt: None,
            jerk_smoothing_passes: 0,
            fd_safety_slack: 0.05,
            max_jerk_jump: None,
            start_sdot: 0.0,
            end_sdot: 0.0,
            max_sdot: 10.0,
        }
    }
}

/// Full constraint bundle consumed by
/// [`Topp3TcpSpline::retime`](crate::Topp3TcpSpline).
#[derive(Debug, Clone)]
pub struct Topp3TcpSplineConstraints<const N: usize> {
    pub joint: JointLimits<N>,
    pub tcp: TcpLimits,
    pub path: SplinePathOptions<N>,
    pub search: SearchOptions,
}

impl<const N: usize> Topp3TcpSplineConstraints<N> {
    /// Symmetric per-axis limits and a (separate) symmetric TCP cap.
    pub fn symmetric(
        joint_v: f64,
        joint_a: f64,
        joint_j: f64,
        tcp_v: f64,
        tcp_a: f64,
        tcp_j: f64,
    ) -> Self {
        Self {
            joint: JointLimits::symmetric(joint_v, joint_a, joint_j),
            tcp: TcpLimits::new(tcp_v, tcp_a, tcp_j),
            path: SplinePathOptions::default(),
            search: SearchOptions::default(),
        }
    }
}
