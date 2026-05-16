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
    /// Output time-step (seconds between consecutive states).
    pub dt: f64,
    /// Sub-step size used to verify constraint satisfaction within each
    /// `dt`. Must satisfy `0 < verify_dt <= dt`.
    pub verify_dt: f64,
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
