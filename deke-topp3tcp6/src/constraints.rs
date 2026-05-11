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
    /// Symmetric bounds with infinite positional range. Velocity, acceleration and jerk are each
    /// set to the provided scalar on every joint.
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

/// Scalar bounds on the translational component of the TCP (tool center point) trajectory.
/// Rotational TCP bounds are out of scope for the v1 retimer.
#[derive(Debug, Clone, Copy)]
pub struct TcpLimits {
    pub v_max: f64,
    pub a_max: f64,
    pub j_max: f64,
}

impl TcpLimits {
    pub fn unbounded() -> Self {
        Self {
            v_max: f64::INFINITY,
            a_max: f64::INFINITY,
            j_max: f64::INFINITY,
        }
    }

    /// Returns true if every bound is either zero or non-finite (infinity / NaN). In that case
    /// the retimer can skip running forward kinematics on every densified waypoint and skip
    /// every TCP constraint in the NLP, which is a big win for TCP-unconstrained problems.
    pub fn is_disabled(&self) -> bool {
        let inactive = |v: f64| v == 0.0 || !v.is_finite();
        inactive(self.v_max) && inactive(self.a_max) && inactive(self.j_max)
    }
}

/// Boundary conditions at the start and end of the trajectory.
/// The user supplies joint-space velocity and acceleration vectors; the retimer projects them
/// onto the path tangent and reports any residual as a pre-flight error.
#[derive(Debug, Clone)]
pub struct BoundaryConditions<const N: usize> {
    pub v_start: SRobotQ<N, f64>,
    pub a_start: SRobotQ<N, f64>,
    pub v_end: SRobotQ<N, f64>,
    pub a_end: SRobotQ<N, f64>,
    /// Maximum allowed perpendicular-component norm during projection, in joint-space units.
    /// Defaults to 1e-4.
    pub projection_tolerance: f64,
}

impl<const N: usize> BoundaryConditions<N> {
    /// Rest-to-rest boundary condition: zero velocity and acceleration at both ends.
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
    /// Upper bound on the joint-space distance between consecutive densified waypoints.
    /// `None` disables densification (rarely desirable).
    pub max_segment_step: Option<f64>,
    /// Hard cap on the number of densified waypoints — the retimer downsamples uniformly if the
    /// densified path exceeds this.
    pub max_samples: usize,
    /// Minimum number of densified waypoints. Small paths are densified at least this far so the
    /// finite-difference path derivatives stay meaningful.
    pub min_samples: usize,
    /// Pre-densification waypoint merge threshold, expressed as a fraction of the mean
    /// segment length of the input path. Any interior waypoint whose chord distance to the
    /// previous kept waypoint is below this threshold is dropped before densification — a
    /// path with one 1e-6 segment and one 0.7 segment otherwise produces an integrator
    /// equality whose `ds[k]` factors range across six orders of magnitude across adjacent
    /// segments, which the IPM cannot scale away.
    ///
    /// Default 5e-3 (drop waypoints less than 0.5% of the mean segment apart). The actual
    /// threshold is `max(min_segment_fraction × mean_segment, 1e-5)` — the absolute floor
    /// catches "all segments are tiny" pathological inputs that would otherwise be
    /// unfilterable. Set to 0.0 to disable merging.
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

/// Numerical options passed through to the sleipnir solver.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    pub tolerance: f64,
    pub max_iterations: i32,
    pub timeout: Option<Duration>,
    pub diagnostics: bool,
    /// Half-width of the slack box on the start/end velocity and acceleration boundary
    /// "equalities". The retimer enforces `|sd[0] - start.sd| ≤ boundary_slack` (and three
    /// more like it) instead of hard `sd[0] == start.sd`, because the IPM behaves badly when
    /// rest-to-rest equalities pin variables to exactly zero at the cone tip — a small slack
    /// box gives the line search room without observable change in output. Defaults to 1e-4.
    pub boundary_slack: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            // Budget tuned so a healthy retime converges well inside it (most tests land
            // in 50–300 iter) and pathological inputs fail fast rather than burning
            // minutes of CPU in the restoration phase. Long smooth paths with many
            // extrema can need ~2k iter to converge under PCHIP — set
            // `solver.max_iterations` higher per-call when you know that's the workload.
            max_iterations: 1500,
            timeout: None,
            diagnostics: false,
            boundary_slack: 1e-4,
        }
    }
}

/// Full constraint bundle consumed by [`crate::Topp3Tcp6::retime`].
#[derive(Debug, Clone)]
pub struct Topp3Tcp6Constraints<const N: usize> {
    pub joint: JointLimits<N>,
    pub tcp: TcpLimits,
    pub boundary: BoundaryConditions<N>,
    pub densification: DensificationOptions,
    pub solver: SolverOptions,
    /// Output sample rate in Hz. The output trajectory uses `dt = 1/sample_rate_hz`.
    pub sample_rate_hz: f64,
    /// Number of joints at the base of the kinematic tree that are held constant at their
    /// starting value. The input path must have those joints identical at every waypoint.
    pub locked_prefix: usize,
    /// If true, the retimed trajectory is validated against the provided `validator` after retiming and rejected if invalid.
    pub post_validation: bool,
}

impl<const N: usize> Topp3Tcp6Constraints<N> {
    /// Convenience constructor: unbounded TCP, symmetric joint limits, rest-to-rest boundary,
    /// no locked joints, 250 Hz output.
    pub fn symmetric(v_max: f64, a_max: f64, j_max: f64) -> Self {
        Self {
            joint: JointLimits::symmetric(v_max, a_max, j_max),
            tcp: TcpLimits::unbounded(),
            boundary: BoundaryConditions::rest_to_rest(),
            densification: DensificationOptions::default(),
            solver: SolverOptions::default(),
            sample_rate_hz: 125.0,
            locked_prefix: 0,
            post_validation: true,
        }
    }
}
