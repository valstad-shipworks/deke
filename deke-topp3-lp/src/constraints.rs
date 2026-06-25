use std::time::Duration;

use deke_types::SRobotQ;

/// Per-axis joint velocity/acceleration/jerk ceilings.
#[derive(Clone, Debug)]
pub struct JointLimits<const N: usize> {
    pub v_max: SRobotQ<N, f64>,
    pub a_max: SRobotQ<N, f64>,
    pub j_max: SRobotQ<N, f64>,
}

impl<const N: usize> JointLimits<N> {
    /// The same v/a/j ceiling on every axis.
    pub fn symmetric(v: f64, a: f64, j: f64) -> Self {
        Self {
            v_max: SRobotQ::splat(v),
            a_max: SRobotQ::splat(a),
            j_max: SRobotQ::splat(j),
        }
    }
}

/// Optional Cartesian cap on the tool-centre-point. Only honoured by
/// [`crate::Topp3LpTcp`] (which carries the FK chain needed to evaluate it);
/// [`crate::Topp3Lp`] errors if a cap is set.
#[derive(Clone, Copy, Debug, Default)]
pub struct TcpLimits {
    /// Cap on TCP linear speed (m/s). `None` leaves the tip speed bounded only by
    /// what the per-joint limits permit through the path geometry.
    pub v_max: Option<f64>,
}

impl TcpLimits {
    pub fn speed(v_max: f64) -> Self {
        Self { v_max: Some(v_max) }
    }
}

/// How the raw joint polyline is turned into the knots the σ-LP times.
#[derive(Clone, Copy, Debug, Default)]
pub enum Conditioning {
    /// Keep the raw polyline — one segment per input edge. Ticks within an edge are
    /// interior (cheap scalar rows); only true corners get the exact per-joint rows.
    /// Densifying a straight edge only adds same-secant bins that bloat the program
    /// for no accuracy gain under the exact-row solver, so this is the default.
    #[default]
    Raw,
    /// Collinear densification at joint-space spacing `res` (radians): insert
    /// knots *on* each chord segment. Every knot is a convex combination of two
    /// original waypoints, so geometric deviation is identically zero. Rarely
    /// needed under the exact-row solver; kept for very coarse inputs.
    Collinear(f64),
}

/// Full constraint bundle consumed by the retimers.
#[derive(Clone, Debug)]
pub struct Topp3LpConstraints<const N: usize> {
    pub joint: JointLimits<N>,
    /// Optional TCP cap (see [`TcpLimits`]); ignored unless using [`crate::Topp3LpTcp`].
    pub tcp: TcpLimits,
    /// Output trajectory sample period.
    pub output_dt: Duration,
    /// How the input polyline is conditioned before timing.
    pub conditioning: Conditioning,
    /// Joint-space turn angle (radians) above which a vertex is a *sharp corner*:
    /// the path is split there into separate runs that start and stop at rest, so
    /// a hard kink is traversed by stopping on it (zero deviation, bounded jerk)
    /// rather than rounding it off. `None` never splits — shallow corners the
    /// timing LP can dip through stay in one run regardless.
    pub sharp_corner_angle: Option<f64>,
}

impl<const N: usize> Topp3LpConstraints<N> {
    /// Symmetric joint limits, no TCP cap, default (raw) conditioning.
    pub fn symmetric(v_max: f64, a_max: f64, j_max: f64, output_dt: Duration) -> Self {
        Self {
            joint: JointLimits::symmetric(v_max, a_max, j_max),
            tcp: TcpLimits::default(),
            output_dt,
            conditioning: Conditioning::default(),
            sharp_corner_angle: Some(30.0_f64.to_radians()),
        }
    }

    /// Add a TCP linear-speed cap (m/s). Use with [`crate::Topp3LpTcp`].
    pub fn with_tcp_speed(mut self, v_max: f64) -> Self {
        self.tcp = TcpLimits::speed(v_max);
        self
    }
}
