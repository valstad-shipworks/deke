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

/// How the raw Cartesian polyline is conditioned into smooth, arc-length runs.
#[derive(Clone, Debug)]
pub struct PathConditioning {
    /// Turn angle (radians) above which a vertex is treated as a *sharp* corner —
    /// the path is split there into separate runs that start/stop at rest.
    pub sharp_corner_angle: f64,
}

impl Default for PathConditioning {
    fn default() -> Self {
        Self {
            sharp_corner_angle: 30.0_f64.to_radians(),
        }
    }
}

/// Knobs for the branch-tracking planner (Stage B).
#[derive(Clone, Debug)]
pub struct PlannerOptions<const N: usize> {
    /// Arc-length spacing (metres) at which the run is sampled and IK'd.
    pub sample_ds: f64,
    /// Weight on the manipulability (singularity-avoidance) node cost.
    pub manip_weight: f64,
    /// Absolute per-sample joint continuity guard (radians): an edge whose worst
    /// per-axis joint jump exceeds this is a reconfiguration. Always active.
    pub max_branch_jump: f64,
    /// TCP speed (m/s) used by the velocity-based reconfiguration test. When
    /// `> 0` and [`Self::joint_v_max`] is finite, an edge that would drive **any**
    /// joint past [`Self::reconfig_vel_fraction`] of its velocity limit at this
    /// speed is treated as a reconfiguration/discontinuity. At weld speeds this is
    /// the signature of a singularity or wrist flip. Set `0.0` to disable.
    pub max_velocity: f64,
    /// Per-joint velocity ceilings for the velocity-based reconfiguration test.
    /// `INFINITY` (the default) disables the test on that axis.
    pub joint_v_max: SRobotQ<N, f64>,
    /// Fraction of `joint_v_max` an edge may demand before it counts as a
    /// reconfiguration (e.g. `0.9` = 90%).
    pub reconfig_vel_fraction: f64,
}

impl<const N: usize> Default for PlannerOptions<N> {
    fn default() -> Self {
        Self {
            sample_ds: 2e-3,
            manip_weight: 1.0,
            max_branch_jump: 0.6,
            max_velocity: 0.0,
            joint_v_max: SRobotQ::splat(f64::INFINITY),
            reconfig_vel_fraction: 0.9,
        }
    }
}

/// Kinematic ceilings + the commanded constant TCP speed (Stage C).
#[derive(Clone, Debug)]
pub struct LinearConstraints<const N: usize> {
    pub joint: JointLimits<N>,
    /// Commanded constant TCP linear speed (m/s), held wherever feasible.
    pub tcp_speed: f64,
    /// Output trajectory sample period.
    pub output_dt: Duration,
    /// When `true`, the speed may only fall below `tcp_speed` during the rest
    /// ramp at the start and end of each run. If the joint v/a/j geometry would
    /// force a dip anywhere in a run's interior (a shallow corner or a
    /// near-singular patch), the retime fails with [`crate::LinearError::SpeedDipRequired`]
    /// instead of slowing down. Sharp corners are unaffected — they are already
    /// split into separate runs whose endpoints are legitimate stops.
    pub forbid_interior_dips: bool,
}
