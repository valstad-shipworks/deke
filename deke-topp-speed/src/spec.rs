//! Kinematic-limit specification and goal description used by both the
//! offline solver and the live pursuer.

use deke_types::{KinScalar, SRobotQ};

use crate::modes::{ControlMode, Coordination, DurationGrid, GoalOutOfBounds};

/// Full input specification for one solve.
///
/// Field naming follows the convention `current_*`, `goal_*` and `max_*` /
/// `min_*` for limits. Most fields are optional or carry a sensible default.
#[derive(Debug, Clone)]
pub struct MotionSpec<const N: usize, F: KinScalar = f32> {
    pub current_pose: SRobotQ<N, F>,
    pub current_vel: SRobotQ<N, F>,
    pub current_accel: SRobotQ<N, F>,

    pub goal_pose: SRobotQ<N, F>,
    pub goal_vel: SRobotQ<N, F>,
    pub goal_accel: SRobotQ<N, F>,

    pub max_vel: SRobotQ<N, F>,
    pub max_accel: SRobotQ<N, F>,
    pub max_jerk: SRobotQ<N, F>,

    /// Optional Cartesian TCP linear-speed ceiling (units/s). When set, the
    /// offline solver post-processes the produced trajectory so the magnitude
    /// of the TCP linear velocity ‖J_v(q)·q̇‖ never exceeds this value. The
    /// constraint is enforced by uniform time-scaling, which means it
    /// preserves the joint path exactly but is conservative (the whole
    /// trajectory is slowed even when only a small region binds).
    pub max_tcp_speed: Option<F>,

    pub min_vel: Option<SRobotQ<N, F>>,
    pub min_accel: Option<SRobotQ<N, F>>,

    pub max_pose: Option<SRobotQ<N, F>>,
    pub min_pose: Option<SRobotQ<N, F>>,

    pub axis_active: [bool; N],

    pub control_mode: ControlMode,
    pub coordination: Coordination,
    pub duration_grid: DurationGrid,
    pub goal_overflow: GoalOutOfBounds,

    pub min_duration: Option<F>,
    pub compute_budget: Option<F>, // microseconds

    pub per_axis_control_mode: Option<[ControlMode; N]>,
    pub per_axis_coordination: Option<[Coordination; N]>,

    pub waypoint_poses: Vec<SRobotQ<N, F>>,

    pub per_section_max_vel: Option<Vec<SRobotQ<N, F>>>,
    pub per_section_max_accel: Option<Vec<SRobotQ<N, F>>>,
    pub per_section_max_jerk: Option<Vec<SRobotQ<N, F>>>,
    pub per_section_min_vel: Option<Vec<SRobotQ<N, F>>>,
    pub per_section_min_accel: Option<Vec<SRobotQ<N, F>>>,
    pub per_section_max_pose: Option<Vec<SRobotQ<N, F>>>,
    pub per_section_min_pose: Option<Vec<SRobotQ<N, F>>>,
    pub per_section_min_duration: Option<Vec<F>>,
}

impl<const N: usize, F: KinScalar> MotionSpec<N, F> {
    /// Construct a spec at the origin with all limits set to one (so the user
    /// must overwrite them before use).
    pub fn new() -> Self {
        Self {
            current_pose: SRobotQ::zeros(),
            current_vel: SRobotQ::zeros(),
            current_accel: SRobotQ::zeros(),
            goal_pose: SRobotQ::zeros(),
            goal_vel: SRobotQ::zeros(),
            goal_accel: SRobotQ::zeros(),
            max_vel: SRobotQ::splat(F::one()),
            max_accel: SRobotQ::splat(F::one()),
            max_jerk: SRobotQ::splat(F::one()),
            max_tcp_speed: None,
            min_vel: None,
            min_accel: None,
            max_pose: None,
            min_pose: None,
            axis_active: [true; N],
            control_mode: ControlMode::default(),
            coordination: Coordination::default(),
            duration_grid: DurationGrid::default(),
            goal_overflow: GoalOutOfBounds::default(),
            min_duration: None,
            compute_budget: None,
            per_axis_control_mode: None,
            per_axis_coordination: None,
            waypoint_poses: Vec::new(),
            per_section_max_vel: None,
            per_section_max_accel: None,
            per_section_max_jerk: None,
            per_section_min_vel: None,
            per_section_min_accel: None,
            per_section_max_pose: None,
            per_section_min_pose: None,
            per_section_min_duration: None,
        }
    }
}

impl<const N: usize, F: KinScalar> Default for MotionSpec<N, F> {
    fn default() -> Self {
        Self::new()
    }
}
