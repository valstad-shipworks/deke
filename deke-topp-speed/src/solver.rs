//! Dispatcher between the single-target and multi-waypoint solvers.
//!
//! Mirrors the C++ `Calculator` driver: hands off to the [`WaypointSolver`]
//! when the spec carries intermediate waypoints and is in position control,
//! and to the [`TargetSolver`] otherwise. After the per-solve runs, optionally
//! enforces the position envelope on the produced plan.

use deke_types::FKScalar;
use num_traits::Float;

use crate::modes::ControlMode;
use crate::plan::Plan;
use crate::spec::MotionSpec;
use crate::status::StepStatus;
use crate::target_solver::TargetSolver;
use crate::waypoint_solver::WaypointSolver;

#[inline]
fn from_f<F: Float>(x: f64) -> F {
    F::from(x).unwrap()
}

/// Combined target / multi-waypoint solver.
///
/// Selects between [`TargetSolver`] and [`WaypointSolver`] based on whether
/// the [`MotionSpec`] carries intermediate waypoints.
#[derive(Debug)]
pub(crate) struct Solver<const N: usize, F: FKScalar> {
    pub target: TargetSolver<N, F>,
    pub waypoint: WaypointSolver<N, F>,
}

impl<const N: usize, F: FKScalar> Solver<N, F> {
    pub fn new() -> Self {
        Self {
            target: TargetSolver::new(),
            waypoint: WaypointSolver::new(),
        }
    }

    /// Returns `true` when the spec should be solved via the multi-waypoint
    /// path: at least one intermediate waypoint and position control selected.
    fn should_use_waypoints(spec: &MotionSpec<N, F>) -> bool {
        !spec.waypoint_poses.is_empty() && spec.control_mode == ControlMode::Position
    }

    /// Verify that the produced plan respects the (optional) per-axis pose
    /// envelope from the spec. Returns `true` when at least one axis steps
    /// outside its allowed range.
    fn exceeds_position_constraints(spec: &MotionSpec<N, F>, plan: &mut Plan<N, F>) -> bool {
        if spec.max_pose.is_none() && spec.min_pose.is_none() {
            return false;
        }
        let eps = from_f::<F>(1e-12);
        let extrema = plan.position_extrema();
        for dof_idx in 0..N {
            if let Some(min_pose) = spec.min_pose.as_ref()
                && extrema[dof_idx].min < min_pose[dof_idx] - eps
            {
                return true;
            }
            if let Some(max_pose) = spec.max_pose.as_ref()
                && extrema[dof_idx].max > max_pose[dof_idx] + eps
            {
                return true;
            }
        }
        false
    }

    /// Solve the trajectory. Dispatches to the per-spec sub-solver and then
    /// runs the post-solve position-envelope check.
    pub fn solve(&mut self, spec: &MotionSpec<N, F>, plan: &mut Plan<N, F>) -> StepStatus {
        let status = if Self::should_use_waypoints(spec) {
            self.waypoint.solve(spec, plan)
        } else {
            self.target.solve(spec, plan, F::zero())
        };
        if status == StepStatus::InProgress && Self::exceeds_position_constraints(spec, plan) {
            return StepStatus::PoseOverrun;
        }
        status
    }
}

impl<const N: usize, F: FKScalar> Default for Solver<N, F> {
    fn default() -> Self {
        Self::new()
    }
}
