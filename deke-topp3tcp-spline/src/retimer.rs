//! Public retimer entry-point.

use std::time::{Duration, Instant};

use deke_types::{
    DekeError, DekeResult, FKChain, Retimer, SRobotPath, SRobotQ, SRobotTraj, Validator,
};

use crate::constraints::Topp3TcpSplineConstraints;
use crate::diagnostic::{SolveStatus, Topp3TcpSplineDiagnostic};
use crate::path::SplineInterpolatedRobotPath;
use crate::trajectory::{Trajectory, set_dt};

/// Discrete TOPP-3TCP retimer using a B-spline path representation and
/// depth-first search over jerk candidates.
///
/// The retimer:
///
/// 1. Builds a clamped B-spline through the input waypoints, refining support
///    density until the spline lies within the configured joint-space
///    deviation tube around the polyline.
/// 2. Runs a depth-first search over discrete jerk values along that spline,
///    simultaneously enforcing per-axis joint v/a/j limits and Cartesian
///    TCP v/a/j limits using the position rows of the geometric Jacobian.
/// 3. Emits a uniformly time-sampled [`SRobotTraj`] at the configured
///    output `dt`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Topp3TcpSpline;

impl<const N: usize> Retimer<N, f64> for Topp3TcpSpline {
    type Diagnostic = Topp3TcpSplineDiagnostic;
    type Constraints = Topp3TcpSplineConstraints<N>;

    fn retime<V: Validator<N, (), f64>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, f64>,
        fk: &impl FKChain<N, f64>,
        _validator: &V,
        _ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, f64>>, Self::Diagnostic) {
        let mut diag = Topp3TcpSplineDiagnostic {
            input_waypoints: path.len(),
            ..Default::default()
        };

        if path.len() < 2 {
            diag.message = Some(format!("path has {} waypoints, need >= 2", path.len()));
            return (Err(DekeError::PathTooShort(path.len())), diag);
        }
        if constraints.search.dt <= 0.0 || constraints.search.verify_dt <= 0.0 {
            diag.message = Some("dt and verify_dt must be positive".to_string());
            return (
                Err(DekeError::RetimerFailed(
                    "dt and verify_dt must be positive".to_string(),
                )),
                diag,
            );
        }

        let spline_path = SplineInterpolatedRobotPath::<N>::from_path(
            path,
            constraints.path.max_deviation,
            constraints.path.max_refine_iters,
            constraints.path.start_direction.as_ref(),
            constraints.path.end_direction.as_ref(),
        );
        diag.deduplicated_waypoints = spline_path.polyline_waypoints().len();

        let t_solve = Instant::now();
        let mut traj = Trajectory::new(fk, &spline_path, constraints);
        let result = traj.optimize();
        diag.solve_time = t_solve.elapsed();

        match result {
            Ok(()) => {
                let dt_val = traj.dt();
                let states = traj.into_states();
                diag.output_states = states.len();
                diag.total_time = Duration::from_secs_f64(
                    (states.len() as f64 - 1.0).max(0.0) * dt_val,
                );
                diag.status = SolveStatus::Success;

                // Reconstruct joint samples at every state by evaluating the
                // spline at `s`. Re-prime the thread-local dt cache so any
                // forward_integrate the user calls afterwards matches.
                set_dt(dt_val);
                let mut waypoints: Vec<SRobotQ<N, f64>> = states
                    .iter()
                    .map(|st| {
                        let (q, _qp, _qpp, _qppp) = spline_path.eval(st.s());
                        q
                    })
                    .collect();
                // Pin endpoints to the user's first/last waypoint exactly:
                // the spline path is built on a deduplicated polyline whose
                // tolerance can drop a near-duplicate final waypoint, and the
                // boundary-condition solver can leave `s_final` shy of 1.0
                // by a few ulp.  Clamp out the drift so downstream consumers
                // see traj.first() == path.first() and traj.last() == path.last().
                if let Some(first) = waypoints.first_mut() {
                    *first = *path.first();
                }
                if let Some(last) = waypoints.last_mut() {
                    *last = *path.last();
                }

                let srpath = match SRobotPath::try_new(waypoints) {
                    Ok(p) => p,
                    Err(e) => {
                        diag.message = Some(format!("{}", e));
                        return (Err(e), diag);
                    }
                };
                let dt_out = Duration::from_secs_f64(dt_val);
                (Ok(SRobotTraj::new(dt_out, srpath)), diag)
            }
            Err(e) => {
                diag.status = SolveStatus::SearchExhausted;
                diag.message = Some(format!("{}", e));
                (Err(e), diag)
            }
        }
    }
}
