//! Public retimer entry-point.

use std::time::{Duration, Instant};

use deke_types::{
    ContinuousFKChain, DekeError, DekeResult, Retimer, SRobotPath, SRobotQ, SRobotTraj,
    Validator,
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
pub struct Topp3TcpSpline<'a, const N: usize, FK: ContinuousFKChain<N, f64>> {
    fk: &'a FK,
}

impl<'a, const N: usize, FK: ContinuousFKChain<N, f64>> Topp3TcpSpline<'a, N, FK> {
    /// Construct the retimer over the forward-kinematics chain it will retime against.
    pub fn new(fk: &'a FK) -> Self {
        Self { fk }
    }
}

impl<'a, const N: usize, FK: ContinuousFKChain<N, f64>> Retimer<N, f64> for Topp3TcpSpline<'a, N, FK> {
    type Diagnostic = Topp3TcpSplineDiagnostic;
    type Constraints = Topp3TcpSplineConstraints<N>;

    fn retime<V: Validator<N, (), f64>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, f64>,
        _validator: &V,
        _ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, f64>>, Self::Diagnostic) {
        let fk = self.fk;
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
                // Phase B-lite: smooth the per-segment jerk schedule to
                // halve the FD-jerk spike the backward-FD stencil reads
                // when it straddles a DFS boundary. Each pass binomially
                // averages interior jerks and renormalizes via uniform
                // time-rescale so `s_final` lands at 1.
                traj.smooth_jerks(constraints.search.jerk_smoothing_passes);

                let dt_val = traj.dt();
                // Decouple the consumer-visible sample step from the DFS
                // search step: if `output_dt` is set, analytically resample
                // the converged constant-jerk segments onto that grid.
                let dt_emit = constraints.search.output_dt.unwrap_or(dt_val);
                let mut states_for_eval = if constraints.search.output_dt.is_some() {
                    traj.resample_to(dt_emit)
                } else {
                    traj.states().to_vec()
                };

                // Phase C: FD-readout safety. Walk the output samples with
                // the consumer-visible stencils; if any per-sample max_u
                // exceeds the slack ceiling, uniformly slow the trajectory.
                // Bounded loop — in practice fires only when smoothing
                // can't absorb the boundary spike entirely.
                let safety_slack = constraints.search.fd_safety_slack;
                for _ in 0..3 {
                    // Build the sample-q list for the overshoot check.
                    let q_samples: Vec<SRobotQ<N, f64>> = states_for_eval
                        .iter()
                        .map(|st| spline_path.eval(st.s()).0)
                        .collect();
                    let alpha = match traj.peak_fd_overshoot_scale(
                        &q_samples,
                        dt_emit,
                        safety_slack,
                    ) {
                        Ok(a) => a,
                        Err(_) => break,
                    };
                    if !(alpha > 1.0 + 1e-9) {
                        break;
                    }
                    traj.rescale_time_in_place(alpha);
                    states_for_eval = match constraints.search.output_dt {
                        Some(h) if h > 0.0 => traj.resample_to(h),
                        _ => traj.states().to_vec(),
                    };
                }

                let states = states_for_eval;
                diag.output_states = states.len();
                diag.total_time = Duration::from_secs_f64(
                    (states.len() as f64 - 1.0).max(0.0) * dt_emit,
                );
                diag.status = SolveStatus::Success;

                // Reconstruct joint samples at every state by evaluating the
                // spline at `s`. Re-prime the thread-local dt cache so any
                // forward_integrate the user calls afterwards matches.
                set_dt(dt_emit);
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
                let dt_out = Duration::from_secs_f64(dt_emit);
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
