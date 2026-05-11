use std::time::Duration;

use deke_types::{
    DekeError, DekeResult, FKChain, Retimer, SRobotPath, SRobotTraj, Validator,
};

use crate::boundary::project;
use crate::constraints::Topp3Tcp6Constraints;
use crate::diagnostic::{LimitingGroup, SolveStatus, Topp3Tcp6Diagnostic};
use crate::nlp::{Solution, build_and_solve};
use crate::path_derivatives::PathDerivatives;
use crate::resample::resample_to_uniform;

/// Time-optimal path-parameterization retimer with per-joint and per-TCP velocity, acceleration
/// and jerk constraints. See the crate-level docs for the mathematical formulation.
#[derive(Debug, Clone, Default)]
pub struct Topp3Tcp6;

impl<const N: usize> Retimer<N, f64> for Topp3Tcp6 {
    type Diagnostic = Topp3Tcp6Diagnostic;
    type Constraints = Topp3Tcp6Constraints<N>;

    fn retime<V: Validator<N, (), f64>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, f64>,
        fk: &impl FKChain<N, f64>,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, f64>>, Self::Diagnostic) {
        let mut diag = Topp3Tcp6Diagnostic::default();

        if let Err(e) = PathDerivatives::<N>::check_locked_prefix(path, constraints.locked_prefix) {
            diag.status = SolveStatus::NotAttempted;
            diag.limiting_constraint = Some(LimitingGroup::BoundaryCondition);
            diag.message = Some(format!("{}", e));
            return (Err(e), diag);
        }

        let densified = match densify_path(path, &constraints.densification) {
            Ok(p) => p,
            Err(e) => {
                diag.message = Some(format!("{}", e));
                return (Err(e), diag);
            }
        };
        diag.densified_samples = densified.len();

        let tcp_disabled = constraints.tcp.is_disabled();
        let deriv = match if tcp_disabled {
            PathDerivatives::<N>::new_without_tcp(&densified)
        } else {
            PathDerivatives::<N>::new(&densified, fk)
        } {
            Ok(d) => d,
            Err(e) => {
                diag.message = Some(format!("{}", e));
                return (Err(e), diag);
            }
        };

        let start = project::<N>(
            &constraints.boundary.v_start,
            &constraints.boundary.a_start,
            &deriv.qp[0],
            &deriv.qpp[0],
        );
        let end_idx = deriv.num_waypoints() - 1;
        let end = project::<N>(
            &constraints.boundary.v_end,
            &constraints.boundary.a_end,
            &deriv.qp[end_idx],
            &deriv.qpp[end_idx],
        );
        let residual = start.max_residual().max(end.max_residual());
        diag.boundary_projection_residual = residual;
        if residual > constraints.boundary.projection_tolerance {
            diag.limiting_constraint = Some(LimitingGroup::BoundaryCondition);
            let err = DekeError::BoundaryInfeasible(residual as f32);
            diag.message = Some(format!("{}", err));
            return (Err(err), diag);
        }

        let solution = match build_and_solve::<N>(&deriv, constraints, start, end) {
            Ok(s) => s,
            Err(e) => {
                diag.message = Some(format!("{}", e));
                return (Err(e), diag);
            }
        };
        diag.status = solution.status;
        diag.iterations = solution.iterations;
        diag.solve_time = solution.solve_time;

        if !matches!(solution.status, SolveStatus::Success) {
            diag.limiting_constraint = infer_limiting_group(&solution, &deriv, constraints);
            let err = DekeError::RetimerFailed(format!("{}", solution.status));
            diag.message = Some(format!("{}", err));
            return (Err(err), diag);
        }

        populate_analytical_peaks(&mut diag, &solution, &deriv, constraints);

        let dt_out = Duration::from_secs_f64(1.0 / constraints.sample_rate_hz);
        let (total_time, samples) = resample_to_uniform(&solution, &deriv, dt_out);
        diag.output_samples = samples.len();
        diag.total_time = total_time;
        let traj_path = match SRobotPath::try_new(samples) {
            Ok(p) => p,
            Err(e) => {
                diag.message = Some(format!("{}", e));
                return (Err(e), diag);
            }
        };

        if constraints.post_validation {
            if let Err(e) = validator.validate_motion(traj_path.iter().as_slice(), ctx) {
                diag.message = Some(format!("validator rejected output: {}", e));
                return (Err(e), diag);
            }
        }

        (Ok(SRobotTraj::new(dt_out, traj_path)), diag)
    }
}

fn densify_path<const N: usize>(
    path: &SRobotPath<N, f64>,
    opts: &crate::constraints::DensificationOptions,
) -> DekeResult<SRobotPath<N, f64>> {
    let merged = merge_near_duplicates(path, opts.min_segment_fraction)?;

    let mut p = if let Some(step) = opts.max_segment_step {
        merged.densify(step)
    } else {
        merged
    };

    if p.len() < opts.min_samples {
        let n = opts.min_samples.max(2);
        let mut wps = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            wps.push(p.sample(t).unwrap_or(*p.first()));
        }
        p = SRobotPath::try_new(wps)?;
    }

    if p.len() > opts.max_samples {
        let n = opts.max_samples.max(2);
        let mut wps = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            wps.push(p.sample(t).unwrap_or(*p.first()));
        }
        p = SRobotPath::try_new(wps)?;
    }

    Ok(p)
}

/// Drops interior waypoints whose chord distance to the previously-kept waypoint is below
/// `max(relative_threshold × mean_segment_length, ABSOLUTE_FLOOR)`. The first and last
/// waypoints are always kept; if the last waypoint is itself within the threshold of the
/// previous interior keep, that interior waypoint is dropped in favor of the user-requested
/// endpoint. A `relative_threshold` of 0 disables merging.
///
/// We use the mean rather than the median because a path with a "quasi-stationary"
/// section (a few normal segments + several deliberate tiny ones) has a tiny median and
/// a relative-to-median threshold then can't see the tiny segments. The mean is dragged
/// down too but stays at least order(of the largest segments / total count) — enough to
/// catch the tinies. The absolute floor (1e-5) catches "all segments are tiny" pathological
/// inputs that would otherwise be unfilterable.
fn merge_near_duplicates<const N: usize>(
    path: &SRobotPath<N, f64>,
    relative_threshold: f64,
) -> DekeResult<SRobotPath<N, f64>> {
    const ABSOLUTE_FLOOR: f64 = 1e-5;
    let m = path.len();
    if m < 3 || relative_threshold <= 0.0 {
        return Ok(path.clone());
    }

    let mut total = 0.0_f64;
    for k in 0..m - 1 {
        total += chord_distance::<N>(path.get(k).unwrap(), path.get(k + 1).unwrap());
    }
    let mean_seg = total / (m - 1) as f64;
    let threshold = (mean_seg * relative_threshold).max(ABSOLUTE_FLOOR);

    let mut kept_indices: Vec<usize> = Vec::with_capacity(m);
    kept_indices.push(0);
    for k in 1..m - 1 {
        let last_idx = *kept_indices.last().unwrap();
        let last = path.get(last_idx).unwrap();
        let cur = path.get(k).unwrap();
        if chord_distance::<N>(last, cur) >= threshold {
            kept_indices.push(k);
        }
    }
    let last_idx = *kept_indices.last().unwrap();
    let last_kept = path.get(last_idx).unwrap();
    let final_wp = path.get(m - 1).unwrap();
    if chord_distance::<N>(last_kept, final_wp) < threshold && kept_indices.len() > 1 {
        // Last interior keep is too close to the user's final waypoint; drop the interior one.
        kept_indices.pop();
    }
    kept_indices.push(m - 1);

    let kept: Vec<_> = kept_indices
        .iter()
        .map(|&i| *path.get(i).unwrap())
        .collect();
    SRobotPath::try_new(kept)
}

fn chord_distance<const N: usize>(
    a: &deke_types::SRobotQ<N, f64>,
    b: &deke_types::SRobotQ<N, f64>,
) -> f64 {
    let mut sq = 0.0_f64;
    for j in 0..N {
        let d = b.0[j] - a.0[j];
        sq += d * d;
    }
    sq.sqrt()
}

fn infer_limiting_group<const N: usize>(
    solution: &Solution,
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
) -> Option<LimitingGroup> {
    if matches!(solution.status, SolveStatus::LocallyInfeasible | SolveStatus::GloballyInfeasible) {
        let lock = constraints.locked_prefix.min(N);
        let m = deriv.num_waypoints();
        let mut worst = (0.0_f64, LimitingGroup::JointVelocity);
        for k in 0..m {
            let sd = solution.sd[k].max(0.0);
            let sdd = solution.sdd[k];
            for j in lock..N {
                let qp = deriv.qp[k][j];
                let qpp = deriv.qpp[k][j];
                let v = (qp * sd).abs();
                let v_max = constraints.joint.v_max.0[j];
                if v_max.is_finite() && v - v_max > worst.0 {
                    worst = (v - v_max, LimitingGroup::JointVelocity);
                }
                let a = (qpp * sd * sd + qp * sdd).abs();
                let a_max = constraints.joint.a_max.0[j];
                if a_max.is_finite() && a - a_max > worst.0 {
                    worst = (a - a_max, LimitingGroup::JointAcceleration);
                }
            }
            if deriv.has_tcp() {
                let pp = &deriv.pp[k];
                let tcp_v = (pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2]).sqrt() * sd;
                let tcp_v_max = constraints.tcp.v_max;
                if tcp_v_max.is_finite() && tcp_v - tcp_v_max > worst.0 {
                    worst = (tcp_v - tcp_v_max, LimitingGroup::TcpVelocity);
                }
            }
        }
        Some(worst.1)
    } else {
        None
    }
}

/// Populates the `peak_*` diagnostic fields directly from the NLP solution using the
/// path-parameter expression for each derivative. These are the quantities the solver actually
/// constrained, unlike a finite-difference on the resampled output which is noisy at segment
/// boundaries because the geometric path is piecewise-linear in joint space.
fn populate_analytical_peaks<const N: usize>(
    diag: &mut Topp3Tcp6Diagnostic,
    solution: &Solution,
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
) {
    let m = deriv.num_waypoints();
    let seg = deriv.num_segments();
    let lock = constraints.locked_prefix.min(N);

    let mut peak_jv = 0.0_f64;
    let mut peak_ja = 0.0_f64;
    let mut peak_jj = 0.0_f64;
    let mut peak_tv = 0.0_f64;
    let mut peak_ta = 0.0_f64;
    let mut peak_tj = 0.0_f64;

    let jv_max: Vec<f64> = (0..N).map(|j| constraints.joint.v_max.0[j]).collect();
    let ja_max: Vec<f64> = (0..N).map(|j| constraints.joint.a_max.0[j]).collect();
    let jj_max: Vec<f64> = (0..N).map(|j| constraints.joint.j_max.0[j]).collect();
    let tv_max = constraints.tcp.v_max;
    let ta_max = constraints.tcp.a_max;
    let tj_max = constraints.tcp.j_max;

    let update_util = |cur: &mut f64, val: f64, limit: f64| {
        if limit.is_finite() && limit > 0.0 {
            let u = val / limit;
            if u > *cur {
                *cur = u;
            }
        }
    };

    let mut util_sum = 0.0_f64;

    for k in 0..m {
        let sd = solution.sd[k];
        let sdd = solution.sdd[k];
        let seg_idx = k.min(seg - 1);
        let sddd = solution.sddd[seg_idx];

        let mut step_util = 0.0_f64;

        for j in lock..N {
            let qp = deriv.qp[k][j];
            let qpp = deriv.qpp[k][j];
            let qppp = deriv.qppp[k][j];
            let jv = (qp * sd).abs();
            let ja = (qpp * sd * sd + qp * sdd).abs();
            let jj = (qppp * sd * sd * sd + 3.0 * qpp * sd * sdd + qp * sddd).abs();
            peak_jv = peak_jv.max(jv);
            peak_ja = peak_ja.max(ja);
            peak_jj = peak_jj.max(jj);
            update_util(&mut step_util, jv, jv_max[j]);
            update_util(&mut step_util, ja, ja_max[j]);
            update_util(&mut step_util, jj, jj_max[j]);
        }

        if deriv.has_tcp() {
            let pp = &deriv.pp[k];
            let ppp = &deriv.ppp[k];
            let pppp = &deriv.pppp[k];

            let vx = pp[0] * sd;
            let vy = pp[1] * sd;
            let vz = pp[2] * sd;
            let tv = (vx * vx + vy * vy + vz * vz).sqrt();
            peak_tv = peak_tv.max(tv);
            update_util(&mut step_util, tv, tv_max);

            let ax = ppp[0] * sd * sd + pp[0] * sdd;
            let ay = ppp[1] * sd * sd + pp[1] * sdd;
            let az = ppp[2] * sd * sd + pp[2] * sdd;
            let ta = (ax * ax + ay * ay + az * az).sqrt();
            peak_ta = peak_ta.max(ta);
            update_util(&mut step_util, ta, ta_max);

            let jx = pppp[0] * sd * sd * sd + 3.0 * ppp[0] * sd * sdd + pp[0] * sddd;
            let jy = pppp[1] * sd * sd * sd + 3.0 * ppp[1] * sd * sdd + pp[1] * sddd;
            let jz = pppp[2] * sd * sd * sd + 3.0 * ppp[2] * sd * sdd + pp[2] * sddd;
            let tj = (jx * jx + jy * jy + jz * jz).sqrt();
            peak_tj = peak_tj.max(tj);
            update_util(&mut step_util, tj, tj_max);
        }

        util_sum += step_util;
    }

    diag.peak_joint_velocity = peak_jv;
    diag.peak_joint_acceleration = peak_ja;
    diag.peak_joint_jerk = peak_jj;
    diag.peak_tcp_velocity = peak_tv;
    diag.peak_tcp_acceleration = peak_ta;
    diag.peak_tcp_jerk = peak_tj;
    diag.average_utilization = if m > 0 { util_sum / m as f64 } else { 0.0 };
}
