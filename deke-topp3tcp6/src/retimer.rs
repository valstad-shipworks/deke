use std::time::{Duration, Instant};

use deke_types::{
    DekeError, DekeResult, FKChain, Retimer, SRobotPath, SRobotTraj, Validator,
};

use crate::boundary::project;
use crate::constraints::Topp3Tcp6Constraints;
use crate::diagnostic::{
    DerivativeStats, LimitingGroup, PathStats, PeakLocation, SolveStatus, TcpStats,
    Topp3Tcp6Diagnostic,
};
use crate::nlp::{Solution, build_and_solve, build_and_solve_warm};
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
        diag.path_stats.input_waypoints = path.len();

        if let Err(e) = PathDerivatives::<N>::check_locked_prefix(path, constraints.locked_prefix) {
            diag.status = SolveStatus::NotAttempted;
            diag.limiting_constraint = Some(LimitingGroup::BoundaryCondition);
            diag.message = Some(format!("{}", e));
            return (Err(e), diag);
        }

        let t_densify = Instant::now();
        let (densified, merged_count) =
            match densify_path(path, &constraints.densification) {
                Ok(out) => out,
                Err(e) => {
                    diag.message = Some(format!("{}", e));
                    diag.phase_timing.densify = t_densify.elapsed();
                    return (Err(e), diag);
                }
            };
        diag.phase_timing.densify = t_densify.elapsed();
        diag.densified_samples = densified.len();
        diag.path_stats.merged_waypoints = merged_count;
        populate_path_geometry::<N>(&mut diag.path_stats, &densified);

        let t_deriv = Instant::now();
        let tcp_disabled = constraints.tcp.is_none();
        let deriv = match if tcp_disabled {
            PathDerivatives::<N>::new_without_tcp(&densified)
        } else {
            PathDerivatives::<N>::new(&densified, fk)
        } {
            Ok(d) => d,
            Err(e) => {
                diag.message = Some(format!("{}", e));
                diag.phase_timing.derivatives = t_deriv.elapsed();
                return (Err(e), diag);
            }
        };
        diag.phase_timing.derivatives = t_deriv.elapsed();
        diag.derivative_stats = derivative_stats_from_deriv::<N>(&deriv, constraints);
        if deriv.has_tcp() {
            diag.tcp_stats = tcp_stats_from_deriv::<N>(&deriv);
        }

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

        // Two-stage warm-start path. When TCP constraints are active and the user has
        // it enabled, first solve the TCP-disabled problem (joint constraints +
        // integrator only) to get a feasible (sd, sdd, sddd, dt) iterate, then run the
        // TCP-enabled solve from that warm start. Stage 1 is cheap (smaller constraint
        // set) and Stage 2 typically converges in <50 iter from the warm point —
        // significantly faster *and* more robust than single-stage on hard paths
        // (8wp/50wp shapes that previously consumed any iter budget then bailed).
        let solution_result = if !tcp_disabled
            && deriv.has_tcp()
            && constraints.solver.two_stage_warm_start
        {
            two_stage_solve::<N>(&densified, &deriv, fk, constraints, start, end)
        } else {
            build_and_solve::<N>(&deriv, constraints, start, end)
        };
        let solution = match solution_result {
            Ok(s) => s,
            Err(e) => {
                diag.message = Some(format!("{}", e));
                return (Err(e), diag);
            }
        };
        diag.status = solution.status;
        diag.iterations = solution.iterations;
        diag.solve_time = solution.solve_time;
        diag.phase_timing.nlp_build = solution.build_time;
        diag.phase_timing.nlp_solve = solution.solve_time;
        diag.constraint_counts = solution.constraint_counts;
        diag.initial_guess = solution.initial_guess;
        diag.boundary_slack_usage = solution.boundary_slack_usage;
        diag.derivative_stats.degenerate_qp_samples = solution.degenerate_qp_samples;
        diag.derivative_stats.min_qp_norm_relative_sq = solution.min_qp_norm_relative_sq;
        diag.derivative_stats.min_qp_norm_sample = solution.min_qp_norm_sample;

        if !matches!(solution.status, SolveStatus::Success) {
            let (group, sample) = infer_limiting_group(&solution, &deriv, constraints);
            diag.limiting_constraint = group;
            diag.limiting_sample = sample;
            let err = DekeError::RetimerFailed(format!("{}", solution.status));
            diag.message = Some(format!("{}", err));
            return (Err(err), diag);
        }

        populate_analytical_peaks(&mut diag, &solution, &deriv, constraints);

        let t_resample = Instant::now();
        let dt_out = Duration::from_secs_f64(1.0 / constraints.sample_rate_hz);
        let (total_time, samples) = resample_to_uniform(&solution, &deriv, dt_out);
        diag.output_samples = samples.len();
        diag.total_time = total_time;
        let traj_path = match SRobotPath::try_new(samples) {
            Ok(p) => p,
            Err(e) => {
                diag.message = Some(format!("{}", e));
                diag.phase_timing.resample = t_resample.elapsed();
                return (Err(e), diag);
            }
        };
        diag.phase_timing.resample = t_resample.elapsed();

        if constraints.post_validation {
            if let Err(e) = validator.validate_motion(traj_path.iter().as_slice(), ctx) {
                diag.message = Some(format!("validator rejected output: {}", e));
                return (Err(e), diag);
            }
        }

        if constraints.check_output_dynamics {
            if let Err(e) = check_dynamics_against_limits::<N>(&solution, &deriv, constraints, dt_out) {
                diag.message = Some(format!("{}", e));
                return (Err(e), diag);
            }
        }

        (Ok(SRobotTraj::new(dt_out, traj_path)), diag)
    }
}

/// Re-evaluates the analytical per-sample kinematics from the converged NLP solution
/// against the configured joint and TCP limits. Each constraint is the same expression
/// the solver enforced, so this is exact up to IPM convergence tolerance — the small
/// relative slack (`solver.tolerance`, defaulting to 1e-6) absorbs that.
///
/// Joint violations use the joint index as `dof`. TCP violations report `dof = u8::MAX`
/// since the bound is on the translational-velocity magnitude rather than a single axis.
fn check_dynamics_against_limits<const N: usize>(
    solution: &Solution,
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
    dt_in: Duration,
) -> DekeResult<()> {
    let m = deriv.num_waypoints();
    let seg = deriv.num_segments();
    let lock = constraints.locked_prefix.min(N);
    let rel_slack = constraints.solver.tolerance.max(0.0);

    let exceeds = |observed: f64, limit: f64| -> bool {
        limit.is_finite() && limit > 0.0 && observed > limit * (1.0 + rel_slack)
    };

    for k in 0..m {
        let sd = solution.sd[k];
        let sdd = solution.sdd[k];
        let seg_idx = k.min(seg - 1);
        let sddd = solution.sddd[seg_idx];

        for j in lock..N {
            let qp = deriv.qp[k][j];
            let qpp = deriv.qpp[k][j];
            let qppp = deriv.qppp[k][j];

            let v = (qp * sd).abs();
            let v_max = constraints.joint.v_max.0[j];
            if exceeds(v, v_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in,
                    limit_type: "joint_velocity",
                    dof: j as u8,
                    limit_value: v_max,
                    observed_value: v,
                });
            }

            let a = (qpp * sd * sd + qp * sdd).abs();
            let a_max = constraints.joint.a_max.0[j];
            if exceeds(a, a_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in,
                    limit_type: "joint_acceleration",
                    dof: j as u8,
                    limit_value: a_max,
                    observed_value: a,
                });
            }

            let jk = (qppp * sd * sd * sd + 3.0 * qpp * sd * sdd + qp * sddd).abs();
            let j_max = constraints.joint.j_max.0[j];
            if exceeds(jk, j_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in,
                    limit_type: "joint_jerk",
                    dof: j as u8,
                    limit_value: j_max,
                    observed_value: jk,
                });
            }
        }

        if let Some(tcp) = constraints.tcp
            && deriv.has_tcp()
        {
            let pp = &deriv.pp[k];
            let ppp = &deriv.ppp[k];
            let pppp = &deriv.pppp[k];

            let vx = pp[0] * sd;
            let vy = pp[1] * sd;
            let vz = pp[2] * sd;
            let tv = (vx * vx + vy * vy + vz * vz).sqrt();
            if exceeds(tv, tcp.v_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in,
                    limit_type: "tcp_velocity",
                    dof: u8::MAX,
                    limit_value: tcp.v_max,
                    observed_value: tv,
                });
            }

            let ax = ppp[0] * sd * sd + pp[0] * sdd;
            let ay = ppp[1] * sd * sd + pp[1] * sdd;
            let az = ppp[2] * sd * sd + pp[2] * sdd;
            let ta = (ax * ax + ay * ay + az * az).sqrt();
            if exceeds(ta, tcp.a_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in,
                    limit_type: "tcp_acceleration",
                    dof: u8::MAX,
                    limit_value: tcp.a_max,
                    observed_value: ta,
                });
            }

            let jx = pppp[0] * sd * sd * sd + 3.0 * ppp[0] * sd * sdd + pp[0] * sddd;
            let jy = pppp[1] * sd * sd * sd + 3.0 * ppp[1] * sd * sdd + pp[1] * sddd;
            let jz = pppp[2] * sd * sd * sd + 3.0 * ppp[2] * sd * sdd + pp[2] * sddd;
            let tj = (jx * jx + jy * jy + jz * jz).sqrt();
            if exceeds(tj, tcp.j_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in,
                    limit_type: "tcp_jerk",
                    dof: u8::MAX,
                    limit_value: tcp.j_max,
                    observed_value: tj,
                });
            }
        }
    }

    Ok(())
}

fn densify_path<const N: usize>(
    path: &SRobotPath<N, f64>,
    opts: &crate::constraints::DensificationOptions,
) -> DekeResult<(SRobotPath<N, f64>, usize)> {
    let merged = merge_near_duplicates(path, opts.min_segment_fraction)?;
    let merged_len = merged.len();

    let mut p = if let Some(step) = opts.max_segment_step {
        densify_with_kink_boost::<N>(&merged, step)
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

    Ok((p, merged_len))
}

/// Fills `chord_length`, `min_segment_length`, `max_segment_length`, and
/// `segment_length_ratio` from the densified path. Leaves the input/merged counts alone
/// (those are populated by the caller earlier in the flow).
fn populate_path_geometry<const N: usize>(stats: &mut PathStats, densified: &SRobotPath<N, f64>) {
    let m = densified.len();
    if m < 2 {
        return;
    }
    let mut total = 0.0_f64;
    let mut min_seg = f64::INFINITY;
    let mut max_seg = 0.0_f64;
    for k in 0..m - 1 {
        let d = chord_distance::<N>(densified.get(k).unwrap(), densified.get(k + 1).unwrap());
        total += d;
        if d < min_seg {
            min_seg = d;
        }
        if d > max_seg {
            max_seg = d;
        }
    }
    stats.chord_length = total;
    stats.min_segment_length = if min_seg.is_finite() { min_seg } else { 0.0 };
    stats.max_segment_length = max_seg;
    stats.segment_length_ratio = if min_seg > 0.0 && min_seg.is_finite() {
        max_seg / min_seg
    } else {
        0.0
    };
}

/// Computes the per-path PCHIP-derivative magnitude stats for failure triage. Does not
/// populate `min_qp_norm_*` or `degenerate_qp_samples` — those are tracked alongside the
/// constraint-build loop in `nlp::build_and_solve`.
fn derivative_stats_from_deriv<const N: usize>(
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
) -> DerivativeStats {
    let m = deriv.num_waypoints();
    let lock = constraints.locked_prefix.min(N);
    let mut out = DerivativeStats::default();
    for k in 0..m {
        for j in lock..N {
            let qpp = deriv.qpp[k][j].abs();
            if qpp > out.max_abs_qpp {
                out.max_abs_qpp = qpp;
                out.max_abs_qpp_sample = k;
                out.max_abs_qpp_joint = j;
            }
            let qppp = deriv.qppp[k][j].abs();
            if qppp > out.max_abs_qppp {
                out.max_abs_qppp = qppp;
                out.max_abs_qppp_sample = k;
                out.max_abs_qppp_joint = j;
            }
        }
    }
    out
}

/// Per-axis `pp` min/max + global max of `pp`/`ppp`/`pppp` — flags the TCP-axis-collapse
/// failure mode.
fn tcp_stats_from_deriv<const N: usize>(deriv: &PathDerivatives<N>) -> TcpStats {
    let m = deriv.num_waypoints();
    let mut out = TcpStats::default();
    out.min_abs_pp_per_axis = [f64::INFINITY; 3];
    for k in 0..m {
        let pp = &deriv.pp[k];
        let ppp = &deriv.ppp[k];
        let pppp = &deriv.pppp[k];
        for d in 0..3 {
            let abs_pp = pp[d].abs();
            if abs_pp > out.max_abs_pp_per_axis[d] {
                out.max_abs_pp_per_axis[d] = abs_pp;
            }
            if abs_pp < out.min_abs_pp_per_axis[d] {
                out.min_abs_pp_per_axis[d] = abs_pp;
            }
            if abs_pp > out.max_abs_pp {
                out.max_abs_pp = abs_pp;
            }
            let abs_ppp = ppp[d].abs();
            if abs_ppp > out.max_abs_ppp {
                out.max_abs_ppp = abs_ppp;
            }
            let abs_pppp = pppp[d].abs();
            if abs_pppp > out.max_abs_pppp {
                out.max_abs_pppp = abs_pppp;
            }
        }
    }
    for d in 0..3 {
        if !out.min_abs_pp_per_axis[d].is_finite() {
            out.min_abs_pp_per_axis[d] = 0.0;
        }
    }
    out
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

/// Chord-by-chord densifier that takes a smaller step in segments adjacent to sharp
/// kinks. A kink at waypoint `k` is a triplet `(p[k-1], p[k], p[k+1])` whose unit chord
/// vectors have a dot product below `-0.5` (i.e. the path direction reverses by more
/// than 120°). PCHIP's `qp` collapses to zero at a true 180° reversal, but the
/// surrounding samples carry the constraint pressure — denser sampling there gives the
/// IPM more rows of `qpp·sd² ≤ a_max` to honor before sd can climb back up.
fn densify_with_kink_boost<const N: usize>(
    path: &deke_types::SRobotPath<N, f64>,
    base_step: f64,
) -> deke_types::SRobotPath<N, f64> {
    let m = path.len();
    if m < 2 || base_step <= 0.0 {
        return path.clone();
    }

    // Per-segment boost factor: 1.0 by default, larger near sharp kinks. cos=-1 ⇒ 8×,
    // cos=-0.5 ⇒ 4×, cos≥-0.5 ⇒ 1× (untouched).
    let mut boost = vec![1.0_f64; m - 1];
    if m >= 3 {
        for k in 1..m - 1 {
            let a = path.get(k - 1).unwrap();
            let b = path.get(k).unwrap();
            let c = path.get(k + 1).unwrap();
            let d1 = chord_distance::<N>(a, b);
            let d2 = chord_distance::<N>(b, c);
            if d1 < 1e-12 || d2 < 1e-12 {
                continue;
            }
            let mut dot = 0.0_f64;
            for j in 0..N {
                dot += (b.0[j] - a.0[j]) * (c.0[j] - b.0[j]);
            }
            let cos = (dot / (d1 * d2)).clamp(-1.0, 1.0);
            // Threshold tuned for 6-DOF random-direction noise: at high N, consecutive
            // chord directions naturally have moderately negative cosines (the typical
            // dot product of two unit vectors in N-D is ~1/√N), so a permissive cutoff
            // floods the densifier on benign paths. Only triggers for clear reversals.
            if cos < -0.7 {
                let factor = 1.0 + (-cos - 0.5) * 6.0; // cos=-0.7→2.2, cos=-1.0→4.0
                if factor > boost[k - 1] {
                    boost[k - 1] = factor;
                }
                if factor > boost[k] {
                    boost[k] = factor;
                }
            }
        }
    }

    let mut out = Vec::with_capacity(m);
    out.push(*path.get(0).unwrap());
    for k in 0..m - 1 {
        let a = path.get(k).unwrap();
        let b = path.get(k + 1).unwrap();
        let d = chord_distance::<N>(a, b);
        if d <= 0.0 {
            out.push(*b);
            continue;
        }
        let effective_step = base_step / boost[k];
        let steps = (d / effective_step).ceil().max(1.0) as usize;
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            out.push(a.interpolate(b, t));
        }
    }

    deke_types::SRobotPath::try_new(out).unwrap_or_else(|_| path.clone())
}

fn infer_limiting_group<const N: usize>(
    solution: &Solution,
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
) -> (Option<LimitingGroup>, Option<usize>) {
    if matches!(solution.status, SolveStatus::LocallyInfeasible | SolveStatus::GloballyInfeasible) {
        let lock = constraints.locked_prefix.min(N);
        let m = deriv.num_waypoints();
        // (excess, group, sample_idx)
        let mut worst: (f64, LimitingGroup, usize) =
            (0.0, LimitingGroup::JointVelocity, 0);
        for k in 0..m {
            let sd = solution.sd[k].max(0.0);
            let sdd = solution.sdd[k];
            for j in lock..N {
                let qp = deriv.qp[k][j];
                let qpp = deriv.qpp[k][j];
                let v = (qp * sd).abs();
                let v_max = constraints.joint.v_max.0[j];
                if v_max.is_finite() && v - v_max > worst.0 {
                    worst = (v - v_max, LimitingGroup::JointVelocity, k);
                }
                let a = (qpp * sd * sd + qp * sdd).abs();
                let a_max = constraints.joint.a_max.0[j];
                if a_max.is_finite() && a - a_max > worst.0 {
                    worst = (a - a_max, LimitingGroup::JointAcceleration, k);
                }
            }
            if let Some(tcp) = constraints.tcp
                && deriv.has_tcp()
            {
                let pp = &deriv.pp[k];
                let tcp_v = (pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2]).sqrt() * sd;
                let tcp_v_max = tcp.v_max;
                if tcp_v_max.is_finite() && tcp_v - tcp_v_max > worst.0 {
                    worst = (tcp_v - tcp_v_max, LimitingGroup::TcpVelocity, k);
                }
            }
        }
        (Some(worst.1), Some(worst.2))
    } else {
        (None, None)
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
    let mut peak_jv_at = (0_usize, 0_usize);
    let mut peak_ja = 0.0_f64;
    let mut peak_ja_at = (0_usize, 0_usize);
    let mut peak_jj = 0.0_f64;
    let mut peak_jj_at = (0_usize, 0_usize);
    let mut peak_tv = 0.0_f64;
    let mut peak_tv_at = 0_usize;
    let mut peak_ta = 0.0_f64;
    let mut peak_ta_at = 0_usize;
    let mut peak_tj = 0.0_f64;
    let mut peak_tj_at = 0_usize;

    let jv_max: Vec<f64> = (0..N).map(|j| constraints.joint.v_max.0[j]).collect();
    let ja_max: Vec<f64> = (0..N).map(|j| constraints.joint.a_max.0[j]).collect();
    let jj_max: Vec<f64> = (0..N).map(|j| constraints.joint.j_max.0[j]).collect();
    // TCP limits are optional; when absent we never call the update helper against these
    // so the actual value doesn't matter — sentinel infinities mean `update_util` no-ops.
    let (tv_max, ta_max, tj_max) = match constraints.tcp {
        Some(tcp) => (tcp.v_max, tcp.a_max, tcp.j_max),
        None => (f64::INFINITY, f64::INFINITY, f64::INFINITY),
    };

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
            if jv > peak_jv {
                peak_jv = jv;
                peak_jv_at = (k, j);
            }
            if ja > peak_ja {
                peak_ja = ja;
                peak_ja_at = (k, j);
            }
            if jj > peak_jj {
                peak_jj = jj;
                peak_jj_at = (k, j);
            }
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
            if tv > peak_tv {
                peak_tv = tv;
                peak_tv_at = k;
            }
            update_util(&mut step_util, tv, tv_max);

            let ax = ppp[0] * sd * sd + pp[0] * sdd;
            let ay = ppp[1] * sd * sd + pp[1] * sdd;
            let az = ppp[2] * sd * sd + pp[2] * sdd;
            let ta = (ax * ax + ay * ay + az * az).sqrt();
            if ta > peak_ta {
                peak_ta = ta;
                peak_ta_at = k;
            }
            update_util(&mut step_util, ta, ta_max);

            let jx = pppp[0] * sd * sd * sd + 3.0 * ppp[0] * sd * sdd + pp[0] * sddd;
            let jy = pppp[1] * sd * sd * sd + 3.0 * ppp[1] * sd * sdd + pp[1] * sddd;
            let jz = pppp[2] * sd * sd * sd + 3.0 * ppp[2] * sd * sdd + pp[2] * sddd;
            let tj = (jx * jx + jy * jy + jz * jz).sqrt();
            if tj > peak_tj {
                peak_tj = tj;
                peak_tj_at = k;
            }
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
    diag.peak_joint_velocity_at = PeakLocation {
        value: peak_jv,
        sample: peak_jv_at.0,
        joint: Some(peak_jv_at.1),
    };
    diag.peak_joint_acceleration_at = PeakLocation {
        value: peak_ja,
        sample: peak_ja_at.0,
        joint: Some(peak_ja_at.1),
    };
    diag.peak_joint_jerk_at = PeakLocation {
        value: peak_jj,
        sample: peak_jj_at.0,
        joint: Some(peak_jj_at.1),
    };
    diag.peak_tcp_velocity_at = PeakLocation {
        value: peak_tv,
        sample: peak_tv_at,
        joint: None,
    };
    diag.peak_tcp_acceleration_at = PeakLocation {
        value: peak_ta,
        sample: peak_ta_at,
        joint: None,
    };
    diag.peak_tcp_jerk_at = PeakLocation {
        value: peak_tj,
        sample: peak_tj_at,
        joint: None,
    };
    diag.average_utilization = if m > 0 { util_sum / m as f64 } else { 0.0 };
}

/// Two-stage solve: first run with TCP disabled to get a feasible warm-start, then run
/// with TCP enabled seeded from the stage-1 solution. Returns the stage-2 result.
///
/// On hard paths the stage-1 solution is in a feasible neighborhood of the full
/// problem's optimum, so stage 2 typically converges in tens of iterations (vs the
/// IPM grinding through max-iter limit then bailing out from a synthetic initial guess).
/// Total wall time is comparable to or better than single-stage even on easy paths
/// because stage 1 is small (no quadratic TCP constraints, no FK calls) and stage 2
/// converges fast from the warm start.
fn two_stage_solve<const N: usize>(
    densified: &SRobotPath<N, f64>,
    deriv_with_tcp: &PathDerivatives<N>,
    fk: &impl FKChain<N, f64>,
    constraints: &Topp3Tcp6Constraints<N>,
    start: crate::boundary::ProjectedBoundary,
    end: crate::boundary::ProjectedBoundary,
) -> DekeResult<Solution> {
    let _ = fk; // FK already consumed by deriv_with_tcp; passed through for clarity at call site.

    // Stage 1: TCP-disabled derivatives + TCP-disabled constraints.
    let deriv_no_tcp = PathDerivatives::<N>::new_without_tcp(densified)?;
    let mut cfg_no_tcp = constraints.clone();
    cfg_no_tcp.tcp = None;
    // Avoid recursion: stage 1 must not itself try to two-stage.
    cfg_no_tcp.solver.two_stage_warm_start = false;

    // Re-project boundaries against TCP-free derivatives. The qp/qpp values are the same
    // (they come from joint waypoints, not FK), but recomputing keeps it explicit.
    let start_no_tcp = crate::boundary::project::<N>(
        &constraints.boundary.v_start,
        &constraints.boundary.a_start,
        &deriv_no_tcp.qp[0],
        &deriv_no_tcp.qpp[0],
    );
    let end_idx = deriv_no_tcp.num_waypoints() - 1;
    let end_no_tcp = crate::boundary::project::<N>(
        &constraints.boundary.v_end,
        &constraints.boundary.a_end,
        &deriv_no_tcp.qp[end_idx],
        &deriv_no_tcp.qpp[end_idx],
    );

    let stage1 = build_and_solve::<N>(&deriv_no_tcp, &cfg_no_tcp, start_no_tcp, end_no_tcp)?;
    if !matches!(stage1.status, SolveStatus::Success) {
        // Stage 1 failed — fall back to single-stage on the full problem. The stage-1
        // failure usually means the path is joint-infeasible, in which case stage 2
        // will fail too, but at least the user gets a meaningful diagnostic.
        return build_and_solve::<N>(deriv_with_tcp, constraints, start, end);
    }

    // Stage 2: TCP-enabled with the stage-1 solution as warm start.
    let stage2 = build_and_solve_warm::<N>(deriv_with_tcp, constraints, start, end, &stage1)?;
    Ok(stage2)
}
