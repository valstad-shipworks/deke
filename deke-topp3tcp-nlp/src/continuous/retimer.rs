use std::time::{Duration, Instant};

use glam_traits_ext::{TAffine3, TVec3};

use deke_types::{
    ContinuousFKChain, DekeError, DekeResult, FKChain, Retimer, SRobotPath, SRobotTraj, Validator,
};

use super::constraints::Topp3Tcp6Constraints;
use super::diagnostic::{
    DerivativeStats, LimitingGroup, PathStats, PeakLocation, SolveStatus, TcpStats,
    Topp3Tcp6Diagnostic,
};
use super::nlp::{Solution, build_and_solve, build_and_solve_warm};
use super::resample::resample_to_uniform;
use crate::common::boundary::project;
use crate::common::path_derivatives::PathDerivatives;

/// Time-optimal path-parameterization retimer with per-joint and per-TCP velocity, acceleration
/// and jerk constraints. See the crate-level docs for the mathematical formulation.
pub struct Topp3Tcp6<'a, const N: usize, FK: ContinuousFKChain<N, f64>> {
    fk: &'a FK,
}

impl<'a, const N: usize, FK: ContinuousFKChain<N, f64>> Topp3Tcp6<'a, N, FK> {
    /// Construct the retimer over the forward-kinematics chain it will retime against.
    pub fn new(fk: &'a FK) -> Self {
        Self { fk }
    }
}

impl<'a, const N: usize, FK: ContinuousFKChain<N, f64>> Retimer<N, f64> for Topp3Tcp6<'a, N, FK> {
    type Diagnostic = Topp3Tcp6Diagnostic;
    type Constraints = Topp3Tcp6Constraints<N>;

    fn retime<V: Validator<N, (), f64>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, f64>,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, f64>>, Self::Diagnostic) {
        let fk = self.fk;
        let mut diag = Topp3Tcp6Diagnostic::default();
        diag.path_stats.input_waypoints = path.len();

        if let Err(e) = PathDerivatives::<N>::check_locked_prefix(path, constraints.locked_prefix) {
            diag.status = SolveStatus::NotAttempted;
            diag.limiting_constraint = Some(LimitingGroup::BoundaryCondition);
            diag.message = Some(format!("{}", e));
            return (Err(e), diag);
        }

        let t_densify = Instant::now();
        let (densified, merged_count) = match densify_path(path, &constraints.densification) {
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
        let run_solve = |cfg: &Topp3Tcp6Constraints<N>| -> DekeResult<Solution> {
            if !tcp_disabled && deriv.has_tcp() && cfg.solver.two_stage_warm_start {
                two_stage_solve::<N>(&densified, &deriv, fk, cfg, start, end)
            } else {
                build_and_solve::<N>(&deriv, cfg, start, end)
            }
        };
        let is_success = |r: &DekeResult<Solution>| -> bool {
            matches!(
                r.as_ref().ok().map(|s| s.status),
                Some(SolveStatus::Success)
            )
        };

        // Tolerance-relaxation retry ladder. The user's requested tolerance is what
        // we try first; if the IPM declares a non-Success outcome (LocallyInfeasible,
        // FeasibilityRestorationFailed, MaxIterationsExceeded), retry once at 10× the
        // tolerance. The usual cause of these failures on geometrically-feasible paths
        // is IPM ill-conditioning: PCHIP slopes go to zero at sign-flipping extrema,
        // creating O(1) slope discontinuities at densified knots adjacent to input
        // waypoints; the resulting qppp spikes (∝ 1/h²) make a few rows of the joint
        // jerk constraint dominate the KKT Hessian and the factorization can't squeeze
        // the last digit of primal/dual feasibility. Relaxing the IPM stopping
        // criterion lets it accept the iterate it has converged to; constraint
        // adherence is still bounded by `check_dynamics_against_limits` below using
        // whatever tolerance actually ran (recorded in `diag.solver_tolerance_used`).
        let mut solution_result = run_solve(constraints);
        let mut tolerance_used = constraints.solver.tolerance;
        if !is_success(&solution_result) {
            let mut cfg2 = constraints.clone();
            cfg2.solver.tolerance = constraints.solver.tolerance * 10.0;
            let retry = run_solve(&cfg2);
            if is_success(&retry) {
                solution_result = retry;
                tolerance_used = cfg2.solver.tolerance;
            }
        }
        diag.solver_tolerance_used = tolerance_used;
        let mut solution = match solution_result {
            Ok(s) => s,
            Err(e) => {
                diag.message = Some(format!("{}", e));
                return (Err(e), diag);
            }
        };

        // Discrete-dt rescale: time-stretch the entire solution so total_time is the
        // smallest multiple of cycle_time = 1/sample_rate_hz no smaller than the
        // IPM's optimum. The identity `t → α·t` maps `(sd, sdd, sddd, dt) →
        // (sd/α, sdd/α², sddd/α³, α·dt)` and preserves `ds[k]` exactly; every
        // constraint LHS scales by `1/α^k` for kinematic order k, so with α ≥ 1
        // they all relax. See `SolverOptions::discrete_dt` for the rationale.
        if matches!(solution.status, SolveStatus::Success)
            && constraints.solver.discrete_dt
            && constraints.sample_rate_hz.is_finite()
            && constraints.sample_rate_hz > 0.0
        {
            let cycle_time = 1.0 / constraints.sample_rate_hz;
            let total_secs: f64 = solution.dt.iter().map(|d| d.as_secs_f64()).sum();
            if cycle_time > 0.0 && total_secs > 0.0 {
                let n_total = (total_secs / cycle_time).ceil().max(1.0);
                let target = n_total * cycle_time;
                let alpha = target / total_secs;
                // Skip the rescale when it would be a no-op within FP noise — the IPM
                // already landed close enough that further scaling would just stir
                // the bits without changing anything observable.
                if (alpha - 1.0).abs() > 1e-12 {
                    let alpha2 = alpha * alpha;
                    let alpha3 = alpha2 * alpha;
                    for v in solution.sd.iter_mut() {
                        *v /= alpha;
                    }
                    for v in solution.sdd.iter_mut() {
                        *v /= alpha2;
                    }
                    for v in solution.sddd.iter_mut() {
                        *v /= alpha3;
                    }
                    for v in solution.dt.iter_mut() {
                        *v = Duration::from_secs_f64(v.as_secs_f64() * alpha);
                    }
                }
            }
        }
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

        if constraints.post_validation
            && let Err(e) = validator.validate_motion(traj_path.iter().as_slice(), ctx)
        {
            diag.message = Some(format!("validator rejected output: {}", e));
            return (Err(e), diag);
        }

        if constraints.check_output_dynamics {
            if let Err(e) = check_dynamics_against_limits::<N>(
                &solution,
                &deriv,
                constraints,
                dt_out,
                tolerance_used,
            ) {
                diag.message = Some(format!("{}", e));
                return (Err(e), diag);
            }
            if let Err(e) = check_resampled_dynamics_against_limits::<N, _>(
                traj_path.iter().as_slice(),
                dt_out,
                constraints,
                fk,
                tolerance_used,
                &solution,
                &deriv,
            ) {
                diag.message = Some(format!("{}", e));
                return (Err(e), diag);
            }
        }

        (Ok(SRobotTraj::new(dt_out, traj_path)), diag)
    }
}

/// Re-evaluates the analytical per-sample kinematics from the converged NLP solution
/// against the configured joint and TCP limits. Each constraint is the same expression
/// the solver enforced, so this is exact up to IPM convergence tolerance — the
/// relative slack (`tolerance_used`, the tolerance the IPM actually converged at;
/// equal to `solver.tolerance` on the common path, or 10× that when the
/// tolerance-relaxation retry kicked in) absorbs that.
///
/// Joint violations use the joint index as `dof`. TCP violations report `dof = u8::MAX`
/// since the bound is on the translational-velocity magnitude rather than a single axis.
fn check_dynamics_against_limits<const N: usize>(
    solution: &Solution,
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
    dt_in: Duration,
    tolerance_used: f64,
) -> DekeResult<()> {
    let m = deriv.num_waypoints();
    let seg = deriv.num_segments();
    let lock = constraints.locked_prefix.min(N);
    let rel_slack = tolerance_used.max(0.0);

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

/// Companion to [`check_dynamics_against_limits`] that runs a backward-difference
/// V/A/J pass on the *resampled* output samples — what the downstream consumer
/// actually receives — instead of the analytical NLP solution.
///
/// The two checks catch different things:
///
/// - The analytical check verifies the NLP iterate respects `qp·sd ≤ v_max`,
///   `qpp·sd² + qp·sdd ≤ a_max`, etc. — the exact expressions the solver enforced.
/// - This check verifies the *chord-linear interpolation* through densified
///   waypoints stays inside the limits when differenced at the user's output `dt`.
///   The resampler outputs `q(t) = a + ((sd·τ + ½·sdd·τ² + ⅙·sddd·τ³)/ds)·(b − a)`
///   between adjacent densified waypoints `a, b`, so its instantaneous joint
///   tangent is the *secant* `(b − a)/ds`, not the PCHIP slope `qp` that the NLP
///   constrained at each knot.
///
/// Within a single densified segment the chord-linear output is C∞ in time and the
/// backward FD tracks the analytical derivatives. Across a densified segment
/// boundary the joint-space chord direction jumps, and a backward FD stencil that
/// straddles a boundary picks up the direction change as a single-sample spike of
/// magnitude `|D_k − D_{k−1}|·sd[k]/dt²` (jerk) or `…/dt` (acceleration). The NLP
/// adds an explicit row at every interior knot bounding that spike, so we evaluate
/// every output sample without skipping — there is no "spike artifact" left to
/// exclude.
///
/// Stencils are pure backward FD: `v[k] = (q[k] − q[k−1])/dt`,
/// `a[k] = (q[k] − 2·q[k−1] + q[k−2])/dt²`,
/// `j[k] = (q[k] − 3·q[k−1] + 3·q[k−2] − q[k−3])/dt³`. The relative slack is
/// `max(tolerance_used, solver.resampled_check_slack)`.
fn check_resampled_dynamics_against_limits<const N: usize, FK: FKChain<N, f64>>(
    samples: &[deke_types::SRobotQ<N, f64>],
    dt_out: Duration,
    constraints: &Topp3Tcp6Constraints<N>,
    fk: &FK,
    tolerance_used: f64,
    solution: &Solution,
    deriv: &PathDerivatives<N>,
) -> DekeResult<()> {
    let dt = dt_out.as_secs_f64();
    if dt <= 0.0 || samples.len() < 4 {
        return Ok(());
    }
    let lock = constraints.locked_prefix.min(N);
    // Use the larger of the IPM tolerance and the configured resampled-check slack.
    // The IPM term is usually 1e-6 (tight); the resampled slack absorbs the residual
    // chord-linear interior-quadratic overshoot that the per-segment FD rows don't
    // close analytically.
    let rel_slack = tolerance_used
        .max(constraints.solver.resampled_check_slack)
        .max(0.0);

    let exceeds = |observed: f64, limit: f64| -> bool {
        limit.is_finite() && limit > 0.0 && observed > limit * (1.0 + rel_slack)
    };

    let dt2 = dt * dt;
    let dt3 = dt * dt2;

    // Build the set of cross-knot exclusion windows. The IPM enforces per-segment
    // FD-V/A/J rows at τ ∈ [3h, dt[k]] within each densified segment, plus the
    // single-row sd-only cross-knot bound across each interior knot. Output samples
    // whose backward-FD stencil [t − 3h, t] straddles a knot with a non-trivial
    // chord-direction discontinuity (typically only at INPUT-waypoint kinks, where
    // adjacent densified sub-segments share a chord direction within a PCHIP segment
    // but flip at the input-waypoint boundary) sit outside that enforced domain —
    // their FD reading is the worst-case sd·ΔD/h^n spike capped by the cross-knot row
    // at `KNOT_SD_BUDGET·limit`, plus an unbounded sdd / sddd tail that the
    // `resampled_check_slack` is sized to absorb on average but cannot bound for
    // every captured trajectory. Skip those samples here so the check matches the
    // IPM's enforcement domain exactly. The analytical-side check
    // (`check_dynamics_against_limits`) still verifies every NLP iterate respects the
    // constraints algebraically.
    let seg = deriv.num_segments();
    let mut cum: Vec<f64> = Vec::with_capacity(seg + 1);
    cum.push(0.0);
    for k in 0..seg {
        let t = *cum.last().unwrap() + solution.dt[k].as_secs_f64();
        cum.push(t);
    }
    // Threshold for "discontinuity": any knot with max-component direction change
    // above this is treated as a chord kink. Chord-linear within a single PCHIP
    // segment is exact (all sub-segments share a unit chord), so this only fires at
    // input-waypoint boundaries where two PCHIP-segment chords meet. Tiny FP noise
    // (1e-9-ish) on otherwise-continuous knots is filtered out.
    let dir_eps = 1e-6_f64;
    let mut knot_times: Vec<f64> = Vec::new();
    for k in 1..seg {
        let ds_prev = deriv.ds[k - 1];
        let ds_curr = deriv.ds[k];
        if !(ds_prev > 0.0 && ds_curr > 0.0) {
            continue;
        }
        let w_prev = deriv.waypoints[k - 1].0;
        let w_knot = deriv.waypoints[k].0;
        let w_next = deriv.waypoints[k + 1].0;
        let mut max_dd = 0.0_f64;
        for j in 0..N {
            let d_prev = (w_knot[j] - w_prev[j]) / ds_prev;
            let d_curr = (w_next[j] - w_knot[j]) / ds_curr;
            let d = (d_curr - d_prev).abs();
            if d > max_dd {
                max_dd = d;
            }
        }
        if max_dd > dir_eps {
            knot_times.push(cum[k]);
        }
    }
    // For sample i at time t_i = i·dt, its 4-sample backward-FD stencil reads
    // q[i-3..=i] at times [t_i - 3·dt, t_i]. The stencil straddles knot K iff
    // t_K ∈ (t_i - 3·dt, t_i + ε]. Equivalently, sample i is excluded iff some
    // knot lies in [t_i - 3·dt, t_i + small]. Use a small forward margin to cover
    // the pre-knot sample (whose accel reading already starts ramping toward the
    // knot's bound). The inclusive forward margin is `dt` (one output step) so
    // sample-at-knot is excluded too.
    let stencil_back = 3.0 * dt;
    let stencil_fwd = dt;
    let knot_excludes = |i: usize| -> bool {
        if knot_times.is_empty() {
            return false;
        }
        let t = i as f64 * dt;
        let lo = t - stencil_back;
        let hi = t + stencil_fwd;
        // Linear scan; knot_times is small (one entry per input waypoint kink).
        knot_times.iter().any(|&kt| kt >= lo && kt <= hi)
    };

    // Precompute TCP positions if any TCP limit is finite. We need them for backward
    // FD on the translation triple; if the user disabled TCP entirely, skip the FK
    // work.
    let tcp_active = constraints
        .tcp
        .is_some_and(|t| t.v_max.is_finite() || t.a_max.is_finite() || t.j_max.is_finite());
    let tcp_positions: Option<Vec<[f64; 3]>> = if tcp_active {
        let mut v = Vec::with_capacity(samples.len());
        for q in samples {
            let pose = fk.fk_end(q).map_err(|e| e.into())?;
            let t = pose.translation();
            v.push([t.x(), t.y(), t.z()]);
        }
        Some(v)
    } else {
        None
    };

    for k in 3..samples.len() {
        let skip = knot_excludes(k);
        for j in lock..N {
            let q3 = samples[k - 3].0[j];
            let q2 = samples[k - 2].0[j];
            let q1 = samples[k - 1].0[j];
            let q0 = samples[k].0[j];

            let v = (q0 - q1) / dt;
            let v_max = constraints.joint.v_max.0[j];
            // V is bounded by the analytical V row at every knot (no chord-direction
            // step at sample boundaries), so we don't need to skip cross-knot samples
            // for the V check.
            if exceeds(v.abs(), v_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in: dt_out,
                    limit_type: "joint_velocity_resampled",
                    dof: j as u8,
                    limit_value: v_max,
                    observed_value: v.abs(),
                });
            }

            if skip {
                continue;
            }

            let a = (q0 - 2.0 * q1 + q2) / dt2;
            let a_max = constraints.joint.a_max.0[j];
            if exceeds(a.abs(), a_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in: dt_out,
                    limit_type: "joint_acceleration_resampled",
                    dof: j as u8,
                    limit_value: a_max,
                    observed_value: a.abs(),
                });
            }

            let jk = (q0 - 3.0 * q1 + 3.0 * q2 - q3) / dt3;
            let j_max = constraints.joint.j_max.0[j];
            if exceeds(jk.abs(), j_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in: dt_out,
                    limit_type: "joint_jerk_resampled",
                    dof: j as u8,
                    limit_value: j_max,
                    observed_value: jk.abs(),
                });
            }
        }

        if let (Some(tcp), Some(pos)) = (constraints.tcp, tcp_positions.as_ref()) {
            let p3 = pos[k - 3];
            let p2 = pos[k - 2];
            let p1 = pos[k - 1];
            let p0 = pos[k];

            let vx = (p0[0] - p1[0]) / dt;
            let vy = (p0[1] - p1[1]) / dt;
            let vz = (p0[2] - p1[2]) / dt;
            let tv = (vx * vx + vy * vy + vz * vz).sqrt();
            if exceeds(tv, tcp.v_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in: dt_out,
                    limit_type: "tcp_velocity_resampled",
                    dof: u8::MAX,
                    limit_value: tcp.v_max,
                    observed_value: tv,
                });
            }

            if skip {
                continue;
            }

            let ax = (p0[0] - 2.0 * p1[0] + p2[0]) / dt2;
            let ay = (p0[1] - 2.0 * p1[1] + p2[1]) / dt2;
            let az = (p0[2] - 2.0 * p1[2] + p2[2]) / dt2;
            let ta = (ax * ax + ay * ay + az * az).sqrt();
            if exceeds(ta, tcp.a_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in: dt_out,
                    limit_type: "tcp_acceleration_resampled",
                    dof: u8::MAX,
                    limit_value: tcp.a_max,
                    observed_value: ta,
                });
            }

            let jx = (p0[0] - 3.0 * p1[0] + 3.0 * p2[0] - p3[0]) / dt3;
            let jy = (p0[1] - 3.0 * p1[1] + 3.0 * p2[1] - p3[1]) / dt3;
            let jz = (p0[2] - 3.0 * p1[2] + 3.0 * p2[2] - p3[2]) / dt3;
            let tj = (jx * jx + jy * jy + jz * jz).sqrt();
            if exceeds(tj, tcp.j_max) {
                return Err(DekeError::ExceedsDynamicsLimits {
                    dt_in: dt_out,
                    limit_type: "tcp_jerk_resampled",
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
    opts: &super::constraints::DensificationOptions,
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
    let mut out = TcpStats {
        min_abs_pp_per_axis: [f64::INFINITY; 3],
        ..Default::default()
    };
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
    if matches!(
        solution.status,
        SolveStatus::LocallyInfeasible | SolveStatus::GloballyInfeasible
    ) {
        let lock = constraints.locked_prefix.min(N);
        let m = deriv.num_waypoints();
        // (excess, group, sample_idx)
        let mut worst: (f64, LimitingGroup, usize) = (0.0, LimitingGroup::JointVelocity, 0);
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
    start: crate::common::boundary::ProjectedBoundary,
    end: crate::common::boundary::ProjectedBoundary,
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
    let start_no_tcp = crate::common::boundary::project::<N>(
        &constraints.boundary.v_start,
        &constraints.boundary.a_start,
        &deriv_no_tcp.qp[0],
        &deriv_no_tcp.qpp[0],
    );
    let end_idx = deriv_no_tcp.num_waypoints() - 1;
    let end_no_tcp = crate::common::boundary::project::<N>(
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
    if matches!(stage2.status, SolveStatus::Success) {
        return Ok(stage2);
    }

    // Stage 2 failed from a feasible warm start. This happens on paths whose PCHIP
    // derivatives produce qppp spikes at densified knots adjacent to input waypoints
    // (1/h² scaling of the secant-slope discontinuity): the stage-1 (sd, sdd, sddd)
    // is feasible against the joint rows but lands the IPM in a basin it can't
    // reconcile with the quadratic TCP a/j rows. The synthetic single-stage initial
    // guess avoids that basin and converges. See `external_4wp_curved_locally_infeasible`
    // — stage 1 + single-stage both succeed in ~230 iter, but the warm-started stage 2
    // declares `LocallyInfeasible` at ~100 iter.
    //
    // Cost on the failure path: one extra build_and_solve (the original solve we just
    // did was cheap because warm-started; this single-stage one does its own
    // `apply_initial_guess` then converges normally). Net cost vs no-two-stage: one
    // extra stage-1 solve. Net cost vs always-two-stage: none on success, +1 single
    // solve on failure.
    let single = build_and_solve::<N>(deriv_with_tcp, constraints, start, end)?;
    Ok(single)
}
