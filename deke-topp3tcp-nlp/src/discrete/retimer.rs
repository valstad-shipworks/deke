use std::time::{Duration, Instant};

use deke_topp_speed::{MotionSpec, ToppSolver};
use deke_types::{
    ContinuousFKChain, DekeError, DekeResult, FKChain, JointValidator, Retimer, SRobotPath, SRobotQ,
    SRobotTraj, Validator,
};

use crate::common::boundary::project;
use super::constraints::{DensificationOptions, Topp3Tcp6DiscreteConstraints};
use super::diagnostic::{
    BisectionStep, LimitingGroup, SolveStatus, Topp3Tcp6DiscreteDiagnostic,
};
use super::nlp::{
    DiscreteSolution, bins_from_sigma, build_and_solve_discrete,
    build_and_solve_discrete_with_bins, build_and_solve_discrete_with_timeout,
};
use crate::common::path_derivatives::PathDerivatives;
use super::verify::verify_output_fd;

/// Discrete-time TOPP3TCP6 retimer.
///
/// Variables are the path-arc-length values at output samples; the IPM directly
/// enforces the same backward-FD bounds that downstream consumers measure, so
/// the strict verifier in [`super::verify`] returns `Ok` to within the IPM
/// tolerance on every successful retime — no `resampled_check_slack` needed.
pub struct Topp3Tcp6Discrete<'a, const N: usize, FK: ContinuousFKChain<N, f64>> {
    fk: &'a FK,
}

impl<'a, const N: usize, FK: ContinuousFKChain<N, f64>> Topp3Tcp6Discrete<'a, N, FK> {
    /// Construct the retimer over the forward-kinematics chain it will retime against.
    pub fn new(fk: &'a FK) -> Self {
        Self { fk }
    }
}

impl<'a, const N: usize, FK: ContinuousFKChain<N, f64>> Retimer<N, f64> for Topp3Tcp6Discrete<'a, N, FK> {
    type Diagnostic = Topp3Tcp6DiscreteDiagnostic;
    type Constraints = Topp3Tcp6DiscreteConstraints<N>;

    fn retime<V: Validator<N, (), f64>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, f64>,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, f64>>, Self::Diagnostic) {
        let fk = self.fk;
        let mut diag = Topp3Tcp6DiscreteDiagnostic::default();
        diag.path_stats.input_waypoints = path.len();

        if let Err(e) = PathDerivatives::<N>::check_locked_prefix(path, constraints.locked_prefix)
        {
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
        populate_path_geometry(&mut diag.path_stats, &densified);

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

        // ── K bisection ───────────────────────────────────────────────────
        let h = 1.0 / constraints.sample_rate_hz;
        let s_total = deriv.total_length();
        let lock = constraints.locked_prefix.min(N);

        // Lower bound on K: ceil(S / v_max / h) + boundary settling.
        // The +3 absorbs the rest-to-rest σ[1]=σ[2]=0 (and similarly at the
        // tail) pinned by the boundary equalities.
        let v_max_path: f64 = (lock..N)
            .map(|j| constraints.joint.v_max.0[j])
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(0.0_f64, f64::max);
        let v_max_path = v_max_path.max(1e-6);
        let mut k_lo = ((s_total / (v_max_path * h)).ceil() as usize) + 4;
        k_lo = k_lo.max(8);
        let mut k_hi = (k_lo * 4).max(k_lo + 2);

        // Seed: optionally run `deke-topp-speed` first to get a feasible σ
        // profile and a tighter `K_hi` initial guess. Topp-speed is fast
        // (sub-ms typically), respects joint V/A/J and TCP V, and produces
        // a jerk-limited trajectory whose sample count is a good upper
        // bound on the discrete crate's bisection K. The σ values projected
        // back onto the densified path are a much better IPM warm start
        // than the uniform initial guess — especially on jerk-tight paths
        // where the optimal σ is highly non-uniform.
        // Seed σ from topp-speed. The seed's *sample count* is unreliable as
        // a `k_hi` estimate because topp-speed ignores TCP-A/J — on
        // TCP-A/J-binding paths it underestimates required K by a large
        // factor, which would send the probe loop doubling for nothing.
        // Use the seed only as the σ warm-start; the bisection's K_hi keeps
        // its v_max-derived initial guess and grows from there.
        let trace_b = std::env::var("DEKE_DISCRETE_TRACE").is_ok();
        let mut warm: Option<Vec<f64>> = None;
        if constraints.solver.seed_from_topp_speed
            && let Some(seed) = topp_speed_seed::<N, _>(&deriv, path, constraints, fk, h)
        {
            // Use 4× topp-speed's sample count as a starting `K_hi`. Topp-
            // speed respects joint V/A/J and TCP V but not TCP A/J, so its
            // `K` underestimates the discrete crate's required `K` on TCP-
            // A/J-binding paths (typically by 1.5–4×). Quadrupling it as the
            // initial `K_hi` lets the probe loop skip the slowest slack-mode
            // probes on those paths — sleipnir's per-solve timeout doesn't
            // preempt within an iter, so each narrowly-infeasible-K probe
            // costs tens of seconds. The bisection narrows back down to the
            // smallest feasible `K` quickly because each strict probe runs
            // in ~100 ms.
            let seed_k_hi = seed.0.saturating_mul(4);
            if seed_k_hi > k_hi {
                k_hi = seed_k_hi;
            }
            warm = Some(seed.1);
            if trace_b {
                eprintln!(
                    "[discrete] topp-speed seed K={} → initial K_hi={}",
                    seed.0, k_hi
                );
            }
        }

        let mut last_feasible: Option<DiscreteSolution> = None;
        let max_iter = constraints.solver.max_bisection_iterations.max(1);
        let tol = constraints.solver.tolerance;

        // Per-probe IPM timeout. The slack-mode bisection only needs a
        // feasible/infeasible verdict; if the IPM enters feasibility
        // restoration and grinds, we'd rather move on and grow K. Capped to
        // the user-supplied `solver.timeout` when present (so users can still
        // tighten further).
        let probe_timeout = match constraints.solver.timeout {
            Some(t) => Some(t.min(Duration::from_millis(500))),
            None => Some(Duration::from_millis(500)),
        };

        // Probe k_hi first; grow if infeasible. Captured-failure trajectories
        // (curved 6-DOF paths near singularities, long rest-to-rest moves with
        // a tight TCP) commonly need `K` 16–64× the v_max-derived lower bound;
        // 8 doublings gives a 256× headroom.
        //
        // When a probe times out (sleipnir's feasibility-restoration sub-iter
        // is uninterruptible — the per-solve timeout option fires at the
        // *start* of each outer iter, so each iter still runs to completion),
        // we treat that `K` as "infeasible and in the slow zone" and jump
        // by 4× instead of 2× to skip ahead past the slow zone faster.
        for probe in 0..8 {
            let t = Instant::now();
            let sol = build_and_solve_discrete_with_timeout::<N>(
                &deriv, constraints, start, end, k_hi, true, warm.as_deref(), probe_timeout,
            );
            if trace_b {
                if let Ok(s) = &sol {
                    eprintln!(
                        "[discrete] probe#{} K={} status={:?} iter={} slack={:.3e} wall={:?}",
                        probe, k_hi, s.status, s.iterations, s.slack, t.elapsed()
                    );
                } else {
                    eprintln!("[discrete] probe#{} K={} ERR wall={:?}", probe, k_hi, t.elapsed());
                }
            }
            match sol {
                Ok(s) => {
                    record_step(&mut diag.bisection_steps, &s);
                    let feasible = matches!(s.status, SolveStatus::Success) && s.slack < tol;
                    let timed_out = matches!(s.status, SolveStatus::Timeout);
                    warm = Some(s.sigma.clone());
                    if feasible {
                        last_feasible = Some(s);
                        break;
                    } else {
                        k_lo = k_hi;
                        k_hi *= if timed_out { 4 } else { 2 };
                    }
                }
                Err(e) => {
                    diag.message = Some(format!("k_hi probe failed: {}", e));
                    diag.status = SolveStatus::NotAttempted;
                    return (Err(e), diag);
                }
            }
        }

        if last_feasible.is_none() {
            diag.status = SolveStatus::LocallyInfeasible;
            diag.limiting_constraint = Some(LimitingGroup::KSampleCountAtCeiling);
            diag.message = Some(format!(
                "no feasible K found up to ceiling {}",
                k_hi
            ));
            let err = DekeError::RetimerFailed(diag.message.clone().unwrap_or_default());
            return (Err(err), diag);
        }

        // Standard bisection. We use strict (no-slack) mode here so the IPM
        // returns `LocallyInfeasible` quickly on infeasible `K` instead of
        // grinding feasibility-restoration sub-iters for tens of seconds.
        // The slack-mode probing above established that `K_hi` is feasible,
        // and the warm σ from that solve is a good interior starting point;
        // strict mode just verifies that smaller `K` values are still
        // feasible (Success) vs not (LocallyInfeasible). Sleipnir's per-iter
        // timeout doesn't preempt feasibility-restoration sub-iters within
        // an iter, so slack-mode bisection probes at narrowly-infeasible K
        // would otherwise blow the wall budget.
        //
        // We also bail out of the bisection if it consumes more wall-time
        // than the strict K_hi probe took. The slow zone in the IPM is
        // typically narrow (a few percent of K), so once we've spent that
        // much extra time on `K_mid` probes we're paying more for marginal
        // K-reduction than the runtime savings justify.
        let bisect_start = Instant::now();
        // Default 2 s bisection budget — enough for ~10–20 strict probes at
        // typical K (100–200 ms each) plus the occasional 500 ms-bounded
        // timeout. Adversarial paths that need many probes bail out with
        // `K_hi` as the final answer (the trajectory is still feasible, just
        // not the K-optimal one). The user can override via
        // `solver.timeout`, which takes precedence here too.
        let bisect_wall_budget = constraints
            .solver
            .timeout
            .unwrap_or(Duration::from_millis(2000));
        for biter in 0..max_iter {
            if k_hi - k_lo <= 1 {
                break;
            }
            if bisect_start.elapsed() > bisect_wall_budget {
                if trace_b {
                    eprintln!(
                        "[discrete] bisect budget exhausted after #{} ({:?} > {:?}); keeping K_hi={}",
                        biter,
                        bisect_start.elapsed(),
                        bisect_wall_budget,
                        k_hi
                    );
                }
                break;
            }
            let k_mid = (k_lo + k_hi) / 2;
            let t = Instant::now();
            let sol = build_and_solve_discrete_with_timeout::<N>(
                &deriv, constraints, start, end, k_mid, false, warm.as_deref(), probe_timeout,
            );
            let s = match sol {
                Ok(s) => s,
                Err(e) => {
                    diag.message = Some(format!("bisection k={} failed: {}", k_mid, e));
                    break;
                }
            };
            if trace_b {
                eprintln!(
                    "[discrete] bisect#{} K={} (lo={} hi={}) status={:?} iter={} slack={:.3e} wall={:?}",
                    biter, k_mid, k_lo, k_hi, s.status, s.iterations, s.slack, t.elapsed()
                );
            }
            record_step(&mut diag.bisection_steps, &s);
            let feasible = matches!(s.status, SolveStatus::Success);
            if feasible {
                warm = Some(s.sigma.clone());
                k_hi = k_mid;
                last_feasible = Some(s);
            } else if matches!(s.status, SolveStatus::Timeout) {
                // The IPM ran out of wall budget at this `K` — treat as
                // infeasible *and* shrink the remaining bisection budget so
                // we don't fall into the same slow zone again.
                k_lo = k_mid;
            } else {
                k_lo = k_mid;
            }
        }

        let lf = last_feasible.unwrap();
        // Strict-feasible margin: try a strict (no-slack) re-solve at
        // `K_bisection + headroom` so the IPM has interior headroom. The
        // slack-feasible σ at the bisection optimum can overshoot a row by
        // up to `slack / rhs` (relative), which on the tightest `j_max·h³`
        // row is ~9% at `tol=1e-6`. A few percent of extra K converts that
        // slack into actual feasibility; if not, the verify-and-bump loop
        // below grows `K` until the chord-FD output passes the strict check.
        let trace = std::env::var("DEKE_DISCRETE_TRACE").is_ok();
        let t_solve = Instant::now();
        let mut final_sol = {
            let strict_k = lf.k + (lf.k / 40).max(2);
            let t = Instant::now();
            let strict_attempt = build_and_solve_discrete::<N>(
                &deriv, constraints, start, end, strict_k, false, Some(&lf.sigma),
            );
            if trace {
                eprintln!("[discrete] strict@K={} took {:?}", strict_k, t.elapsed());
            }
            match strict_attempt {
                Ok(s) if matches!(s.status, SolveStatus::Success) => s,
                _ => lf,
            }
        };

        // Re-bin loop. The build-time bins came from `proportional_bins`,
        // which is just an initial guess. If the IPM's σ values fall into
        // different densified segments than assumed, the FD-row coefficients
        // (which use the build-time bins) won't match the actual chord-linear
        // reconstruction — the downstream FD check then sees spikes at the
        // mis-binned samples. Re-solve with the corrected bins; converges in
        // 1–3 iterations on typical paths.
        let t = Instant::now();
        let iters = rebin_to_convergence::<N>(&deriv, constraints, start, end, &mut final_sol, 16);
        if trace {
            eprintln!("[discrete] rebin took {:?} ({} iters)", t.elapsed(), iters);
        }

        // Verify-and-bump loop. Even after the rebin converges, the FD-output
        // can violate joint v/a/j when `slack` at the bisection optimum was
        // close to `tolerance` — the slack-feasible σ overshoots its bound by
        // a small amount that the strict check catches as a real violation.
        // Bump `K` (≥+2, +5% of K) and re-solve; the larger sample count
        // gives the IPM more room to find a strictly feasible σ. We cap the
        // bump loop at 4 attempts so adversarial paths fail cleanly rather
        // than spinning.
        let dt_out = Duration::from_secs_f64(h);
        let mut verify_residual = super::diagnostic::PerLimitResidual::default();
        let mut verify_result: DekeResult<()> = Ok(());
        for bump in 0..5 {
            let samples_try = sigma_to_samples(&final_sol.sigma, &deriv);
            let (r, vr) = verify_output_fd(
                &samples_try, dt_out, constraints, fk, constraints.solver.tolerance,
            );
            verify_residual = r;
            verify_result = vr;
            if !constraints.check_output_dynamics || verify_result.is_ok() {
                if trace {
                    eprintln!("[discrete] verify passed after {} bumps", bump);
                }
                break;
            }
            let next_k = final_sol.k + (final_sol.k / 20).max(2);
            let t = Instant::now();
            let bumped = build_and_solve_discrete::<N>(
                &deriv, constraints, start, end, next_k, false, Some(&final_sol.sigma),
            );
            if trace {
                eprintln!("[discrete] bump@K={} took {:?}", next_k, t.elapsed());
            }
            match bumped {
                Ok(s) if matches!(s.status, SolveStatus::Success) => {
                    final_sol = s;
                    let t = Instant::now();
                    let iters = rebin_to_convergence::<N>(
                        &deriv, constraints, start, end, &mut final_sol, 16,
                    );
                    if trace {
                        eprintln!(
                            "[discrete] bump rebin took {:?} ({} iters)",
                            t.elapsed(),
                            iters
                        );
                    }
                }
                _ => break,
            }
        }

        diag.final_k = final_sol.k;
        diag.phase_timing.nlp_solve = t_solve.elapsed();
        diag.iterations = final_sol.iterations;
        diag.solve_time = final_sol.solve_time;
        diag.phase_timing.nlp_build = final_sol.build_time;
        diag.constraint_counts = final_sol.constraint_counts;
        diag.solver_tolerance_used = constraints.solver.tolerance;
        diag.status = SolveStatus::Success;

        // Reconstruct output joint samples by chord-linear interpolation σ → q.
        let samples = sigma_to_samples(&final_sol.sigma, &deriv);
        diag.output_samples = samples.len();
        diag.total_time = Duration::from_secs_f64((final_sol.k - 1) as f64 * h);
        let traj_path = match SRobotPath::try_new(samples.clone()) {
            Ok(p) => p,
            Err(e) => {
                diag.message = Some(format!("{}", e));
                return (Err(e), diag);
            }
        };

        // Strict verification result is already computed by the bump loop above.
        let t_verify = Instant::now();
        diag.output_fd_residual = verify_residual;
        diag.phase_timing.verify = t_verify.elapsed();
        if constraints.check_output_dynamics && let Err(e) = verify_result {
            diag.message = Some(format!("{}", e));
            return (Err(e), diag);
        }

        if constraints.post_validation {
            if let Err(e) = validator.validate_motion(traj_path.iter().as_slice(), ctx) {
                diag.message = Some(format!("validator rejected output: {}", e));
                return (Err(e), diag);
            }
        }

        populate_peak_kinematics::<N, _>(
            &mut diag,
            traj_path.iter().as_slice(),
            dt_out,
            constraints,
            fk,
            lock,
        );

        (Ok(SRobotTraj::new(dt_out, traj_path)), diag)
    }
}

/// Re-solve with corrected bins until the IPM's σ values land in the same
/// densified segment they were assigned to at build time. Without this, the
/// FD-row coefficients (built against the build-time bins) don't match the
/// chord-linear reconstruction at samples whose σ shifted across a segment
/// boundary, and the strict FD-verify catches the discrepancy as a violation.
fn rebin_to_convergence<const N: usize>(
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6DiscreteConstraints<N>,
    start: crate::common::boundary::ProjectedBoundary,
    end: crate::common::boundary::ProjectedBoundary,
    sol: &mut DiscreteSolution,
    max_iter: usize,
) -> usize {
    let mut iters = 0usize;
    for _ in 0..max_iter {
        let actual_bins = bins_from_sigma(&deriv.s, &sol.sigma);
        if actual_bins == sol.bins_used {
            break;
        }
        iters += 1;
        match build_and_solve_discrete_with_bins::<N>(
            deriv,
            constraints,
            start,
            end,
            sol.k,
            false,
            Some(&sol.sigma),
            Some(&actual_bins),
        ) {
            Ok(s) if matches!(s.status, SolveStatus::Success) => {
                *sol = s;
            }
            _ => break,
        }
    }
    iters
}

fn record_step(steps: &mut Vec<BisectionStep>, sol: &DiscreteSolution) {
    steps.push(BisectionStep {
        k: sol.k,
        status: sol.status,
        slack_sum: sol.slack,
        iter: sol.iterations,
        solve_time: sol.solve_time,
    });
}

/// Reconstruct chord-linear joint positions q[i] from σ[i].
fn sigma_to_samples<const N: usize>(
    sigma: &[f64],
    deriv: &PathDerivatives<N>,
) -> Vec<SRobotQ<N, f64>> {
    let mut out = Vec::with_capacity(sigma.len());
    let m = deriv.num_waypoints();
    let mut b = 0usize;
    for &s in sigma {
        let s_clamped = s.clamp(0.0, deriv.total_length());
        while b + 1 < m - 1 && deriv.s[b + 1] <= s_clamped {
            b += 1;
        }
        while b > 0 && deriv.s[b] > s_clamped {
            b -= 1;
        }
        let u = ((s_clamped - deriv.s[b]) / deriv.ds[b]).clamp(0.0, 1.0);
        out.push(deriv.waypoints[b].interpolate(&deriv.waypoints[b + 1], u));
    }
    // Pin first/last samples to user waypoints exactly (clamp out FP noise).
    if let Some(first) = out.first_mut() {
        *first = deriv.waypoints[0];
    }
    if let Some(last) = out.last_mut() {
        *last = *deriv.waypoints.last().unwrap();
    }
    out
}

fn populate_peak_kinematics<const N: usize, FK: FKChain<N, f64>>(
    diag: &mut Topp3Tcp6DiscreteDiagnostic,
    samples: &[SRobotQ<N, f64>],
    dt_out: Duration,
    constraints: &Topp3Tcp6DiscreteConstraints<N>,
    fk: &FK,
    lock: usize,
) {
    let dt = dt_out.as_secs_f64();
    if dt <= 0.0 || samples.len() < 4 {
        return;
    }
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let tcp_active = constraints.tcp.is_some_and(|t| {
        t.v_max.is_finite() || t.a_max.is_finite() || t.j_max.is_finite()
    });
    let tcp_positions: Option<Vec<[f64; 3]>> = if tcp_active {
        let mut v = Vec::with_capacity(samples.len());
        for q in samples {
            match fk.fk_end(q) {
                Ok(pose) => {
                    use glam_traits_ext::{TAffine3, TVec3};
                    let t = pose.translation();
                    v.push([t.x(), t.y(), t.z()]);
                }
                Err(_) => return,
            }
        }
        Some(v)
    } else {
        None
    };

    let mut p_jv = 0.0_f64;
    let mut p_ja = 0.0_f64;
    let mut p_jj = 0.0_f64;
    let mut p_tv = 0.0_f64;
    let mut p_ta = 0.0_f64;
    let mut p_tj = 0.0_f64;
    let mut util_sum = 0.0_f64;
    let mut util_count = 0usize;

    for k in 3..samples.len() {
        let mut util = 0.0_f64;
        for j in lock..N {
            let q3 = samples[k - 3].0[j];
            let q2 = samples[k - 2].0[j];
            let q1 = samples[k - 1].0[j];
            let q0 = samples[k].0[j];
            let v = ((q0 - q1) / dt).abs();
            let a = ((q0 - 2.0 * q1 + q2) / dt2).abs();
            let jk = ((q0 - 3.0 * q1 + 3.0 * q2 - q3) / dt3).abs();
            if v > p_jv {
                p_jv = v;
            }
            if a > p_ja {
                p_ja = a;
            }
            if jk > p_jj {
                p_jj = jk;
            }
            let v_max = constraints.joint.v_max.0[j];
            if v_max.is_finite() && v_max > 0.0 {
                util = util.max(v / v_max);
            }
            let a_max = constraints.joint.a_max.0[j];
            if a_max.is_finite() && a_max > 0.0 {
                util = util.max(a / a_max);
            }
            let j_max = constraints.joint.j_max.0[j];
            if j_max.is_finite() && j_max > 0.0 {
                util = util.max(jk / j_max);
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
            if tv > p_tv {
                p_tv = tv;
            }
            if tcp.v_max.is_finite() && tcp.v_max > 0.0 {
                util = util.max(tv / tcp.v_max);
            }
            let ax = (p0[0] - 2.0 * p1[0] + p2[0]) / dt2;
            let ay = (p0[1] - 2.0 * p1[1] + p2[1]) / dt2;
            let az = (p0[2] - 2.0 * p1[2] + p2[2]) / dt2;
            let ta = (ax * ax + ay * ay + az * az).sqrt();
            if ta > p_ta {
                p_ta = ta;
            }
            if tcp.a_max.is_finite() && tcp.a_max > 0.0 {
                util = util.max(ta / tcp.a_max);
            }
            let jx = (p0[0] - 3.0 * p1[0] + 3.0 * p2[0] - p3[0]) / dt3;
            let jy = (p0[1] - 3.0 * p1[1] + 3.0 * p2[1] - p3[1]) / dt3;
            let jz = (p0[2] - 3.0 * p1[2] + 3.0 * p2[2] - p3[2]) / dt3;
            let tj = (jx * jx + jy * jy + jz * jz).sqrt();
            if tj > p_tj {
                p_tj = tj;
            }
            if tcp.j_max.is_finite() && tcp.j_max > 0.0 {
                util = util.max(tj / tcp.j_max);
            }
        }
        util_sum += util;
        util_count += 1;
    }

    diag.peak_joint_velocity = p_jv;
    diag.peak_joint_acceleration = p_ja;
    diag.peak_joint_jerk = p_jj;
    diag.peak_tcp_velocity = p_tv;
    diag.peak_tcp_acceleration = p_ta;
    diag.peak_tcp_jerk = p_tj;
    diag.average_utilization = if util_count > 0 {
        util_sum / util_count as f64
    } else {
        0.0
    };
}

fn populate_path_geometry<const N: usize>(
    stats: &mut super::diagnostic::PathStats,
    densified: &SRobotPath<N, f64>,
) {
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

// ── deke-topp-speed seed ────────────────────────────────────────────────────

/// Runs `deke-topp-speed`'s `ToppSolver` on the user's (densified) waypoint
/// list to produce an initial trajectory, then projects each output sample
/// back onto the densified path's arc-length parameter `σ`. The returned
/// `(K, σ[0..K])` is used as a warm start by the bisection driver.
///
/// Returns `None` on any failure — the discrete retimer falls back to the
/// uniform-σ initial guess in that case. Topp-speed is fast (sub-ms on
/// typical paths), so the cost when it succeeds is negligible.
fn topp_speed_seed<const N: usize, FK: ContinuousFKChain<N, f64>>(
    deriv: &PathDerivatives<N>,
    path: &SRobotPath<N, f64>,
    constraints: &Topp3Tcp6DiscreteConstraints<N>,
    fk: &FK,
    h: f64,
) -> Option<(usize, Vec<f64>)> {
    // Feed topp-speed the *user's* original waypoints, not the discrete
    // crate's densified version. Topp-speed's solve cost scales with
    // waypoint count (the waypoint solver runs a coordinated-stop step per
    // intermediate waypoint), and densification can inflate 10 waypoints
    // into 100+ samples — easily a 50× slowdown that defeats the seeding.
    let mut spec = MotionSpec::<N, f64>::new();
    spec.current_pose = *path.first();
    spec.goal_pose = *path.last();
    spec.waypoint_poses.clear();
    for i in 1..path.len() - 1 {
        if let Some(p) = path.get(i) {
            spec.waypoint_poses.push(*p);
        }
    }
    spec.max_vel = constraints.joint.v_max;
    spec.max_accel = constraints.joint.a_max;
    spec.max_jerk = constraints.joint.j_max;
    if let Some(tcp) = constraints.tcp
        && tcp.v_max.is_finite()
        && tcp.v_max > 0.0
    {
        spec.max_tcp_speed = Some(tcp.v_max);
    }
    spec.current_vel = constraints.boundary.v_start;
    spec.current_accel = constraints.boundary.a_start;
    spec.goal_vel = constraints.boundary.v_end;
    spec.goal_accel = constraints.boundary.a_end;

    let dt = Duration::from_secs_f64(h);
    let solver = ToppSolver::<N, f64, FK>::new(dt, fk);
    // Use a permissive joint validator — topp-speed only consults it for
    // post-validation, and we want any seed regardless of whether it lies
    // inside the user's q bounds (the discrete crate enforces those
    // separately).
    let validator = JointValidator::<N, f64>::new(
        SRobotQ::from_array([f64::NEG_INFINITY; N]),
        SRobotQ::from_array([f64::INFINITY; N]),
    );
    let (result, _diag) = solver.retime(&spec, path, &validator, &());
    let traj = result.ok()?;

    let samples: Vec<SRobotQ<N, f64>> = traj.path().iter().copied().collect();
    let k = samples.len();
    if k < 4 {
        return None;
    }

    // Project each sample back onto the densified path's chord-linear arc
    // parameter. Topp-speed's samples lie on the same chord-linear path the
    // discrete crate models, so the projection error is bounded by FP noise
    // for any sample that landed inside a chord (a few samples right at the
    // start/end may land at u=0 or u=1).
    let sigma = project_samples_to_sigma::<N>(&samples, deriv)?;
    Some((k, sigma))
}

fn project_samples_to_sigma<const N: usize>(
    samples: &[SRobotQ<N, f64>],
    deriv: &PathDerivatives<N>,
) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(samples.len());
    let m = deriv.num_waypoints();
    let total = deriv.total_length();
    // Walk segments left-to-right with a moving hint; trajectory samples
    // are monotone in arc-length so we never revisit a finished segment.
    let mut hint = 0usize;
    for q in samples {
        // Look for the segment whose chord contains `q`. Scan from the
        // hint forward; if no segment matches (sample is past the end or
        // off-path due to FP noise), clamp.
        let mut best_b = hint;
        let mut best_u = 0.0_f64;
        let mut best_err_sq = f64::INFINITY;
        // Don't scan all m-1 segments every time; bound to a small window
        // around the hint. Topp-speed samples are monotone-ish along the
        // path, so the relevant segment is near `hint`.
        let lo = hint.saturating_sub(2);
        let hi = (hint + 4).min(m - 1);
        for b in lo..hi {
            let a = deriv.waypoints[b].0;
            let bv = deriv.waypoints[b + 1].0;
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for j in 0..N {
                let dq = bv[j] - a[j];
                num += (q.0[j] - a[j]) * dq;
                den += dq * dq;
            }
            let u_raw = if den > 0.0 { num / den } else { 0.0 };
            let u = u_raw.clamp(0.0, 1.0);
            let mut err_sq = 0.0_f64;
            for j in 0..N {
                let interp = a[j] + u * (bv[j] - a[j]);
                let d = q.0[j] - interp;
                err_sq += d * d;
            }
            if err_sq < best_err_sq {
                best_err_sq = err_sq;
                best_b = b;
                best_u = u;
            }
        }
        let sigma = (deriv.s[best_b] + best_u * deriv.ds[best_b]).clamp(0.0, total);
        out.push(sigma);
        hint = best_b;
    }
    // Pin endpoints to 0 / S exactly.
    if let Some(f) = out.first_mut() {
        *f = 0.0;
    }
    if let Some(l) = out.last_mut() {
        *l = total;
    }
    Some(out)
}

// densification

fn densify_path<const N: usize>(
    path: &SRobotPath<N, f64>,
    opts: &DensificationOptions,
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

fn densify_with_kink_boost<const N: usize>(
    path: &SRobotPath<N, f64>,
    base_step: f64,
) -> SRobotPath<N, f64> {
    let m = path.len();
    if m < 2 || base_step <= 0.0 {
        return path.clone();
    }

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
            if cos < -0.7 {
                let factor = 1.0 + (-cos - 0.5) * 6.0;
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

    SRobotPath::try_new(out).unwrap_or_else(|_| path.clone())
}

