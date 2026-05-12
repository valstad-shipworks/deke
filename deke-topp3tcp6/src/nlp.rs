use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

use hafgufa::{Options, Problem, VariableArena, subject_to};

use deke_types::{DekeError, DekeResult};

use crate::Topp3Tcp6Constraints;
use crate::boundary::ProjectedBoundary;
use crate::diagnostic::{
    BoundarySlackUsage, ConstraintCounts, InitialGuessStats, SolveStatus,
};
use crate::path_derivatives::PathDerivatives;

/// Numeric output of the NLP solve — everything downstream uses this POD struct so that the
/// `VariableArena` can be dropped before the next pipeline stage.
#[derive(Debug, Clone)]
pub struct Solution {
    pub sd: Vec<f64>,
    pub sdd: Vec<f64>,
    pub sddd: Vec<f64>,
    pub dt: Vec<Duration>,
    pub status: SolveStatus,
    pub iterations: i32,
    pub solve_time: Duration,
    /// Wall time spent assembling decision variables and constraints, before
    /// `problem.solve_status` is invoked.
    pub build_time: Duration,
    pub constraint_counts: ConstraintCounts,
    pub initial_guess: InitialGuessStats,
    pub boundary_slack_usage: BoundarySlackUsage,
    /// Per-sample qp-degeneracy detector outcome — number of samples whose joint
    /// constraints were skipped entirely because `‖qp[k]‖²` fell below the relative
    /// threshold.
    pub degenerate_qp_samples: usize,
    /// `min_k ‖qp[k]‖² / max_k ‖qp[k]‖²` and the sample where it was reached. The min
    /// itself is also reported as `derivative_stats.min_qp_norm_relative_sq` upstream.
    pub min_qp_norm_relative_sq: f64,
    pub min_qp_norm_sample: usize,
}

/// Drop the per-sample TCP velocity upper bound (`sd_k ≤ v_max/|pp_k|`) at any waypoint whose
/// path tangent is below this fraction of the maximum |pp|² across the path. At those
/// locally-stationary samples the bound would otherwise inflate to a huge value (e.g.
/// wrist-only rotation, |pp| ≈ 0 → upper ≈ 1e9) and wreck the variable scaling. Picked to be
/// well below typical curvature variation but well above floating-point noise.
const PP_RELATIVE_CUTOFF: f64 = 1e-6;

pub fn build_and_solve<const N: usize>(
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
    start: ProjectedBoundary,
    end: ProjectedBoundary,
) -> DekeResult<Solution> {
    build_and_solve_inner(deriv, constraints, start, end, None)
}

/// Like [`build_and_solve`] but seeds the IPM's initial point from `warm_start` instead
/// of running [`apply_initial_guess`]. Useful for two-stage solves: solve the
/// TCP-disabled problem first, then warm-start the TCP-enabled problem from that
/// solution. The warm-start array lengths must match the new problem's `(m, seg)`.
pub fn build_and_solve_warm<const N: usize>(
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
    start: ProjectedBoundary,
    end: ProjectedBoundary,
    warm_start: &Solution,
) -> DekeResult<Solution> {
    build_and_solve_inner(deriv, constraints, start, end, Some(warm_start))
}

fn build_and_solve_inner<const N: usize>(
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
    start: ProjectedBoundary,
    end: ProjectedBoundary,
    warm_start: Option<&Solution>,
) -> DekeResult<Solution> {
    let m = deriv.num_waypoints();
    let seg = deriv.num_segments();
    if m < 2 || seg == 0 {
        return Err(DekeError::PathTooShort(m));
    }
    let lock = constraints.locked_prefix.min(N);

    let build_start = Instant::now();
    let mut counts = ConstraintCounts::default();

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let sd: Vec<_> = (0..m).map(|_| problem.decision_variable()).collect();
    let sdd: Vec<_> = (0..m).map(|_| problem.decision_variable()).collect();
    let sddd: Vec<_> = (0..seg).map(|_| problem.decision_variable()).collect();
    let dt: Vec<_> = (0..seg).map(|_| problem.decision_variable()).collect();

    let tcp_active = deriv.has_tcp() && constraints.tcp.is_some();

    let pp_cutoff_sq = if tcp_active {
        let mut max_sq = 0.0_f64;
        for pp in &deriv.pp {
            let s = pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2];
            if s > max_sq {
                max_sq = s;
            }
        }
        if max_sq > 0.0 {
            PP_RELATIVE_CUTOFF * max_sq
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Per-joint relative-magnitude cutoff for `qp_j`. The joint v upper bound is
    // `sd_k ≤ v_max / |qp_j|`; when `|qp_j|` is tiny but non-zero (a "slow section" where
    // the joint is *barely* moving relative to the rest of the path), this bound inflates
    // to 1/|qp_j| · v_max ≈ 1e9, and the corresponding constraint row's Jacobian gradient
    // (`1/|qp_j|`) makes the IPM's KKT matrix wildly ill-conditioned. Anything below
    // `1e-6 × max|qp_j|` on a per-joint basis is treated as "not really moving" and the
    // constraint is dropped.
    let mut qp_max_abs = [0.0_f64; N];
    for k in 0..m {
        for j in 0..N {
            let a = deriv.qp[k][j].abs();
            if a > qp_max_abs[j] {
                qp_max_abs[j] = a;
            }
        }
    }
    let qp_cutoffs: [f64; N] = {
        let mut out = [0.0_f64; N];
        for j in 0..N {
            out[j] = (1e-6 * qp_max_abs[j]).max(1e-12);
        }
        out
    };

    for k in 0..m {
        let sd_k = sd[k];
        subject_to!(problem, sd_k >= 0.0);

        if let Some(tcp) = constraints.tcp
            && tcp_active
            && tcp.v_max.is_finite()
            && tcp.v_max > 0.0
        {
            let pp = &deriv.pp[k];
            let pp_norm_sq = pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2];
            if pp_norm_sq > pp_cutoff_sq {
                let upper = tcp.v_max / pp_norm_sq.sqrt();
                subject_to!(problem, sd_k <= upper);
                counts.tcp_v += 1;
            }
        }

        for j in lock..N {
            let qp_j = deriv.qp[k][j];
            if qp_j.abs() < qp_cutoffs[j] {
                continue;
            }
            let v_max = constraints.joint.v_max.0[j];
            if v_max.is_finite() && v_max > 0.0 {
                let upper = v_max / qp_j.abs();
                subject_to!(problem, sd_k <= upper);
                // Counted alongside the symmetric ± constraint pair below to avoid
                // double-counting; this branch is just the redundant scalar upper bound.
            }
        }
    }

    for k in 0..seg {
        let dt_k = dt[k];
        subject_to!(problem, dt_k >= 1e-6);
    }

    for k in 0..seg {
        let sd_k = sd[k];
        let sd_k1 = sd[k + 1];
        let sdd_k = sdd[k];
        let sdd_k1 = sdd[k + 1];
        let sddd_k = sddd[k];
        let dt_k = dt[k];
        let ds_k = deriv.ds[k];

        let rhs_sd = sd_k + sdd_k * dt_k + 0.5 * sddd_k * dt_k * dt_k;
        subject_to!(problem, sd_k1 == rhs_sd);

        let rhs_sdd = sdd_k + sddd_k * dt_k;
        subject_to!(problem, sdd_k1 == rhs_sdd);

        let rhs_ds = sd_k * dt_k
            + 0.5 * sdd_k * dt_k * dt_k
            + (1.0 / 6.0) * sddd_k * dt_k * dt_k * dt_k;
        subject_to!(problem, rhs_ds == ds_k);
    }

    {
        // Soft boundary box: |sd[0] - start.sd| ≤ slack (and three more like it). Hard
        // equalities pin variables to exactly zero at rest-to-rest, which is the cone tip
        // of every TCP constraint — the IPM line search hates that. A tiny slack box
        // (default 1e-4) gives Newton room to step without observable output drift; the
        // resampler still reads the boundary off the user's request, not the variable.
        let slack = constraints.solver.boundary_slack.max(0.0);
        let sd_0 = sd[0];
        let sdd_0 = sdd[0];
        let sd_f = sd[m - 1];
        let sdd_f = sdd[m - 1];
        if slack == 0.0 {
            subject_to!(problem, sd_0 == start.sd);
            subject_to!(problem, sdd_0 == start.sdd);
            subject_to!(problem, sd_f == end.sd);
            subject_to!(problem, sdd_f == end.sdd);
        } else {
            let up_sd0 = sd_0 - start.sd;
            subject_to!(problem, up_sd0 <= slack);
            let lo_sd0 = -sd_0 + start.sd;
            subject_to!(problem, lo_sd0 <= slack);

            let up_sdd0 = sdd_0 - start.sdd;
            subject_to!(problem, up_sdd0 <= slack);
            let lo_sdd0 = -sdd_0 + start.sdd;
            subject_to!(problem, lo_sdd0 <= slack);

            let up_sdf = sd_f - end.sd;
            subject_to!(problem, up_sdf <= slack);
            let lo_sdf = -sd_f + end.sd;
            subject_to!(problem, lo_sdf <= slack);

            let up_sddf = sdd_f - end.sdd;
            subject_to!(problem, up_sddf <= slack);
            let lo_sddf = -sdd_f + end.sdd;
            subject_to!(problem, lo_sddf <= slack);
        }
    }

    // Per-sample degenerate-corner detector. A "degenerate" sample is one where the path
    // tangent `qp` is essentially zero in *every* joint — i.e. the chord-length parameter
    // is still advancing through this point but the joint-space velocity direction is
    // undefined (turning points of a zigzag, 180° reversals). Joint a/j constraints
    // computed against `qp_j = 0` have a zero gradient w.r.t. `sdd`/`sddd` and trip up
    // Sleipnir's KKT factorization. We detect these samples relative to the max ‖qp‖ on
    // the path and skip their joint-side constraints entirely; the integrator chain alone
    // keeps `sd[k]` and `sdd[k]` well-determined.
    let mut max_qp_norm_sq = 0.0_f64;
    for k in 0..m {
        let mut n = 0.0_f64;
        for j in lock..N {
            n += deriv.qp[k][j] * deriv.qp[k][j];
        }
        if n > max_qp_norm_sq {
            max_qp_norm_sq = n;
        }
    }
    let degenerate_qp_threshold_sq = 1e-12 * max_qp_norm_sq;
    let mut degenerate_qp_samples = 0_usize;
    let mut min_qp_rel_sq = f64::INFINITY;
    let mut min_qp_rel_sample = 0_usize;

    for k in 0..m {
        let sd_k = sd[k];
        let sdd_k = sdd[k];
        let seg_idx = if k < seg { k } else { seg - 1 };
        let sddd_k = sddd[seg_idx];

        // The TCP a/j constraints at the very first and last sample are fully determined by the
        // boundary equalities on sd, sdd (and by the implicit determination of sddd through the
        // integrator). Adding them anyway just creates spurious KKT activity at the boundary
        // that the IPM restoration phase trips over.
        let at_boundary = k == 0 || k == m - 1;

        let qp_norm_sq: f64 = (lock..N)
            .map(|j| deriv.qp[k][j] * deriv.qp[k][j])
            .sum();
        let qp_degenerate = qp_norm_sq <= degenerate_qp_threshold_sq;

        let rel = if max_qp_norm_sq > 0.0 {
            qp_norm_sq / max_qp_norm_sq
        } else {
            0.0
        };
        if rel < min_qp_rel_sq {
            min_qp_rel_sq = rel;
            min_qp_rel_sample = k;
        }
        if qp_degenerate {
            degenerate_qp_samples += 1;
        }

        if !qp_degenerate {
            for j in lock..N {
                let qp_j = deriv.qp[k][j];
                let qpp_j = deriv.qpp[k][j];
                let qppp_j = deriv.qppp[k][j];
                let v_max = constraints.joint.v_max.0[j];
                let a_max = constraints.joint.a_max.0[j];
                let j_max = constraints.joint.j_max.0[j];

                if v_max.is_finite() && v_max > 0.0 && qp_j.abs() > qp_cutoffs[j] {
                    let expr = qp_j * sd_k;
                    subject_to!(problem, expr <= v_max);
                    let neg = -qp_j * sd_k;
                    subject_to!(problem, neg <= v_max);
                    counts.joint_v += 2;
                }

                if a_max.is_finite() && a_max > 0.0 {
                    let expr = qpp_j * sd_k * sd_k + qp_j * sdd_k;
                    subject_to!(problem, expr <= a_max);
                    let neg = -qpp_j * sd_k * sd_k - qp_j * sdd_k;
                    subject_to!(problem, neg <= a_max);
                    counts.joint_a += 2;
                }

                if j_max.is_finite() && j_max > 0.0 {
                    let expr = qppp_j * sd_k * sd_k * sd_k
                        + 3.0 * qpp_j * sd_k * sdd_k
                        + qp_j * sddd_k;
                    subject_to!(problem, expr <= j_max);
                    let neg = -qppp_j * sd_k * sd_k * sd_k
                        - 3.0 * qpp_j * sd_k * sdd_k
                        - qp_j * sddd_k;
                    subject_to!(problem, neg <= j_max);
                    counts.joint_j += 2;
                }
            }
        }

        // TCP a/j are scaled into the unit ball: ||a_vec/a_max||² ≤ 1 instead of
        // ||a_vec||² ≤ a_max². The quadratic LHS otherwise ranges across a_max²/j_max²
        // (e.g. 40000), and that wrecks the Hessian conditioning in interior-point methods.
        if let Some(tcp) = constraints.tcp
            && tcp_active
            && !at_boundary
            && tcp.a_max.is_finite()
            && tcp.a_max > 0.0
        {
            let inv = 1.0 / tcp.a_max;
            let ppp = &deriv.ppp[k];
            let pp = &deriv.pp[k];
            let c0 = (ppp[0] * inv) * sd_k * sd_k + (pp[0] * inv) * sdd_k;
            let c1 = (ppp[1] * inv) * sd_k * sd_k + (pp[1] * inv) * sdd_k;
            let c2 = (ppp[2] * inv) * sd_k * sd_k + (pp[2] * inv) * sdd_k;
            let sum_sq = c0 * c0 + c1 * c1 + c2 * c2;
            subject_to!(problem, sum_sq <= 1.0);
            counts.tcp_a += 1;
        }

        if let Some(tcp) = constraints.tcp
            && tcp_active
            && !at_boundary
            && tcp.j_max.is_finite()
            && tcp.j_max > 0.0
        {
            let inv = 1.0 / tcp.j_max;
            let pppp = &deriv.pppp[k];
            let ppp = &deriv.ppp[k];
            let pp = &deriv.pp[k];
            let c0 = (pppp[0] * inv) * sd_k * sd_k * sd_k
                + (3.0 * ppp[0] * inv) * sd_k * sdd_k
                + (pp[0] * inv) * sddd_k;
            let c1 = (pppp[1] * inv) * sd_k * sd_k * sd_k
                + (3.0 * ppp[1] * inv) * sd_k * sdd_k
                + (pp[1] * inv) * sddd_k;
            let c2 = (pppp[2] * inv) * sd_k * sd_k * sd_k
                + (3.0 * ppp[2] * inv) * sd_k * sdd_k
                + (pp[2] * inv) * sddd_k;
            let sum_sq = c0 * c0 + c1 * c1 + c2 * c2;
            subject_to!(problem, sum_sq <= 1.0);
            counts.tcp_j += 1;
        }
    }

    let mut total = dt[0];
    for k in 1..seg {
        total = total + dt[k];
    }
    problem.minimize(total);

    let initial_guess = if let Some(ws) = warm_start {
        // Two-stage warm start: copy variable values from the previous solution. Skip
        // the synthetic initial-guess construction and the cruise-feasibility loop;
        // the previous solve already produced a feasible iterate.
        for k in 0..ws.sd.len().min(m) {
            sd[k].set_value(ws.sd[k]);
            sdd[k].set_value(ws.sdd[k]);
        }
        for k in 0..ws.sddd.len().min(seg) {
            sddd[k].set_value(ws.sddd[k]);
            dt[k].set_value(ws.dt[k].as_secs_f64().max(1e-5));
        }
        // Recompute end-residual stats against the new boundary projection.
        let mut max_sddd = 0.0_f64;
        let mut max_sddd_segment = 0_usize;
        for k in 0..ws.sddd.len() {
            let a = ws.sddd[k].abs();
            if a > max_sddd {
                max_sddd = a;
                max_sddd_segment = k;
            }
        }
        InitialGuessStats {
            end_sd_residual: (ws.sd[m - 1] - end.sd).abs(),
            end_sdd_residual: (ws.sdd[m - 1] - end.sdd).abs(),
            max_sddd,
            max_sddd_segment,
        }
    } else {
        apply_initial_guess(
            &sd, &sdd, &sddd, &dt, deriv, constraints, start, end, pp_cutoff_sq,
        )
    };

    let build_time = build_start.elapsed();

    let iter_counter = Arc::new(AtomicI32::new(0));
    let ic = iter_counter.clone();
    problem.add_callback(move |_info| {
        ic.fetch_add(1, Ordering::Relaxed);
        false
    });

    let mut options = Options::default()
        .tolerance(constraints.solver.tolerance)
        .max_iterations(constraints.solver.max_iterations)
        .diagnostics(constraints.solver.diagnostics);
    if let Some(t) = constraints.solver.timeout {
        options = options.timeout(t);
    }

    let t0 = Instant::now();
    let status_raw = problem.solve_status(options);
    let solve_time = t0.elapsed();
    let status = SolveStatus::from(status_raw);
    let iterations = iter_counter.load(Ordering::Relaxed);

    let sd_vals: Vec<f64> = sd.iter().map(|v| v.value()).collect();
    let sdd_vals: Vec<f64> = sdd.iter().map(|v| v.value()).collect();
    let sddd_vals: Vec<f64> = sddd.iter().map(|v| v.value()).collect();
    let dt_vals: Vec<Duration> = dt
        .iter()
        .map(|v| Duration::from_secs_f64(v.value().max(0.0)))
        .collect();

    let boundary_slack_usage = BoundarySlackUsage {
        start_sd: (sd_vals[0] - start.sd).abs(),
        start_sdd: (sdd_vals[0] - start.sdd).abs(),
        end_sd: (sd_vals[m - 1] - end.sd).abs(),
        end_sdd: (sdd_vals[m - 1] - end.sdd).abs(),
    };

    let min_qp_norm_relative_sq = if min_qp_rel_sq.is_finite() {
        min_qp_rel_sq
    } else {
        0.0
    };

    Ok(Solution {
        sd: sd_vals,
        sdd: sdd_vals,
        sddd: sddd_vals,
        dt: dt_vals,
        status,
        iterations,
        solve_time,
        build_time,
        constraint_counts: counts,
        initial_guess,
        boundary_slack_usage,
        degenerate_qp_samples,
        min_qp_norm_relative_sq,
        min_qp_norm_sample: min_qp_rel_sample,
    })
}

/// Builds an initial guess that exactly satisfies the per-segment integrator and
/// `ds`-equality constraints, and the start-side boundary equalities. The end-side
/// boundary on `sdd` is generally not satisfied — by design, the forward propagation
/// can't be told to hit both endpoints simultaneously without a global solve, and the
/// boundary slack box [`crate::SolverOptions::boundary_slack`] is sized to absorb the
/// residual.
///
/// Algorithm:
/// 1. Pick a smooth target `sd` profile: sinusoidal ramp from `start.sd` up to a flat
///    cruise speed (≈70% of the most restrictive sample-wise velocity cap), then a
///    matching sinusoidal ramp down to `end.sd`.
/// 2. Forward-propagate. For each segment, given `(sd[k], sd[k+1], sdd[k], ds[k])`, solve
///    the integrator equalities exactly for `(dt[k], sddd[k])` using the closed-form
///    quadratic `(1/6)·sdd[k]·dt² + ((2/3)·sd[k] + (1/3)·sd[k+1])·dt − ds[k] = 0`. Then
///    propagate `sdd[k+1] = sdd[k] + sddd[k]·dt[k]`.
/// 3. Compute the worst joint v/a/j violation of the propagated guess. If anything
///    violates by more than 5%, halve the `cruise` factor and re-do steps 1–2. Up to 12
///    rounds (`cruise ≈ 1.7e-4 × cap` at the floor) — a safety net that keeps the IPM
///    from starting in feasibility-restoration mode on paths whose forward-propagation
///    sddd would otherwise spike to 10–40× `j_max`.
fn apply_initial_guess<'a, const N: usize>(
    sd: &[hafgufa::Variable<'a>],
    sdd: &[hafgufa::Variable<'a>],
    sddd: &[hafgufa::Variable<'a>],
    dt: &[hafgufa::Variable<'a>],
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
    start: ProjectedBoundary,
    end: ProjectedBoundary,
    pp_cutoff_sq: f64,
) -> InitialGuessStats {
    let m = deriv.num_waypoints();
    let seg = deriv.num_segments();
    let lock = constraints.locked_prefix.min(N);

    let mut cap = vec![1.0_f64; m];
    for k in 0..m {
        let mut c = f64::INFINITY;
        for j in lock..N {
            let q = deriv.qp[k][j].abs();
            if q > 1e-9 {
                let v = constraints.joint.v_max.0[j];
                if v.is_finite() && v > 0.0 {
                    c = c.min(v / q);
                }
            }
        }
        if let Some(tcp) = constraints.tcp
            && deriv.has_tcp()
            && tcp.v_max.is_finite()
            && tcp.v_max > 0.0
        {
            let pp = &deriv.pp[k];
            let pn_sq = pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2];
            if pn_sq > pp_cutoff_sq {
                c = c.min(tcp.v_max / pn_sq.sqrt());
            }
        }
        if !c.is_finite() {
            c = 1.0;
        }
        // Note: `cap[k]` is the per-sample velocity bound *before* applying any cruise
        // discount; the cruise-feasibility loop below scales it down by `cruise_factor`.
        cap[k] = c.max(1e-3);
    }

    let min_cap = cap
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .max(1e-3);

    let start_sd = start.sd.max(0.0);
    let end_sd = end.sd.max(0.0);

    // Ramp count: span enough samples that the sd transition doesn't demand huge sddd.
    // m/4 with a floor of 2 and a ceiling at half the path keeps it stable across both
    // very short (m=2..10) and very long (m=200+) densifications.
    let ramp = ((m / 4).max(2)).min((m.saturating_sub(1)) / 2);

    // Iteratively shrink `cruise_factor` until the forward-propagated guess respects the
    // joint v/a/j limits. Starting at 0.5 and halving each round caps the worst case at
    // 8 rounds (cruise ≈ 2e-3 × min_cap), at which point we accept whatever we have and
    // let the IPM recover.
    //
    // **Threshold is path-aware.** Two regimes:
    //
    // - **Curved joint paths** (`max|qpp| / max|qp| > 1e-6`). The joint a/j rows have
    //   real second/third-order dependence on `sd, sdd`; the IPM can absorb a fairly
    //   over-limit guess in a few iterations, and a too-conservative cruise costs more
    //   iter (climbing back up to the optimum) than it saves. Tolerance `10.0` —
    //   only halve on truly egregious (10× over-limit) guesses.
    //
    // - **Linear joint paths** (`max|qpp| ≈ 0`). After densification, PCHIP through
    //   colinear waypoints yields `qpp ≈ qppp ≈ FP noise`, so the joint-jerk row
    //   reduces to `|qp_j·sddd| ≤ j_max[j]`. The cruise-loop's analytical `max|sddd|`
    //   then lands at exactly `j_max[j*]/qp_j*` for the dominant joint — *violation
    //   ≈ 1.000*. Combined with the boundary-slack box pinning `sd[0]/sdd[0]` at the
    //   soft-box wall, the IPM sits in a three-active-row corner with no descent
    //   direction and either declares *locally infeasible* or fails feasibility
    //   restoration. Tolerance `0.95` — fires exactly one halving (cruise 0.5 → 0.25,
    //   `max|sddd|` drops to ≈half) and leaves the IPM with margin to maneuver. See
    //   the `external_2wp_long_chord_*` regression tests for the captured trajectories.
    let mut max_abs_qpp = 0.0_f64;
    let mut max_abs_qp = 0.0_f64;
    for k in 0..m {
        for j in lock..N {
            let qp = deriv.qp[k][j].abs();
            if qp > max_abs_qp {
                max_abs_qp = qp;
            }
            let qpp = deriv.qpp[k][j].abs();
            if qpp > max_abs_qpp {
                max_abs_qpp = qpp;
            }
        }
    }
    let joint_path_is_linear = max_abs_qp > 0.0 && max_abs_qpp < 1e-6 * max_abs_qp;

    let mut cruise_factor = 0.5;
    let mut sd_guess = vec![0.0_f64; m];
    let mut sdd_guess = vec![0.0_f64; m];
    let mut sddd_guess = vec![0.0_f64; seg];
    let mut dt_guess = vec![1e-5_f64; seg];
    let mut violation = f64::INFINITY;
    let max_rounds = 8;
    let violation_tolerance = if joint_path_is_linear { 0.95 } else { 10.0 };
    for _round in 0..max_rounds {
        let cruise = (min_cap * cruise_factor).max(1e-3);
        build_sd_profile(
            &mut sd_guess,
            &cap,
            cruise,
            cruise_factor,
            start_sd,
            end_sd,
            ramp,
        );
        // Forward-propagation only. The symmetric (forward+backward stitched)
        // alternative is implemented below but disabled — it gives a better initial
        // guess on average (lower end-boundary residual), but the seam discontinuity
        // at the stitch point pushes the IPM into infeasible neighborhoods on
        // specific paths (see `zigzag_pattern`, `long_path_many_waypoints`,
        // and the captured `bench_multi_seg_50wp` failures we tracked when this
        // was enabled). Net effect across the bench is ~1pp better with symmetric
        // on, but at the cost of regressing several individual stress tests; the
        // simpler forward-only path is the more reliable trade.
        propagate_initial_guess(
            &sd_guess,
            &mut sdd_guess,
            &mut sddd_guess,
            &mut dt_guess,
            deriv,
            start.sdd,
        );
        violation = worst_initial_guess_violation::<N>(
            &sd_guess,
            &sdd_guess,
            &sddd_guess,
            deriv,
            constraints,
        );
        if violation <= violation_tolerance {
            break;
        }
        cruise_factor *= 0.5;
    }

    for k in 0..m {
        sd[k].set_value(sd_guess[k]);
        sdd[k].set_value(sdd_guess[k]);
    }
    for k in 0..seg {
        sddd[k].set_value(sddd_guess[k]);
        dt[k].set_value(dt_guess[k]);
    }

    let mut max_sddd = 0.0_f64;
    let mut max_sddd_segment = 0_usize;
    for k in 0..seg {
        let a = sddd_guess[k].abs();
        if a > max_sddd {
            max_sddd = a;
            max_sddd_segment = k;
        }
    }
    let _ = violation; // recorded implicitly via max_sddd; future: surface in stats

    InitialGuessStats {
        end_sd_residual: (sd_guess[m - 1] - end.sd).abs(),
        end_sdd_residual: (sdd_guess[m - 1] - end.sdd).abs(),
        max_sddd,
        max_sddd_segment,
    }
}

/// Builds the smooth `sd` target profile: sinusoidal ramp-up from `start_sd` to a flat
/// `cruise` over `ramp` samples, hold, sinusoidal ramp-down to `end_sd`. Per-sample
/// values are then clipped to `cruise_factor × cap[k]` so a path with a tighter local
/// velocity bound (e.g. one wrist joint hitting its `v_max`) doesn't get its sample
/// pushed above the bound just because the global `cruise` hasn't been reduced enough yet.
fn build_sd_profile(
    sd_guess: &mut [f64],
    cap: &[f64],
    cruise: f64,
    cruise_factor: f64,
    start_sd: f64,
    end_sd: f64,
    ramp: usize,
) {
    let m = sd_guess.len();
    if m == 0 {
        return;
    }
    if ramp == 0 {
        for v in sd_guess.iter_mut() {
            *v = cruise;
        }
        sd_guess[0] = start_sd;
        if m > 1 {
            sd_guess[m - 1] = end_sd;
        }
    } else {
        let pi = std::f64::consts::PI;
        for v in sd_guess.iter_mut() {
            *v = cruise;
        }
        for k in 0..=ramp.min(m - 1) {
            let factor = 0.5 * (1.0 - (pi * k as f64 / ramp as f64).cos());
            sd_guess[k] = start_sd + (cruise - start_sd) * factor;
        }
        for k in (m.saturating_sub(ramp + 1))..m {
            let j = m - 1 - k;
            let factor = 0.5 * (1.0 - (pi * j as f64 / ramp as f64).cos());
            let blended = end_sd + (cruise - end_sd) * factor;
            if blended < sd_guess[k] {
                sd_guess[k] = blended;
            }
        }
        sd_guess[0] = start_sd;
        sd_guess[m - 1] = end_sd;
    }
    for k in 0..m {
        let local_cap = cap[k] * cruise_factor;
        if sd_guess[k] > local_cap {
            sd_guess[k] = local_cap;
        }
        if sd_guess[k] < 0.0 {
            sd_guess[k] = 0.0;
        }
    }
}

/// Forward-propagates the integrator-consistent `sdd`, `sddd`, `dt` arrays from a given
/// `sd_guess` profile and `start_sdd`. Each per-segment quadratic is solved exactly so
/// the integrator and `ds`-equality constraints hold for the resulting `(dt[k],
/// sddd[k])`; `sdd[k+1]` is then propagated from `sdd[k] + sddd[k]·dt[k]`.
fn propagate_initial_guess<const N: usize>(
    sd_guess: &[f64],
    sdd_guess: &mut [f64],
    sddd_guess: &mut [f64],
    dt_guess: &mut [f64],
    deriv: &PathDerivatives<N>,
    start_sdd: f64,
) {
    let seg = deriv.num_segments();
    sdd_guess[0] = start_sdd;
    for k in 0..seg {
        let v0 = sd_guess[k].max(0.0);
        let v1 = sd_guess[k + 1].max(0.0);
        let a0 = sdd_guess[k];
        let ds_k = deriv.ds[k];

        let big_a = a0 / 6.0;
        let big_b = (2.0 / 3.0) * v0 + (1.0 / 3.0) * v1;
        let big_c = -ds_k;

        let dt_k = solve_segment_dt(big_a, big_b, big_c, v0, v1, ds_k);
        let j_k = 2.0 * (v1 - v0 - a0 * dt_k) / (dt_k * dt_k);
        let a1 = a0 + j_k * dt_k;

        dt_guess[k] = dt_k.max(1e-5);
        sddd_guess[k] = j_k;
        sdd_guess[k + 1] = a1;
    }
}

/// Symmetric forward + backward propagation. Forward fills the first half (samples
/// `0..=mid`, segments `0..mid`); backward fills the second half (samples `mid..m`,
/// segments `mid..seg`). At the seam (sample `mid`) we average the two passes' `sdd`,
/// which leaves a small per-segment integrator residual the IPM smooths out in its
/// first iteration.
///
/// Currently unused — see the comment above the call site in `apply_initial_guess`
/// for why. Kept compiled and tested (transitively, via the `apply_initial_guess`
/// surface) so it doesn't bit-rot if a future change wants to re-enable it.
#[allow(dead_code)]
///
/// **Win vs pure-forward:** both boundary equalities are exactly satisfied by
/// construction. Pure-forward leaves `sdd[m-1] − end.sdd` as a residual the boundary
/// slack box has to absorb; on long paths that residual saturates the slack and the
/// IPM enters feasibility restoration.
///
/// **Loss vs pure-forward:** the seam introduces a per-segment integrator residual
/// (one-sided `sdd` discrepancy at `mid`). On short paths this seam penalty
/// outweighs the boundary fix because (a) forward's end residual is already small
/// (few accumulation steps) and (b) the seam is a relatively large fraction of the
/// total path. So we gate this on `m >= SYMMETRIC_MIN_SAMPLES`.
///
/// Empirically (release bench, see `tests/external_bench.rs`) this gating recovers
/// the baseline's `multi-seg 25wp#3` failure without introducing the short-path
/// regressions seen with always-on symmetric.
fn propagate_symmetric_initial_guess<const N: usize>(
    sd_guess: &[f64],
    sdd_guess: &mut [f64],
    sddd_guess: &mut [f64],
    dt_guess: &mut [f64],
    deriv: &PathDerivatives<N>,
    start_sdd: f64,
    end_sdd: f64,
) {
    let m = sd_guess.len();
    let seg = deriv.num_segments();
    if seg < 2 {
        propagate_initial_guess(sd_guess, sdd_guess, sddd_guess, dt_guess, deriv, start_sdd);
        return;
    }

    let mut sdd_fwd = vec![0.0_f64; m];
    let mut sddd_fwd = vec![0.0_f64; seg];
    let mut dt_fwd = vec![1e-5_f64; seg];
    propagate_initial_guess(
        sd_guess,
        &mut sdd_fwd,
        &mut sddd_fwd,
        &mut dt_fwd,
        deriv,
        start_sdd,
    );

    let mut sdd_bwd = vec![0.0_f64; m];
    let mut sddd_bwd = vec![0.0_f64; seg];
    let mut dt_bwd = vec![1e-5_f64; seg];
    propagate_initial_guess_backward(
        sd_guess,
        &mut sdd_bwd,
        &mut sddd_bwd,
        &mut dt_bwd,
        deriv,
        end_sdd,
    );

    // Pick the seam at the sample where forward and backward `sdd` values agree best,
    // *constrained to the middle third* of the path. Two reasons for the constraint:
    // (1) very early or very late seams put the discontinuity in the ramp-up or
    // ramp-down region where the IPM is most sensitive. (2) on long smooth paths,
    // forward and backward agree closely *everywhere*; an unconstrained search picks
    // an essentially-arbitrary location that happens to flip between failure and
    // success on different paths.
    let lo = (seg / 3).max(1);
    let hi = ((2 * seg) / 3).min(seg - 1);
    let mut mid = seg / 2;
    let mut best_diff = (sdd_fwd[mid] - sdd_bwd[mid]).abs();
    for k in lo..=hi {
        let diff = (sdd_fwd[k] - sdd_bwd[k]).abs();
        if diff < best_diff {
            best_diff = diff;
            mid = k;
        }
    }

    for k in 0..=mid {
        sdd_guess[k] = sdd_fwd[k];
    }
    sdd_guess[mid] = 0.5 * (sdd_fwd[mid] + sdd_bwd[mid]);
    for k in (mid + 1)..m {
        sdd_guess[k] = sdd_bwd[k];
    }
    for k in 0..mid {
        sddd_guess[k] = sddd_fwd[k];
        dt_guess[k] = dt_fwd[k];
    }
    for k in mid..seg {
        sddd_guess[k] = sddd_bwd[k];
        dt_guess[k] = dt_bwd[k];
    }
}

/// Backward analogue of [`propagate_initial_guess`]. Starts from `end_sdd` at the last
/// sample and walks backward, solving per-segment for `(dt[k], sddd[k], sdd[k])` such
/// that the integrator equalities hold over each segment. Derivation: with
/// `(v0, v1, a1, ds)` known and `(dt, j, a0)` unknown, substituting `a0 = a1 − j·dt`
/// into the sd-integrator gives `j = 2·(v0 − v1 + a1·dt)/dt²`, and substituting that into
/// the ds-equality yields the quadratic `(1/6)·a1·dt² − ((1/3)·v0 + (2/3)·v1)·dt + ds = 0`.
///
/// Currently unused (only called by the disabled symmetric path).
#[allow(dead_code)]
fn propagate_initial_guess_backward<const N: usize>(
    sd_guess: &[f64],
    sdd_guess: &mut [f64],
    sddd_guess: &mut [f64],
    dt_guess: &mut [f64],
    deriv: &PathDerivatives<N>,
    end_sdd: f64,
) {
    let m = sd_guess.len();
    let seg = deriv.num_segments();
    sdd_guess[m - 1] = end_sdd;
    for k in (0..seg).rev() {
        let v0 = sd_guess[k].max(0.0);
        let v1 = sd_guess[k + 1].max(0.0);
        let a1 = sdd_guess[k + 1];
        let ds_k = deriv.ds[k];

        let big_a = a1 / 6.0;
        let big_b = -((1.0 / 3.0) * v0 + (2.0 / 3.0) * v1);
        let big_c = ds_k;

        let dt_k = solve_segment_dt(big_a, big_b, big_c, v0, v1, ds_k);
        let j_k = 2.0 * (v0 - v1 + a1 * dt_k) / (dt_k * dt_k);
        let a0 = a1 - j_k * dt_k;

        dt_guess[k] = dt_k.max(1e-5);
        sddd_guess[k] = j_k;
        sdd_guess[k] = a0;
    }
}

// `SYMMETRIC_MIN_SAMPLES` and `SYMMETRIC_RESIDUAL_THRESHOLD` previously gated when
// the symmetric forward+backward initial-guess fired. Both removed when we disabled
// the symmetric path above; left as a note in case someone wants to re-enable.

/// Returns the worst per-joint v/a/j violation ratio across the entire propagated guess,
/// where `1.0` means "exactly at the limit" and `> 1.0` means "over the limit". Used by
/// the cruise-feasibility loop to decide whether to halve `cruise` and re-propagate.
fn worst_initial_guess_violation<const N: usize>(
    sd_guess: &[f64],
    sdd_guess: &[f64],
    sddd_guess: &[f64],
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6Constraints<N>,
) -> f64 {
    let m = sd_guess.len();
    let seg = sddd_guess.len();
    let mut worst = 0.0_f64;
    for k in 0..m {
        let sd = sd_guess[k];
        let sdd = sdd_guess[k];
        let seg_idx = k.min(seg.saturating_sub(1));
        let sddd = sddd_guess[seg_idx];
        for j in 0..N {
            let qp = deriv.qp[k][j];
            let qpp = deriv.qpp[k][j];
            let qppp = deriv.qppp[k][j];

            let v_max = constraints.joint.v_max.0[j];
            if v_max.is_finite() && v_max > 0.0 {
                let r = (qp * sd).abs() / v_max;
                if r > worst {
                    worst = r;
                }
            }
            let a_max = constraints.joint.a_max.0[j];
            if a_max.is_finite() && a_max > 0.0 {
                let r = (qpp * sd * sd + qp * sdd).abs() / a_max;
                if r > worst {
                    worst = r;
                }
            }
            let j_max = constraints.joint.j_max.0[j];
            if j_max.is_finite() && j_max > 0.0 {
                let r = (qppp * sd * sd * sd + 3.0 * qpp * sd * sdd + qp * sddd).abs()
                    / j_max;
                if r > worst {
                    worst = r;
                }
            }
        }
    }
    worst
}

/// Solves `(1/6)·a·u² + b·u + c = 0` for the smallest positive real root `u = dt`, with
/// graceful fallbacks for degenerate cases (`a ≈ 0`, no real roots, no positive root).
/// `v0, v1, ds_k` are used only to compute a trapezoidal fallback when the closed form
/// can't return a usable root.
fn solve_segment_dt(a: f64, b: f64, c: f64, v0: f64, v1: f64, ds_k: f64) -> f64 {
    let fallback = || {
        let avg = (0.5 * (v0 + v1)).max(1e-6);
        (ds_k / avg).max(1e-5)
    };

    if a.abs() < 1e-12 * (b.abs() + 1.0) {
        if b.abs() < 1e-9 {
            return fallback();
        }
        let u = -c / b;
        return if u > 1e-9 { u } else { fallback() };
    }

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return fallback();
    }
    let sqd = disc.sqrt();
    let u1 = (-b + sqd) / (2.0 * a);
    let u2 = (-b - sqd) / (2.0 * a);
    let best = [u1, u2]
        .iter()
        .copied()
        .filter(|&u| u > 1e-9 && u.is_finite())
        .fold(f64::INFINITY, f64::min);
    if best.is_finite() { best } else { fallback() }
}
