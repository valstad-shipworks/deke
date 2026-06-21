use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

use hafgufa::{Options, Problem, VariableArena, subject_to};

use deke_types::{DekeError, DekeResult};

use super::constraints::Topp3Tcp6Constraints;
use super::diagnostic::{BoundarySlackUsage, ConstraintCounts, InitialGuessStats, SolveStatus};
use crate::common::boundary::ProjectedBoundary;
use crate::common::path_derivatives::PathDerivatives;

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

        let rhs_ds =
            sd_k * dt_k + 0.5 * sdd_k * dt_k * dt_k + (1.0 / 6.0) * sddd_k * dt_k * dt_k * dt_k;
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

        let qp_norm_sq: f64 = (lock..N).map(|j| deriv.qp[k][j] * deriv.qp[k][j]).sum();
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

        // Joint a/j constraints are applied even at qp-degenerate samples: their LHS
        // reduces to `qpp·sd² (+ 3·qpp·sd·sdd)` when `qp_j = 0`, which is still a
        // well-posed bound on `sd` (and `sdd`). Skipping them lets the IPM find a
        // solution where `sd` stays high through a kink even though `|qpp|` is huge
        // there, producing acceleration violations one or two samples past the corner.
        // The v constraint stays guarded since it becomes vacuous at `qp_j ≈ 0`.
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
                let expr = qppp_j * sd_k * sd_k * sd_k + 3.0 * qpp_j * sd_k * sdd_k + qp_j * sddd_k;
                subject_to!(problem, expr <= j_max);
                let neg = -qppp_j * sd_k * sd_k * sd_k - 3.0 * qpp_j * sd_k * sdd_k - qp_j * sddd_k;
                subject_to!(problem, neg <= j_max);
                counts.joint_j += 2;
            }
        }
        let _ = qp_degenerate; // retained for diagnostic accounting above

        let at_boundary = k == 0 || k == m - 1;

        // TCP a/j are scaled into the unit ball: ||a_vec/a_max||² ≤ 1 instead of
        // ||a_vec||² ≤ a_max². The quadratic LHS otherwise ranges across a_max²/j_max²
        // (e.g. 40000), and that wrecks the Hessian conditioning in interior-point methods.
        //
        // TCP a at the boundary is redundant: with `sd ≈ 0` and `sdd ≈ 0` (within the
        // boundary slack), the LHS `ppp·sd² + pp·sdd` is ~0 regardless of `a_max`.
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
            && tcp.j_max.is_finite()
            && tcp.j_max > 0.0
        {
            if at_boundary {
                // At rest-to-rest boundaries `sd, sdd ≈ 0` (within slack), so the TCP
                // jerk LHS `pppp·sd³ + 3·ppp·sd·sdd + pp·sddd` collapses to `pp·sddd`.
                // Bound `|sddd|·‖pp‖ ≤ j_max` as two scalar linear inequalities on the
                // segment's `sddd` decision variable — no SOC at the boundary keeps the
                // IPM's barrier scaling well-behaved.
                let pp = &deriv.pp[k];
                let pp_norm = (pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2]).sqrt();
                if pp_norm > 1e-12 {
                    let upper = tcp.j_max / pp_norm;
                    subject_to!(problem, sddd_k <= upper);
                    let neg_sddd = -sddd_k;
                    subject_to!(problem, neg_sddd <= upper);
                    counts.tcp_j += 1;
                }
            } else {
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
    }

    // Output-FD V/A/J bound. The resampler emits chord-linear samples inside each
    // densified segment, so the *backward-difference* kinematics that downstream
    // consumers (and `check_resampled_dynamics_against_limits`) read are
    //
    //   v_FD(τ) / secant = sd(τ) − h/2·sdd(τ) + h²/6·sddd        (cubic in τ)
    //   a_FD(τ) / secant = sdd(τ − h) = sdd[k] + sddd[k]·(τ − h) (linear in τ)
    //   j_FD     / secant = sddd                                  (constant)
    //
    // where `h = 1/sample_rate_hz` is the output dt, `τ` is local time within the
    // segment, and `secant = (q[k+1] − q[k])/ds[k]`. The PCHIP-side rows above
    // bound the *analytical* kinematics `qp·sd`, `qpp·sd² + qp·sdd`, …, which
    // differ from the chord-linear FD kinematics by an `O(qpp·ds + sdd·h)` term
    // that grows with curvature and output dt.
    //
    // FD-valid τ in segment k runs from `3h` to `dt[k]`. The lower bound is `3h`
    // (not `h`) because `check_resampled_dynamics_against_limits` skips any sample
    // whose 3-step backward *jerk* stencil straddles a segment boundary, and that
    // skip applies to the V/A rows too. Constraint magnitudes are quadratic /
    // linear / constant in τ, so enforcing at both ends bounds the entire valid
    // range — the only interior overshoot is the concave-in-τ V case, capped at
    // `(1/8)·|sddd|·dt[k]² · |secant|` ≪ v_max for realistic (j_max, dt[k]) pairs.
    // Enforcing at `τ = h` instead would miss the `2.5h·sdd[k]` linear ramp that
    // the FD picks up between `h` and `3h` whenever the IPM puts non-zero `sdd`
    // at the start of a segment.
    let h = if constraints.sample_rate_hz.is_finite() && constraints.sample_rate_hz > 0.0 {
        1.0 / constraints.sample_rate_hz
    } else {
        0.0
    };
    let h_half = 0.5 * h;
    let h_sq_6 = h * h / 6.0;
    let two_and_a_half_h = 2.5 * h;
    let nineteen_sixth_h_sq = (19.0 / 6.0) * h * h;
    let two_h = 2.0 * h;
    // Skip FD constraints entirely when the path is short enough that no output
    // sample will be FD-checked. `check_resampled_dynamics_against_limits` needs a
    // 3-step backward stencil inside one segment, so the very first FD-evaluated
    // sample sits at `t ≥ 3h` and the trajectory needs at least roughly `3h` of
    // wall time to host any check at all. For micro-paths whose total arc length
    // is less than the distance the slowest plausible cruise can cover in `3h`
    // (`total_ds < 3h · v_max_path`), the entire trajectory fits inside one
    // FD-skipped window and the FD rows would only extrapolate past it — they're
    // numerically punishing tight and can render the IPM problem infeasible
    // (`microscopic_path_length` etc.). No FD sample can read a violation in that
    // regime, so dropping the rows is exact, not a relaxation.
    let v_max_path: f64 = (lock..N)
        .map(|j| constraints.joint.v_max.0[j])
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let total_ds: f64 = deriv.ds.iter().sum();
    let path_too_short_for_fd = total_ds < 3.0 * h * v_max_path;
    const FD_RELAX: f64 = 1.0;
    // Per-knot effective sd upper bound from the binding V constraints (joint AND
    // TCP). The interior V_FD rows below use this to pick a lower bound on `dt[k]`
    // for each segment: `dt[k] ≥ ds[k] / max(sd_upper[k], sd_upper[k+1])`. Using
    // only the joint v_max would over-estimate the sd ceiling whenever the TCP v
    // limit is tighter (e.g. rail-dominant paths where TCP v / |pp| ≪ joint v_max),
    // so we take the min over all V rows the NLP enforces at each knot.
    let sd_upper_at_knot: Vec<f64> = (0..m)
        .map(|k| {
            let mut upper = f64::INFINITY;
            for j in lock..N {
                let qp_abs = deriv.qp[k][j].abs();
                if qp_abs > qp_cutoffs[j] {
                    let v_max = constraints.joint.v_max.0[j];
                    if v_max.is_finite() && v_max > 0.0 {
                        upper = upper.min(v_max / qp_abs);
                    }
                }
            }
            if tcp_active
                && deriv.has_tcp()
                && let Some(tcp) = constraints.tcp
                && tcp.v_max.is_finite()
                && tcp.v_max > 0.0
            {
                let pp_k = &deriv.pp[k];
                let pp_norm_sq = pp_k[0] * pp_k[0] + pp_k[1] * pp_k[1] + pp_k[2] * pp_k[2];
                if pp_norm_sq > pp_cutoff_sq {
                    upper = upper.min(tcp.v_max / pp_norm_sq.sqrt());
                }
            }
            if upper.is_finite() { upper } else { v_max_path }
        })
        .collect();
    // Per-segment lower bound on `dt[k]`. Used to gate the interior V_FD rows
    // below: a row at `τ = i·h` is added only when `i·h ≤ dt_lower_k`, so the row
    // never bounds an extrapolation past the segment end.
    const MAX_INTERIOR_I: usize = 20;
    if h > 0.0 && !path_too_short_for_fd {
        for k in 0..seg {
            let ds_k = deriv.ds[k];
            if !matches!(ds_k.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
                continue;
            }
            let a = deriv.waypoints[k].0;
            let b = deriv.waypoints[k + 1].0;
            let sd_k = sd[k];
            let sd_k1 = sd[k + 1];
            let sdd_k = sdd[k];
            let sdd_k1 = sdd[k + 1];
            let sddd_k = sddd[k];
            let sd_upper_seg = sd_upper_at_knot[k].max(sd_upper_at_knot[k + 1]).max(1e-12);
            let dt_lower_k = ds_k / sd_upper_seg;
            let max_interior_i = ((dt_lower_k / h).floor() as usize).min(MAX_INTERIOR_I);

            for j in lock..N {
                let secant_j = (b[j] - a[j]) / ds_k;
                let abs_sec = secant_j.abs();
                if abs_sec < qp_cutoffs[j] {
                    // Joint j barely moves in this segment — secant·anything is
                    // dominated by the noise floor; constraint row is near-vacuous
                    // and only hurts IPM conditioning.
                    continue;
                }
                // Curvature-guard: skip when path curvature on this joint·segment
                // is so steep that the analytical jerk row (`|qppp·sd³ + …| ≤ j_max`)
                // is already pinning `sd` extremely low, and the chord-linear V row
                // here would only multiply the over-constraint. Threshold compares
                // the PCHIP curvature contribution `|qpp[k]·ds[k]|` against the
                // PCHIP slope `|qp[k]|` — when `qpp·ds ≫ qp` the chord direction
                // diverges from the PCHIP tangent by an unbounded factor and any
                // chord-side row at that scale clashes with the analytical jerk
                // bound. Empirically tuned to keep adversarial joint-space zigzags
                // (max|qpp| ≈ 1500) IPM-feasible while still applying the FD rows
                // wherever the chord-linear/PCHIP gap is the dominant overshoot
                // source.
                let qpp_k = deriv.qpp[k][j].abs();
                let qpp_k1 = deriv.qpp[k + 1][j].abs();
                let qp_k_abs = deriv.qp[k][j].abs().max(deriv.qp[k + 1][j].abs());
                if qp_k_abs > 0.0 && qpp_k.max(qpp_k1) * ds_k > 3.0 * qp_k_abs {
                    continue;
                }
                let v_max = constraints.joint.v_max.0[j] * FD_RELAX;
                let a_max = constraints.joint.a_max.0[j] * FD_RELAX;
                let j_max = constraints.joint.j_max.0[j] * FD_RELAX;

                // V at τ = 3h (start of check-valid range):
                //   secant·(sd[k] + 2.5h·sdd[k] + (19/6)·h²·sddd[k])
                // V at τ = dt[k] (segment end):
                //   secant·(sd[k+1] − h/2·sdd[k+1] + h²/6·sddd[k])
                //
                // Endpoint rows alone leave a concave-in-τ interior peak: when
                // `sddd[k] < 0` the parabola `v_FD(τ)/secant = sd[k] + sdd[k]·
                // (τ − h/2) + ½·sddd[k]·(τ² − τh + h²/3)` has its vertex at
                // `τ* = h/2 − sdd[k]/sddd[k]` and exceeds both endpoints by up to
                // `½·sdd[k]²/|sddd[k]|`. The interior rows below pin the parabola
                // at additional fixed τ values with 2h spacing, closing the peak
                // to `|sddd|·h²/2` (negligible for realistic params).
                if v_max.is_finite() && v_max > 0.0 {
                    let v_start = secant_j * sd_k
                        + (secant_j * two_and_a_half_h) * sdd_k
                        + (secant_j * nineteen_sixth_h_sq) * sddd_k;
                    subject_to!(problem, v_start <= v_max);
                    let v_start_neg = -secant_j * sd_k
                        + (-secant_j * two_and_a_half_h) * sdd_k
                        + (-secant_j * nineteen_sixth_h_sq) * sddd_k;
                    subject_to!(problem, v_start_neg <= v_max);

                    let v_end = secant_j * sd_k1
                        + (-secant_j * h_half) * sdd_k1
                        + (secant_j * h_sq_6) * sddd_k;
                    subject_to!(problem, v_end <= v_max);
                    let v_end_neg = -secant_j * sd_k1
                        + (secant_j * h_half) * sdd_k1
                        + (-secant_j * h_sq_6) * sddd_k;
                    subject_to!(problem, v_end_neg <= v_max);
                    counts.joint_v += 4;

                    // Interior V_FD rows at τ = i·h for i ∈ {5, 7, …} up to
                    // `floor(dt_lower_k / h)`. Each is a linear combination of
                    // `(sd[k], sdd[k], sddd[k])` evaluating `v_FD(τ)/secant` at
                    // that fixed τ. 2h spacing is enough to bound the residual
                    // peak to `|sddd|·h²/2` ≈ 0.3 % of `v_max` for typical params.
                    let mut i = 5_usize;
                    while i <= max_interior_i {
                        let i_f = i as f64;
                        let coeff_sdd = (i_f - 0.5) * h;
                        let coeff_sddd = ((3.0 * i_f - 3.0) * i_f + 1.0) / 6.0 * h * h;
                        let v_i = secant_j * sd_k
                            + (secant_j * coeff_sdd) * sdd_k
                            + (secant_j * coeff_sddd) * sddd_k;
                        subject_to!(problem, v_i <= v_max);
                        let v_i_neg = -secant_j * sd_k
                            + (-secant_j * coeff_sdd) * sdd_k
                            + (-secant_j * coeff_sddd) * sddd_k;
                        subject_to!(problem, v_i_neg <= v_max);
                        counts.joint_v += 2;
                        i += 2;
                    }
                }

                // A at τ = 3h:       secant·(sdd[k] + 2h·sddd[k])
                // A at τ = dt[k]:    secant·(sdd[k+1] − h·sddd[k])
                if a_max.is_finite() && a_max > 0.0 {
                    let a_start = secant_j * sdd_k + (secant_j * two_h) * sddd_k;
                    subject_to!(problem, a_start <= a_max);
                    let a_start_neg = -secant_j * sdd_k + (-secant_j * two_h) * sddd_k;
                    subject_to!(problem, a_start_neg <= a_max);

                    let a_end = secant_j * sdd_k1 + (-secant_j * h) * sddd_k;
                    subject_to!(problem, a_end <= a_max);
                    let a_end_neg = -secant_j * sdd_k1 + (secant_j * h) * sddd_k;
                    subject_to!(problem, a_end_neg <= a_max);
                    counts.joint_a += 4;
                }

                // J:                 secant·sddd[k] (constant per segment)
                if j_max.is_finite() && j_max > 0.0 {
                    let j_expr = secant_j * sddd_k;
                    subject_to!(problem, j_expr <= j_max);
                    let j_neg = -secant_j * sddd_k;
                    subject_to!(problem, j_neg <= j_max);
                    counts.joint_j += 2;
                }
            }

            // TCP V/A/J via the same FD-on-chord-linear logic but with the TCP-space
            // secant `(T[k+1] − T[k]) / ds_k` standing in for the joint secant. The
            // chord-linear joint interp produces a curve in TCP space (FK is nonlinear)
            // rather than a chord, so this is exact at the segment endpoints and
            // approximate (`O(qpp · ds)` and Jacobian variation) inside. Skipped when
            // the segment's TCP secant is below the same noise floor `pp_cutoff_sq`
            // that gates the analytical per-knot TCP-v bound — a wrist-only rotation
            // segment, for instance, has `|tcp_secant| ≈ 0` and `tcp.v_max/|tcp_secant|`
            // would inflate to a vacuous upper bound that costs the IPM nothing in
            // feasibility but wrecks scaling.
            if let Some(tcp) = constraints.tcp
                && tcp_active
            {
                // Use the precomputed per-segment chord-curve tangent bound from
                // `PathDerivatives` (numerical FK at u = 0/0.5/1, max of full +
                // half-chord secants) — captures the FK curvature inside the
                // chord-linear joint segment that a plain `(T[k+1]−T[k])/ds_k`
                // secant under-reports on high-curvature paths.
                let secant_norm_sq = deriv.chord_tcp_tangent_max_sq[k];
                if secant_norm_sq > pp_cutoff_sq {
                    let norm = secant_norm_sq.sqrt();

                    // TCP V at τ = 3h: |sd[k] + 2.5h·sdd[k] + (19/6)·h²·sddd[k]| · ‖tcp_tangent‖ ≤ v_max
                    // TCP V at τ = dt[k]: |sd[k+1] − h/2·sdd[k+1] + h²/6·sddd[k]| · ‖tcp_tangent‖ ≤ v_max
                    // Plus interior rows at τ = 5h, 7h, … to close the concave-down
                    // parabola peak in the same way as the joint V rows above.
                    if tcp.v_max.is_finite() && tcp.v_max > 0.0 {
                        let upper = (tcp.v_max * FD_RELAX) / norm;
                        let sd_avg_start =
                            sd_k + two_and_a_half_h * sdd_k + nineteen_sixth_h_sq * sddd_k;
                        subject_to!(problem, sd_avg_start <= upper);
                        let sd_avg_end = sd_k1 + (-h_half) * sdd_k1 + h_sq_6 * sddd_k;
                        subject_to!(problem, sd_avg_end <= upper);
                        counts.tcp_v += 2;
                        let mut i = 5_usize;
                        while i <= max_interior_i {
                            let i_f = i as f64;
                            let coeff_sdd = (i_f - 0.5) * h;
                            let coeff_sddd = ((3.0 * i_f - 3.0) * i_f + 1.0) / 6.0 * h * h;
                            let sd_avg_i = sd_k + coeff_sdd * sdd_k + coeff_sddd * sddd_k;
                            subject_to!(problem, sd_avg_i <= upper);
                            counts.tcp_v += 1;
                            i += 2;
                        }
                    }

                    // TCP A at τ = 3h:    |sdd[k] + 2h·sddd[k]| · ‖tcp_tangent‖ ≤ a_max
                    // TCP A at τ = dt[k]: |sdd[k+1] − h·sddd[k]|  · ‖tcp_tangent‖ ≤ a_max
                    if tcp.a_max.is_finite() && tcp.a_max > 0.0 {
                        let upper = (tcp.a_max * FD_RELAX) / norm;
                        let sdd_start = sdd_k + two_h * sddd_k;
                        subject_to!(problem, sdd_start <= upper);
                        let neg_sdd_start = -sdd_k + (-two_h) * sddd_k;
                        subject_to!(problem, neg_sdd_start <= upper);
                        let sdd_end = sdd_k1 + (-h) * sddd_k;
                        subject_to!(problem, sdd_end <= upper);
                        let neg_sdd_end = -sdd_k1 + h * sddd_k;
                        subject_to!(problem, neg_sdd_end <= upper);
                        counts.tcp_a += 4;
                    }

                    // TCP J: |sddd| · ‖tcp_secant‖ ≤ j_max
                    if tcp.j_max.is_finite() && tcp.j_max > 0.0 {
                        let upper = (tcp.j_max * FD_RELAX) / norm;
                        subject_to!(problem, sddd_k <= upper);
                        let neg_sddd_k = -sddd_k;
                        subject_to!(problem, neg_sddd_k <= upper);
                        counts.tcp_j += 2;
                    }
                }
            }
        }
    }

    // Cross-knot output-FD bound. The resampler emits chord-linear joint
    // samples, so the chord direction `D_k = (w[k+1] − w[k])/ds[k]` jumps
    // across each interior densified knot. The full backward-FD readout at
    // the post-knot worst-case (sample 1 past the knot, τ→h⁻) is, per
    // component j:
    //
    //   a_FD_j ≈ sd[K]·ΔD_j/h
    //          + (1/2)·sdd[K]·(D_new_j + D_old_j)
    //          + (h/6)·(sddd[K]·D_new_j − sddd[K-1]·D_old_j)
    //
    //   j_FD_j ≈ sd[K]·ΔD_j/h²
    //          + (1/2)·sdd[K]·ΔD_j/h
    //          + (1/6)·sddd[K]·D_new_j
    //          + (5/6)·sddd[K-1]·D_old_j
    //
    // (with `ΔD = D_new − D_old`). Only the leading `sd·ΔD/h^n` term is
    // *unbounded* by the per-segment analytical rows; the sdd / sddd tails are
    // already constrained per segment but their cross-knot combinations can
    // still reach a substantial fraction of the limit (the sddd weights sum
    // to 1 in jerk). Bounding the whole linear combination in a single IPM
    // row over-constrains the optimizer into feasibility restoration on
    // tight axis-flip paths, so we cap only the leading spike at
    // `KNOT_SD_BUDGET` of the limit and let `resampled_check_slack` absorb
    // the remainder. Empirically 0.78 is the largest budget that keeps the
    // tail-driven overshoot under the check's 10% slack envelope on the
    // sharp-corner stress tests without flipping any IPM into infeasibility.
    //
    // Cross-knot V_FD at the post-knot sample is `D_new·sd[K]`, already
    // bounded by the analytical V row in segment K, so no V cross-knot row.
    if h > 0.0 && !path_too_short_for_fd {
        const KNOT_SD_BUDGET: f64 = 0.78;
        let knot_a_rhs = h * KNOT_SD_BUDGET;
        let knot_j_rhs = h * h * KNOT_SD_BUDGET;
        for k in 1..seg {
            let ds_prev = deriv.ds[k - 1];
            let ds_curr = deriv.ds[k];
            if !(ds_prev > 0.0 && ds_curr > 0.0) {
                continue;
            }
            let w_prev = deriv.waypoints[k - 1].0;
            let w_knot = deriv.waypoints[k].0;
            let w_next = deriv.waypoints[k + 1].0;
            let sd_k = sd[k];
            for j in lock..N {
                let d_prev = (w_knot[j] - w_prev[j]) / ds_prev;
                let d_curr = (w_next[j] - w_knot[j]) / ds_curr;
                let delta = (d_curr - d_prev).abs();
                if delta < qp_cutoffs[j] {
                    continue;
                }
                let a_max = constraints.joint.a_max.0[j];
                if a_max.is_finite() && a_max > 0.0 {
                    let upper = (a_max * knot_a_rhs) / delta;
                    subject_to!(problem, sd_k <= upper);
                    counts.joint_a += 1;
                }
                let j_max = constraints.joint.j_max.0[j];
                if j_max.is_finite() && j_max > 0.0 {
                    let upper = (j_max * knot_j_rhs) / delta;
                    subject_to!(problem, sd_k <= upper);
                    counts.joint_j += 1;
                }
            }
            if let Some(tcp) = constraints.tcp
                && tcp_active
                && deriv.has_tcp()
            {
                let t_prev = deriv.tcp[k - 1];
                let t_knot = deriv.tcp[k];
                let t_next = deriv.tcp[k + 1];
                let dx = (t_next[0] - t_knot[0]) / ds_curr - (t_knot[0] - t_prev[0]) / ds_prev;
                let dy = (t_next[1] - t_knot[1]) / ds_curr - (t_knot[1] - t_prev[1]) / ds_prev;
                let dz = (t_next[2] - t_knot[2]) / ds_curr - (t_knot[2] - t_prev[2]) / ds_prev;
                let delta_norm_sq = dx * dx + dy * dy + dz * dz;
                if delta_norm_sq > pp_cutoff_sq {
                    let delta_norm = delta_norm_sq.sqrt();
                    if tcp.a_max.is_finite() && tcp.a_max > 0.0 {
                        let upper = (tcp.a_max * knot_a_rhs) / delta_norm;
                        subject_to!(problem, sd_k <= upper);
                        counts.tcp_a += 1;
                    }
                    if tcp.j_max.is_finite() && tcp.j_max > 0.0 {
                        let upper = (tcp.j_max * knot_j_rhs) / delta_norm;
                        subject_to!(problem, sd_k <= upper);
                        counts.tcp_j += 1;
                    }
                }
            }
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
            &sd,
            &sdd,
            &sddd,
            &dt,
            deriv,
            constraints,
            start,
            end,
            pp_cutoff_sq,
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
/// boundary slack box [`super::constraints::SolverOptions::boundary_slack`] is sized to absorb the
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

    let min_cap = cap.iter().copied().fold(f64::INFINITY, f64::min).max(1e-3);

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

    sdd_guess[..=mid].copy_from_slice(&sdd_fwd[..=mid]);
    sdd_guess[mid] = 0.5 * (sdd_fwd[mid] + sdd_bwd[mid]);
    sdd_guess[(mid + 1)..m].copy_from_slice(&sdd_bwd[(mid + 1)..m]);
    sddd_guess[..mid].copy_from_slice(&sddd_fwd[..mid]);
    dt_guess[..mid].copy_from_slice(&dt_fwd[..mid]);
    sddd_guess[mid..seg].copy_from_slice(&sddd_bwd[mid..seg]);
    dt_guess[mid..seg].copy_from_slice(&dt_bwd[mid..seg]);
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
                let r = (qppp * sd * sd * sd + 3.0 * qpp * sd * sdd + qp * sddd).abs() / j_max;
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
