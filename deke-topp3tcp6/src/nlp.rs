use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

use hafgufa::{Options, Problem, VariableArena, subject_to};

use deke_types::{DekeError, DekeResult};

use crate::Topp3Tcp6Constraints;
use crate::boundary::ProjectedBoundary;
use crate::diagnostic::SolveStatus;
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
    let m = deriv.num_waypoints();
    let seg = deriv.num_segments();
    if m < 2 || seg == 0 {
        return Err(DekeError::PathTooShort(m));
    }
    let lock = constraints.locked_prefix.min(N);

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let sd: Vec<_> = (0..m).map(|_| problem.decision_variable()).collect();
    let sdd: Vec<_> = (0..m).map(|_| problem.decision_variable()).collect();
    let sddd: Vec<_> = (0..seg).map(|_| problem.decision_variable()).collect();
    let dt: Vec<_> = (0..seg).map(|_| problem.decision_variable()).collect();

    let tcp_active = deriv.has_tcp() && !constraints.tcp.is_disabled();

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

    for k in 0..m {
        let sd_k = sd[k];
        subject_to!(problem, sd_k >= 0.0);

        if tcp_active && constraints.tcp.v_max.is_finite() && constraints.tcp.v_max > 0.0 {
            let pp = &deriv.pp[k];
            let pp_norm_sq = pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2];
            if pp_norm_sq > pp_cutoff_sq {
                let upper = constraints.tcp.v_max / pp_norm_sq.sqrt();
                subject_to!(problem, sd_k <= upper);
            }
        }

        for j in lock..N {
            let qp_j = deriv.qp[k][j];
            if qp_j.abs() < 1e-12 {
                continue;
            }
            let v_max = constraints.joint.v_max.0[j];
            if v_max.is_finite() && v_max > 0.0 {
                let upper = v_max / qp_j.abs();
                subject_to!(problem, sd_k <= upper);
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

        for j in lock..N {
            let qp_j = deriv.qp[k][j];
            let qpp_j = deriv.qpp[k][j];
            let qppp_j = deriv.qppp[k][j];
            let v_max = constraints.joint.v_max.0[j];
            let a_max = constraints.joint.a_max.0[j];
            let j_max = constraints.joint.j_max.0[j];

            if v_max.is_finite() && v_max > 0.0 && qp_j.abs() > 1e-12 {
                let expr = qp_j * sd_k;
                subject_to!(problem, expr <= v_max);
                let neg = -qp_j * sd_k;
                subject_to!(problem, neg <= v_max);
            }

            if a_max.is_finite() && a_max > 0.0 {
                let expr = qpp_j * sd_k * sd_k + qp_j * sdd_k;
                subject_to!(problem, expr <= a_max);
                let neg = -qpp_j * sd_k * sd_k - qp_j * sdd_k;
                subject_to!(problem, neg <= a_max);
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
            }
        }

        // TCP a/j are scaled into the unit ball: ||a_vec/a_max||² ≤ 1 instead of
        // ||a_vec||² ≤ a_max². The quadratic LHS otherwise ranges across a_max²/j_max²
        // (e.g. 40000), and that wrecks the Hessian conditioning in interior-point methods.
        if tcp_active
            && !at_boundary
            && constraints.tcp.a_max.is_finite()
            && constraints.tcp.a_max > 0.0
        {
            let inv = 1.0 / constraints.tcp.a_max;
            let ppp = &deriv.ppp[k];
            let pp = &deriv.pp[k];
            let c0 = (ppp[0] * inv) * sd_k * sd_k + (pp[0] * inv) * sdd_k;
            let c1 = (ppp[1] * inv) * sd_k * sd_k + (pp[1] * inv) * sdd_k;
            let c2 = (ppp[2] * inv) * sd_k * sd_k + (pp[2] * inv) * sdd_k;
            let sum_sq = c0 * c0 + c1 * c1 + c2 * c2;
            subject_to!(problem, sum_sq <= 1.0);
        }

        if tcp_active
            && !at_boundary
            && constraints.tcp.j_max.is_finite()
            && constraints.tcp.j_max > 0.0
        {
            let inv = 1.0 / constraints.tcp.j_max;
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
        }
    }

    let mut total = dt[0];
    for k in 1..seg {
        total = total + dt[k];
    }
    problem.minimize(total);

    apply_initial_guess(&sd, &sdd, &sddd, &dt, deriv, constraints, start, end, pp_cutoff_sq);

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

    Ok(Solution {
        sd: sd_vals,
        sdd: sdd_vals,
        sddd: sddd_vals,
        dt: dt_vals,
        status,
        iterations,
        solve_time,
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
) {
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
        if deriv.has_tcp()
            && constraints.tcp.v_max.is_finite()
            && constraints.tcp.v_max > 0.0
        {
            let pp = &deriv.pp[k];
            let pn_sq = pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2];
            if pn_sq > pp_cutoff_sq {
                c = c.min(constraints.tcp.v_max / pn_sq.sqrt());
            }
        }
        if !c.is_finite() {
            c = 1.0;
        }
        cap[k] = (c * 0.7).max(1e-3);
    }

    let cruise = cap
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

    let mut sd_guess = vec![cruise; m];
    if ramp == 0 {
        sd_guess[0] = start_sd;
        if m > 1 {
            sd_guess[m - 1] = end_sd;
        }
    } else {
        let pi = std::f64::consts::PI;
        for k in 0..=ramp.min(m - 1) {
            let factor = 0.5 * (1.0 - (pi * k as f64 / ramp as f64).cos());
            sd_guess[k] = start_sd + (cruise - start_sd) * factor;
        }
        for k in (m.saturating_sub(ramp + 1))..m {
            let j = m - 1 - k;
            let factor = 0.5 * (1.0 - (pi * j as f64 / ramp as f64).cos());
            let blended = end_sd + (cruise - end_sd) * factor;
            // Don't reduce below the ramp-up value — keep cruise wherever the two ramps
            // would overlap.
            if blended < sd_guess[k] {
                sd_guess[k] = blended;
            }
        }
        sd_guess[0] = start_sd;
        sd_guess[m - 1] = end_sd;
    }
    for k in 0..m {
        if sd_guess[k] > cap[k] {
            sd_guess[k] = cap[k];
        }
        if sd_guess[k] < 0.0 {
            sd_guess[k] = 0.0;
        }
    }

    let mut sdd_guess = vec![0.0_f64; m];
    sdd_guess[0] = start.sdd;
    let mut sddd_guess = vec![0.0_f64; seg];
    let mut dt_guess = vec![1e-5_f64; seg];

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

    for k in 0..m {
        sd[k].set_value(sd_guess[k]);
        sdd[k].set_value(sdd_guess[k]);
    }
    for k in 0..seg {
        sddd[k].set_value(sddd_guess[k]);
        dt[k].set_value(dt_guess[k]);
    }
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
