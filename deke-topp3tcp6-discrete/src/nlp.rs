//! Discrete-time NLP builder.
//!
//! # Variables
//!
//! For a fixed sample count `K` (chosen by the bisection driver in
//! [`crate::retimer`]):
//!
//! - `sigma[i]` for `i ∈ 0..K`: cumulative path-arc-length at each output
//!   sample, with `sigma[0] = 0` and `sigma[K-1] = S` pinned by equality.
//! - `slack ≥ 0`: a single non-negative slack scalar added in bisection mode.
//!   Every FD row is relaxed by `+slack`; the objective adds
//!   `bisection_slack_penalty · slack`. At convergence, `slack ≈ 0` means the
//!   chosen `K` is feasible; otherwise it gives a continuous infeasibility
//!   magnitude.
//!
//! # Sample → segment binning (relaxed-box)
//!
//! Pre-assigned at build time via the proportional-arc-length heuristic
//! `sigma_guess[i] = i · S / (K-1)`, then `bin(i) = argmax_k { s[k] ≤ sigma_guess[i] }`.
//!
//! Each `Δᵐ q[i]_j` row is built against the *pre-assigned* bins of the samples
//! it spans. [`crate::retimer`] verifies post-solve that each `sigma[i]`
//! landed in its bin's relaxed box `[s[bin(i)-1], s[bin(i)+2]]`; if any drifted
//! out, the retimer re-bins and re-solves.
//!
//! # Linear coefficient tables for chord-linear differences
//!
//! Inside densified segment `b` the chord-linear joint position is
//!
//! ```text
//! q[i]_j = q_seg[b]_j + secant_j[b] · (sigma[i] − s[b])
//! ```
//!
//! where `secant_j[b] = (q_seg[b+1]_j − q_seg[b]_j) / ds[b]`.
//!
//! ## 1-step difference  Δ¹ q[i]_j = q[i]_j − q[i−1]_j
//!
//! Let `b_lo = bin(i−1)`, `b_hi = bin(i)`. For a `c`-boundary crossing
//! (`b_hi = b_lo + c`):
//!
//! - `c = 0` (same segment):  `Δ¹ = secant_j[b] · (sigma[i] − sigma[i−1])`
//! - `c ≥ 1`:
//!   ```text
//!   Δ¹ = secant_j[b_lo] · (s[b_lo+1] − sigma[i−1])
//!      + Σ_{m=b_lo+1..b_hi-1} secant_j[m] · ds[m]
//!      + secant_j[b_hi] · (sigma[i] − s[b_hi])
//!   ```
//!
//! All cases collapse into a single linear form
//! `c_curr · sigma[i] + c_prev · sigma[i−1] + beta`.
//!
//! ## Higher orders
//!
//! `Δ² q[i] = Δ¹(i) − Δ¹(i−1)` and `Δ³ q[i] = Δ²(i) − Δ²(i−1)`. Each is a
//! linear combination of the σ values it touches plus a constant. The
//! recursive builder below (`step_coeffs`) constructs the unified form
//! `Σ_t α[t] · sigma[i−t] + beta ≤ rhs` for any order `m ∈ {1, 2, 3}`.
//!
//! # Inequalities
//!
//! Per joint `j ∈ [lock_prefix, N)` and per output sample `i ∈ [m, K)`:
//!
//! ```text
//! |Δ¹ q[i]_j| ≤ v_max[j] · h
//! |Δ² q[i]_j| ≤ a_max[j] · h²
//! |Δ³ q[i]_j| ≤ j_max[j] · h³
//! ```
//!
//! Rows whose linear coefficients are all below the per-joint `qp` cutoff
//! (same threshold the continuous-time crate applies at
//! `deke-topp3tcp6/src/nlp.rs:472`) are dropped.
//!
//! # TCP rows
//!
//! TCP V/A/J use the per-segment numerical bound
//! `chord_tcp_tangent_max_sq[b]` already built by [`PathDerivatives`]. For a
//! stencil whose samples span segments `b_lo..=b_hi`, we use the max over
//! those segments as the tangent norm upper bound. The same Δ¹/Δ²/Δ³
//! coefficient builder (with `secant_j` replaced by the scalar bound)
//! produces the rows.
//!
//! # Boundary handling
//!
//! `sigma[0] = 0` and `sigma[K-1] = S` are pinned hard (no slack). Non-trivial
//! `v_start`/`v_end`/`a_start`/`a_end` are matched via additional equality
//! rows on the corresponding `Δᵐ q` expressions. The plan's "drop these
//! equalities if rest-to-rest" optimization is *not* taken here for
//! correctness: with rest-to-rest, the equalities simplify to `Δᵐ q = 0` at
//! the endpoints, which forces the leading/trailing σ values to coincide
//! with the endpoint σ (a valid feasibility constraint).

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

use deke_types::{DekeError, DekeResult};
use hafgufa::{Options, Problem, Variable, VariableArena, subject_to};

use crate::boundary::ProjectedBoundary;
use crate::constraints::Topp3Tcp6DiscreteConstraints;
use crate::diagnostic::{ConstraintCounts, SolveStatus};
use crate::path_derivatives::PathDerivatives;

/// Output of one discrete NLP solve at a fixed sample count `K`.
#[derive(Debug, Clone)]
pub struct DiscreteSolution {
    pub sigma: Vec<f64>,
    pub k: usize,
    pub status: SolveStatus,
    pub iterations: i32,
    pub solve_time: Duration,
    pub build_time: Duration,
    /// Value of the global slack scalar at convergence (0 if `with_slacks=false`).
    pub slack: f64,
    pub constraint_counts: ConstraintCounts,
    /// Per-sample densified-segment bin used at constraint-build time. Exposed
    /// so the retimer's re-bin loop can detect σ that drifted into a
    /// different segment than assumed.
    pub bins_used: Vec<usize>,
}

/// Builds and solves the discrete NLP at the supplied sample count `K`.
///
/// - `with_slacks = true` adds a single non-negative slack scalar that relaxes
///   every FD row; the objective is augmented by `slack_penalty · slack`. Used
///   by the bisection driver: at the optimum, `slack < tolerance` means `K` is
///   feasible.
/// - `with_slacks = false` is the strict mode — no slack, equalities and
///   inequalities all hard.
/// - `warm_sigma`: optional previous solution. Linearly interpolated to the
///   new `K` and used as the initial guess.
pub fn build_and_solve_discrete<const N: usize>(
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6DiscreteConstraints<N>,
    start: ProjectedBoundary,
    end: ProjectedBoundary,
    k_samples: usize,
    with_slacks: bool,
    warm_sigma: Option<&[f64]>,
) -> DekeResult<DiscreteSolution> {
    build_and_solve_discrete_with_bins(
        deriv, constraints, start, end, k_samples, with_slacks, warm_sigma, None,
    )
}

/// Variant of [`build_and_solve_discrete`] that overrides the per-solve IPM
/// timeout. Used by the bisection driver to short-circuit feasibility-
/// restoration hangs at adversarial `K` values without affecting the strict
/// post-bisection solves.
pub fn build_and_solve_discrete_with_timeout<const N: usize>(
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6DiscreteConstraints<N>,
    start: ProjectedBoundary,
    end: ProjectedBoundary,
    k_samples: usize,
    with_slacks: bool,
    warm_sigma: Option<&[f64]>,
    timeout: Option<Duration>,
) -> DekeResult<DiscreteSolution> {
    build_and_solve_discrete_inner(
        deriv, constraints, start, end, k_samples, with_slacks, warm_sigma, None, timeout,
    )
}

/// Variant of [`build_and_solve_discrete`] that lets the caller override the
/// per-sample bin assignment. Used by [`crate::retimer`] for the post-solve
/// re-bin loop: after solving with proportional bins, the actual `σ` values
/// may fall in different densified segments than assumed; if so, re-solve
/// with the corrected bins.
pub fn build_and_solve_discrete_with_bins<const N: usize>(
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6DiscreteConstraints<N>,
    start: ProjectedBoundary,
    end: ProjectedBoundary,
    k_samples: usize,
    with_slacks: bool,
    warm_sigma: Option<&[f64]>,
    bins_override: Option<&[usize]>,
) -> DekeResult<DiscreteSolution> {
    build_and_solve_discrete_inner(
        deriv, constraints, start, end, k_samples, with_slacks, warm_sigma, bins_override, None,
    )
}

fn build_and_solve_discrete_inner<const N: usize>(
    deriv: &PathDerivatives<N>,
    constraints: &Topp3Tcp6DiscreteConstraints<N>,
    start: ProjectedBoundary,
    end: ProjectedBoundary,
    k_samples: usize,
    with_slacks: bool,
    warm_sigma: Option<&[f64]>,
    bins_override: Option<&[usize]>,
    timeout_override: Option<Duration>,
) -> DekeResult<DiscreteSolution> {
    if k_samples < 4 {
        return Err(DekeError::PathTooShort(k_samples));
    }
    let m = deriv.num_waypoints();
    if m < 2 {
        return Err(DekeError::PathTooShort(m));
    }
    let s_total = deriv.total_length();
    if !(s_total > 0.0) {
        return Err(DekeError::PathTooShort(m));
    }
    let lock = constraints.locked_prefix.min(N);
    let h = if constraints.sample_rate_hz.is_finite() && constraints.sample_rate_hz > 0.0 {
        1.0 / constraints.sample_rate_hz
    } else {
        return Err(DekeError::RetimerFailed(
            "sample_rate_hz must be finite positive".into(),
        ));
    };
    let h2 = h * h;
    let h3 = h2 * h;

    let build_start = Instant::now();
    let mut counts = ConstraintCounts::default();

    let bins: Vec<usize> = match bins_override {
        Some(b) if b.len() == k_samples => b.to_vec(),
        _ => proportional_bins(&deriv.s, k_samples),
    };

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    // σ variables. We allocate K of them; σ[0] and σ[K-1] are pinned by
    // equality below.
    let sigma: Vec<Variable> = problem.decision_variables(k_samples);
    let sigma_0 = sigma[0];
    let sigma_last = sigma[k_samples - 1];
    subject_to!(problem, sigma_0 == 0.0);
    subject_to!(problem, sigma_last == s_total);

    // Monotonicity σ[i] − σ[i−1] ≥ 0. Equivalent to Δσ ≥ 0 and gives the
    // banded structure mentioned in the plan.
    for i in 1..k_samples {
        let s_prev = sigma[i - 1];
        let s_curr = sigma[i];
        let diff = s_curr - s_prev;
        subject_to!(problem, diff >= 0.0);
    }

    // Global slack scalar — gated by `with_slacks`. Building it
    // unconditionally and pinning to zero in strict mode keeps the variable
    // layout identical across bisection iterations and the strict final
    // solve (lets the IPM warm-cache the symbolic Hessian sparsity).
    let slack = problem.decision_variable();
    if with_slacks {
        let s = slack;
        subject_to!(problem, s >= 0.0);
        slack.set_value(0.0);
    } else {
        subject_to!(problem, slack == 0.0);
    }

    // Per-joint relative-magnitude cutoff (same heuristic as the
    // continuous-time crate). Drops rows where every secant in the stencil
    // is below the noise floor.
    let mut qp_max_abs = [0.0_f64; N];
    for b in 0..deriv.num_segments() {
        for j in 0..N {
            let sec = secant_j(deriv, b, j).abs();
            if sec > qp_max_abs[j] {
                qp_max_abs[j] = sec;
            }
        }
    }
    let mut qp_cutoffs = [0.0_f64; N];
    for j in 0..N {
        qp_cutoffs[j] = (1e-6 * qp_max_abs[j]).max(1e-12);
    }

    // FD-V, FD-A, FD-J rows per joint per sample.
    for i in 1..k_samples {
        for j in lock..N {
            let stencil = step_coeffs::<N>(deriv, j, &bins, i, 1);
            if let Some(row) = stencil {
                let v_max = constraints.joint.v_max.0[j];
                if v_max.is_finite() && v_max > 0.0
                    && row.max_abs_alpha() > qp_cutoffs[j]
                {
                    let rhs = v_max * h;
                    let up = build_row_expr(&sigma, &row, slack, with_slacks, 1.0, rhs);
                    subject_to!(problem, up <= rhs);
                    let lo = build_row_expr(&sigma, &row, slack, with_slacks, -1.0, rhs);
                    subject_to!(problem, lo <= rhs);
                    counts.joint_v += 2;
                }
            }
        }
    }
    for i in 2..k_samples {
        for j in lock..N {
            let stencil = step_coeffs::<N>(deriv, j, &bins, i, 2);
            if let Some(row) = stencil {
                let a_max = constraints.joint.a_max.0[j];
                if a_max.is_finite() && a_max > 0.0
                    && row.max_abs_alpha() > qp_cutoffs[j]
                {
                    let rhs = a_max * h2;
                    let up = build_row_expr(&sigma, &row, slack, with_slacks, 1.0, rhs);
                    subject_to!(problem, up <= rhs);
                    let lo = build_row_expr(&sigma, &row, slack, with_slacks, -1.0, rhs);
                    subject_to!(problem, lo <= rhs);
                    counts.joint_a += 2;
                }
            }
        }
    }
    for i in 3..k_samples {
        for j in lock..N {
            let stencil = step_coeffs::<N>(deriv, j, &bins, i, 3);
            if let Some(row) = stencil {
                let j_max = constraints.joint.j_max.0[j];
                if j_max.is_finite() && j_max > 0.0
                    && row.max_abs_alpha() > qp_cutoffs[j]
                {
                    let rhs = j_max * h3;
                    let up = build_row_expr(&sigma, &row, slack, with_slacks, 1.0, rhs);
                    subject_to!(problem, up <= rhs);
                    let lo = build_row_expr(&sigma, &row, slack, with_slacks, -1.0, rhs);
                    subject_to!(problem, lo <= rhs);
                    counts.joint_j += 2;
                }
            }
        }
    }

    // TCP V/A/J rows. Replaces per-joint secant_j with the per-segment
    // numerical chord-tangent bound. The stencil is built across the bins
    // spanned by samples (i-order..=i); we use the max bound over those bins.
    if let Some(tcp) = constraints.tcp
        && deriv.has_tcp()
    {
        let mut max_tan_sq = 0.0_f64;
        for &v in &deriv.chord_tcp_tangent_max_sq {
            if v > max_tan_sq {
                max_tan_sq = v;
            }
        }
        let tan_cutoff_sq = 1e-12 * max_tan_sq;

        for i in 1..k_samples {
            if let Some(row) = tcp_stencil_coeffs(deriv, &bins, i, 1, tan_cutoff_sq) {
                if tcp.v_max.is_finite() && tcp.v_max > 0.0 {
                    let rhs = tcp.v_max * h;
                    let up = build_row_expr(&sigma, &row, slack, with_slacks, 1.0, rhs);
                    subject_to!(problem, up <= rhs);
                    let lo = build_row_expr(&sigma, &row, slack, with_slacks, -1.0, rhs);
                    subject_to!(problem, lo <= rhs);
                    counts.tcp_v += 2;
                }
            }
        }
        for i in 2..k_samples {
            if let Some(row) = tcp_stencil_coeffs(deriv, &bins, i, 2, tan_cutoff_sq) {
                if tcp.a_max.is_finite() && tcp.a_max > 0.0 {
                    let rhs = tcp.a_max * h2;
                    let up = build_row_expr(&sigma, &row, slack, with_slacks, 1.0, rhs);
                    subject_to!(problem, up <= rhs);
                    let lo = build_row_expr(&sigma, &row, slack, with_slacks, -1.0, rhs);
                    subject_to!(problem, lo <= rhs);
                    counts.tcp_a += 2;
                }
            }
        }
        for i in 3..k_samples {
            if let Some(row) = tcp_stencil_coeffs(deriv, &bins, i, 3, tan_cutoff_sq) {
                if tcp.j_max.is_finite() && tcp.j_max > 0.0 {
                    let rhs = tcp.j_max * h3;
                    let up = build_row_expr(&sigma, &row, slack, with_slacks, 1.0, rhs);
                    subject_to!(problem, up <= rhs);
                    let lo = build_row_expr(&sigma, &row, slack, with_slacks, -1.0, rhs);
                    subject_to!(problem, lo <= rhs);
                    counts.tcp_j += 2;
                }
            }
        }
    }

    // Boundary V/A: pin the leading/trailing Δ¹/Δ² expressions to the
    // projected start/end values. This is correct for both rest-to-rest
    // (v_start=0 ⇒ first Δ¹ ≡ 0 ⇒ σ[1]=σ[0]=0) and non-rest cases.
    //
    // Use the projected scalar `start.sd` / `start.sdd`: those are the path-
    // tangent components of the requested joint-space boundary velocity and
    // acceleration. Map them onto an equivalent constraint on the FD row in
    // the dominant-secant joint (the joint with the largest |secant| at the
    // first/last segment), expressed as an exact equality on the σ chain.
    //
    // Use slack-relaxed equalities (within `solver.boundary_slack`) to keep
    // the IPM line search away from cone-tip pathologies at rest-to-rest.
    let bslack = constraints.solver.boundary_slack.max(0.0);
    // Pick the dominant-secant joint at each end and pin Δ¹/Δ² there.
    for side in [BoundarySide::Start, BoundarySide::End] {
        let seg_idx = match side {
            BoundarySide::Start => 0,
            BoundarySide::End => deriv.num_segments() - 1,
        };
        let mut j_star = lock;
        let mut sec_star = 0.0_f64;
        for j in lock..N {
            let s = secant_j(deriv, seg_idx, j).abs();
            if s > sec_star {
                sec_star = s;
                j_star = j;
            }
        }
        if !(sec_star > qp_cutoffs[j_star]) {
            continue;
        }
        let (sd_t, sdd_t) = match side {
            BoundarySide::Start => (start.sd, start.sdd),
            BoundarySide::End => (end.sd, end.sdd),
        };
        let sec_signed = secant_j(deriv, seg_idx, j_star);
        let target_v = sd_t * sec_signed * h;
        let target_a = sdd_t * sec_signed * h2;
        let (i_v, i_a) = match side {
            BoundarySide::Start => (1, 2),
            BoundarySide::End => (k_samples - 1, k_samples - 1),
        };
        if let Some(row) = step_coeffs::<N>(deriv, j_star, &bins, i_v, 1) {
            // Σ alpha · σ + beta − target_v ∈ [−bslack, +bslack]
            let mut e = Variable::constant_in(&arena, row.beta - target_v);
            for (t, &a) in row.coeffs.iter().enumerate() {
                if a != 0.0 {
                    e = e + a * sigma[row.i - t];
                }
            }
            if bslack <= 0.0 {
                subject_to!(problem, e == 0.0);
            } else {
                let up = e;
                subject_to!(problem, up <= bslack);
                let lo = -e;
                subject_to!(problem, lo <= bslack);
            }
        }
        if i_a < k_samples && let Some(row) = step_coeffs::<N>(deriv, j_star, &bins, i_a, 2) {
            let mut e = Variable::constant_in(&arena, row.beta - target_a);
            for (t, &a) in row.coeffs.iter().enumerate() {
                if a != 0.0 {
                    e = e + a * sigma[row.i - t];
                }
            }
            if bslack <= 0.0 {
                subject_to!(problem, e == 0.0);
            } else {
                let up = e;
                subject_to!(problem, up <= bslack);
                let lo = -e;
                subject_to!(problem, lo <= bslack);
            }
        }
    }

    // Objective: Σ_{i=1..K-1} (σ[i] − target_i)² + penalty · slack
    //   target_i = i · S / (K−1)
    //
    // The squared-deviation term regularizes the σ profile toward uniform
    // progress along the path; combined with monotonicity and the FD bounds
    // it gives the IPM a well-conditioned descent direction even on
    // pure-feasibility problems.
    let mut obj: Variable = Variable::constant_in(&arena, 0.0);
    for i in 1..k_samples - 1 {
        let target = (i as f64) * s_total / ((k_samples - 1) as f64);
        let dev = sigma[i] - target;
        obj = obj + dev * dev;
    }
    if with_slacks {
        let pen = constraints.solver.bisection_slack_penalty.max(0.0);
        obj = obj + pen * slack;
    }
    problem.minimize(obj);

    // Initial guess.
    if let Some(ws) = warm_sigma {
        let kp = ws.len();
        for i in 0..k_samples {
            let u = (i as f64) * ((kp - 1) as f64) / ((k_samples - 1).max(1) as f64);
            let lo = (u.floor() as usize).min(kp - 2);
            let f = (u - lo as f64).clamp(0.0, 1.0);
            sigma[i].set_value(ws[lo] + f * (ws[lo + 1] - ws[lo]));
        }
    } else {
        for i in 0..k_samples {
            let u = (i as f64) / ((k_samples - 1) as f64);
            sigma[i].set_value(u * s_total);
        }
    }
    slack.set_value(0.0);

    let build_time = build_start.elapsed();

    let iter_counter = Arc::new(AtomicI32::new(0));
    let ic = iter_counter.clone();
    let effective_timeout = timeout_override.or(constraints.solver.timeout);
    let cb_deadline: Option<Instant> = effective_timeout.map(|t| Instant::now() + t);
    problem.add_callback(move |_info| {
        ic.fetch_add(1, Ordering::Relaxed);
        // Return `true` to request the IPM to stop. Sleipnir's
        // `options.timeout` only fires at the *start* of each outer iter and
        // never preempts the slow feasibility-restoration sub-iters, so we
        // also check the deadline here on every callback hit.
        if let Some(d) = cb_deadline
            && Instant::now() >= d
        {
            return true;
        }
        false
    });

    let mut options = Options::default()
        .tolerance(constraints.solver.tolerance)
        .max_iterations(constraints.solver.max_iterations)
        .diagnostics(constraints.solver.diagnostics);
    if let Some(t) = effective_timeout {
        options = options.timeout(t);
    }

    let t0 = Instant::now();
    let status_raw = problem.solve_status(options);
    let solve_time = t0.elapsed();
    let status = SolveStatus::from(status_raw);
    let iterations = iter_counter.load(Ordering::Relaxed);

    let sigma_vals: Vec<f64> = sigma.iter().map(|v| v.value()).collect();
    let slack_val = slack.value();

    Ok(DiscreteSolution {
        sigma: sigma_vals,
        k: k_samples,
        status,
        iterations,
        solve_time,
        build_time,
        slack: slack_val,
        constraint_counts: counts,
        bins_used: bins,
    })
}

/// Re-compute the segment bin for each `σ[i]` value. Returns `bin[i] = b` such
/// that `s[b] ≤ σ[i] < s[b+1]` (clamped to `[0, num_segments-1]`).
pub fn bins_from_sigma(s: &[f64], sigma: &[f64]) -> Vec<usize> {
    let num_segs = s.len() - 1;
    let mut out = Vec::with_capacity(sigma.len());
    let mut b = 0usize;
    for &sig in sigma {
        let sc = sig.clamp(0.0, *s.last().unwrap());
        while b + 1 < num_segs && s[b + 1] <= sc {
            b += 1;
        }
        while b > 0 && s[b] > sc {
            b -= 1;
        }
        out.push(b);
    }
    out
}

// ── stencil coefficient builders ─────────────────────────────────────────────

/// Linear coefficients for one `Δᵐ q[i]_j ≤ rhs` row, in the unified form
/// `Σ_t α[t] · sigma[i − t] + beta`.
#[derive(Debug, Clone)]
pub(crate) struct StencilRow {
    /// `coeffs[t]` multiplies `sigma[i − t]` for `t = 0..order+1`.
    pub coeffs: Vec<f64>,
    pub beta: f64,
    pub i: usize,
}

impl StencilRow {
    pub fn max_abs_alpha(&self) -> f64 {
        self.coeffs.iter().map(|c| c.abs()).fold(0.0_f64, f64::max)
    }
}

/// Builds a per-joint stencil row for sample `i`, kinematic order `m ∈ {1,2,3}`.
pub(crate) fn step_coeffs<const N: usize>(
    deriv: &PathDerivatives<N>,
    j: usize,
    bins: &[usize],
    i: usize,
    order: usize,
) -> Option<StencilRow> {
    if i < order {
        return None;
    }
    // Build Δ¹ at i, i-1, ..., i-order+1, then combine.
    // We use the standard centered-coefficient recurrence:
    //   Δ^m q[i] = Σ_{k=0..m} (-1)^k · C(m, k) · q[i − k]
    // We build it by composing one-step differences: Δ^m = Δ^(m-1) ∘ Δ¹.
    let mut alpha = vec![0.0_f64; order + 1];
    let mut beta = 0.0_f64;
    // The signed combination for Δ^m q[i] = Σ_k (-1)^k C(m,k) q[i-k].
    // We expand each q[i-k] in terms of σ using the (b_{lo}, b_{hi})=(bin[i-k-1], bin[i-k])
    // step1 form? No — q[i-k] alone (not Δ¹) is a chord-linear function of σ[i-k] in
    // its own bin: q[i-k]_j = secant_j[bin[i-k]] · sigma[i-k] + offset_k.
    //
    // That's simpler. q[i-k]_j = q_seg[bin[i-k]]_j + secant_j[bin[i-k]] · (σ[i-k] − s[bin[i-k]]).
    // So α[k] = (-1)^k · C(m, k) · secant_j[bin[i-k]]; β = Σ_k (-1)^k · C(m,k) · (q_seg[bin[i-k]]_j − secant_j[bin[i-k]] · s[bin[i-k]]).
    let mut comb = 1.0_f64; // C(m, 0)
    for k in 0..=order {
        let idx = i - k;
        let b = bins[idx].min(deriv.num_segments() - 1);
        let sec = secant_j(deriv, b, j);
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let factor = sign * comb;
        alpha[k] = factor * sec;
        beta += factor * (deriv.waypoints[b].0[j] - sec * deriv.s[b]);
        // Update comb for next k: C(m, k+1) = C(m, k) · (m − k) / (k + 1)
        if k < order {
            comb = comb * ((order - k) as f64) / ((k + 1) as f64);
        }
    }
    Some(StencilRow { coeffs: alpha, beta, i })
}

/// TCP stencil row: same Δᵐ recurrence but with `secant_j` replaced by the
/// per-segment numerical chord-tangent magnitude bound. Returns `None` when
/// every spanned segment's bound is below the noise cutoff.
pub(crate) fn tcp_stencil_coeffs<const N: usize>(
    deriv: &PathDerivatives<N>,
    bins: &[usize],
    i: usize,
    order: usize,
    cutoff_sq: f64,
) -> Option<StencilRow> {
    if i < order || deriv.chord_tcp_tangent_max_sq.is_empty() {
        return None;
    }
    let mut max_tan = 0.0_f64;
    for k in 0..=order {
        let b = bins[i - k].min(deriv.num_segments() - 1);
        let t_sq = deriv.chord_tcp_tangent_max_sq[b];
        if t_sq > max_tan {
            max_tan = t_sq;
        }
    }
    if max_tan <= cutoff_sq {
        return None;
    }
    let tan = max_tan.sqrt();
    // Apply the Δᵐ recurrence with `sec = tan` (same across the stencil — this
    // is a deliberate upper bound).
    let mut alpha = vec![0.0_f64; order + 1];
    let mut beta = 0.0_f64;
    let mut comb = 1.0_f64;
    for k in 0..=order {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let factor = sign * comb;
        alpha[k] = factor * tan;
        // For the TCP magnitude bound we treat the path as having a constant
        // tangent magnitude `tan` across the stencil; the constant offset
        // collapses to zero because Σ_k (-1)^k C(m, k) = 0 for m ≥ 1.
        beta += 0.0;
        if k < order {
            comb = comb * ((order - k) as f64) / ((k + 1) as f64);
        }
    }
    Some(StencilRow { coeffs: alpha, beta, i })
}

// ── row emitters ─────────────────────────────────────────────────────────────

fn build_row_expr<'a>(
    sigma: &[Variable<'a>],
    row: &StencilRow,
    slack: Variable<'a>,
    with_slacks: bool,
    sign: f64,
    rhs: f64,
) -> Variable<'a> {
    // With `with_slacks`, the row reads `expr - rhs·slack ≤ rhs`, i.e. the
    // global `slack` is in *relative* units — the per-row infeasibility
    // magnitude is `slack` regardless of how small `rhs` is. This matters
    // when constraint scales differ wildly (e.g. `v_max·h` ≈ 1e-2 vs
    // `j_max·h³` ≈ 1e-6); an absolute slack of `tol = 1e-6` would translate
    // to ~50% relative overshoot on the tightest jerk rows.
    let mut expr = Variable::constant_in(sigma[0].arena(), sign * row.beta);
    for (t, &a) in row.coeffs.iter().enumerate() {
        if a != 0.0 {
            expr = expr + (sign * a) * sigma[row.i - t];
        }
    }
    if with_slacks {
        expr = expr - rhs * slack;
    }
    expr
}

// ── boundary pinning ─────────────────────────────────────────────────────────

#[derive(Copy, Clone)]
enum BoundarySide {
    Start,
    End,
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Computes the binning for `K` output samples against the densified-segment
/// boundaries `s[0..m]`. Returns `bin[i] = b` such that
/// `s[b] ≤ sigma_guess[i] < s[b+1]`.
pub(crate) fn proportional_bins(s: &[f64], k_samples: usize) -> Vec<usize> {
    let m = s.len();
    debug_assert!(m >= 2 && k_samples >= 2);
    let total = *s.last().unwrap();
    let mut bins = Vec::with_capacity(k_samples);
    let mut b = 0usize;
    for i in 0..k_samples {
        let target = (i as f64) * total / ((k_samples - 1) as f64);
        while b + 1 < m - 1 && s[b + 1] <= target {
            b += 1;
        }
        bins.push(b.min(m - 2));
    }
    bins
}

/// Closed-form 1-step coefficients (kept for unit tests; the production
/// builder uses [`step_coeffs`] which handles all orders uniformly).
pub(crate) fn step1_coeffs<const N: usize>(
    deriv: &PathDerivatives<N>,
    j: usize,
    b_lo: usize,
    b_hi: usize,
) -> (f64, f64, f64) {
    if b_hi == b_lo {
        let sec = secant_j(deriv, b_lo, j);
        return (sec, -sec, 0.0);
    }
    let sec_lo = secant_j(deriv, b_lo, j);
    let sec_hi = secant_j(deriv, b_hi, j);
    let mut beta = sec_lo * deriv.s[b_lo + 1] - sec_hi * deriv.s[b_hi];
    for m in (b_lo + 1)..b_hi {
        beta += secant_j(deriv, m, j) * deriv.ds[m];
    }
    (sec_hi, -sec_lo, beta)
}

fn secant_j<const N: usize>(deriv: &PathDerivatives<N>, b: usize, j: usize) -> f64 {
    let a = deriv.waypoints[b].0[j];
    let bv = deriv.waypoints[b + 1].0[j];
    (bv - a) / deriv.ds[b]
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use deke_types::SRobotPath;

    fn deriv_from_path<const N: usize>(path: &SRobotPath<N, f64>) -> PathDerivatives<N> {
        PathDerivatives::<N>::new_without_tcp(path).unwrap()
    }

    #[test]
    fn step1_same_segment_is_secant_times_dsigma() {
        let wps = vec![
            deke_types::SRobotQ::from_array([0.0_f64]),
            deke_types::SRobotQ::from_array([1.0_f64]),
            deke_types::SRobotQ::from_array([3.0_f64]),
        ];
        let path = SRobotPath::try_new(wps).unwrap();
        let d = deriv_from_path::<1>(&path);
        let (cc, cp, beta) = step1_coeffs::<1>(&d, 0, 0, 0);
        assert_eq!(beta, 0.0);
        assert!((cc - 1.0).abs() < 1e-12);
        assert!((cp + 1.0).abs() < 1e-12);
    }

    #[test]
    fn step1_one_boundary_cross_is_piecewise_linear() {
        let wps = vec![
            deke_types::SRobotQ::from_array([0.0_f64, 0.0]),
            deke_types::SRobotQ::from_array([1.0_f64, 0.0]),
            deke_types::SRobotQ::from_array([1.0_f64, 1.0]),
        ];
        let path = SRobotPath::try_new(wps).unwrap();
        let d = deriv_from_path::<2>(&path);
        let (cc, cp, beta) = step1_coeffs::<2>(&d, 0, 0, 1);
        assert!(cc.abs() < 1e-12, "cc={}", cc);
        assert!((cp + 1.0).abs() < 1e-12, "cp={}", cp);
        assert!((beta - 1.0).abs() < 1e-12, "beta={}", beta);
        let (cc, cp, beta) = step1_coeffs::<2>(&d, 1, 0, 1);
        assert!((cc - 1.0).abs() < 1e-12, "cc={}", cc);
        assert!(cp.abs() < 1e-12, "cp={}", cp);
        assert!((beta + 1.0).abs() < 1e-12, "beta={}", beta);
    }

    #[test]
    fn proportional_bins_monotone() {
        let s = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let bins = proportional_bins(&s, 5);
        assert_eq!(bins[0], 0);
        assert_eq!(*bins.last().unwrap(), 3);
        for w in bins.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn step_coeffs_order1_matches_step1() {
        // Recurrence-based step_coeffs (using per-sample-bin q-expansion) must
        // match the explicit step1_coeffs formula on the same bins.
        let wps = vec![
            deke_types::SRobotQ::from_array([0.0_f64, 0.0]),
            deke_types::SRobotQ::from_array([1.0_f64, 0.0]),
            deke_types::SRobotQ::from_array([1.0_f64, 1.0]),
        ];
        let path = SRobotPath::try_new(wps).unwrap();
        let d = deriv_from_path::<2>(&path);
        // Manually fix bins so sample 0 is in segment 0 and sample 1 in segment 1.
        let bins = vec![0_usize, 1];
        let r = step_coeffs::<2>(&d, 0, &bins, 1, 1).unwrap();
        // Same expectations as the explicit form for joint 0 across the L-corner.
        // The recurrence form expands each q in its own bin, so the σ[i-1] coefficient
        // is -secant_j[bin[i-1]] = -1, σ[i] coefficient is secant_j[bin[i]] = 0,
        // beta = (q_seg[1]_0 - secant·s[1]) - (q_seg[0]_0 - secant·s[0])
        //      = (1 - 0·1) - (0 - 1·0) = 1.
        assert!((r.coeffs[0] - 0.0).abs() < 1e-12, "α0={}", r.coeffs[0]);
        assert!((r.coeffs[1] - (-1.0)).abs() < 1e-12, "α1={}", r.coeffs[1]);
        assert!((r.beta - 1.0).abs() < 1e-12, "β={}", r.beta);
    }

    #[test]
    fn step_coeffs_order2_summed_alpha_is_zero_on_uniform_path() {
        // On a uniform-secant path the Δ² recurrence should give Σ α = 0 (constant
        // function in σ ⇒ Δ² ≡ 0).
        let wps = (0..5)
            .map(|i| deke_types::SRobotQ::from_array([i as f64]))
            .collect();
        let path = SRobotPath::try_new(wps).unwrap();
        let d = deriv_from_path::<1>(&path);
        let bins = vec![0, 1, 2, 3, 3];
        let r = step_coeffs::<1>(&d, 0, &bins, 3, 2).unwrap();
        let sum: f64 = r.coeffs.iter().sum();
        // Σ α[k] = secant · Σ (-1)^k C(2,k) = secant · (1 − 2 + 1) = 0.
        assert!(sum.abs() < 1e-12, "Σα={}", sum);
    }
}
