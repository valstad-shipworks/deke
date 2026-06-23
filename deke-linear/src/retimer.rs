//! Stage C — time-parameterise a joint path at a commanded constant TCP speed.
//!
//! This is a CNC-style constant-feedrate planner. The timing is found by a
//! **discrete convex program**: the variable is the cumulative arc length
//! `σ[k]` at each output tick `k·dt`, and the per-joint velocity/acceleration/
//! jerk limits are written directly as bounds on the *finite differences* of the
//! emitted joint samples — the exact quantities a downstream controller
//! reconstructs. So the limits are honoured **by construction** rather than by a
//! continuous bound plus a margin, and the solver never touches a fragile
//! joint-space third derivative.
//!
//! The joint path is first smoothed (a cubic spline, densely resampled) so the
//! chord-linear interpolation the solver times is C²-smooth at the tick scale;
//! `σ` is solved with a small banded LP (Clarabel); the result is reconstructed,
//! and a final finite-difference check against the *true* limits is the airtight
//! backstop. A path that physically cannot fit under the limits fails.

use std::time::Duration;

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use deke_types::glam::DVec3;
use deke_types::{
    ContinuousFKChain, DekeError, DekeResult, Retimer, SRobotPath, SRobotQ, SRobotTraj, Validator,
};

use crate::constraints::{JointLimits, LinearConstraints};
use crate::diagnostic::LinearRetimerDiagnostic;
use crate::error::LinearError;

/// The per-tick limit caps are planned at `margin·limit` so the small cross-bin
/// leak of the chord-linear reconstruction stays under the true limit; the final
/// finite-difference verify against the *true* limits is the airtight backstop.
const LIMIT_MARGIN: f64 = 0.97;

/// Finest arc-length spacing (metres) the smoothing spline is resampled at, so
/// the chord-linear path the solver times is C²-smooth at the output-tick scale
/// and its FK arc length tracks the commanded Cartesian arc closely.
const SMOOTH_STEP: f64 = 1e-4;

/// A converged solve: the arc-length profile `σ`, the live-sample count, the
/// reconstructed joint samples, and the realized TCP accel/jerk overshoot ratios.
type SolvedProfile<const N: usize> = (Vec<f64>, usize, Vec<SRobotQ<N, f64>>, f64, f64);

/// Constant-feedrate, jerk-limited retimer over a joint path.
#[derive(Clone, Debug)]
pub struct ConstantSpeedRetimer<'a, const N: usize, FK> {
    fk: &'a FK,
}

impl<'a, const N: usize, FK> ConstantSpeedRetimer<'a, N, FK>
where
    FK: ContinuousFKChain<N, f64>,
{
    pub fn new(fk: &'a FK) -> Self {
        Self { fk }
    }

    pub(crate) fn retime_path(
        &self,
        c: &LinearConstraints<N>,
        path: &SRobotPath<N, f64>,
        run_idx: usize,
    ) -> Result<(SRobotTraj<N, f64>, LinearRetimerDiagnostic), LinearError> {
        let raw: Vec<SRobotQ<N, f64>> = path.iter().copied().collect();
        let dt = c.output_dt.as_secs_f64().max(1e-6);

        // Smooth, densely-resampled joint path + its Cartesian arc length. The
        // solver works on the chord-linear interpolation of these knots, so
        // smoothing keeps the secant changes between knots tiny (small cross-bin
        // leak) and removes the IK jitter that would otherwise spike the jerk.
        let (knots, s) = smooth_path(self.fk, &raw, c.corner_smoothing)?;
        let nb = knots.len();
        let total = if nb == 0 { 0.0 } else { s[nb - 1] };
        if nb < 2 || total < 1e-9 {
            let traj = SRobotTraj::new(c.output_dt, path.clone());
            return Ok((traj, degenerate_diag(nb, dt, total, c.tcp.speed)));
        }

        // Per-segment secant slope dq/ds — the chord-linear path derivative. The
        // joint velocity/accel/jerk of the output are exactly `secant·Δᵐσ/dtᵐ`
        // within a segment, so capping the σ-differences bounds the per-joint
        // finite differences the consumer measures, with no path derivatives.
        let secant: Vec<[f64; N]> = (0..nb - 1)
            .map(|b| {
                let ds = (s[b + 1] - s[b]).max(1e-12);
                std::array::from_fn(|j| (knots[b + 1].0[j] - knots[b].0[j]) / ds)
            })
            .collect();

        let v_cmd = c.tcp.speed.max(1e-9);
        let bin_of = |sx: f64| -> usize {
            let sx = sx.clamp(0.0, total);
            let mut b = 0;
            while b + 1 < nb - 1 && s[b + 1] <= sx {
                b += 1;
            }
            b
        };
        // `min_j limit_j / |secant_j|` on a segment — the projected scalar limit.
        let proj = |b: usize, lim: &SRobotQ<N, f64>| -> f64 {
            (0..N)
                .map(|j| lim.0[j] / secant[b][j].abs().max(1e-12))
                .fold(f64::INFINITY, f64::min)
        };

        // Constant-speed contract: if the joint velocity limits force the
        // feasible speed below the command anywhere interior and the caller
        // forbids dips, fail loudly rather than slow down.
        if c.forbid_interior_dips {
            let mut worst: Option<(usize, f64)> = None;
            #[allow(clippy::needless_range_loop)]
            for b in 1..nb - 1 {
                let g = proj(b, &c.joint.v_max);
                if g < v_cmd * (1.0 - 1e-3) && worst.is_none_or(|(_, gw)| g < gw) {
                    worst = Some((b, g));
                }
            }
            if let Some((b, g)) = worst {
                return Err(LinearError::SpeedDipRequired {
                    run: run_idx,
                    s: s[b],
                    feasible_speed: g,
                    commanded: v_cmd,
                });
            }
        }

        // Generous tick count: cruise time + a few jerk-limited ramp lengths,
        // using the *effective* accel/jerk floor (joint projection intersected
        // with the TCP caps, which can be far tighter and lengthen the ramps).
        let mg = LIMIT_MARGIN;
        let a_eff = (0..nb - 1)
            .map(|b| proj(b, &c.joint.a_max))
            .fold(f64::INFINITY, f64::min)
            .min(c.tcp.accel.unwrap_or(f64::INFINITY))
            .max(1e-6);
        let j_eff = (0..nb - 1)
            .map(|b| proj(b, &c.joint.j_max))
            .fold(f64::INFINITY, f64::min)
            .min(c.tcp.jerk.unwrap_or(f64::INFINITY))
            .max(1e-6);
        let ramp_t = v_cmd / a_eff + a_eff / j_eff;
        let mut kk = ((total / (v_cmd * dt)).ceil() as usize + (4.0 * ramp_t / dt) as usize + 64).max(8);

        // The TCP tangential accel/jerk a controller measures is the finite
        // difference of the FK Cartesian speed, which the chord-linear
        // reconstruction over-reads vs the arc-length `Δσ` the program bounds (the
        // third difference — jerk — most). Rather than guess a fixed derate, plan
        // the TCP caps at the joint margin, measure the *realized* FK-FD overshoot,
        // and tighten each TCP quantity by exactly its overshoot before re-solving.
        // This converges in 1–2 passes, derates only what actually over-reads (so
        // accel isn't throttled needlessly), and is robust to path/robot curvature.
        let mut a_der = mg;
        let mut j_der = mg;
        let mut solved: Option<SolvedProfile<N>> = None;
        for _tcp_iter in 0..5 {
            // Solve, growing the horizon if the program is infeasible (a too-small
            // tick count, distinct from a path that genuinely can't be timed). The
            // chord-linear coefficients depend on which segment each σ[k] lands in,
            // so re-bin and re-solve to a fixed point (stable in 1–3 passes).
            let mut sigma: Option<Vec<f64>> = None;
            for _grow in 0..6 {
                let mut sg: Vec<f64> = (0..kk).map(|k| k as f64 / (kk - 1) as f64 * total).collect();
                let mut prev_bins: Vec<usize> = Vec::new();
                let mut feasible = true;
                for _pass in 0..4 {
                    let bins: Vec<usize> = sg.iter().map(|&sx| bin_of(sx)).collect();
                    if bins == prev_bins {
                        break;
                    }
                    let cap_v: Vec<f64> = bins
                        .iter()
                        .map(|&b| (mg * proj(b, &c.joint.v_max)).min(v_cmd) * dt)
                        .collect();
                    let cap_a: Vec<f64> = bins
                        .iter()
                        .map(|&b| (mg * proj(b, &c.joint.a_max)).min(c.tcp.accel.map_or(f64::INFINITY, |t| a_der * t)) * dt * dt)
                        .collect();
                    let cap_j: Vec<f64> = bins
                        .iter()
                        .map(|&b| (mg * proj(b, &c.joint.j_max)).min(c.tcp.jerk.map_or(f64::INFINITY, |t| j_der * t)) * dt * dt * dt)
                        .collect();
                    match solve_sigma(kk, total, &cap_v, &cap_a, &cap_j) {
                        Some(next) => sg = next,
                        None => {
                            feasible = false;
                            break;
                        }
                    }
                    prev_bins = bins;
                }
                if feasible {
                    sigma = Some(sg);
                    break;
                }
                kk = (kk as f64 * 1.6) as usize + 16;
            }
            let sigma = sigma.ok_or(LinearError::Stalled { run: run_idx, s: 0.0 })?;

            // Reconstruct chord-linear joint samples; trim the trailing stationary
            // tail (ticks parked at `total` after the motion finished).
            let recon = |sx: f64| -> SRobotQ<N, f64> {
                let b = bin_of(sx);
                SRobotQ(std::array::from_fn(|j| knots[b].0[j] + secant[b][j] * (sx - s[b])))
            };
            let mut end = kk;
            while end > 2 && (total - sigma[end - 2]).abs() < 1e-9 {
                end -= 1;
            }
            let samples: Vec<SRobotQ<N, f64>> = sigma[..end].iter().map(|&sx| recon(sx)).collect();

            // Measure the realized FK TCP accel/jerk overshoot (FK finite
            // difference ÷ true cap; `0` when a cap is unset). `1.0` is exactly at.
            let (a_over, j_over) = tcp_fd_ratios(self.fk, &samples, c.tcp.accel, c.tcp.jerk, dt)?;
            let tol = 1.0 + 1e-6;
            let converged = a_over <= tol && j_over <= tol;
            solved = Some((sigma, end, samples, a_over, j_over));
            if converged {
                break;
            }
            // Tighten only the quantity that over-reads, by its overshoot plus a
            // small safety factor for the residual nonlinearity, then re-solve.
            if a_over > tol {
                a_der /= a_over * 1.01;
            }
            if j_over > tol {
                j_der /= j_over * 1.01;
            }
        }
        let (sigma, end, samples, a_over, j_over) =
            solved.ok_or(LinearError::Stalled { run: run_idx, s: 0.0 })?;

        // Airtight backstop: the *finite differences* of the emitted samples
        // against the *true* (un-margined) per-joint limits. If any exceeds, the
        // path can't be timed under the limits at this speed — fail.
        if let Some((kind, joint, value, limit, idx)) = verify_fd(&samples, &c.joint, dt) {
            return Err(LinearError::LimitExceeded {
                run: run_idx,
                s: sigma.get(idx).copied().unwrap_or(total),
                joint,
                kind,
                value,
                limit,
            });
        }
        // Same backstop for the TCP caps: if the adaptation above could not bring
        // the realized FK accel/jerk under the true cap, fail rather than emit a
        // trajectory that exceeds it.
        if a_over > 1.0 + 1e-6 {
            let limit = c.tcp.accel.expect("accel cap set when its ratio is positive");
            return Err(LinearError::TcpLimitExceeded {
                run: run_idx,
                kind: "acceleration",
                value: a_over * limit,
                limit,
            });
        }
        if j_over > 1.0 + 1e-6 {
            let limit = c.tcp.jerk.expect("jerk cap set when its ratio is positive");
            return Err(LinearError::TcpLimitExceeded {
                run: run_idx,
                kind: "jerk",
                value: j_over * limit,
                limit,
            });
        }

        let out_samples = samples.len();
        let (_, pk_a, pk_j) = fd_peaks(&samples, dt);
        let peak_speed = (1..end)
            .map(|k| (sigma[k] - sigma[k - 1]) / dt)
            .fold(0.0, f64::max);
        let path_out = SRobotPath::try_new(samples).map_err(LinearError::from)?;
        let traj = SRobotTraj::new(c.output_dt, path_out);
        Ok((
            traj,
            LinearRetimerDiagnostic {
                output_samples: out_samples,
                duration: Duration::from_secs_f64((out_samples.saturating_sub(1)) as f64 * dt),
                arc_length: total,
                commanded_speed: c.tcp.speed,
                peak_speed,
                peak_joint_accel: pk_a,
                peak_joint_jerk: pk_j,
            },
        ))
    }
}

impl<'a, const N: usize, FK> Retimer<N, f64> for ConstantSpeedRetimer<'a, N, FK>
where
    FK: ContinuousFKChain<N, f64>,
{
    type Diagnostic = LinearRetimerDiagnostic;
    type Constraints = LinearConstraints<N>;

    fn retime<V: Validator<N, (), f64>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, f64>,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, f64>>, Self::Diagnostic) {
        match self.retime_path(constraints, path, 0) {
            Ok((traj, diag)) => {
                let samples: Vec<SRobotQ<N, f64>> = traj.path().iter().copied().collect();
                if let Err(e) = validator.validate_motion(&samples, ctx) {
                    return (Err(e), diag);
                }
                (Ok(traj), diag)
            }
            Err(e) => (
                Err(e.into()),
                LinearRetimerDiagnostic {
                    output_samples: 0,
                    duration: Duration::ZERO,
                    arc_length: 0.0,
                    commanded_speed: constraints.tcp.speed,
                    peak_speed: 0.0,
                    peak_joint_accel: 0.0,
                    peak_joint_jerk: 0.0,
                },
            ),
        }
    }
}

fn degenerate_diag(nb: usize, dt: f64, total: f64, speed: f64) -> LinearRetimerDiagnostic {
    LinearRetimerDiagnostic {
        output_samples: nb,
        duration: Duration::from_secs_f64((nb.saturating_sub(1)) as f64 * dt),
        arc_length: total,
        commanded_speed: speed,
        peak_speed: 0.0,
        peak_joint_accel: 0.0,
        peak_joint_jerk: 0.0,
    }
}

/// Smooth and densely resample the joint path, returning the knots and their
/// cumulative Cartesian arc length. Coincident-arc knots are dropped first (a
/// zero-length chord makes the cubic singular). `None` (or too few knots) keeps
/// the raw deduped polyline.
fn smooth_path<const N: usize, FK: ContinuousFKChain<N, f64>>(
    fk: &FK,
    raw: &[SRobotQ<N, f64>],
    res: Option<f64>,
) -> Result<(Vec<SRobotQ<N, f64>>, Vec<f64>), LinearError> {
    let arc = |pts: &[SRobotQ<N, f64>]| -> Result<Vec<f64>, LinearError> {
        let pos: Vec<DVec3> = pts
            .iter()
            .map(|q| fk.fk_end(q).map(|t| t.translation))
            .collect::<Result<_, DekeError>>()?;
        let mut s = vec![0.0f64; pts.len()];
        for i in 1..pts.len() {
            s[i] = s[i - 1] + pos[i].distance(pos[i - 1]);
        }
        Ok(s)
    };
    let s_raw = arc(raw)?;
    let mut sd: Vec<f64> = Vec::with_capacity(raw.len());
    let mut qd: Vec<SRobotQ<N, f64>> = Vec::with_capacity(raw.len());
    for i in 0..raw.len() {
        if qd.is_empty() || s_raw[i] - sd[sd.len() - 1] > 1e-9 {
            sd.push(s_raw[i]);
            qd.push(raw[i]);
        }
    }
    let n = qd.len();
    let step = match res {
        Some(r) if r > 0.0 && n >= 3 => r.min(SMOOTH_STEP),
        _ => {
            let s = arc(&qd)?;
            return Ok((qd, s));
        }
    };
    let h: Vec<f64> = (0..n - 1).map(|i| (sd[i + 1] - sd[i]).max(1e-12)).collect();
    let mm = natural_cubic(&h, &qd);
    let mut out = vec![qd[0]];
    for i in 0..n - 1 {
        let k = ((h[i] / step).ceil() as usize).max(1);
        for ss in 1..=k {
            let uu = sd[i] + h[i] * (ss as f64) / (k as f64);
            let a = (sd[i + 1] - uu) / h[i];
            let b = (uu - sd[i]) / h[i];
            out.push(SRobotQ(std::array::from_fn(|j| {
                qd[i].0[j] * a
                    + qd[i + 1].0[j] * b
                    + (mm[i][j] * (a * a * a - a) + mm[i + 1][j] * (b * b * b - b)) * (h[i] * h[i] / 6.0)
            })));
        }
    }
    let s = arc(&out)?;
    Ok((out, s))
}

/// Natural-cubic-spline second derivatives per joint via the Thomas algorithm
/// (`M_0 = M_{n-1} = 0`); the scalar tridiagonal sweep is shared by every joint.
#[allow(clippy::needless_range_loop)]
fn natural_cubic<const N: usize>(h: &[f64], y: &[SRobotQ<N, f64>]) -> Vec<[f64; N]> {
    let n = y.len();
    let mut cp = vec![0.0f64; n];
    let mut dp = vec![[0.0f64; N]; n];
    for i in 1..n - 1 {
        let (a, b, cc) = (h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]);
        let denom = b - a * cp[i - 1];
        cp[i] = cc / denom;
        for j in 0..N {
            let rhs = ((y[i + 1].0[j] - y[i].0[j]) / h[i] - (y[i].0[j] - y[i - 1].0[j]) / h[i - 1]) * 6.0;
            dp[i][j] = (rhs - dp[i - 1][j] * a) / denom;
        }
    }
    let mut mm = vec![[0.0f64; N]; n];
    for i in (1..n - 1).rev() {
        for j in 0..N {
            mm[i][j] = dp[i][j] - mm[i + 1][j] * cp[i];
        }
    }
    mm
}

/// Solve `maximise Σσ` subject to `σ[0]=0`, `σ[n-1]=total`, rest (v=a=0) at both
/// ends, monotonic advance, and the per-tick first/second/third-difference caps.
/// Returns the σ profile, or `None` if the program is infeasible.
#[allow(clippy::needless_range_loop, clippy::field_reassign_with_default)]
fn solve_sigma(n: usize, total: f64, cap_v: &[f64], cap_a: &[f64], cap_j: &[f64]) -> Option<Vec<f64>> {
    // Non-dimensionalize: solve in σ̃ = σ/total ∈ [0,1] so the variables are O(1)
    // and the (tiny) per-tick difference caps are well-conditioned against them.
    let inv = 1.0 / total;
    let mut t: Vec<(usize, usize, f64)> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut row = 0usize;
    let eq = |t: &mut Vec<(usize, usize, f64)>, b: &mut Vec<f64>, row: &mut usize, e: &[(usize, f64)], rhs: f64| {
        for &(c, v) in e {
            t.push((*row, c, v));
        }
        b.push(rhs);
        *row += 1;
    };
    eq(&mut t, &mut b, &mut row, &[(0, 1.0)], 0.0);
    eq(&mut t, &mut b, &mut row, &[(n - 1, 1.0)], 1.0);
    eq(&mut t, &mut b, &mut row, &[(1, 1.0), (0, -1.0)], 0.0);
    eq(&mut t, &mut b, &mut row, &[(2, 1.0), (1, -1.0)], 0.0);
    eq(&mut t, &mut b, &mut row, &[(n - 1, 1.0), (n - 2, -1.0)], 0.0);
    eq(&mut t, &mut b, &mut row, &[(n - 2, 1.0), (n - 3, -1.0)], 0.0);
    let n_eq = row;
    for k in 1..n {
        t.push((row, k - 1, 1.0));
        t.push((row, k, -1.0));
        b.push(0.0);
        row += 1;
        t.push((row, k, 1.0));
        t.push((row, k - 1, -1.0));
        b.push(cap_v[k] * inv);
        row += 1;
    }
    for k in 2..n {
        for sgn in [1.0, -1.0] {
            t.push((row, k, sgn));
            t.push((row, k - 1, -2.0 * sgn));
            t.push((row, k - 2, sgn));
            b.push(cap_a[k] * inv);
            row += 1;
        }
    }
    for k in 3..n {
        for sgn in [1.0, -1.0] {
            t.push((row, k, sgn));
            t.push((row, k - 1, -3.0 * sgn));
            t.push((row, k - 2, 3.0 * sgn));
            t.push((row, k - 3, -sgn));
            b.push(cap_j[k] * inv);
            row += 1;
        }
    }
    let m = row;
    // Row-scale each inequality so its (tiny) cap RHS becomes ±1: the difference
    // caps span ~1e-6…1, and normalizing them to O(1) lets the interior-point
    // solver hit the tight ramp constraints in far fewer iterations.
    let mut scale = vec![1.0f64; m];
    for r in n_eq..m {
        if b[r].abs() > 1e-12 {
            scale[r] = 1.0 / b[r].abs();
        }
    }
    for entry in t.iter_mut() {
        entry.2 *= scale[entry.0];
    }
    for r in 0..m {
        b[r] *= scale[r];
    }
    let a = triplets_to_csc(m, n, t);
    let p = CscMatrix::new(n, n, vec![0; n + 1], vec![], vec![]);
    let q = vec![-1.0f64; n];
    let cones = [SupportedConeT::ZeroConeT(n_eq), SupportedConeT::NonnegativeConeT(m - n_eq)];
    let mut set = DefaultSettings::default();
    set.verbose = false;
    set.max_iter = 200;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, set).ok()?;
    solver.solve();
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            Some(solver.solution.x.iter().map(|x| x * total).collect())
        }
        _ => None,
    }
}

/// Build a column-compressed sparse matrix from `(row, col, val)` triplets.
fn triplets_to_csc(m: usize, n: usize, mut t: Vec<(usize, usize, f64)>) -> CscMatrix<f64> {
    t.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut colptr = vec![0usize; n + 1];
    let mut rowval = Vec::with_capacity(t.len());
    let mut nzval = Vec::with_capacity(t.len());
    for &(r, c, v) in &t {
        colptr[c + 1] += 1;
        rowval.push(r);
        nzval.push(v);
    }
    for c in 0..n {
        colptr[c + 1] += colptr[c];
    }
    CscMatrix::new(m, n, colptr, rowval, nzval)
}

/// Worst per-joint finite-difference violation of the *true* limits, or `None`
/// if every difference is within limit. Returns `(kind, joint, value, limit,
/// sample_index)`.
fn verify_fd<const N: usize>(
    q: &[SRobotQ<N, f64>],
    lim: &JointLimits<N>,
    dt: f64,
) -> Option<(&'static str, usize, f64, f64, usize)> {
    let n = q.len();
    let mut worst: Option<(f64, &'static str, usize, f64, f64, usize)> = None;
    let mut consider = |val: f64, limit: f64, kind: &'static str, j: usize, idx: usize| {
        if val > limit * (1.0 + 1e-6) {
            let r = val / limit;
            if worst.is_none_or(|w| r > w.0) {
                worst = Some((r, kind, j, val, limit, idx));
            }
        }
    };
    for i in 1..n {
        for j in 0..N {
            consider((q[i].0[j] - q[i - 1].0[j]).abs() / dt, lim.v_max.0[j], "velocity", j, i);
        }
    }
    for i in 2..n {
        for j in 0..N {
            let a = (q[i].0[j] - 2.0 * q[i - 1].0[j] + q[i - 2].0[j]).abs() / (dt * dt);
            consider(a, lim.a_max.0[j], "acceleration", j, i);
        }
    }
    for i in 3..n {
        for j in 0..N {
            let jk = (q[i].0[j] - 3.0 * q[i - 1].0[j] + 3.0 * q[i - 2].0[j] - q[i - 3].0[j]).abs() / (dt * dt * dt);
            consider(jk, lim.j_max.0[j], "jerk", j, i);
        }
    }
    worst.map(|(_, kind, j, val, limit, idx)| (kind, j, val, limit, idx))
}

/// Realized TCP tangential acceleration/jerk overshoot — the finite differences
/// of the FK Cartesian speed stream divided by their caps (`(accel, jerk)`, each
/// `0.0` when its cap is unset). `1.0` is exactly at the cap; `> 1` over it.
fn tcp_fd_ratios<const N: usize, FK: ContinuousFKChain<N, f64>>(
    fk: &FK,
    q: &[SRobotQ<N, f64>],
    accel_cap: Option<f64>,
    jerk_cap: Option<f64>,
    dt: f64,
) -> Result<(f64, f64), LinearError> {
    if (accel_cap.is_none() && jerk_cap.is_none()) || q.len() < 3 {
        return Ok((0.0, 0.0));
    }
    let pos: Vec<DVec3> = q
        .iter()
        .map(|qi| fk.fk_end(qi).map(|t| t.translation))
        .collect::<Result<_, DekeError>>()?;
    let sp: Vec<f64> = (0..pos.len() - 1).map(|i| pos[i + 1].distance(pos[i]) / dt).collect();
    let a_ratio = accel_cap.map_or(0.0, |a| {
        (1..sp.len()).fold(0.0f64, |w, i| w.max((sp[i] - sp[i - 1]).abs() / dt)) / a
    });
    let j_ratio = jerk_cap.map_or(0.0, |j| {
        (2..sp.len()).fold(0.0f64, |w, i| w.max((sp[i] - 2.0 * sp[i - 1] + sp[i - 2]).abs() / (dt * dt)))
            / j
    });
    Ok((a_ratio, j_ratio))
}

/// Peak per-joint finite-difference velocity/acceleration/jerk of the output.
fn fd_peaks<const N: usize>(q: &[SRobotQ<N, f64>], dt: f64) -> (f64, f64, f64) {
    let n = q.len();
    let (mut v, mut a, mut jk) = (0.0f64, 0.0f64, 0.0f64);
    for i in 1..n {
        for j in 0..N {
            v = v.max((q[i].0[j] - q[i - 1].0[j]).abs() / dt);
        }
    }
    for i in 2..n {
        for j in 0..N {
            a = a.max((q[i].0[j] - 2.0 * q[i - 1].0[j] + q[i - 2].0[j]).abs() / (dt * dt));
        }
    }
    for i in 3..n {
        for j in 0..N {
            jk = jk.max((q[i].0[j] - 3.0 * q[i - 1].0[j] + 3.0 * q[i - 2].0[j] - q[i - 3].0[j]).abs() / (dt * dt * dt));
        }
    }
    (v, a, jk)
}
