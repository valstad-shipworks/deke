use std::time::Duration;

use deke_types::glam::DVec3;
use deke_types::{
    ContinuousFKChain, DekeResult, Retimer, SRobotPath, SRobotQ, SRobotTraj, Validator,
};

use crate::chord;
use crate::constraints::{JointLimits, Topp3LpConstraints};
use crate::diagnostic::{RetimeRecovery, Topp3LpDiagnostic};
use crate::error::Topp3LpError;
use crate::solve::solve_sigma;

/// The per-tick limit caps are planned at `margin·limit` so the small cross-bin
/// leak of the chord-linear reconstruction stays under the true limit; the final
/// finite-difference verify against the *true* limits is the airtight backstop.
const LIMIT_MARGIN: f64 = 0.999;

/// Upper bound on the output grid size. A path whose limits would need more ticks
/// than this on the `dt` grid (e.g. a near-zero velocity cap over a long path) is
/// treated as un-timeable rather than allocating an enormous LP.
const MAX_TICKS: usize = 200_000;

/// Cap on recursive run bisection in the un-timeable-run recovery. A run halves on
/// each level, so this bounds the recovery to at most `2^depth` rest-to-rest pieces
/// — far more than any real run needs before it either times or bottoms out at a
/// 2-waypoint segment.
const MAX_SEGMENT_DEPTH: usize = 12;

/// Joint-space, path-exact, jerk-limited retimer. Needs no FK; rejects a TCP cap.
#[derive(Clone, Copy, Debug)]
pub struct Topp3Lp<const N: usize>;

impl<const N: usize> Topp3Lp<N> {
    pub const fn new() -> Self {
        Self
    }

    fn retime_inner(
        &self,
        c: &Topp3LpConstraints<N>,
        path: &SRobotPath<N, f64>,
    ) -> Result<(SRobotTraj<N, f64>, Topp3LpDiagnostic), Topp3LpError> {
        if c.tcp.v_max.is_some() {
            return Err(Topp3LpError::TcpNeedsFk);
        }
        let raw: Vec<SRobotQ<N, f64>> = path.iter().copied().collect();
        if raw.len() < 2 {
            return Err(Topp3LpError::TooShort(raw.len()));
        }
        let dt = validate_inputs(c, &raw)?;
        let deduped = chord::dedup(&raw);
        if deduped.len() < 2 {
            return Err(Topp3LpError::Degenerate);
        }
        let runs = match c.sharp_corner_angle {
            Some(angle) => chord::split_sharp(&deduped, angle),
            None => vec![deduped],
        };

        let mut all: Vec<SRobotQ<N, f64>> = Vec::new();
        let mut total_arc = 0.0;
        let mut recovery = RetimeRecovery::default();
        for run in &runs {
            let (knots, s) = chord::condition(run, c.conditioning);
            let total = check_timeable(&knots, &s)?;
            let secant = chord::secants(&knots, &s);
            let (samples, run_recovery) = time_chord(&knots, &s, &secant, &c.joint, dt, None)?;
            recovery.merge(run_recovery);
            total_arc += total;
            concat_run(&mut all, samples);
        }
        if let Some((kind, joint_idx, value, limit, _)) = verify_joint_fd(&all, &c.joint, dt) {
            return Err(Topp3LpError::JointLimitExceeded {
                joint: joint_idx,
                kind,
                value,
                limit,
            });
        }
        build(all, total_arc, dt, c.output_dt, 0.0, recovery)
    }
}

/// Reject malformed inputs up front and return the output period in seconds.
/// Guards the hard-FD invariant at the boundary: a sub-microsecond/zero/non-finite
/// `output_dt` would otherwise be silently clamped (so the verified dt and the
/// stamped dt disagree), and non-finite waypoints or non-positive limits would
/// propagate NaN/inf into the solver and the FD check.
fn validate_inputs<const N: usize>(
    c: &Topp3LpConstraints<N>,
    raw: &[SRobotQ<N, f64>],
) -> Result<f64, Topp3LpError> {
    let dt = c.output_dt.as_secs_f64();
    if !dt.is_finite() || dt < 1e-6 {
        return Err(Topp3LpError::InvalidOutputDt);
    }
    let positive = |q: &SRobotQ<N, f64>| q.0.iter().all(|x| x.is_finite() && *x > 0.0);
    if !(positive(&c.joint.v_max) && positive(&c.joint.a_max) && positive(&c.joint.j_max)) {
        return Err(Topp3LpError::InvalidLimits);
    }
    if let Some(v) = c.tcp.v_max
        && !(v.is_finite() && v > 0.0)
    {
        return Err(Topp3LpError::InvalidLimits);
    }
    if raw.iter().any(|q| q.0.iter().any(|x| !x.is_finite())) {
        return Err(Topp3LpError::NonFinitePath);
    }
    Ok(dt)
}

/// Append a run's samples to the accumulated trajectory, dropping the duplicated
/// shared corner vertex (the first sample of every run after the first).
fn concat_run<const N: usize>(all: &mut Vec<SRobotQ<N, f64>>, samples: Vec<SRobotQ<N, f64>>) {
    if all.is_empty() {
        all.extend(samples);
    } else {
        all.extend(samples.into_iter().skip(1));
    }
}

impl<const N: usize> Default for Topp3Lp<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Retimer<N, f64> for Topp3Lp<N> {
    type Diagnostic = Topp3LpDiagnostic;
    type Constraints = Topp3LpConstraints<N>;

    fn retime<V: Validator<N, (), f64>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, f64>,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, f64>>, Self::Diagnostic) {
        match self.retime_inner(constraints, path) {
            Ok((traj, diag)) => {
                let samples: Vec<SRobotQ<N, f64>> = traj.path().iter().copied().collect();
                if let Err(e) = validator.validate_motion(&samples, ctx) {
                    return (Err(e), Topp3LpDiagnostic::zeroed());
                }
                (Ok(traj), diag)
            }
            Err(e) => (Err(e.into()), Topp3LpDiagnostic::zeroed()),
        }
    }
}

/// Joint-space retimer with a Cartesian TCP linear-speed cap. The cap becomes a
/// per-segment ceiling on `σ̇` from the FK Jacobian (`‖J_lin·secant‖`), and the
/// realised FK speed is verified against the true cap.
#[derive(Clone, Debug)]
pub struct Topp3LpTcp<'a, const N: usize, FK> {
    fk: &'a FK,
}

impl<'a, const N: usize, FK> Topp3LpTcp<'a, N, FK>
where
    FK: ContinuousFKChain<N, f64>,
{
    pub fn new(fk: &'a FK) -> Self {
        Self { fk }
    }

    fn retime_inner(
        &self,
        c: &Topp3LpConstraints<N>,
        path: &SRobotPath<N, f64>,
    ) -> Result<(SRobotTraj<N, f64>, Topp3LpDiagnostic), Topp3LpError> {
        let raw: Vec<SRobotQ<N, f64>> = path.iter().copied().collect();
        if raw.len() < 2 {
            return Err(Topp3LpError::TooShort(raw.len()));
        }
        let dt = validate_inputs(c, &raw)?;
        let deduped = chord::dedup(&raw);
        if deduped.len() < 2 {
            return Err(Topp3LpError::Degenerate);
        }
        let runs = match c.sharp_corner_angle {
            Some(angle) => chord::split_sharp(&deduped, angle),
            None => vec![deduped],
        };

        let mut all: Vec<SRobotQ<N, f64>> = Vec::new();
        let mut total_arc = 0.0;
        let mut recovery = RetimeRecovery::default();
        for run in &runs {
            let (samples, total, run_recovery) = self.time_run_segmented(c, run, dt, 0)?;
            recovery.merge(run_recovery);
            total_arc += total;
            concat_run(&mut all, samples);
        }
        if let Some((kind, joint_idx, value, limit, _)) = verify_joint_fd(&all, &c.joint, dt) {
            return Err(Topp3LpError::JointLimitExceeded {
                joint: joint_idx,
                kind,
                value,
                limit,
            });
        }
        let peak = self.tcp_speed_peak(&all, dt)?;
        if let Some(v_tcp) = c.tcp.v_max
            && (peak.is_nan() || peak > v_tcp * (1.0 + 1e-6))
        {
            return Err(Topp3LpError::TcpLimitExceeded {
                value: peak,
                limit: v_tcp,
            });
        }
        build(all, total_arc, dt, c.output_dt, peak, recovery)
    }

    /// Time a run, recovering from an un-timeable (too-curved / too-tight) run by
    /// bisecting it at its middle waypoint and timing each half rest-to-rest.
    ///
    /// The split point is an existing waypoint on the chord, so every emitted sample
    /// still lies exactly on the planned polyline — the recovery never deviates from
    /// the path. It only brings the move to rest at that waypoint, trading a brief
    /// stop for a solution where a single pass was infeasible (a sharp wrist
    /// reversal, a near-singular reconfiguration). Bisection recurses until each
    /// piece times or a piece is too short to split (&lt; 3 waypoints), in which case
    /// the original timing error surfaces. Only timing failures are recovered;
    /// malformed-input errors (degenerate, non-finite, bad limits) propagate.
    fn time_run_segmented(
        &self,
        c: &Topp3LpConstraints<N>,
        run: &[SRobotQ<N, f64>],
        dt: f64,
        depth: usize,
    ) -> Result<(Vec<SRobotQ<N, f64>>, f64, RetimeRecovery), Topp3LpError> {
        match self.time_run(c, run, dt) {
            Ok(out) => Ok(out),
            Err(e) => {
                let recoverable = matches!(
                    e,
                    Topp3LpError::Infeasible
                        | Topp3LpError::TickBudgetExceeded { .. }
                        | Topp3LpError::JointLimitExceeded { .. }
                        | Topp3LpError::TcpLimitExceeded { .. }
                );
                if !recoverable || depth >= MAX_SEGMENT_DEPTH || run.len() < 3 {
                    return Err(e);
                }
                let mid = run.len() / 2;
                let (mut head, t_head, mut rec) =
                    self.time_run_segmented(c, &run[..=mid], dt, depth + 1)?;
                let (tail, t_tail, tail_rec) =
                    self.time_run_segmented(c, &run[mid..], dt, depth + 1)?;
                concat_run(&mut head, tail);
                rec.merge(tail_rec);
                // This level's split inserted one on-chord rest stop (the halves'
                // own splits are already counted in their recoveries).
                rec.bisections += 1;
                Ok((head, t_head + t_tail, rec))
            }
        }
    }

    /// Time one rest-to-rest run, honouring the TCP velocity cap when set. Returns
    /// the run's samples, its arc length, and which recovery backstops were needed.
    fn time_run(
        &self,
        c: &Topp3LpConstraints<N>,
        run: &[SRobotQ<N, f64>],
        dt: f64,
    ) -> Result<(Vec<SRobotQ<N, f64>>, f64, RetimeRecovery), Topp3LpError> {
        let (knots, s) = chord::condition(run, c.conditioning);
        let total = check_timeable(&knots, &s)?;
        let secant = chord::secants(&knots, &s);

        let Some(v_tcp) = c.tcp.v_max else {
            let (samples, recovery) = time_chord(&knots, &s, &secant, &c.joint, dt, None)?;
            return Ok((samples, total, recovery));
        };

        // `‖dp/ds‖` per segment from the Jacobian: ṗ = J_lin·secant·σ̇, so the
        // per-tick velocity cap from the TCP limit is v_tcp / κ_b.
        let kappa = self.kappa_per_segment(&knots, &secant)?;

        // The FK speed a controller differences is the chord secant of the FK
        // *positions*, which the analytic κ underestimates by the FK curvature
        // across a tick. Plan at the cap, measure the realised overshoot, and
        // derate by exactly that before re-solving (1–2 passes); fail if it can't
        // be brought under the true cap rather than emit an over-cap trajectory.
        let mut derate = 1.0;
        let mut last_peak = 0.0;
        for _ in 0..6 {
            let tcp_cap: Vec<f64> = kappa
                .iter()
                .map(|k| v_tcp * derate / k.max(1e-12))
                .collect();
            let (samples, recovery) = time_chord(&knots, &s, &secant, &c.joint, dt, Some(&tcp_cap))?;
            let peak = self.tcp_speed_peak(&samples, dt)?;
            last_peak = peak;
            if peak <= v_tcp * (1.0 + 1e-6) {
                return Ok((samples, total, recovery));
            }
            derate /= (peak / v_tcp) * 1.01;
        }
        Err(Topp3LpError::TcpLimitExceeded {
            value: last_peak,
            limit: v_tcp,
        })
    }

    /// Per-segment `κ = ‖J_lin·secant‖`, the TCP linear speed per unit `σ̇`. The
    /// Jacobian varies along a segment, so sampling only the start knot
    /// under-estimates `κ` on a long chord that reconfigures the arm a lot — the
    /// realised mid-chord TCP speed then overshoots the cap and the derate loop
    /// in `time_run` can't recover a few-percent overshoot. Sample the Jacobian
    /// at several interior points of each segment and take the max, so the
    /// per-segment cap is conservative (bounds the worst-case TCP gain) and the
    /// first solve already respects the cap. Segments are short under `Collinear`
    /// conditioning, so the extra evals there are cheap and `κ` ≈ the endpoint.
    fn kappa_per_segment(
        &self,
        knots: &[SRobotQ<N, f64>],
        secant: &[[f64; N]],
    ) -> Result<Vec<f64>, Topp3LpError> {
        const SAMPLES: usize = 5;
        secant
            .iter()
            .enumerate()
            .map(|(b, sec)| {
                let q0 = &knots[b];
                let q1 = &knots[b + 1];
                let mut kmax = 0.0f64;
                for si in 0..SAMPLES {
                    let t = si as f64 / (SAMPLES - 1) as f64;
                    let q = SRobotQ(std::array::from_fn(|j| q0.0[j] * (1.0 - t) + q1.0[j] * t));
                    let jac = self.fk.jacobian(&q)?;
                    let mut v = [0.0f64; 3];
                    for (r, vr) in v.iter_mut().enumerate() {
                        *vr = (0..N).map(|col| jac[r][col] * sec[col]).sum();
                    }
                    kmax = kmax.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                }
                Ok(kmax)
            })
            .collect()
    }

    fn tcp_speed_peak(&self, samples: &[SRobotQ<N, f64>], dt: f64) -> Result<f64, Topp3LpError> {
        if samples.len() < 2 {
            return Ok(0.0);
        }
        let mut prev: DVec3 = self.fk.fk_end(&samples[0])?.translation;
        let mut peak = 0.0f64;
        for q in &samples[1..] {
            let p = self.fk.fk_end(q)?.translation;
            peak = peak.max(prev.distance(p) / dt);
            prev = p;
        }
        Ok(peak)
    }
}

impl<'a, const N: usize, FK> Retimer<N, f64> for Topp3LpTcp<'a, N, FK>
where
    FK: ContinuousFKChain<N, f64>,
{
    type Diagnostic = Topp3LpDiagnostic;
    type Constraints = Topp3LpConstraints<N>;

    fn retime<V: Validator<N, (), f64>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, f64>,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, f64>>, Self::Diagnostic) {
        match self.retime_inner(constraints, path) {
            Ok((traj, diag)) => {
                let samples: Vec<SRobotQ<N, f64>> = traj.path().iter().copied().collect();
                if let Err(e) = validator.validate_motion(&samples, ctx) {
                    return (Err(e), Topp3LpDiagnostic::zeroed());
                }
                (Ok(traj), diag)
            }
            Err(e) => (Err(e.into()), Topp3LpDiagnostic::zeroed()),
        }
    }
}

/// Joint-space total arc length, or `Degenerate` when the conditioned path has no
/// length (all waypoints coincide).
fn check_timeable<const N: usize>(
    knots: &[SRobotQ<N, f64>],
    s: &[f64],
) -> Result<f64, Topp3LpError> {
    let nb = knots.len();
    let total = if nb == 0 { 0.0 } else { s[nb - 1] };
    if nb < 2 || total < 1e-9 {
        return Err(Topp3LpError::Degenerate);
    }
    Ok(total)
}

/// Build the output trajectory and its diagnostic from the timed samples.
fn build<const N: usize>(
    samples: Vec<SRobotQ<N, f64>>,
    total: f64,
    dt: f64,
    output_dt: Duration,
    peak_tcp_speed: f64,
    recovery: RetimeRecovery,
) -> Result<(SRobotTraj<N, f64>, Topp3LpDiagnostic), Topp3LpError> {
    let (v, a, jk) = fd_peaks(&samples, dt);
    let diag = Topp3LpDiagnostic {
        output_samples: samples.len(),
        duration: Duration::from_secs_f64(samples.len().saturating_sub(1) as f64 * dt),
        arc_length: total,
        peak_joint_vel: v,
        peak_joint_accel: a,
        peak_joint_jerk: jk,
        peak_tcp_speed,
        recovery,
    };
    let path = SRobotPath::try_new(samples)?;
    Ok((SRobotTraj::new(output_dt, path), diag))
}

/// Time the chord-linear path onto a uniform-`dt` grid: plan per-tick σ-difference
/// caps from the projected joint limits (intersected with the optional TCP cap),
/// solve the σ-LP growing the horizon and re-binning to a fixed point, reconstruct
/// the on-chord samples, and verify the output finite differences against the
/// *true* joint limits. Returns the reconstructed samples.
#[allow(clippy::needless_range_loop)]
fn time_chord<const N: usize>(
    knots: &[SRobotQ<N, f64>],
    s: &[f64],
    secant: &[[f64; N]],
    joint: &JointLimits<N>,
    dt: f64,
    tcp_vel_cap: Option<&[f64]>,
) -> Result<(Vec<SRobotQ<N, f64>>, RetimeRecovery), Topp3LpError> {
    let nb = knots.len();
    let total = s[nb - 1];

    let bin_of = |sx: f64| -> usize {
        let sx = sx.clamp(0.0, total);
        let mut b = 0;
        while b + 1 < nb - 1 && s[b + 1] <= sx {
            b += 1;
        }
        b
    };
    let proj = |b: usize, lim: &SRobotQ<N, f64>| -> f64 {
        (0..N)
            .map(|j| lim.0[j] / secant[b][j].abs().max(1e-12))
            .fold(f64::INFINITY, f64::min)
    };
    let tcp_at = |b: usize| -> f64 { tcp_vel_cap.map_or(f64::INFINITY, |t| t[b]) };

    let mg = LIMIT_MARGIN;
    let v_eff = (0..nb - 1)
        .map(|b| proj(b, &joint.v_max).min(tcp_at(b)))
        .fold(f64::INFINITY, f64::min)
        .max(1e-9);
    let a_eff = (0..nb - 1)
        .map(|b| proj(b, &joint.a_max))
        .fold(f64::INFINITY, f64::min)
        .max(1e-6);
    let j_eff = (0..nb - 1)
        .map(|b| proj(b, &joint.j_max))
        .fold(f64::INFINITY, f64::min)
        .max(1e-6);
    let ramp_t = v_eff / a_eff + a_eff / j_eff;
    let mut kk =
        ((total / (v_eff * dt)).ceil() as usize + (4.0 * ramp_t / dt) as usize + 64).max(8);

    // Reconstruct on-chord samples from a σ profile, trim the parked tail, and pin
    // the endpoints exactly (`σ[0]=0`, `σ[end-1]=total` are equality constrained, so
    // the tiny LP residual is snapped to the waypoints — zero deviation *at* the
    // waypoints, not merely near them).
    let recon_samples = |sg: &[f64]| -> Vec<SRobotQ<N, f64>> {
        // Trim the parked-at-total tail but keep three at-total samples so the
        // terminal velocity AND acceleration finite-differences are exactly zero
        // (a genuine rest, symmetric with the pinned σ[0]=σ[1]=σ[2]=0 start). The
        // old keep-one trim left a non-rest endpoint (terminal v ≈ a_max·dt).
        let mut end = sg.len();
        while end > 4 && (total - sg[end - 4]).abs() < 1e-9 {
            end -= 1;
        }
        // Snap the rest runs (σ within ε of 0 / total) to the exact endpoint knots,
        // so the start and end are bit-exact AND the terminal velocity/acceleration
        // FD are exactly zero (the LP leaves a ~1e-8 residual at the pinned ticks).
        // ε is far below any real decel step (~proj_a·dt²), so motion is untouched.
        let eps = 1e-7 * total;
        let mut samples: Vec<SRobotQ<N, f64>> = sg[..end]
            .iter()
            .map(|&sx| {
                if sx <= eps {
                    knots[0]
                } else if sx >= total - eps {
                    knots[nb - 1]
                } else {
                    let b = bin_of(sx);
                    SRobotQ(std::array::from_fn(|j| {
                        knots[b].0[j] + secant[b][j] * (sx - s[b])
                    }))
                }
            })
            .collect();
        if let Some(first) = samples.first_mut() {
            *first = knots[0];
        }
        if let Some(last) = samples.last_mut() {
            *last = knots[nb - 1];
        }
        samples
    };

    // The exact per-joint FD rows are exact only once the per-tick bin assignment
    // has converged; at a sharp corner it may not in a few passes, leaking a little
    // cross-bin jerk. The true-limit `verify_joint_fd` is the airtight gate: grow
    // the horizon (more ticks at the corner ⇒ lower σ̇ ⇒ smaller leak) and re-solve
    // until it passes, rather than emit an over-limit trajectory.
    let mut last_violation: Option<(&'static str, usize, f64, f64)> = None;
    // First feasible reconstruction kept around so that, if growing the horizon
    // never clears the FD verify, we can fall back to uniformly slowing it (see
    // `time_scale_to_limits`) instead of failing outright.
    let mut best: Option<Vec<SRobotQ<N, f64>>> = None;
    // The σ-LP plans against these limits, which start at the true caps and are
    // tightened in place whenever a reconstruction's realized FD overruns the true
    // cap (see the derate at the end of the loop).
    let mut plan = joint.clone();
    // The net planned-limit derate at the point a solve is accepted: `(joint, kind,
    // factor)` for every axis whose planned cap ended up below its true cap.
    let derates_of = |plan: &JointLimits<N>| -> Vec<(usize, &'static str, f64)> {
        let mut out = Vec::new();
        for j in 0..N {
            for (kind, p, t) in [
                ("velocity", plan.v_max.0[j], joint.v_max.0[j]),
                ("acceleration", plan.a_max.0[j], joint.a_max.0[j]),
                ("jerk", plan.j_max.0[j], joint.j_max.0[j]),
            ] {
                if t > 0.0 && p < t * (1.0 - 1e-9) {
                    out.push((j, kind, p / t));
                }
            }
        }
        out
    };
    for _grow in 0..8 {
        if kk > MAX_TICKS {
            // The grid the limits demand at this dt exceeds the budget — this is a
            // tick-budget problem (raise dt / relax limits), distinct from a path
            // that is genuinely too curved to time (JointLimitExceeded below).
            return Err(Topp3LpError::TickBudgetExceeded { max: MAX_TICKS });
        }
        let mut sg: Vec<f64> = (0..kk)
            .map(|k| k as f64 / (kk - 1) as f64 * total)
            .collect();
        let mut prev_bins: Vec<usize> = Vec::new();
        let mut feasible = true;
        for _pass in 0..8 {
            let bins: Vec<usize> = sg.iter().map(|&sx| bin_of(sx)).collect();
            if bins == prev_bins {
                break;
            }
            // Per-tick affine reconstruction `q[k] = c[k] + m[k]·σ[k]` from each
            // tick's segment, so the exact-FD rows the LP enforces match what the
            // consumer differences — including across a corner where bins differ.
            let mvals: Vec<[f64; N]> = bins.iter().map(|&b| secant[b]).collect();
            let cvals: Vec<[f64; N]> = bins
                .iter()
                .map(|&b| std::array::from_fn(|j| knots[b].0[j] - secant[b][j] * s[b]))
                .collect();
            let tcp_ds: Option<Vec<f64>> =
                tcp_vel_cap.map(|_| bins.iter().map(|&b| tcp_at(b) * dt).collect());
            match solve_sigma(
                kk,
                total,
                &mvals,
                &cvals,
                &plan.v_max,
                &plan.a_max,
                &plan.j_max,
                tcp_ds.as_deref(),
                None,
                None,
                &bins,
                mg,
                dt,
            ) {
                Some(next) => sg = next,
                None => {
                    feasible = false;
                    break;
                }
            }
            prev_bins = bins;
        }
        if feasible {
            // Fast path: if the unboxed solution already passes the true-limit FD
            // verify (the common case — interior-dominated paths with no cross-bin
            // leak), take it without a second solve.
            let samples = recon_samples(&sg);
            match verify_joint_fd(&samples, joint, dt) {
                None => {
                    return Ok((
                        samples,
                        RetimeRecovery {
                            derates: derates_of(&plan),
                            ..Default::default()
                        },
                    ));
                }
                Some((kind, joint_idx, value, limit, _idx)) => {
                    last_violation = Some((kind, joint_idx, value, limit));
                    if best.is_none() {
                        best = Some(samples);
                    }
                }
            }

            // Otherwise a stencil straddled a corner: re-solve with each tick boxed
            // into the segment it landed in so the reconstruction bins match the rows
            // exactly (no leak). If the boxed solve is verified it returns; otherwise
            // we fall through and grow the horizon — we never emit an unverified `sg`.
            let bins: Vec<usize> = sg.iter().map(|&sx| bin_of(sx)).collect();
            let mvals: Vec<[f64; N]> = bins.iter().map(|&b| secant[b]).collect();
            let cvals: Vec<[f64; N]> = bins
                .iter()
                .map(|&b| std::array::from_fn(|j| knots[b].0[j] - secant[b][j] * s[b]))
                .collect();
            let tcp_ds: Option<Vec<f64>> =
                tcp_vel_cap.map(|_| bins.iter().map(|&b| tcp_at(b) * dt).collect());
            let lo: Vec<f64> = bins.iter().map(|&b| s[b]).collect();
            let hi: Vec<f64> = bins.iter().map(|&b| s[b + 1]).collect();
            let boxed = solve_sigma(
                kk,
                total,
                &mvals,
                &cvals,
                &plan.v_max,
                &plan.a_max,
                &plan.j_max,
                tcp_ds.as_deref(),
                Some(&lo),
                Some(&hi),
                &bins,
                mg,
                dt,
            );
            if let Some(boxed) = boxed {
                let samples = recon_samples(&boxed);
                match verify_joint_fd(&samples, joint, dt) {
                    None => {
                        return Ok((
                            samples,
                            RetimeRecovery {
                                derates: derates_of(&plan),
                                ..Default::default()
                            },
                        ));
                    }
                    Some((kind, joint_idx, value, limit, _idx)) => {
                        last_violation = Some((kind, joint_idx, value, limit));
                        if best.is_none() {
                            best = Some(samples);
                        }
                    }
                }
            }
        }
        // The σ-LP solves its v/a/j rows only to the convex solver's tolerance. On
        // the jerk rows — whose RHS is `limit·dt³`, vanishingly small — that
        // tolerance can leave the realized FD a few-to-20% over the true cap, an
        // overrun that is precision (not a cross-bin leak) and so survives any
        // horizon growth. Derate the offending joint's planned limit by the overrun
        // (with a little headroom) and re-solve: the next reconstruction lands under
        // the true cap while keeping the LP's smooth, minimally-slowed profile —
        // instead of falling through to a uniform time-stretch that slows the whole
        // move many-fold to clear one ramp's jerk spike.
        if let Some((kind, jidx, value, limit)) = last_violation {
            let shrink = (limit / value) / 1.02;
            if shrink.is_finite() && shrink < 1.0 {
                match kind {
                    "velocity" => plan.v_max.0[jidx] *= shrink,
                    "acceleration" => plan.a_max.0[jidx] *= shrink,
                    _ => plan.j_max.0[jidx] *= shrink,
                }
            }
        }
        kk = (kk as f64 * 1.6) as usize + 16;
    }
    // Growing the horizon failed to clear the verify. On a straight, slow-axis-
    // dominated chord that is expected: the LP ramps the binding joint at its
    // limit regardless of horizon, and an ill-conditioned solve can leave its own
    // σ rows slightly (sometimes grossly) violated, which more ticks never shrink.
    // Uniformly slow the best feasible reconstruction until the true-limit FD
    // verify passes — always safe (it stays exactly on the chord and only lowers
    // every derivative) and independent of solver precision.
    if let Some(best) = best
        && let Some(scaled) = time_scale_to_limits(&best, joint, dt)
    {
        return Ok((
            scaled,
            RetimeRecovery {
                derates: derates_of(&plan),
                time_scaled: true,
                ..Default::default()
            },
        ));
    }
    match last_violation {
        Some((kind, joint_idx, value, limit)) => Err(Topp3LpError::JointLimitExceeded {
            joint: joint_idx,
            kind,
            value,
            limit,
        }),
        None => Err(Topp3LpError::Infeasible),
    }
}

/// Worst per-joint finite-difference violation of the *true* limits, or `None` if
/// every difference is within limit. Returns `(kind, joint, value, limit, idx)`.
fn verify_joint_fd<const N: usize>(
    q: &[SRobotQ<N, f64>],
    lim: &JointLimits<N>,
    dt: f64,
) -> Option<(&'static str, usize, f64, f64, usize)> {
    let n = q.len();
    let mut worst: Option<(f64, &'static str, usize, f64, f64, usize)> = None;
    let mut consider = |val: f64, limit: f64, kind: &'static str, j: usize, idx: usize| {
        // Flag NaN explicitly (fail-closed): `NaN > x` is false, which would
        // silently pass a NaN finite difference.
        if val.is_nan() || val > limit * (1.0 + 1e-6) {
            let r = val / limit;
            if worst.is_none_or(|w| r.is_nan() || r > w.0) {
                worst = Some((r, kind, j, val, limit, idx));
            }
        }
    };
    for i in 1..n {
        for j in 0..N {
            consider(
                (q[i].0[j] - q[i - 1].0[j]).abs() / dt,
                lim.v_max.0[j],
                "velocity",
                j,
                i,
            );
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
            let jk = (q[i].0[j] - 3.0 * q[i - 1].0[j] + 3.0 * q[i - 2].0[j] - q[i - 3].0[j]).abs()
                / (dt * dt * dt);
            consider(jk, lim.j_max.0[j], "jerk", j, i);
        }
    }
    worst.map(|(_, kind, j, val, limit, idx)| (kind, j, val, limit, idx))
}

/// Uniformly slow `samples` until every finite-difference v/a/j is under the true
/// limit, returning the slowed trajectory (or `None` if it cannot converge). A
/// time stretch by `s` scales the order-`d` finite difference by `1/sᵈ`, so this
/// always reduces the violation and — because it only resamples along the chord
/// the samples already lie on — never leaves the planned path. It is the robust
/// backstop for an ill-conditioned σ-LP whose returned profile overruns its own
/// planned caps: correctness no longer hinges on the convex solver's precision.
fn time_scale_to_limits<const N: usize>(
    samples: &[SRobotQ<N, f64>],
    joint: &JointLimits<N>,
    dt: f64,
) -> Option<Vec<SRobotQ<N, f64>>> {
    let mut cur = samples.to_vec();
    for _ in 0..16 {
        match verify_joint_fd(&cur, joint, dt) {
            None => return Some(cur),
            Some((kind, _j, value, limit, _idx)) => {
                let order = match kind {
                    "velocity" => 1.0,
                    "acceleration" => 2.0,
                    _ => 3.0,
                };
                // Slow just enough to bring the worst derivative under, with a
                // little headroom so the next verify clears rather than lands on
                // the bound.
                let s = (value / limit).powf(1.0 / order) * 1.01;
                if !s.is_finite() || s <= 1.0 {
                    return None;
                }
                cur = resample_stretch(&cur, s);
            }
        }
    }
    None
}

/// Resample `samples` onto a `s×`-longer uniform grid (same `dt`): the identical
/// on-chord polyline traversed `s` times slower, endpoints pinned. New tick `i`
/// reads the old trajectory at time `i/s`.
fn resample_stretch<const N: usize>(samples: &[SRobotQ<N, f64>], s: f64) -> Vec<SRobotQ<N, f64>> {
    let len = samples.len();
    if len < 2 {
        return samples.to_vec();
    }
    let last = len - 1;
    let m = ((last as f64) * s).round() as usize + 1;
    (0..m)
        .map(|i| {
            let x = (i as f64) / s;
            if x >= last as f64 {
                return samples[last];
            }
            let lo = x.floor() as usize;
            let frac = x - lo as f64;
            SRobotQ(std::array::from_fn(|j| {
                samples[lo].0[j] * (1.0 - frac) + samples[lo + 1].0[j] * frac
            }))
        })
        .collect()
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
            jk = jk.max(
                (q[i].0[j] - 3.0 * q[i - 1].0[j] + 3.0 * q[i - 2].0[j] - q[i - 3].0[j]).abs()
                    / (dt * dt * dt),
            );
        }
    }
    (v, a, jk)
}
