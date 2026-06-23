//! Stage C — time-parameterise a joint path at constant TCP speed.
//!
//! This is a CNC-style constant-feedrate planner, not a TOPP retimer. The
//! feasible-speed ceiling along the path (the "maximum velocity curve") comes from
//! the per-joint v/a/j limits projected onto the path tangent `q'(s)`. The
//! commanded speed `tcp.speed` is held flat wherever that ceiling allows; near a
//! singularity `|q'(s)| → ∞` so the ceiling collapses and the feedrate dips to zero
//! smoothly instead of demanding infinite joint speed. The profile is built by a
//! backward+forward acceleration-bounded pass (zero speed at both ends) followed by
//! a forward jerk-limited time integration that tracks it.
//!
//! Joint velocity is enforced exactly; acceleration and jerk are enforced through
//! the tangent projection — the `q''(s)·ṡ²` curvature cross-term is a deliberate
//! first-pass approximation, softened by the jerk-limited integrator.

use std::time::Duration;

use deke_types::glam::DVec3;
use deke_types::{
    ContinuousFKChain, DekeError, DekeResult, Retimer, SRobotPath, SRobotQ, SRobotTraj, Validator,
};

use crate::constraints::LinearConstraints;
use crate::diagnostic::LinearRetimerDiagnostic;
use crate::error::LinearError;

const BIG: f64 = 1e9;

/// Safety derating applied to every joint and TCP accel/jerk limit the solver
/// plans against. The integrator bounds the *continuous* v/a/j exactly, but the
/// discrete finite differences a controller reconstructs from the sampled output
/// read a little higher (half-step integration, the secant-vs-tangent gap across
/// knots); planning at `margin·limit` keeps those reconstructions under the true
/// limit. It does not derate the commanded TCP speed, which is a target rather
/// than a ceiling to retreat from.
const LIMIT_MARGIN: f64 = 0.95;

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
        let q = match c.corner_smoothing {
            Some(res) => spline_resample(&raw, res),
            None => raw,
        };
        let m = q.len();
        let dt = c.output_dt.as_secs_f64().max(1e-6);

        // Plan against limits derated by `LIMIT_MARGIN`, leaving the headroom the
        // sampled output needs: the integrator bounds the continuous v/a/j, but
        // the discrete finite differences a controller reconstructs read a little
        // higher (half-step integration, the secant-vs-tangent gap between knots).
        // Planning at `margin·limit` keeps those within the true limit. The
        // commanded `tcp.speed` is a target, not a ceiling to back off from, so it
        // is left underated.
        let v_max = c.joint.v_max * LIMIT_MARGIN;
        let a_max = c.joint.a_max * LIMIT_MARGIN;
        let j_max = c.joint.j_max * LIMIT_MARGIN;
        let tcp_accel = c.tcp.accel.map(|x| x * LIMIT_MARGIN);
        let tcp_jerk = c.tcp.jerk.map(|x| x * LIMIT_MARGIN);

        // Cartesian arc length from FK end positions (true metres for `tcp.speed`).
        let pos: Vec<DVec3> = q
            .iter()
            .map(|qi| self.fk.fk_end(qi).map(|t| t.translation))
            .collect::<Result<_, DekeError>>()?;
        let mut s = vec![0.0f64; m];
        for i in 1..m {
            s[i] = s[i - 1] + pos[i].distance(pos[i - 1]);
        }
        let total = s[m - 1];
        if m < 2 || total < 1e-9 {
            let traj = SRobotTraj::new(c.output_dt, path.clone());
            return Ok((
                traj,
                LinearRetimerDiagnostic {
                    output_samples: m,
                    duration: Duration::from_secs_f64((m.saturating_sub(1)) as f64 * dt),
                    arc_length: total,
                    commanded_speed: c.tcp.speed,
                    peak_speed: 0.0,
                    peak_joint_accel: 0.0,
                    peak_joint_jerk: 0.0,
                },
            ));
        }

        // Path derivatives wrt arc length by central difference: q'(s), q''(s),
        // q'''(s). The higher derivatives carry the joint-space path curvature
        // that turns Cartesian motion into joint accel/jerk via the chain rule
        //   q̇  = q'·v
        //   q̈  = q'·a + q''·v²
        //   q⃛ = q'·j_s + 3·q''·a·v + q'''·v³
        // so a straight-Cartesian line can still load the joints when q bends.
        let central = |arr: &[SRobotQ<N, f64>], i: usize| -> SRobotQ<N, f64> {
            let (lo, hi) = if i == 0 {
                (0, 1)
            } else if i == m - 1 {
                (m - 2, m - 1)
            } else {
                (i - 1, i + 1)
            };
            let ds = (s[hi] - s[lo]).max(1e-12);
            (arr[hi] - arr[lo]) * (1.0 / ds)
        };
        let qp: Vec<SRobotQ<N, f64>> = (0..m).map(|i| central(&q, i)).collect();
        let qpp: Vec<SRobotQ<N, f64>> = (0..m).map(|i| central(&qp, i)).collect();
        let qppp: Vec<SRobotQ<N, f64>> = (0..m).map(|i| central(&qpp, i)).collect();

        // Velocity-limit curve: the joint velocity limit plus the centripetal
        // caps where path curvature alone (at zero tangential accel/jerk) would
        // breach a joint's accel/jerk limit — `|q''|·v² ≤ a_max` and
        // `|q'''|·v³ ≤ j_max` — all intersected with the commanded TCP speed.
        // Holds the speed down through joint-space bends.
        let v_ceiling = |i: usize| {
            project_min(&qp[i], &v_max)
                .min(project_min(&qpp[i], &a_max).sqrt())
                .min(project_min(&qppp[i], &j_max).cbrt())
                .min(c.tcp.speed)
        };

        // An interior dip below the command is forced by the joint v/a/j limits
        // and path curvature (a shallow corner or near-singular patch) — distinct
        // from the temporal rest ramps the profile adds at the ends. With
        // `forbid_interior_dips` the caller would rather fail than slow, so report
        // the worst offending sample against the full feasible-speed ceiling.
        if c.forbid_interior_dips {
            let mut worst: Option<(usize, f64)> = None;
            #[allow(clippy::needless_range_loop)]
            for i in 1..m - 1 {
                let g = v_ceiling(i);
                if g < c.tcp.speed * (1.0 - 1e-3) && worst.is_none_or(|(_, gw)| g < gw) {
                    worst = Some((i, g));
                }
            }
            if let Some((i, g)) = worst {
                return Err(LinearError::SpeedDipRequired {
                    run: run_idx,
                    s: s[i],
                    feasible_speed: g,
                    commanded: c.tcp.speed,
                });
            }
        }

        let a_path: Vec<f64> = (0..m).map(|i| project_min(&qp[i], &a_max)).collect();
        let j_path: Vec<f64> = (0..m).map(|i| project_min(&qp[i], &j_max)).collect();

        // Acceleration-bounded velocity ceiling for interior corners. The end is
        // NOT pinned to rest here: pinning it to 0 makes the in-segment linear
        // interpolation decelerate `v` to rest across the entire final segment
        // (an unbounded-time crawl on a coarse segment). The terminal stop is
        // instead enforced per step by the jerk-limited `jerk_stop_speed`
        // ceiling, which holds cruise until the physical stopping distance.
        // Start-from-rest is the integrator's initial condition (v = 0).
        let mut vc: Vec<f64> = (0..m).map(v_ceiling).collect();
        for i in (0..m - 1).rev() {
            let ds = s[i + 1] - s[i];
            vc[i] = vc[i].min((vc[i + 1] * vc[i + 1] + 2.0 * a_path[i] * ds).sqrt());
        }
        for i in 1..m {
            let ds = s[i] - s[i - 1];
            vc[i] = vc[i].min((vc[i - 1] * vc[i - 1] + 2.0 * a_path[i - 1] * ds).sqrt());
        }

        // Per-segment reciprocal lengths and value slopes. Precomputing these
        // turns every inner-loop lookup into a fused `base + slope·f` — no
        // division and no subtraction in the hot path — and the joint sample
        // becomes `q[i] + dq[i]·f`.
        let seg_n = m - 1;
        let mut inv_ds = vec![0.0f64; seg_n];
        let mut vc_d = vec![0.0f64; seg_n];
        let mut a_d = vec![0.0f64; seg_n];
        let mut j_d = vec![0.0f64; seg_n];
        let mut dq = vec![SRobotQ::<N, f64>::zeros(); seg_n];
        let mut qp_d = vec![SRobotQ::<N, f64>::zeros(); seg_n];
        let mut qpp_d = vec![SRobotQ::<N, f64>::zeros(); seg_n];
        let mut qppp_d = vec![SRobotQ::<N, f64>::zeros(); seg_n];
        for i in 0..seg_n {
            let ds = s[i + 1] - s[i];
            inv_ds[i] = if ds > 0.0 { 1.0 / ds } else { 0.0 };
            vc_d[i] = vc[i + 1] - vc[i];
            a_d[i] = a_path[i + 1] - a_path[i];
            j_d[i] = j_path[i + 1] - j_path[i];
            dq[i] = q[i + 1] - q[i];
            qp_d[i] = qp[i + 1] - qp[i];
            qpp_d[i] = qpp[i + 1] - qpp[i];
            qppp_d[i] = qppp[i + 1] - qppp[i];
        }

        // Forward jerk-limited time integration tracking the ceiling. The flat
        // estimate `total / (tcp.speed·dt)` is a lower bound on the step count
        // (real speed never exceeds the command); doubling it covers the rest
        // ramps so the buffer almost never reallocates mid-sweep.
        let est = (total / (c.tcp.speed.max(1e-6) * dt)) as usize;
        let mut samples: Vec<SRobotQ<N, f64>> = Vec::with_capacity(est * 2 + 16);
        samples.push(q[0]);
        let mut sx = 0.0f64;
        let mut v = 0.0f64;
        let mut a = 0.0f64;
        let mut peak = 0.0f64;
        let mut pk_ja = 0.0f64;
        let mut pk_jj = 0.0f64;
        // Worst per-joint limit overrun against the *true* (un-derated) limits,
        // `(ratio, value, limit, arc_length, joint, kind)`. Tracked so a run the
        // curvature drives past a velocity/accel/jerk limit fails rather than
        // emitting a trajectory the arm cannot execute. `LIMIT_MARGIN` keeps the
        // common case clear; this catches what the margin cannot.
        let mut overrun: Option<(f64, f64, f64, f64, usize, &'static str)> = None;
        let max_iters = (est + m) * 8 + 100_000;

        // `sx` only ever advances, so a single forward cursor (`seg`) brackets
        // every lookup in amortised O(1). The bracket landed on at the end of a
        // step is exactly where the next step's ceiling is read, so it is carried
        // across iterations — one `seg` call per step serves both the sample and
        // the next ceiling read.
        let mut cur = 0usize;
        let mut i = 0usize;
        let mut f = 0.0f64;
        let mut iters = 0usize;
        while sx < total - 1e-9 {
            iters += 1;
            if iters > max_iters {
                return Err(LinearError::Stalled {
                    run: run_idx,
                    s: sx,
                });
            }
            let alim = a_path[i] + a_d[i] * f;
            let jlim = j_path[i] + j_d[i] * f;
            // Effective tangential ceilings: the joint-projected scalar bound
            // intersected with the optional Cartesian TCP accel/jerk caps. These
            // shape the terminal stop envelope and the emergency fallbacks below.
            let alim_eff = tcp_accel.map_or(alim, |t| alim.min(t));
            let jlim_eff = tcp_jerk.map_or(jlim, |t| jlim.min(t));

            // Interior corner ceiling (`vc`) intersected with the jerk-limited
            // stopping envelope to the path end, so the terminal decel takes the
            // physical S-curve distance rather than the whole final segment. The
            // stop is planned at `STOP_JERK_FRACTION` of the available jerk.
            let vlim = (vc[i] + vc_d[i] * f)
                .min(jerk_stop_speed(total - sx, alim_eff, STOP_JERK_FRACTION * jlim_eff))
                .max(0.0);

            // Joint dynamics: bound the path accel `a` (= s̈) and path jerk `j_s`
            // (= s⃛) so the chain-rule joint accel `q'·a + q''·v²` and joint jerk
            // `q'·j_s + 3·q''·a·v + q'''·v³` stay within the per-joint limits, then
            // tighten by the optional Cartesian TCP accel/jerk caps (`s̈`/`s⃛` are
            // the tangential TCP accel/jerk, since `s` is Cartesian arc length).
            // The velocity-limit curve keeps `a = 0` joint-feasible; under extreme
            // curvature the jerk interval can pin, in which case slew `a` back
            // toward zero as hard as the (capped) jerk allows.
            let qp_c = qp[i] + qp_d[i] * f;
            let qpp_c = qpp[i] + qpp_d[i] * f;
            let qppp_c = qppp[i] + qppp_d[i] * f;
            let (aj_lo, aj_hi) = feasible_interval(&qp_c, &(qpp_c * (v * v)), &a_max);
            let (a_lo, a_hi) = cap_interval(aj_lo, aj_hi, tcp_accel);
            let (a_lo, a_hi) = if a_lo <= a_hi {
                (a_lo, a_hi)
            } else if aj_lo <= aj_hi {
                // Joints feasible but the TCP cap excludes the whole interval: the
                // joint limit is hard, so take the joint endpoint nearest zero and
                // accept the TCP-cap overshoot rather than stalling.
                let a_edge = if aj_lo > 0.0 { aj_lo } else { aj_hi };
                (a_edge, a_edge)
            } else {
                (-alim, -alim)
            };

            let jc = qpp_c * (3.0 * a * v) + qppp_c * (v * v * v);
            let (jj_lo, jj_hi) = feasible_interval(&qp_c, &jc, &j_max);
            let (js_lo, js_hi) = cap_interval(jj_lo, jj_hi, c.tcp.jerk);

            let a_des = ((vlim - v) / dt).clamp(a_lo, a_hi);
            let j_s = if js_lo <= js_hi {
                ((a_des - a) / dt).clamp(js_lo, js_hi)
            } else {
                (-a / dt).clamp(-jlim_eff, jlim_eff)
            };
            a = (a + j_s * dt).clamp(a_lo, a_hi);
            v = (v + a * dt).clamp(0.0, vlim);
            peak = peak.max(v);

            // Continuous chain-rule joint accel/jerk actually realized this step
            // — bounded by the limits by construction of the interval clamps.
            let jv = qp_c * v;
            let ja = qp_c * a + qpp_c * (v * v);
            let jj = qp_c * j_s + qpp_c * (3.0 * a * v) + qppp_c * (v * v * v);
            pk_ja = pk_ja.max(ja.0.iter().fold(0.0, |m, &x| m.max(x.abs())));
            pk_jj = pk_jj.max(jj.0.iter().fold(0.0, |m, &x| m.max(x.abs())));
            for k in 0..N {
                for (value, limit, kind) in [
                    (jv.0[k].abs(), c.joint.v_max.0[k], "velocity"),
                    (ja.0[k].abs(), c.joint.a_max.0[k], "acceleration"),
                    (jj.0[k].abs(), c.joint.j_max.0[k], "jerk"),
                ] {
                    if value > limit * (1.0 + 1e-6) {
                        let ratio = value / limit;
                        if overrun.is_none_or(|(w, ..)| ratio > w) {
                            overrun = Some((ratio, value, limit, sx, k, kind));
                        }
                    }
                }
            }
            sx += v * dt;
            (i, f) = seg(&s, &inv_ds, &mut cur, sx.min(total));
            samples.push(q[i] + dq[i] * f);

            // Terminal decel has bled to rest within a sub-sample of the end:
            // the `vc[m-1] = 0` ceiling drives `v → 0` slightly before `sx`
            // reaches `total`, after which `sx += v·dt` only crawls the geometric
            // tail toward the `total - 1e-9` margin, emitting hundreds of
            // effectively-stationary samples. Stop; the exact endpoint is
            // appended below. Bounded to the end (`total - sx` small) so a
            // mid-path singularity still trips the stall guard.
            if v < 1e-6 && total - sx < c.tcp.speed.max(1e-6) * dt {
                break;
            }

            // Guard against a stall at a vanishing ceiling (true singularity).
            if v < 1e-9 && vlim < 1e-9 && sx < total - 1e-6 {
                return Err(LinearError::Stalled {
                    run: run_idx,
                    s: sx,
                });
            }
        }
        if let Some((_, value, limit, s_at, joint, kind)) = overrun {
            return Err(LinearError::LimitExceeded {
                run: run_idx,
                s: s_at,
                joint,
                kind,
                value,
                limit,
            });
        }

        if samples.last().map(|l| l.distance(&q[m - 1])).unwrap_or(1.0) > 1e-9 {
            samples.push(q[m - 1]);
        }

        let out_samples = samples.len();
        let path_out = SRobotPath::try_new(samples).map_err(LinearError::from)?;
        let traj = SRobotTraj::new(c.output_dt, path_out);
        Ok((
            traj,
            LinearRetimerDiagnostic {
                output_samples: out_samples,
                duration: Duration::from_secs_f64((out_samples.saturating_sub(1)) as f64 * dt),
                arc_length: total,
                commanded_speed: c.tcp.speed,
                peak_speed: peak,
                peak_joint_accel: pk_ja,
                peak_joint_jerk: pk_jj,
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

/// `min_j limit_j / |qp_j|` over axes that actually move; `BIG` if none do.
fn project_min<const N: usize>(qp: &SRobotQ<N, f64>, limit: &SRobotQ<N, f64>) -> f64 {
    let mut m = BIG;
    let mut any = false;
    for j in 0..N {
        let g = qp.0[j].abs();
        if g > 1e-9 {
            any = true;
            m = m.min(limit.0[j] / g);
        }
    }
    if any { m } else { BIG }
}

/// Feasible interval for a scalar path-rate control `x` under the per-joint
/// affine constraints `|qp_k·x + c_k| ≤ lim_k` (the chain-rule joint accel or
/// jerk written as `qp·x + const`). Returns `(lo, hi)`; `lo > hi` signals that
/// the constant terms alone already breach a limit — the caller backs off.
#[inline]
fn feasible_interval<const N: usize>(
    qp: &SRobotQ<N, f64>,
    c: &SRobotQ<N, f64>,
    lim: &SRobotQ<N, f64>,
) -> (f64, f64) {
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    for k in 0..N {
        let g = qp.0[k];
        let l = -lim.0[k] - c.0[k]; // qp_k·x ≥ l
        let h = lim.0[k] - c.0[k]; //  qp_k·x ≤ h
        if g > 1e-9 {
            lo = lo.max(l / g);
            hi = hi.min(h / g);
        } else if g < -1e-9 {
            lo = lo.max(h / g);
            hi = hi.min(l / g);
        } else if l > 0.0 || h < 0.0 {
            // qp_k ≈ 0 and 0 ∉ [l, h]: |c_k| > lim_k, infeasible at this speed.
            return (1.0, -1.0);
        }
    }
    (lo, hi)
}

/// Intersect a feasible interval `[lo, hi]` with the symmetric cap `[-c, c]`
/// when `cap` is `Some(c)`; pass it through unchanged when `None`. The result
/// may come back empty (`lo > hi`) if the cap excludes the whole interval — the
/// caller decides how to back off.
#[inline]
fn cap_interval(lo: f64, hi: f64, cap: Option<f64>) -> (f64, f64) {
    match cap {
        Some(c) => (lo.max(-c), hi.min(c)),
        None => (lo, hi),
    }
}

/// Fraction of the available jerk used when planning the terminal stop, so the
/// deceleration is ~80% of the time-optimal jerk and the integrator keeps a
/// margin to the joint jerk limit instead of riding it.
const STOP_JERK_FRACTION: f64 = 0.8;

/// Highest speed from which a jerk- and acceleration-limited deceleration can
/// reach rest within distance `d`. This is the closed-form inverse of the
/// S-curve stopping distance under limits `a`, `j`:
///
/// - `Δv ≤ a²/j` (triangular accel profile, never saturating `a`):
///   `d = v^{3/2} / √j` ⇒ `v = ∛(d²·j)`.
/// - otherwise (a trapezoidal profile with a constant-`a` phase):
///   `d = v²/(2a) + v·a/(2j)` ⇒ the positive root below.
///
/// Used as a per-step velocity ceiling toward the path end so the decel takes
/// the physical jerk-limited distance instead of being dragged to rest across
/// a whole (possibly coarse) input segment.
#[inline]
fn jerk_stop_speed(d: f64, a: f64, j: f64) -> f64 {
    let a = a.max(1e-9);
    let j = j.max(1e-9);
    let v_tri = (d * d * j).cbrt();
    if v_tri <= a * a / j {
        v_tri
    } else {
        let aj = a * a / j;
        0.5 * (-aj + (aj * aj + 8.0 * a * d).sqrt())
    }
}

/// Resample a joint path with a natural cubic spline through the waypoints,
/// emitting points no more than `res` apart in joint-space chord length. The
/// spline interpolates the inputs (zero deviation at the waypoints) and is C²,
/// so the densely-sampled curve has continuous curvature — bounded joint jerk
/// once retimed — and tracks the intended smooth path more closely than the raw
/// piecewise-linear polyline. Endpoints are preserved exactly.
fn spline_resample<const N: usize>(raw: &[SRobotQ<N, f64>], res: f64) -> Vec<SRobotQ<N, f64>> {
    if raw.len() < 3 || res <= 0.0 {
        return raw.to_vec();
    }
    // Drop coincident knots first. A zero-length chord makes the natural cubic
    // spline's tridiagonal system singular (its RHS carries a `1/h` term), and
    // the interpolant then bows wildly off the path — quadrupling the executed
    // arc length on an otherwise straight run. Duplicates arise where the
    // planner samples a segment boundary twice.
    let mut q: Vec<SRobotQ<N, f64>> = Vec::with_capacity(raw.len());
    q.push(raw[0]);
    for &p in &raw[1..] {
        if p.distance(q.last().unwrap()) > 1e-9 {
            q.push(p);
        }
    }
    let m = q.len();
    if m < 3 {
        return q;
    }
    // Parameterize by cumulative joint-space chord length.
    let mut u = vec![0.0f64; m];
    for i in 1..m {
        u[i] = u[i - 1] + q[i].distance(&q[i - 1]);
    }
    if u[m - 1] < 1e-12 {
        return q;
    }
    let h: Vec<f64> = (0..m - 1).map(|i| (u[i + 1] - u[i]).max(1e-12)).collect();
    // Natural cubic spline second derivatives via the Thomas algorithm. The
    // tridiagonal coefficients are scalar (shared by every joint); only the RHS
    // is a vector, so one sweep solves all dimensions. M[0] = M[m-1] = 0.
    let mut cp = vec![0.0f64; m];
    let mut dp = vec![SRobotQ::<N, f64>::zeros(); m];
    for i in 1..m - 1 {
        let (a, b, cc) = (h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]);
        let rhs = ((q[i + 1] - q[i]) * (1.0 / h[i]) - (q[i] - q[i - 1]) * (1.0 / h[i - 1])) * 6.0;
        let denom = b - a * cp[i - 1];
        cp[i] = cc / denom;
        dp[i] = (rhs - dp[i - 1] * a) * (1.0 / denom);
    }
    let mut mm = vec![SRobotQ::<N, f64>::zeros(); m];
    for i in (1..m - 1).rev() {
        mm[i] = dp[i] - mm[i + 1] * cp[i];
    }
    let eval = |i: usize, uu: f64| -> SRobotQ<N, f64> {
        let a = (u[i + 1] - uu) / h[i];
        let b = (uu - u[i]) / h[i];
        q[i] * a
            + q[i + 1] * b
            + (mm[i] * (a * a * a - a) + mm[i + 1] * (b * b * b - b)) * (h[i] * h[i] / 6.0)
    };
    let mut out = Vec::with_capacity(m + (u[m - 1] / res) as usize + 1);
    out.push(q[0]);
    for i in 0..m - 1 {
        let k = ((h[i] / res).ceil() as usize).max(1);
        for ss in 1..=k {
            out.push(eval(i, u[i] + h[i] * (ss as f64) / (k as f64)));
        }
    }
    out
}

/// Bracket `x` against the ascending grid `s`, advancing the forward-only cursor
/// `cur` (kept in `0..s.len()-1`). Returns the segment index `i` with
/// `s[i] <= x <= s[i+1]` and the in-segment fraction `f`, both clamped to the
/// grid range. `inv_ds[i]` is the reciprocal segment length, so the fraction
/// costs a multiply, not a divide. Amortised O(1) over the monotonic sweep.
#[inline]
fn seg(s: &[f64], inv_ds: &[f64], cur: &mut usize, x: f64) -> (usize, f64) {
    let last = s.len() - 1;
    let x = x.clamp(s[0], s[last]);
    while *cur < last - 1 && s[*cur + 1] <= x {
        *cur += 1;
    }
    let f = (x - s[*cur]) * inv_ds[*cur];
    (*cur, f)
}
