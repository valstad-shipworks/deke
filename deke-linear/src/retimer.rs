//! Stage C — time-parameterise a joint path at constant TCP speed.
//!
//! This is a CNC-style constant-feedrate planner, not a TOPP retimer. The
//! feasible-speed ceiling along the path (the "maximum velocity curve") comes from
//! the per-joint v/a/j limits projected onto the path tangent `q'(s)`. The
//! commanded speed `tcp_speed` is held flat wherever that ceiling allows; near a
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
        let q: Vec<SRobotQ<N, f64>> = path.iter().copied().collect();
        let m = q.len();
        let dt = c.output_dt.as_secs_f64().max(1e-6);

        // Cartesian arc length from FK end positions (true metres for `tcp_speed`).
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
                    commanded_speed: c.tcp_speed,
                    peak_speed: 0.0,
                },
            ));
        }

        // q'(s) by central difference over arc length.
        let mut qp = vec![SRobotQ::<N, f64>::zeros(); m];
        for (i, qpi) in qp.iter_mut().enumerate() {
            let (lo, hi) = if i == 0 {
                (0, 1)
            } else if i == m - 1 {
                (m - 2, m - 1)
            } else {
                (i - 1, i + 1)
            };
            let ds = (s[hi] - s[lo]).max(1e-12);
            *qpi = (q[hi] - q[lo]) * (1.0 / ds);
        }

        // Raw geometric speed ceiling per sample (before clamping to the command).
        // A dip below the command here is forced by joint v-limits + path curvature,
        // i.e. a corner or near-singular patch — distinct from the temporal rest
        // ramps the MVC adds at the ends.
        if c.forbid_interior_dips {
            let mut worst: Option<(usize, f64)> = None;
            #[allow(clippy::needless_range_loop)]
            for i in 1..m - 1 {
                let g = project_min(&qp[i], &c.joint.v_max);
                if g < c.tcp_speed * (1.0 - 1e-3) && worst.is_none_or(|(_, gw)| g < gw) {
                    worst = Some((i, g));
                }
            }
            if let Some((i, g)) = worst {
                return Err(LinearError::SpeedDipRequired {
                    run: run_idx,
                    s: s[i],
                    feasible_speed: g,
                    commanded: c.tcp_speed,
                });
            }
        }

        let v_ceiling = |i: usize| project_min(&qp[i], &c.joint.v_max).min(c.tcp_speed);
        let a_path: Vec<f64> = (0..m)
            .map(|i| project_min(&qp[i], &c.joint.a_max))
            .collect();
        let j_path: Vec<f64> = (0..m)
            .map(|i| project_min(&qp[i], &c.joint.j_max))
            .collect();

        // Acceleration-bounded velocity ceiling. Only the end is pinned to rest;
        // start-from-rest is the integrator's initial condition (v = 0), not a
        // ceiling — pinning the start too would forbid ever accelerating.
        let mut vc: Vec<f64> = (0..m).map(v_ceiling).collect();
        vc[m - 1] = 0.0;
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
        for i in 0..seg_n {
            let ds = s[i + 1] - s[i];
            inv_ds[i] = if ds > 0.0 { 1.0 / ds } else { 0.0 };
            vc_d[i] = vc[i + 1] - vc[i];
            a_d[i] = a_path[i + 1] - a_path[i];
            j_d[i] = j_path[i + 1] - j_path[i];
            dq[i] = q[i + 1] - q[i];
        }

        // Forward jerk-limited time integration tracking the ceiling. The flat
        // estimate `total / (tcp_speed·dt)` is a lower bound on the step count
        // (real speed never exceeds the command); doubling it covers the rest
        // ramps so the buffer almost never reallocates mid-sweep.
        let est = (total / (c.tcp_speed.max(1e-6) * dt)) as usize;
        let mut samples: Vec<SRobotQ<N, f64>> = Vec::with_capacity(est * 2 + 16);
        samples.push(q[0]);
        let mut sx = 0.0f64;
        let mut v = 0.0f64;
        let mut a = 0.0f64;
        let mut peak = 0.0f64;
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
            let vlim = (vc[i] + vc_d[i] * f).max(0.0);
            let alim = a_path[i] + a_d[i] * f;
            let jlim = j_path[i] + j_d[i] * f;

            let a_des = ((vlim - v) / dt).clamp(-alim, alim);
            a = a_des.clamp(a - jlim * dt, a + jlim * dt).clamp(-alim, alim);
            v = (v + a * dt).clamp(0.0, vlim);
            peak = peak.max(v);
            sx += v * dt;
            (i, f) = seg(&s, &inv_ds, &mut cur, sx.min(total));
            samples.push(q[i] + dq[i] * f);

            // Guard against a stall at a vanishing ceiling (true singularity).
            if v < 1e-9 && vlim < 1e-9 && sx < total - 1e-6 {
                return Err(LinearError::Stalled {
                    run: run_idx,
                    s: sx,
                });
            }
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
                commanded_speed: c.tcp_speed,
                peak_speed: peak,
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
                    commanded_speed: constraints.tcp_speed,
                    peak_speed: 0.0,
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
