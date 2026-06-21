//! Jerk-limited time-optimal trajectory along a [`SplineInterpolatedRobotPath`].

use crate::bspline::solve_dense;
use crate::constraints::Topp3TcpSplineConstraints;
use crate::path::SplineInterpolatedRobotPath;
use deke_types::{ContinuousFKChain, DekeError, DekeResult, SRobotQ};
use glam_traits_ext::{TAffine3, TVec3};
use std::f64::consts::PI;

thread_local! {
    static DT_CACHE: std::cell::Cell<(f64, f64, f64)> =
        const { std::cell::Cell::new((0.001, 0.001 * 0.001 / 2.0, 0.001 * 0.001 * 0.001 / 6.0)) };
}

pub(crate) fn set_dt(dt: f64) {
    DT_CACHE.with(|c| c.set((dt, dt * dt / 2.0, dt * dt * dt / 6.0)));
}
fn dt() -> f64 {
    DT_CACHE.with(|c| c.get().0)
}
fn dt2on2() -> f64 {
    DT_CACHE.with(|c| c.get().1)
}
fn dt3on6() -> f64 {
    DT_CACHE.with(|c| c.get().2)
}

/// Path-parameter state `(s, ṡ, s̈, s⃛)` treated as a piecewise cubic
/// between discrete time-steps of size `dt`.
#[derive(Clone, Copy, Debug)]
pub struct TrajPCS {
    /// `[s, sdot, sddot, sdddot]`
    pub state: [f64; 4],
}

impl Default for TrajPCS {
    fn default() -> Self {
        Self::new()
    }
}

impl TrajPCS {
    pub fn new() -> Self {
        Self { state: [0.0; 4] }
    }

    pub fn from_state(s: f64, sdot: f64, sddot: f64, sdddot: f64) -> Self {
        Self {
            state: [s, sdot, sddot, sdddot],
        }
    }

    #[inline]
    pub fn s(&self) -> f64 {
        self.state[0]
    }
    #[inline]
    pub fn sdot(&self) -> f64 {
        self.state[1]
    }
    #[inline]
    pub fn sddot(&self) -> f64 {
        self.state[2]
    }
    #[inline]
    pub fn sdddot(&self) -> f64 {
        self.state[3]
    }

    pub fn forward_integrate(&self) -> TrajPCS {
        let [s, sdot, sddot, sdddot] = self.state;
        TrajPCS {
            state: [
                s + sdot * dt() + sddot * dt2on2() + sdddot * dt3on6(),
                sdot + sddot * dt() + sdddot * dt2on2(),
                sddot + sdddot * dt(),
                sdddot,
            ],
        }
    }

    pub fn forward_integrate_dt(&self, h: f64) -> TrajPCS {
        let [s, sdot, sddot, sdddot] = self.state;
        let h2 = h * h;
        let h3 = h2 * h;
        TrajPCS {
            state: [
                s + sdot * h + sddot * h2 / 2.0 + sdddot * h3 / 6.0,
                sdot + sddot * h + sdddot * h2 / 2.0,
                sddot + sdddot * h,
                sdddot,
            ],
        }
    }

    pub fn jerk_to_reach_s(&self, target_s: f64) -> f64 {
        let [s, sdot, sddot, _] = self.state;
        (target_s - s - dt() * sdot - dt2on2() * sddot) / dt3on6()
    }

    pub fn jerk_to_reach_sdot(&self, target_sdot: f64) -> f64 {
        let [_, sdot, sddot, _] = self.state;
        (target_sdot - sdot - dt() * sddot) / dt2on2()
    }

    pub fn jerk_to_reach_sddot(&self, target_sddot: f64) -> f64 {
        let [_, _, sddot, _] = self.state;
        (target_sddot - sddot) / dt()
    }

    /// Solve for three constant-jerk segments connecting `start` to `end`.
    /// Returns four states: the start of each segment plus the final state.
    pub fn solve_boundary_condition(start: &TrajPCS, end: &TrajPCS) -> [TrajPCS; 4] {
        let h = dt();
        let [s0, v0, a0, _] = start.state;
        let [sf, vf, af, _] = end.state;

        let mut mat = vec![
            vec![1.0, 1.0, 1.0],
            vec![5.0, 3.0, 1.0],
            vec![19.0, 7.0, 1.0],
        ];
        let h2 = h * h;
        let h3 = h2 * h;
        let mut rhs = vec![
            (af - a0) / h,
            (vf - v0 - 3.0 * a0 * h) / (0.5 * h2),
            (sf - s0 - 3.0 * v0 * h - 4.5 * a0 * h2) / (h3 / 6.0),
        ];
        solve_dense(&mut mat, &mut rhs);
        let [j0, j1, j2] = [rhs[0], rhs[1], rhs[2]];

        let mut seg0 = *start;
        seg0.state[3] = j0;
        let mut state1 = seg0.forward_integrate();

        state1.state[3] = j1;
        let mut state2 = state1.forward_integrate();

        state2.state[3] = j2;
        let state3 = state2.forward_integrate();

        [seg0, state1, state2, state3]
    }
}

#[inline]
fn vec3_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn vec3_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn vec3_norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline]
fn vec3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Multiply the first 3 (position) rows of a 6×N Jacobian by an N-vector.
#[inline]
fn j_pos_dot<const N: usize>(j: &[[f64; N]; 6], v: &SRobotQ<N, f64>) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for i in 0..3 {
        let mut s = 0.0f64;
        for (jik, vk) in j[i].iter().zip(&v.0) {
            s += jik * vk;
        }
        out[i] = s;
    }
    out
}

/// Jerk-limited trajectory along a [`SplineInterpolatedRobotPath`].
pub struct Trajectory<'a, const N: usize, FK: ContinuousFKChain<N, f64>> {
    pub(crate) fk: &'a FK,
    pub(crate) path: &'a SplineInterpolatedRobotPath<N>,
    pub(crate) dt_val: f64,
    pub(crate) start_sdot: f64,
    pub(crate) end_sdot: f64,
    #[allow(dead_code)]
    pub(crate) max_sdot: f64,
    pub(crate) verify_dt: f64,
    pub(crate) constraints: &'a Topp3TcpSplineConstraints<N>,
    /// Upper bound on `|sdddot|` used by `state_could_possibly_reach_target`'s
    /// pruning cutoff. Derived from the configured joint/TCP jerk limits.
    pub(crate) cached_jerk_threshold: f64,
    pub(crate) states: Vec<TrajPCS>,
}

impl<'a, const N: usize, FK: ContinuousFKChain<N, f64>> Trajectory<'a, N, FK> {
    pub fn new(
        fk: &'a FK,
        path: &'a SplineInterpolatedRobotPath<N>,
        constraints: &'a Topp3TcpSplineConstraints<N>,
    ) -> Self {
        let dt_val = constraints.search.dt;
        set_dt(dt_val);
        // The heuristic in `state_could_possibly_reach_target` rejects states
        // whose minimum-deceleration jerk `sddot²/(2|dv|)` exceeds what the
        // constraints physically allow. We use the largest finite per-axis
        // joint jerk as the bound; TCP jerk doesn't translate directly to an
        // sdddot ceiling because qp scaling enters, so we exclude it. A 2×
        // safety factor keeps the cutoff slightly loose so corner cases don't
        // get pruned by FP noise. Defaults to 1.0 when no finite limit exists
        // (matches the reference's behavior when all jerk limits are set to
        // small unit defaults).
        let j_max_axis = constraints
            .joint
            .j_max
            .0
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(0.0_f64, f64::max);
        let cached_jerk_threshold = if j_max_axis > 0.0 {
            2.0 * j_max_axis
        } else {
            1.0
        };
        Self {
            fk,
            path,
            dt_val,
            start_sdot: constraints.search.start_sdot,
            end_sdot: constraints.search.end_sdot,
            max_sdot: constraints.search.max_sdot,
            verify_dt: constraints.search.verify_dt,
            constraints,
            cached_jerk_threshold,
            states: Vec::new(),
        }
    }

    pub fn dt(&self) -> f64 {
        self.dt_val
    }

    pub fn time_to_complete(&self) -> f64 {
        if self.states.is_empty() {
            return 0.0;
        }
        (self.states.len() - 1) as f64 * self.dt_val
    }

    /// Evaluate joint-space trajectory at given times.
    pub fn joint_fk(
        &self,
        times: &[f64],
    ) -> (
        Vec<SRobotQ<N, f64>>,
        Vec<SRobotQ<N, f64>>,
        Vec<SRobotQ<N, f64>>,
        Vec<SRobotQ<N, f64>>,
    ) {
        let t_max = self.time_to_complete();
        let n_states = self.states.len();
        let mut q_out = Vec::with_capacity(times.len());
        let mut qd_out = Vec::with_capacity(times.len());
        let mut qdd_out = Vec::with_capacity(times.len());
        let mut qddd_out = Vec::with_capacity(times.len());

        for &ti in times {
            let ti_c = ti.clamp(0.0, t_max);
            let idx = if n_states > 0 {
                (ti_c / self.dt_val).floor() as usize
            } else {
                0
            }
            .min(n_states.saturating_sub(1));
            let offset = ti_c - idx as f64 * self.dt_val;
            let state = self.states[idx].forward_integrate_dt(offset);
            let [s, sdot, sddot, sdddot] = state.state;

            let (qi, qp, qpp, qppp) = self.path.eval(s);
            let qdot = qp * sdot;
            let qddot = qpp * (sdot * sdot) + qp * sddot;
            let qdddot = qppp * sdot.powi(3) + qpp * (3.0 * sdot * sddot) + qp * sdddot;

            q_out.push(qi);
            qd_out.push(qdot);
            qdd_out.push(qddot);
            qddd_out.push(qdddot);
        }
        (q_out, qd_out, qdd_out, qddd_out)
    }

    fn jerk_range_from_jerk_constraints(&self, state: &TrajPCS) -> DekeResult<(f64, f64)> {
        let [s, sdot, sddot, _] = state.state;
        let (q, qp, qpp, qppp) = self.path.eval(s);
        let eps = 1e-12;

        let qdot = qp * sdot;
        let qddot = qpp * (sdot * sdot) + qp * sddot;

        let sdddot_max_from_position = state.jerk_to_reach_s(1.0);

        // Joint jerk: jerk_i = qp_i * sdddot + (qppp_i*sdot³ + qpp_i*3*sdot*sddot)
        let joint_base = qppp * (sdot * sdot * sdot) + qpp * (3.0 * sdot * sddot);
        let joint_coeff = qp;

        let mut sdddot_min_j = f64::NEG_INFINITY;
        let mut sdddot_max_j = f64::INFINITY;
        for i in 0..N {
            let base = joint_base.0[i];
            let coeff = joint_coeff.0[i];
            let lim = self.constraints.joint.j_max.0[i];
            if coeff.abs() <= eps {
                if base.abs() > lim + eps {
                    return Ok((0.0, 0.0));
                }
                continue;
            }
            let mut r0 = (-lim - base) / coeff;
            let mut r1 = (lim - base) / coeff;
            if r0 > r1 {
                std::mem::swap(&mut r0, &mut r1);
            }
            sdddot_min_j = sdddot_min_j.max(r0);
            sdddot_max_j = sdddot_max_j.min(r1);
        }

        // TCP jerk: a quadratic in sdddot.  tcp_jerk = J·qdddot + 2·J̇·qddot + J̈·qdot
        // where qdddot = qp*sdddot + joint_base; the qp*sdddot part folds into the
        // linear coefficient and joint_base into the constant.
        let j_mat = self.fk.jacobian(&q)?;
        let jd_mat = self.fk.jacobian_dot(&q, &qdot)?;
        let jdd_mat = self.fk.jacobian_ddot(&q, &qdot, &qddot)?;

        let tcp_base = {
            let a = j_pos_dot(&j_mat, &joint_base);
            let b = vec3_scale(j_pos_dot(&jd_mat, &qddot), 2.0);
            let c = j_pos_dot(&jdd_mat, &qdot);
            vec3_add(vec3_add(a, b), c)
        };
        let tcp_coeff = j_pos_dot(&j_mat, &qp);

        let max_tcp_jrk = self.constraints.tcp.j_max;
        let aa = vec3_dot(tcp_coeff, tcp_coeff);
        let bb = 2.0 * vec3_dot(tcp_base, tcp_coeff);
        let cc = vec3_dot(tcp_base, tcp_base) - max_tcp_jrk * max_tcp_jrk;

        let mut sdddot_min_tcp = f64::NEG_INFINITY;
        let mut sdddot_max_tcp = f64::INFINITY;
        if aa > eps {
            let disc = bb * bb - 4.0 * aa * cc;
            if disc < 0.0 {
                return Ok((0.0, 0.0));
            }
            let sd = disc.sqrt();
            let mut r0 = (-bb - sd) / (2.0 * aa);
            let mut r1 = (-bb + sd) / (2.0 * aa);
            if r0 > r1 {
                std::mem::swap(&mut r0, &mut r1);
            }
            sdddot_min_tcp = r0;
            sdddot_max_tcp = r1;
        } else if cc > 0.0 {
            return Ok((0.0, 0.0));
        }

        let sdddot_min = sdddot_min_j.max(sdddot_min_tcp);
        let sdddot_max = sdddot_max_from_position
            .min(sdddot_max_j)
            .min(sdddot_max_tcp);
        Ok((sdddot_min, sdddot_max))
    }

    /// Heuristic feasibility cutoff. With `dv = target.sdot − state.sdot` and
    /// `state.sddot ≠ 0`, the quantity `sddot²/(2·|dv|)` is the minimum
    /// path-jerk magnitude required to bring `sddot` to zero before `sdot`
    /// passes `target.sdot` (deceleration-only braking). If that minimum
    /// exceeds the largest sdddot the constraints allow, no feasible
    /// continuation exists, so the DFS prunes the branch.
    ///
    /// The reference implementation hardcodes the threshold to `175.0`,
    /// which only works when the configured joint/TCP jerk limits happen
    /// to land near that value (the reference's defaults were
    /// `j_max ≈ 100`). For arbitrary user limits the constant must scale
    /// with them — otherwise short, feasible paths get pruned at the root
    /// and the search exhausts immediately, even though the trajectory is
    /// trivially achievable.
    fn state_could_possibly_reach_target(&self, state: &TrajPCS, target: &TrajPCS) -> bool {
        let dv = target.sdot() - state.sdot();
        if dv.abs() < 1e-30 {
            return true;
        }
        // Only the deceleration case (dv < 0, i.e. state.sdot > target.sdot)
        // needs the bound. When dv > 0 we're accelerating toward the target
        // and the heuristic doesn't apply.
        if dv >= 0.0 {
            return true;
        }
        let filter = state.sddot().powi(2) / (-2.0 * dv);
        filter <= self.cached_jerk_threshold
    }

    fn fused_utilization_at(&self, state: &TrajPCS) -> DekeResult<f64> {
        let [s, sdot, sddot, sdddot] = state.state;
        let (q, qp, qpp, qppp) = self.path.eval(s);

        let sdot2 = sdot * sdot;
        let qdot = qp * sdot;
        let qddot = qpp * sdot2 + qp * sddot;
        let qdddot = qppp * (sdot2 * sdot) + qpp * (3.0 * sdot * sddot) + qp * sdddot;

        let joint_vel_util = qdot
            .elementwise_div(&self.constraints.joint.v_max)
            .abs()
            .max_element();
        let joint_acc_util = qddot
            .elementwise_div(&self.constraints.joint.a_max)
            .abs()
            .max_element();
        let joint_jrk_util = qdddot
            .elementwise_div(&self.constraints.joint.j_max)
            .abs()
            .max_element();
        let joint_util = joint_vel_util.max(joint_acc_util).max(joint_jrk_util);

        // Always computed alongside `joint_util` rather than gated behind a
        // `joint_util < 1.0` early-exit. Skipping TCP when joint was already
        // saturated meant the TCP cap could never *co-bind* with joint at
        // ≥ 1.0 — the DFS would prune candidates the moment joint hit limit,
        // even when those candidates were equally-binding on TCP. The cost
        // is three FK Jacobian calls per intermediate-check; with the
        // `verify_dt = output_dt` tightening this is the dominant cost
        // anyway, and the resulting search picks jerks that respect both
        // limit families simultaneously.
        let j_mat = self.fk.jacobian(&q)?;
        let jd_mat = self.fk.jacobian_dot(&q, &qdot)?;
        let jdd_mat = self.fk.jacobian_ddot(&q, &qdot, &qddot)?;

        let tcp = &self.constraints.tcp;
        let tcp_vel = j_pos_dot(&j_mat, &qdot);
        let tcp_vel_util = vec3_norm(tcp_vel) / tcp.v_max;

        let tcp_acc = vec3_add(j_pos_dot(&j_mat, &qddot), j_pos_dot(&jd_mat, &qdot));
        let tcp_acc_util = vec3_norm(tcp_acc) / tcp.a_max;

        let tcp_jrk = vec3_add(
            vec3_add(
                j_pos_dot(&j_mat, &qdddot),
                vec3_scale(j_pos_dot(&jd_mat, &qddot), 2.0),
            ),
            j_pos_dot(&jdd_mat, &qdot),
        );
        let tcp_jrk_util = vec3_norm(tcp_jrk) / tcp.j_max;
        let tcp_util = tcp_vel_util.max(tcp_acc_util).max(tcp_jrk_util);

        // Return the binding side; downstream pruning only cares about
        // `≥ 1.0`, so the relative ordering of joint vs. TCP doesn't matter
        // beyond which one carries the larger reading.
        Ok(joint_util.max(tcp_util))
    }

    /// Return `(times, s, sdot, sddot, sdddot)` arrays.
    pub fn get_s_state_arrays(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.states.len();
        let mut times = Vec::with_capacity(n);
        let mut s = Vec::with_capacity(n);
        let mut sdot = Vec::with_capacity(n);
        let mut sddot = Vec::with_capacity(n);
        let mut sdddot = Vec::with_capacity(n);
        for (i, st) in self.states.iter().enumerate() {
            times.push(i as f64 * self.dt_val);
            s.push(st.state[0]);
            sdot.push(st.state[1]);
            sddot.push(st.state[2]);
            sdddot.push(st.state[3]);
        }
        (times, s, sdot, sddot, sdddot)
    }

    pub fn states(&self) -> &[TrajPCS] {
        &self.states
    }

    pub fn into_states(self) -> Vec<TrajPCS> {
        self.states
    }

    /// Apply `n_passes` of binomial-kernel smoothing to the per-segment
    /// `sdddot` schedule, re-integrating the state chain from the start each
    /// pass. Each pass replaces interior `sdddot[k]` with
    /// `(sdddot[k-1] + 2·sdddot[k] + sdddot[k+1]) / 4`; endpoints are left
    /// alone so the boundary jerks stay at their DFS-validated values.
    ///
    /// After smoothing, the chain integrates to a different `s_final` than
    /// the DFS landed on; we uniformly rescale time so it lands at exactly
    /// `1` (matching the path endpoint). The rescale identity
    /// `t → α·t → (sdot/α, sddot/α², sdddot/α³)` preserves every analytical
    /// derivative under its limit when `α ≥ 1` (slowing) and only relaxes
    /// the constraints with `α < 1` while the path-traversal stays valid —
    /// either way, smoothing + rescale stays inside the limits the DFS
    /// already verified.
    ///
    /// The net effect: jerk discontinuities at segment boundaries are
    /// halved per pass, which directly halves the spike that the
    /// 4-point backward-FD jerk stencil reads on the output grid when it
    /// straddles a boundary.
    pub fn smooth_jerks(&mut self, n_passes: u32) {
        if n_passes == 0 || self.states.len() < 3 {
            return;
        }
        for _ in 0..n_passes {
            self.smooth_jerks_once();
        }
    }

    fn smooth_jerks_once(&mut self) {
        let n = self.states.len();
        debug_assert!(n >= 3);
        // Pull out the current jerks; leave endpoint segments alone so the
        // start/end boundary states keep their DFS-validated jerk.
        let mut new_jerks: Vec<f64> = self.states.iter().map(|s| s.state[3]).collect();
        #[allow(clippy::needless_range_loop)]
        for k in 1..(n - 1) {
            let j_prev = self.states[k - 1].state[3];
            let j_curr = self.states[k].state[3];
            let j_next = self.states[k + 1].state[3];
            new_jerks[k] = 0.25 * j_prev + 0.5 * j_curr + 0.25 * j_next;
        }
        // Re-integrate from the start with the smoothed schedule.
        let dt_val = self.dt_val;
        set_dt(dt_val);
        let mut rebuilt = Vec::with_capacity(n);
        let mut state = self.states[0];
        state.state[3] = new_jerks[0];
        rebuilt.push(state);
        for k in 0..(n - 1) {
            let mut anchor = rebuilt[k];
            anchor.state[3] = new_jerks[k];
            let mut next = anchor.forward_integrate();
            next.state[3] = new_jerks[k + 1];
            rebuilt.push(next);
        }
        // Uniformly rescale time so `s_final` lands at 1.0 exactly. With
        // smoothing changing the integrated arc length by a few percent at
        // most, `alpha` is close to 1 and the rescaled state's kinematic
        // limits (which scale as `1/α^k` for k-th derivative) stay within
        // the DFS-validated bounds when α ≥ 1 (slowing).
        let s_final = rebuilt.last().unwrap().state[0];
        if s_final > 1e-9 && (s_final - 1.0).abs() > 1e-9 {
            // We have `s(α·t_max) = 1`. Path-parameter state at α·t scales
            // as `(s, sdot/α, sddot/α², sdddot/α³)`. To get `s_final → 1`
            // we need a different α at each kinematic order — but the
            // closed form `α = 1/s_final` only works if we *also*
            // rescale the dt grid by α. We do that by leaving the
            // per-state values in place and reinterpreting the time
            // axis: the new dt is `α · dt_val`. But the rest of the
            // pipeline assumes a uniform dt_val, so easier to scale the
            // state values in place and keep dt_val the same. Apply the
            // rescale `t → α·t` so the schedule still covers
            // `[0, n_segments · dt_val]` but reaches `s = 1` at the end.
            let alpha = s_final; // s_final · sdot_after = 1, sdot_after = sdot_before / α, where α = ?
            // Actually the correct scaling: if we keep the dt_val grid but
            // rescale state values, every `sdot[k]` scales by `1/α`,
            // every `sddot[k]` by `1/α²`, every `sdddot[k]` by `1/α³`,
            // and the integrated `s[k]` by `1/α`. We want s_final → 1, so
            // we need `α = s_final`.
            let alpha2 = alpha * alpha;
            let alpha3 = alpha2 * alpha;
            for state in &mut rebuilt {
                state.state[0] /= alpha;
                state.state[1] /= alpha;
                state.state[2] /= alpha2;
                state.state[3] /= alpha3;
            }
            // After this rescale, `s_final = 1.0` (within FP). All
            // kinematic readings are smaller in magnitude (when alpha > 1
            // / slowing) so analytical limits remain satisfied.
        }
        self.states = rebuilt;
    }

    /// Measure the peak backward-FD readout overshoot across the resampled
    /// output. Returns the smallest time-rescale factor `α ≥ 1` such that
    /// after the trajectory is slowed by `α`, every sample's backward-FD
    /// V/A/J reading lands at or below `1.0 + slack` of its limit.
    ///
    /// Returns `1.0` when no rescale is needed.
    ///
    /// Stencils: 2-point backward FD for joint & TCP velocity, 3-point for
    /// joint acceleration, 4-point for joint jerk. Order-aware scaling:
    /// a velocity overshoot `r` needs `α = r`, acceleration needs
    /// `α = √r`, jerk needs `α = ³√r` (since `v` scales as `1/α`, `a`
    /// as `1/α²`, `j` as `1/α³`). The returned `α` is the max across
    /// all three orders.
    pub fn peak_fd_overshoot_scale(
        &self,
        samples: &[SRobotQ<N, f64>],
        output_dt: f64,
        slack: f64,
    ) -> DekeResult<f64> {
        if samples.len() < 4 || output_dt <= 0.0 {
            return Ok(1.0);
        }
        let limit = 1.0 + slack.max(0.0);
        let v_max = &self.constraints.joint.v_max;
        let a_max = &self.constraints.joint.a_max;
        let j_max = &self.constraints.joint.j_max;
        let tcp = &self.constraints.tcp;
        let dt = output_dt;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let mut scale = 1.0_f64;
        // Cache FK end-effector positions for the TCP-velocity FD reading.
        // The pose is needed at every sample regardless of constraint, so
        // hoist out of the inner loop.
        let mut tcp_pos: Vec<[f64; 3]> = Vec::with_capacity(samples.len());
        for q in samples {
            let pose = self.fk.fk_end(q)?;
            let t = pose.translation();
            tcp_pos.push([t.x(), t.y(), t.z()]);
        }
        for k in 3..samples.len() {
            let q0 = samples[k];
            let q1 = samples[k - 1];
            let q2 = samples[k - 2];
            let q3 = samples[k - 3];
            for j in 0..N {
                let v = (q0.0[j] - q1.0[j]) / dt;
                let u = v.abs() / v_max.0[j];
                if u > limit {
                    scale = scale.max(u / limit);
                }
                let a = (q0.0[j] - 2.0 * q1.0[j] + q2.0[j]) / dt2;
                let u = a.abs() / a_max.0[j];
                if u > limit {
                    scale = scale.max((u / limit).sqrt());
                }
                let jk = (q0.0[j] - 3.0 * q1.0[j] + 3.0 * q2.0[j] - q3.0[j]) / dt3;
                let u = jk.abs() / j_max.0[j];
                if u > limit {
                    scale = scale.max((u / limit).cbrt());
                }
            }
            let p0 = tcp_pos[k];
            let p1 = tcp_pos[k - 1];
            let dv = [
                (p0[0] - p1[0]) / dt,
                (p0[1] - p1[1]) / dt,
                (p0[2] - p1[2]) / dt,
            ];
            let v = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
            let u = v / tcp.v_max;
            if u > limit {
                scale = scale.max(u / limit);
            }
        }
        Ok(scale)
    }

    /// Uniformly slow the state schedule by factor `alpha ≥ 1`. The path
    /// is preserved exactly (every `s` value unchanged); the time axis
    /// stretches by `α`, which scales `sdot → sdot/α`, `sddot → sddot/α²`,
    /// `sdddot → sdddot/α³`. After this rescale, calling `resample_to`
    /// produces a sample sequence that takes `α × original_time` and
    /// whose backward-FD readings scale down accordingly.
    pub fn rescale_time_in_place(&mut self, alpha: f64) {
        if !matches!(alpha.partial_cmp(&1.0), Some(std::cmp::Ordering::Greater))
            || !alpha.is_finite()
        {
            return;
        }
        let inv = 1.0 / alpha;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        for state in &mut self.states {
            // s unchanged
            state.state[1] *= inv;
            state.state[2] *= inv2;
            state.state[3] *= inv3;
        }
        self.dt_val *= alpha;
    }

    /// Resample the converged state list onto an arbitrary `output_dt` grid.
    ///
    /// The DFS produces one state per `self.dt_val` (the search step). Each
    /// segment carries constant `sdddot`, so within a segment the state is a
    /// closed-form cubic in elapsed time — we just call
    /// [`TrajPCS::forward_integrate_dt`] from the segment anchor.
    ///
    /// Output samples are placed at `t = 0, output_dt, 2·output_dt, …` up to
    /// and including the trajectory end (the final state is pinned exactly,
    /// regardless of whether `total_time` lands on the `output_dt` grid).
    /// If `output_dt <= 0` or there's nothing to resample, a clone of the
    /// existing states is returned unchanged.
    pub fn resample_to(&self, output_dt: f64) -> Vec<TrajPCS> {
        let dt_dfs = self.dt_val;
        if output_dt <= 0.0 || self.states.len() < 2 || dt_dfs <= 0.0 {
            return self.states.clone();
        }
        let n_segments = self.states.len() - 1;
        let total_time = n_segments as f64 * dt_dfs;
        let n_samples = ((total_time / output_dt).floor() as usize).max(1) + 1;
        let mut out = Vec::with_capacity(n_samples + 1);
        for i in 0..n_samples {
            let t = i as f64 * output_dt;
            let mut k = (t / dt_dfs).floor() as usize;
            if k >= n_segments {
                k = n_segments - 1;
            }
            let tau = t - k as f64 * dt_dfs;
            out.push(self.states[k].forward_integrate_dt(tau));
        }
        // Pin the endpoint to the exact final DFS state. Either append (if
        // `total_time` didn't land on the output grid) or overwrite the
        // last sample (if it did, within FP tolerance).
        let last_t = (n_samples - 1) as f64 * output_dt;
        if (total_time - last_t).abs() > 1e-9 {
            out.push(*self.states.last().unwrap());
        } else {
            *out.last_mut().unwrap() = *self.states.last().unwrap();
        }
        out
    }

    /// Sample TCP position (linear part of `fk_end`) at joint configuration `q`.
    #[allow(dead_code)]
    fn tcp_position(&self, q: &SRobotQ<N, f64>) -> DekeResult<[f64; 3]> {
        let pose = self.fk.fk_end(q)?;
        let t = pose.translation();
        Ok([t.x(), t.y(), t.z()])
    }

    /// Run the time-optimal trajectory search.  Populates `self.states`.
    pub fn optimize(&mut self) -> DekeResult<()> {
        set_dt(self.dt_val);

        let mut start = TrajPCS::new();
        start.state[1] = self.start_sdot;
        let mut target = TrajPCS::new();
        target.state[0] = 1.0;
        target.state[1] = self.end_sdot;

        type NodeIter = std::vec::IntoIter<f64>;
        let mut stack: Vec<(TrajPCS, NodeIter)> = Vec::new();
        let mut path: Vec<TrajPCS> = vec![start];

        let (sdddot_min, sdddot_max) = self.jerk_range_from_jerk_constraints(&start)?;
        let jerks = Self::make_jerk_candidates(sdddot_min, sdddot_max, &[0.0]);
        stack.push((start, jerks.into_iter()));

        let verify_dt = self.verify_dt;
        let dt_val = self.dt_val;

        while let Some((node_state, node_iter)) = stack.last_mut() {
            let Some(sdddot_next) = node_iter.next() else {
                stack.pop();
                path.pop();
                continue;
            };

            node_state.state[3] = sdddot_next;
            let ns = *node_state;
            let next_node = ns.forward_integrate();

            if let Some(last) = path.last_mut() {
                last.state[3] = sdddot_next;
            }

            let path_len = path.len();
            let t_base = (path_len as f64) * dt_val;
            let phase = t_base % verify_dt;
            let first_check = if phase < 1e-9 || (verify_dt - phase) < 1e-9 {
                verify_dt
            } else {
                verify_dt - phase
            };

            let mut skip = false;
            let mut check_dt = first_check;
            while check_dt < dt_val + 1e-12 {
                let intermediate = ns.forward_integrate_dt(check_dt);
                if self.fused_utilization_at(&intermediate)? >= 1.0 {
                    skip = true;
                    break;
                }
                check_dt += verify_dt;
            }
            if skip {
                continue;
            }

            if next_node.sdot() > 0.0
                && next_node.s() < 1.0
                && self.fused_utilization_at(&next_node)? < 0.99
            {
                if next_node.s() >= 0.7 {
                    let last_steps = TrajPCS::solve_boundary_condition(&next_node, &target);
                    let mut valid = true;
                    for (si, st) in last_steps.iter().enumerate() {
                        if st.sdot() < 0.0 || self.fused_utilization_at(st)? > 1.0 {
                            valid = false;
                            break;
                        }
                        if si == 0 {
                            continue;
                        }
                        // Integrate intermediates from the start of this
                        // boundary segment using *its* constant jerk.
                        // last_steps[0] (seg0) carries jerk j0; last_steps[1]
                        // carries j1; last_steps[2] carries j2.  Using
                        // `next_node` here instead of `last_steps[0]` would
                        // integrate with the prior DFS step's jerk, not j0.
                        let prev = &last_steps[si - 1];
                        let mut ck = verify_dt;
                        while ck < dt_val - 1e-9 {
                            let inter = prev.forward_integrate_dt(ck);
                            if self.fused_utilization_at(&inter)? >= 1.0 {
                                valid = false;
                                break;
                            }
                            ck += verify_dt;
                        }
                        if !valid {
                            break;
                        }
                    }
                    if valid {
                        path.extend_from_slice(&last_steps);
                        self.states = path;
                        return Ok(());
                    }
                }

                let (sdddot_min, sdddot_max) = self.jerk_range_from_jerk_constraints(&next_node)?;
                // Optional jerk-jump cap: restrict the next segment's
                // jerk to within `max_jerk_jump` of the prior segment's
                // jerk (= `next_node.state[3]`, the jerk just applied
                // to produce `next_node`). Bounds the FD-jerk spike at
                // the upcoming segment boundary, which scales as
                // `|qp| × |Δsdddot|`. Smaller caps reduce spikes but
                // also shrink the DFS's effective branching factor.
                let (sdddot_min, sdddot_max) = match self.constraints.search.max_jerk_jump {
                    Some(max_jump) if max_jump > 0.0 => {
                        let prev_jerk = next_node.state[3];
                        let lo = (prev_jerk - max_jump).max(sdddot_min);
                        let hi = (prev_jerk + max_jump).min(sdddot_max);
                        (lo, hi)
                    }
                    _ => (sdddot_min, sdddot_max),
                };
                if sdddot_max < sdddot_min {
                    continue;
                }
                if !self.state_could_possibly_reach_target(&next_node, &target) {
                    continue;
                }

                let jerk_to_zero_acc = next_node.jerk_to_reach_sddot(0.0);
                let mut specials = vec![0.0];
                if jerk_to_zero_acc > sdddot_min && jerk_to_zero_acc < sdddot_max {
                    specials.push(jerk_to_zero_acc);
                }
                let jerks = Self::make_jerk_candidates(sdddot_min, sdddot_max, &specials);

                path.push(next_node);
                stack.push((next_node, jerks.into_iter()));
            }
        }

        Err(DekeError::RetimerFailed(
            "topp3tcp-spline: depth-first search exhausted all jerk candidates".to_string(),
        ))
    }

    fn make_jerk_candidates(min: f64, max: f64, specials: &[f64]) -> Vec<f64> {
        let (cmin, cmax) = match (min.is_finite(), max.is_finite()) {
            (true, true) => (min, max),
            (false, true) => (-max.abs(), max),
            (true, false) => (min, min.abs()),
            (false, false) => (-1.0, 1.0),
        };
        let n = 12usize;
        let mut vals: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                let cos_t = (t * PI).cos();
                let u = (1.0 - cos_t) / 2.0;
                cmax + u * (cmin - cmax)
            })
            .collect();
        vals.extend_from_slice(specials);
        vals.retain(|v| v.is_finite());
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup_by(|a, b| (*a - *b).abs() < 1e-14);
        vals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_integrate() {
        set_dt(0.01);
        let s = TrajPCS::from_state(0.0, 1.0, 0.0, 0.0);
        let next = s.forward_integrate();
        assert!((next.s() - 0.01).abs() < 1e-12);
        assert!((next.sdot() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jerk_to_reach() {
        set_dt(0.01);
        let s = TrajPCS::from_state(0.0, 0.0, 0.0, 0.0);
        let j = s.jerk_to_reach_s(0.5);
        let mut s2 = s;
        s2.state[3] = j;
        let next = s2.forward_integrate();
        assert!((next.s() - 0.5).abs() < 1e-10);
    }
}
