//! Jerk-limited time-optimal trajectory along a [`SplineInterpolatedRobotPath`].

use crate::bspline::solve_dense;
use crate::constraints::Topp3TcpSplineConstraints;
use crate::path::SplineInterpolatedRobotPath;
use deke_types::{DekeError, DekeResult, FKChain, SRobotQ};
use glam_traits_ext::{TAffine3, TVec3};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// TrajPCS
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Small fixed-size vector helpers
// ---------------------------------------------------------------------------

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
        for k in 0..N {
            s += j[i][k] * v.0[k];
        }
        out[i] = s;
    }
    out
}

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

/// Jerk-limited trajectory along a [`SplineInterpolatedRobotPath`].
pub struct Trajectory<'a, const N: usize, FK: FKChain<N, f64>> {
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

impl<'a, const N: usize, FK: FKChain<N, f64>> Trajectory<'a, N, FK> {
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
        let j_mat = self.fk.jacobian(&q).map_err(|e| e.into())?;
        let jd_mat = self.fk.jacobian_dot(&q, &qdot).map_err(|e| e.into())?;
        let jdd_mat = self
            .fk
            .jacobian_ddot(&q, &qdot, &qddot)
            .map_err(|e| e.into())?;

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
        if joint_util >= 1.0 {
            return Ok(joint_util);
        }

        let j_mat = self.fk.jacobian(&q).map_err(|e| e.into())?;
        let jd_mat = self.fk.jacobian_dot(&q, &qdot).map_err(|e| e.into())?;
        let jdd_mat = self
            .fk
            .jacobian_ddot(&q, &qdot, &qddot)
            .map_err(|e| e.into())?;

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

        Ok(joint_util
            .max(tcp_vel_util)
            .max(tcp_acc_util)
            .max(tcp_jrk_util))
    }

    /// Return `(times, s, sdot, sddot, sdddot)` arrays.
    pub fn get_s_state_arrays(
        &self,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
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

    /// Sample TCP position (linear part of `fk_end`) at joint configuration `q`.
    #[allow(dead_code)]
    fn tcp_position(&self, q: &SRobotQ<N, f64>) -> DekeResult<[f64; 3]> {
        let pose = self.fk.fk_end(q).map_err(|e| e.into())?;
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

                let (sdddot_min, sdddot_max) =
                    self.jerk_range_from_jerk_constraints(&next_node)?;
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
            "topp3tcp-spline: depth-first search exhausted all jerk candidates"
                .to_string(),
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
