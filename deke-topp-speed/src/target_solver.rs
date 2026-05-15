//! Single-target trajectory solver (no intermediate waypoints).
//!
//! Builds a per-axis minimum-duration profile, then time-synchronises the
//! per-axis profiles into a single trajectory section honouring the
//! coordination mode requested by the caller. The phase-locked path also
//! looks for a single signed-jerk ratio that lets every axis ride the same
//! shape as the limiting axis; if found, the cross-axis Step-B math is
//! skipped entirely.

use deke_types::{FKScalar, SRobotQ};

use crate::feasible::Feasible;
use crate::halt_segment::{second_order_pose, third_order_pose, third_order_vel};
use crate::kin_state::{
    KinFirstPose, KinSecondPose, KinSecondVel, KinThirdPose, KinThirdVel, LimitsFirstPose,
    LimitsSecondPose, LimitsSecondVel, LimitsThirdPose, LimitsThirdVel,
};
use crate::modes::{ControlMode, Coordination, DurationGrid, GoalOutOfBounds};
use crate::plan::Plan;
use crate::pose_math;
use crate::segment::{Segment, Sweep};
use crate::spec::MotionSpec;
use crate::status::StepStatus;
use crate::vel_math;

/// Single-target solver state.
///
/// Holds the per-DoF caches reused across the per-axis Step-A pass and the
/// cross-axis Step-B time-synchronisation pass: feasible windows, resolved
/// per-axis modes, and the bookkeeping arrays for candidate sync times.
#[derive(Debug)]
pub(crate) struct TargetSolver<const N: usize, F: FKScalar> {
    /// Phase-sync scale ratios. Populated by [`Self::set_phase_sync`].
    p_d: SRobotQ<N, F>,
    /// Signed pose offset cache, `goal - current` per axis.
    position_from_target: SRobotQ<N, F>,
    /// Candidate sync times; capacity `3 * N + 1`.
    candidate_times: Vec<F>,
    /// Candidate sync time indices; capacity `3 * N + 1`.
    candidate_indices: Vec<usize>,
    /// Per-axis feasible windows produced by the per-axis Step-A pass.
    dof_blocks: [Feasible<F>; N],
    /// Effective per-axis minimum velocity (defaults to `-max_vel`).
    inp_min_velocity: SRobotQ<N, F>,
    /// Effective per-axis minimum acceleration (defaults to `-max_accel`).
    inp_min_acceleration: SRobotQ<N, F>,
    /// Resolved per-axis control mode (after applying per-axis overrides).
    dof_cntrl: [ControlMode; N],
    /// Resolved per-axis coordination (after applying per-axis overrides).
    dof_sync: [Coordination; N],
    /// Long-lived scratch used by every `StepA::get_profile_with_scratch` call
    /// to avoid the ~3 KB stack memset that the convenience entry point does.
    step_a_scratch: [crate::segment::Segment<F>; 6],
}

impl<const N: usize, F: FKScalar> TargetSolver<N, F> {
    /// Build a fresh solver with all per-axis caches zeroed.
    pub fn new() -> Self {
        let capacity = 3 * N + 1;
        Self {
            p_d: SRobotQ::zeros(),
            position_from_target: SRobotQ::zeros(),
            candidate_times: vec![F::zero(); capacity],
            candidate_indices: vec![0usize; capacity],
            dof_blocks: core::array::from_fn(|_| Feasible::empty()),
            inp_min_velocity: SRobotQ::zeros(),
            inp_min_acceleration: SRobotQ::zeros(),
            dof_cntrl: [ControlMode::Position; N],
            dof_sync: [Coordination::TimeLocked; N],
            step_a_scratch: [crate::segment::Segment::empty(); 6],
        }
    }

    /// Identify a single global signed-jerk scale that takes every phase-locked
    /// axis from its current state to its target state. Populates
    /// [`Self::p_d`] on success; returns `false` when no such scale exists.
    ///
    /// The probe axis is the first phase-locked axis that has any non-zero
    /// work scalar (pose offset, current vel/accel, target vel/accel). The
    /// ratio between each axis' work scalar and the probe's becomes the phase
    /// ratio. If any axis' state cannot be reconstructed from that ratio, the
    /// phase sync is rejected.
    pub fn set_phase_sync(
        &mut self,
        spec: &MotionSpec<N, F>,
        dof_idx_max: usize,
        phase_sync_dir: Sweep,
    ) -> bool {
        let eps = F::epsilon();
        for dof_idx in 0..N {
            self.position_from_target[dof_idx] =
                spec.goal_pose[dof_idx] - spec.current_pose[dof_idx];
        }
        // The probe axis selection scans every phase-locked axis until it
        // finds one whose work scalar is non-zero. Track which array (and
        // index) the chosen scalar lives in so the per-axis scale ratios can
        // be reconstructed below.
        enum WorkArr {
            PoseOff,
            CurVel,
            CurAccel,
            GoalVel,
            GoalAccel,
        }
        let mut dof_needs_work: Option<(usize, WorkArr)> = None;
        for dof_idx in 0..N {
            if self.dof_sync[dof_idx] != Coordination::PhaseLocked {
                continue;
            }
            if self.dof_cntrl[dof_idx] == ControlMode::Position
                && self.position_from_target[dof_idx].abs() > eps
            {
                dof_needs_work = Some((dof_idx, WorkArr::PoseOff));
                break;
            } else if spec.current_vel[dof_idx].abs() > eps {
                dof_needs_work = Some((dof_idx, WorkArr::CurVel));
                break;
            } else if spec.current_accel[dof_idx].abs() > eps {
                dof_needs_work = Some((dof_idx, WorkArr::CurAccel));
                break;
            } else if spec.goal_vel[dof_idx].abs() > eps {
                dof_needs_work = Some((dof_idx, WorkArr::GoalVel));
                break;
            } else if spec.goal_accel[dof_idx].abs() > eps {
                dof_needs_work = Some((dof_idx, WorkArr::GoalAccel));
                break;
            }
        }
        let (probe_idx, probe_arr) = match dof_needs_work {
            Some(v) => v,
            None => return false,
        };
        let probe_value = match probe_arr {
            WorkArr::PoseOff => self.position_from_target[probe_idx],
            WorkArr::CurVel => spec.current_vel[probe_idx],
            WorkArr::CurAccel => spec.current_accel[probe_idx],
            WorkArr::GoalVel => spec.goal_vel[probe_idx],
            WorkArr::GoalAccel => spec.goal_accel[probe_idx],
        };
        let scale_pos = self.position_from_target[probe_idx] / probe_value;
        let scale_vc = spec.current_vel[probe_idx] / probe_value;
        let scale_vt = spec.goal_vel[probe_idx] / probe_value;
        let scale_ac = spec.current_accel[probe_idx] / probe_value;
        let scale_at = spec.goal_accel[probe_idx] / probe_value;
        let vmax_ref = match probe_arr {
            WorkArr::PoseOff => self.position_from_target[dof_idx_max],
            WorkArr::CurVel => spec.current_vel[dof_idx_max],
            WorkArr::CurAccel => spec.current_accel[dof_idx_max],
            WorkArr::GoalVel => spec.goal_vel[dof_idx_max],
            WorkArr::GoalAccel => spec.goal_accel[dof_idx_max],
        };
        let mut signed_jerk_ref = match phase_sync_dir {
            Sweep::Up => spec.max_jerk[dof_idx_max],
            Sweep::Down => -spec.max_jerk[dof_idx_max],
        };
        if spec.max_jerk[dof_idx_max].is_infinite() {
            signed_jerk_ref = match phase_sync_dir {
                Sweep::Up => spec.max_accel[dof_idx_max],
                Sweep::Down => self.inp_min_acceleration[dof_idx_max],
            };
        }
        for dof_idx in 0..N {
            if self.dof_sync[dof_idx] != Coordination::PhaseLocked {
                continue;
            }
            let work_val = match probe_arr {
                WorkArr::PoseOff => self.position_from_target[dof_idx],
                WorkArr::CurVel => spec.current_vel[dof_idx],
                WorkArr::CurAccel => spec.current_accel[dof_idx],
                WorkArr::GoalVel => spec.goal_vel[dof_idx],
                WorkArr::GoalAccel => spec.goal_accel[dof_idx],
            };
            let cond_a = self.dof_cntrl[dof_idx] == ControlMode::Position
                && (self.position_from_target[dof_idx] - scale_pos * work_val).abs() > eps;
            let cond_b = (spec.current_vel[dof_idx] - scale_vc * work_val).abs() > eps;
            let cond_c = (spec.current_accel[dof_idx] - scale_ac * work_val).abs() > eps;
            let cond_d = (spec.goal_vel[dof_idx] - scale_vt * work_val).abs() > eps;
            let cond_e = (spec.goal_accel[dof_idx] - scale_at * work_val).abs() > eps;
            if cond_a || cond_b || cond_c || cond_d || cond_e {
                return false;
            }
            self.p_d[dof_idx] = signed_jerk_ref * work_val / vmax_ref;
        }
        true
    }

    /// Round `value` up to the next multiple of `step`. Returns `value`
    /// unchanged when it already sits exactly on a step boundary.
    fn discretize_up(value: F, step: F) -> F {
        let remainder = value - (value / step).floor() * step;
        let eps = F::epsilon();
        if remainder > eps {
            return value + (step - remainder);
        }
        value
    }

    /// Find the first per-axis profile time that is both a `step` multiple
    /// and not blocked by a feasibility gap.
    fn find_discrete_time(block: &Feasible<F>, step: F) -> F {
        let mut time = Self::discretize_up(block.t_min, step);
        if !block.is_blocked(time) || block.blocked_interval_a.is_none() {
            return time;
        }
        time = Self::discretize_up(block.blocked_interval_a.unwrap().right_time, step);
        if !block.is_blocked(time) || block.blocked_interval_b.is_none() {
            return time;
        }
        Self::discretize_up(block.blocked_interval_b.unwrap().right_time, step)
    }

    /// Fill `indices` over the half-open range `[0, end)` with the values
    /// `start..start+len`.
    fn fill_indices(indices: &mut [usize], end: usize, start: usize) {
        for (offset, slot) in indices.iter_mut().take(end).enumerate() {
            *slot = start + offset;
        }
    }

    /// Locate the earliest synchronisation time that no axis blocks.
    ///
    /// On success writes the chosen time into `*t_sync` and the index of the
    /// new "limiting" axis (if any) into `new_dof_idx`. When the chosen time
    /// is sourced from the global `min_duration` slot rather than a per-axis
    /// candidate, `new_dof_idx` is set to `None`.
    fn find_feasible_time(
        &mut self,
        min_duration: Option<F>,
        t_sync: &mut F,
        new_dof_idx: &mut Option<usize>,
        profiles: &mut [Segment<F>; N],
        discrete: bool,
        step: F,
    ) -> bool {
        let infinity = F::infinity();
        let mut has_blocked = false;
        for dof_idx in 0..N {
            if self.dof_sync[dof_idx] == Coordination::Independent {
                self.candidate_times[dof_idx] = F::zero();
                self.candidate_times[N + dof_idx] = infinity;
                self.candidate_times[2 * N + dof_idx] = infinity;
                continue;
            }
            self.candidate_times[dof_idx] = self.dof_blocks[dof_idx].t_min;
            self.candidate_times[N + dof_idx] = match self.dof_blocks[dof_idx].blocked_interval_a {
                Some(span) => span.right_time,
                None => infinity,
            };
            self.candidate_times[2 * N + dof_idx] =
                match self.dof_blocks[dof_idx].blocked_interval_b {
                    Some(span) => span.right_time,
                    None => infinity,
                };
            has_blocked |= self.dof_blocks[dof_idx].blocked_interval_a.is_some()
                || self.dof_blocks[dof_idx].blocked_interval_b.is_some();
        }
        self.candidate_times[3 * N] = min_duration.unwrap_or(infinity);
        has_blocked |= min_duration.is_some();
        // Fast path: no blocked intervals and no min_duration. The chosen time
        // is simply the largest `t_min` across non-Independent axes — no
        // sorting, no clone, no per-DoF is_blocked scan.
        if !has_blocked && !discrete {
            let mut found = false;
            let mut max_idx = 0usize;
            let mut max_t = F::zero();
            for i in 0..N {
                if self.dof_sync[i] == Coordination::Independent {
                    continue;
                }
                let t = self.dof_blocks[i].t_min;
                if !found || t > max_t {
                    max_t = t;
                    max_idx = i;
                    found = true;
                }
            }
            if !found {
                return false;
            }
            *t_sync = max_t;
            *new_dof_idx = Some(max_idx);
            profiles[max_idx] = self.dof_blocks[max_idx].p_min;
            return true;
        }
        if discrete {
            for slot in self.candidate_times.iter_mut() {
                if slot.is_infinite() {
                    continue;
                }
                *slot = Self::discretize_up(*slot, step);
            }
        }
        let end_count = if has_blocked { 3 * N + 1 } else { N };
        Self::fill_indices(&mut self.candidate_indices[..end_count], end_count, 0);
        let times = &self.candidate_times;
        self.candidate_indices[..end_count].sort_by(|a, b| {
            times[*a]
                .partial_cmp(&times[*b])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        // Scan starts from the `N - 1`-th sorted entry: smaller candidates
        // can't satisfy the cross-axis synchronisation requirement.
        let start_offset = if N == 0 { 0 } else { N - 1 };
        let min_t = min_duration.unwrap_or(F::zero());
        for sorted_pos in start_offset..end_count {
            let slot_idx = self.candidate_indices[sorted_pos];
            let t = self.candidate_times[slot_idx];
            let mut blocked = false;
            for dof_idx2 in 0..N {
                if self.dof_sync[dof_idx2] == Coordination::Independent {
                    continue;
                }
                if self.dof_blocks[dof_idx2].is_blocked(t) {
                    blocked = true;
                    break;
                }
            }
            if blocked || t < min_t || t.is_infinite() {
                continue;
            }
            *t_sync = t;
            if slot_idx == 3 * N {
                *new_dof_idx = None;
                return true;
            }
            let quot = slot_idx / N;
            let rem = slot_idx % N;
            *new_dof_idx = Some(rem);
            match quot {
                0 => {
                    profiles[rem] = self.dof_blocks[rem].p_min;
                }
                1 => {
                    if let Some(span) = self.dof_blocks[rem].blocked_interval_a {
                        profiles[rem] = span.profile_at_right;
                    }
                }
                2 => {
                    if let Some(span) = self.dof_blocks[rem].blocked_interval_b {
                        profiles[rem] = span.profile_at_right;
                    }
                }
                _ => {}
            }
            return true;
        }
        false
    }

    /// Solve a single-target trajectory from the supplied [`MotionSpec`].
    ///
    /// On success the result is written into the first section of `plan`;
    /// the returned [`StepStatus`] is `InProgress` for any non-error outcome.
    /// `delta_time` is only consulted when [`DurationGrid::Quantized`] is
    /// active; pass `F::zero()` otherwise.
    pub fn solve(
        &mut self,
        spec: &MotionSpec<N, F>,
        plan: &mut Plan<N, F>,
        delta_time: F,
    ) -> StepStatus {
        // Reset the plan in-place so we reuse the caller's Vec capacity
        // instead of dropping/re-allocating two Vecs every solve.
        plan.profiles.clear();
        plan.profiles.push([Segment::empty(); N]);
        plan.intermediate_durations.clear();
        plan.intermediate_durations.push(F::zero());
        plan.independent_min_durations = SRobotQ::zeros();
        plan.duration = F::zero();
        plan.waypoint_iterations = 0;
        let infinity = F::infinity();
        let eps = F::epsilon();

        // ---- Per-axis Step-A: minimum-duration shaping. -------------------
        for dof_idx in 0..N {
            self.inp_min_velocity[dof_idx] = match spec.min_vel {
                Some(arr) => arr[dof_idx],
                None => -spec.max_vel[dof_idx],
            };
            self.inp_min_acceleration[dof_idx] = match spec.min_accel {
                Some(arr) => arr[dof_idx],
                None => -spec.max_accel[dof_idx],
            };
            self.dof_cntrl[dof_idx] = match spec.per_axis_control_mode {
                Some(arr) => arr[dof_idx],
                None => spec.control_mode,
            };
            self.dof_sync[dof_idx] = match spec.per_axis_coordination {
                Some(arr) => arr[dof_idx],
                None => spec.coordination,
            };
            let p = &mut plan.profiles[0][dof_idx];
            if !spec.axis_active[dof_idx] {
                p.p[7] = spec.current_pose[dof_idx];
                p.v[7] = spec.current_vel[dof_idx];
                p.a[7] = spec.current_accel[dof_idx];
                p.duration = F::zero();
                self.dof_blocks[dof_idx].t_min = F::zero();
                self.dof_blocks[dof_idx].blocked_interval_a = None;
                self.dof_blocks[dof_idx].blocked_interval_b = None;
                continue;
            }
            let mut goal_pose = spec.goal_pose[dof_idx];
            let mut goal_vel = spec.goal_vel[dof_idx];
            let mut goal_accel = spec.goal_accel[dof_idx];
            if spec.goal_overflow == GoalOutOfBounds::Clip {
                if goal_vel < self.inp_min_velocity[dof_idx] {
                    goal_vel = self.inp_min_velocity[dof_idx];
                }
                if goal_vel > spec.max_vel[dof_idx] {
                    goal_vel = spec.max_vel[dof_idx];
                }
                if goal_accel < self.inp_min_acceleration[dof_idx] {
                    goal_accel = self.inp_min_acceleration[dof_idx];
                }
                if goal_accel > spec.max_accel[dof_idx] {
                    goal_accel = spec.max_accel[dof_idx];
                }
            }
            // Pre-shape halt ramps for out-of-bounds initial states.
            match self.dof_cntrl[dof_idx] {
                ControlMode::Position => {
                    if !spec.max_jerk[dof_idx].is_infinite() {
                        p.halt = third_order_pose::get_profile(
                            spec.current_vel[dof_idx],
                            spec.current_accel[dof_idx],
                            LimitsThirdPose::new(
                                spec.max_vel[dof_idx],
                                self.inp_min_velocity[dof_idx],
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                                spec.max_jerk[dof_idx],
                            ),
                        );
                    } else if !spec.max_accel[dof_idx].is_infinite() {
                        p.halt = second_order_pose::get_profile(
                            spec.current_vel[dof_idx],
                            LimitsSecondPose::new(
                                spec.max_vel[dof_idx],
                                self.inp_min_velocity[dof_idx],
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                            ),
                        );
                    }
                    if spec.goal_overflow == GoalOutOfBounds::Clip {
                        if let Some(min_pose) = spec.min_pose
                            && goal_pose < min_pose[dof_idx]
                        {
                            goal_pose = min_pose[dof_idx];
                        }
                        if let Some(max_pose) = spec.max_pose
                            && goal_pose > max_pose[dof_idx]
                        {
                            goal_pose = max_pose[dof_idx];
                        }
                    }
                    p.set_boundary_explicit(
                        spec.current_pose[dof_idx],
                        spec.current_vel[dof_idx],
                        spec.current_accel[dof_idx],
                        goal_pose,
                        goal_vel,
                        goal_accel,
                    );
                }
                ControlMode::Velocity => {
                    if !spec.max_jerk[dof_idx].is_infinite() {
                        p.halt = third_order_vel::get_profile(
                            spec.current_accel[dof_idx],
                            LimitsThirdVel::new(
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                                spec.max_jerk[dof_idx],
                            ),
                        );
                    }
                    p.set_initial(
                        spec.current_pose[dof_idx],
                        spec.current_vel[dof_idx],
                        spec.current_accel[dof_idx],
                        goal_vel,
                        goal_accel,
                    );
                }
            }
            if !spec.max_jerk[dof_idx].is_infinite() {
                let (np, nv, na) = p.halt.finalize_second_order(p.p[0], p.v[0], p.a[0]);
                p.p[0] = np;
                p.v[0] = nv;
                p.a[0] = na;
            } else if !spec.max_accel[dof_idx].is_infinite() {
                // Second-order halt: integrate the single constant-accel
                // section forward to produce the post-halt boundary state.
                let (np, nv, na) = p.halt.finalize_second_order(p.p[0], p.v[0], p.a[0]);
                p.p[0] = np;
                p.v[0] = nv;
                p.a[0] = na;
            }

            // Per-axis Step-A: dispatch on order and control mode.
            // We borrow the segment immutably during the StepA call; the
            // mutable borrow on `p` is dropped before passing `&*p` here.
            let p_ref: &Segment<F> = &*p;
            let step1_ok = match self.dof_cntrl[dof_idx] {
                ControlMode::Position => {
                    if !spec.max_jerk[dof_idx].is_infinite() {
                        let step = pose_math::third_order::StepA::new(
                            KinThirdPose::new(p_ref.p[0], p_ref.v[0], p_ref.a[0]),
                            KinThirdPose::new(p_ref.pf, p_ref.vf, p_ref.af),
                            LimitsThirdPose::new(
                                spec.max_vel[dof_idx],
                                self.inp_min_velocity[dof_idx],
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                                spec.max_jerk[dof_idx],
                            ),
                        );
                        step.get_profile_with_scratch(
                            p_ref,
                            &mut self.dof_blocks[dof_idx],
                            &mut self.step_a_scratch,
                        )
                    } else if !spec.max_accel[dof_idx].is_infinite() {
                        let step = pose_math::second_order::StepA::new(
                            KinSecondPose::new(p_ref.p[0], p_ref.v[0]),
                            KinSecondPose::new(p_ref.pf, p_ref.vf),
                            LimitsSecondPose::new(
                                spec.max_vel[dof_idx],
                                self.inp_min_velocity[dof_idx],
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                            ),
                        );
                        step.get_profile(p_ref, &mut self.dof_blocks[dof_idx])
                    } else {
                        let step = pose_math::first_order::StepA::new(
                            KinFirstPose::new(p_ref.p[0]),
                            KinFirstPose::new(p_ref.pf),
                            LimitsFirstPose::new(
                                spec.max_vel[dof_idx],
                                self.inp_min_velocity[dof_idx],
                            ),
                        );
                        step.get_profile(p_ref, &mut self.dof_blocks[dof_idx])
                    }
                }
                ControlMode::Velocity => {
                    if !spec.max_jerk[dof_idx].is_infinite() {
                        let step = vel_math::third_order::StepA::new(
                            KinThirdVel::new(p_ref.v[0], p_ref.a[0]),
                            KinThirdVel::new(p_ref.vf, p_ref.af),
                            LimitsThirdVel::new(
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                                spec.max_jerk[dof_idx],
                            ),
                        );
                        step.get_profile(p_ref, &mut self.dof_blocks[dof_idx])
                    } else {
                        let step = vel_math::second_order::StepA::new(
                            KinSecondVel::new(p_ref.v[0]),
                            KinSecondVel::new(p_ref.vf),
                            LimitsSecondVel::new(
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                            ),
                        );
                        step.get_profile(p_ref, &mut self.dof_blocks[dof_idx])
                    }
                }
            };
            if !step1_ok {
                let zero_limits_dof = spec.max_accel[dof_idx] == F::zero()
                    || self.inp_min_acceleration[dof_idx] == F::zero()
                    || spec.max_jerk[dof_idx] == F::zero();
                if zero_limits_dof {
                    return StepStatus::ZeroLimit;
                }
                return StepStatus::StepOneFailed;
            }
            plan.independent_min_durations[dof_idx] = self.dof_blocks[dof_idx].t_min;
        }

        let discrete = spec.duration_grid == DurationGrid::Quantized;
        if N == 1 && spec.min_duration.is_none() && !discrete {
            plan.duration = self.dof_blocks[0].t_min;
            plan.profiles[0][0] = self.dof_blocks[0].p_min;
            plan.intermediate_durations[0] = plan.duration;
            return StepStatus::InProgress;
        }

        // ---- Cross-axis Step-B: pick a feasible global duration. ----------
        let mut limiting_dof: Option<usize> = None;
        let (sec0, _) = plan
            .profiles
            .split_first_mut()
            .expect("plan has one section");
        let feasible = self.find_feasible_time(
            spec.min_duration,
            &mut plan.duration,
            &mut limiting_dof,
            sec0,
            discrete,
            delta_time,
        );
        if !feasible {
            let mut zero_limits = false;
            for dof_idx in 0..N {
                if spec.max_accel[dof_idx] == F::zero()
                    || self.inp_min_acceleration[dof_idx] == F::zero()
                    || spec.max_jerk[dof_idx] == F::zero()
                {
                    zero_limits = true;
                    break;
                }
            }
            if zero_limits {
                return StepStatus::ZeroLimit;
            }
            return StepStatus::DurationInfeasible;
        }

        // Stretch the global duration if any independent-axis profile demands
        // it. Mirrors the C++ pass that lifts `traj.duration` against the
        // independent axes.
        for dof_idx in 0..N {
            if spec.axis_active[dof_idx] && self.dof_sync[dof_idx] == Coordination::Independent {
                plan.profiles[0][dof_idx] = self.dof_blocks[dof_idx].p_min;
                let mut dof_duration = self.dof_blocks[dof_idx].t_min;
                if discrete {
                    dof_duration = Self::find_discrete_time(&self.dof_blocks[dof_idx], delta_time);
                }
                if dof_duration > plan.duration {
                    plan.duration = dof_duration;
                    limiting_dof = Some(dof_idx);
                }
            }
        }
        plan.intermediate_durations[0] = plan.duration;
        let huge_duration_limit = F::from(7.6e3).unwrap_or_else(F::max_value);
        if plan.duration > huge_duration_limit {
            return StepStatus::DurationInfeasible;
        }
        if plan.duration == F::zero() {
            for dof_idx in 0..N {
                plan.profiles[0][dof_idx] = self.dof_blocks[dof_idx].p_min;
            }
            return StepStatus::InProgress;
        }
        let mut all_independent = true;
        for dof_idx in 0..N {
            if self.dof_sync[dof_idx] != Coordination::Independent {
                all_independent = false;
                break;
            }
        }
        if !discrete && all_independent {
            return StepStatus::InProgress;
        }

        // ---- Phase-locked fast path. --------------------------------------
        let mut has_phase = false;
        for dof_idx in 0..N {
            if self.dof_sync[dof_idx] == Coordination::PhaseLocked {
                has_phase = true;
                break;
            }
        }
        if let (Some(ref_dof), true) = (limiting_dof, has_phase) {
            let ref_profile = plan.profiles[0][ref_dof];
            let dir = ref_profile.sweep;
            if self.set_phase_sync(spec, ref_dof, dir) {
                let mut ok_phase = true;
                for dof_idx in 0..N {
                    if !spec.axis_active[dof_idx]
                        || dof_idx == ref_dof
                        || self.dof_sync[dof_idx] != Coordination::PhaseLocked
                    {
                        continue;
                    }
                    let mut profile_p = plan.profiles[0][dof_idx];
                    let t_section =
                        plan.duration - profile_p.halt.duration - profile_p.accel_halt.duration;
                    profile_p.t = ref_profile.t;
                    profile_p.sign_block = ref_profile.sign_block;
                    let signs = profile_p.sign_block;
                    let pd_val = self.p_d[dof_idx];
                    match self.dof_cntrl[dof_idx] {
                        ControlMode::Position => {
                            if !spec.max_jerk[dof_idx].is_infinite() {
                                let limits = LimitsThirdPose::new(
                                    spec.max_vel[dof_idx],
                                    self.inp_min_velocity[dof_idx],
                                    spec.max_accel[dof_idx],
                                    self.inp_min_acceleration[dof_idx],
                                    spec.max_jerk[dof_idx],
                                );
                                ok_phase &= crate::check::third_order_pose::check_profile2_jerk(
                                    &mut profile_p,
                                    signs,
                                    crate::segment::Touched::None,
                                    true,
                                    t_section,
                                    pd_val,
                                    &limits,
                                );
                            } else if !spec.max_accel[dof_idx].is_infinite() {
                                let limits = LimitsSecondPose::new(
                                    spec.max_vel[dof_idx],
                                    self.inp_min_velocity[dof_idx],
                                    spec.max_accel[dof_idx],
                                    self.inp_min_acceleration[dof_idx],
                                );
                                ok_phase &= crate::check::second_order_pose::check_profile2_accels(
                                    &mut profile_p,
                                    signs,
                                    crate::segment::Touched::None,
                                    t_section,
                                    pd_val,
                                    -pd_val,
                                    &limits,
                                );
                            } else {
                                let limits = LimitsFirstPose::new(
                                    spec.max_vel[dof_idx],
                                    self.inp_min_velocity[dof_idx],
                                );
                                ok_phase &= crate::check::first_order_pose::check_profile2(
                                    &mut profile_p,
                                    signs,
                                    crate::segment::Touched::None,
                                    t_section,
                                    pd_val,
                                    &limits,
                                );
                            }
                        }
                        ControlMode::Velocity => {
                            if !spec.max_jerk[dof_idx].is_infinite() {
                                let limits = LimitsThirdVel::new(
                                    spec.max_accel[dof_idx],
                                    self.inp_min_acceleration[dof_idx],
                                    spec.max_jerk[dof_idx],
                                );
                                ok_phase &= crate::check::third_order_vel::check_profile2(
                                    &mut profile_p,
                                    signs,
                                    crate::segment::Touched::None,
                                    t_section,
                                    pd_val,
                                    &limits,
                                );
                            } else {
                                let limits = LimitsSecondVel::new(
                                    spec.max_accel[dof_idx],
                                    self.inp_min_acceleration[dof_idx],
                                );
                                ok_phase &= crate::check::second_order_vel::check_profile2(
                                    &mut profile_p,
                                    signs,
                                    crate::segment::Touched::None,
                                    t_section,
                                    pd_val,
                                    &limits,
                                );
                            }
                        }
                    }
                    profile_p.touched = ref_profile.touched;
                    plan.profiles[0][dof_idx] = profile_p;
                }
                let mut ok_sync = true;
                for dof_idx in 0..N {
                    if self.dof_sync[dof_idx] != Coordination::PhaseLocked
                        && self.dof_sync[dof_idx] != Coordination::Independent
                    {
                        ok_sync = false;
                        break;
                    }
                }
                if ok_phase && ok_sync {
                    let _ = (eps, infinity);
                    return StepStatus::InProgress;
                }
            }
        }

        // ---- Per-axis Step-B: synchronise each axis to the global time. ---
        for dof_idx in 0..N {
            let skip = (Some(dof_idx) == limiting_dof
                || self.dof_sync[dof_idx] == Coordination::Independent)
                && !discrete;
            if !spec.axis_active[dof_idx] || skip {
                continue;
            }
            let mut duration = plan.duration;
            if discrete && self.dof_sync[dof_idx] == Coordination::Independent {
                duration = Self::find_discrete_time(&self.dof_blocks[dof_idx], delta_time);
            }
            let mut p = plan.profiles[0][dof_idx];
            let t_section = duration - p.halt.duration - p.accel_halt.duration;
            if self.dof_sync[dof_idx] == Coordination::TimeLockedSoft
                && spec.goal_vel[dof_idx].abs() < eps
                && spec.goal_accel[dof_idx].abs() < eps
            {
                plan.profiles[0][dof_idx] = self.dof_blocks[dof_idx].p_min;
                continue;
            }
            let tol = F::from(2.0).unwrap() * eps;
            if (t_section - self.dof_blocks[dof_idx].t_min).abs() < tol {
                plan.profiles[0][dof_idx] = self.dof_blocks[dof_idx].p_min;
                continue;
            }
            if let Some(span) = self.dof_blocks[dof_idx].blocked_interval_a
                && (t_section - span.right_time).abs() < tol
            {
                plan.profiles[0][dof_idx] = span.profile_at_right;
                continue;
            }
            if let Some(span) = self.dof_blocks[dof_idx].blocked_interval_b
                && (t_section - span.right_time).abs() < tol
            {
                plan.profiles[0][dof_idx] = span.profile_at_right;
                continue;
            }
            // Materialise the timed profile.
            let step2_ok = match self.dof_cntrl[dof_idx] {
                ControlMode::Position => {
                    if !spec.max_jerk[dof_idx].is_infinite() {
                        let mut step = pose_math::third_order::StepB::new(
                            t_section,
                            KinThirdPose::new(p.p[0], p.v[0], p.a[0]),
                            KinThirdPose::new(p.pf, p.vf, p.af),
                            LimitsThirdPose::new(
                                spec.max_vel[dof_idx],
                                self.inp_min_velocity[dof_idx],
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                                spec.max_jerk[dof_idx],
                            ),
                        );
                        step.get_profile(&mut p)
                    } else if !spec.max_accel[dof_idx].is_infinite() {
                        let step = pose_math::second_order::StepB::new(
                            t_section,
                            KinSecondPose::new(p.p[0], p.v[0]),
                            KinSecondPose::new(p.pf, p.vf),
                            LimitsSecondPose::new(
                                spec.max_vel[dof_idx],
                                self.inp_min_velocity[dof_idx],
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                            ),
                        );
                        step.get_profile(&mut p)
                    } else {
                        let step = pose_math::first_order::StepB::new(
                            t_section,
                            KinFirstPose::new(p.p[0]),
                            KinFirstPose::new(p.pf),
                            LimitsFirstPose::new(
                                spec.max_vel[dof_idx],
                                self.inp_min_velocity[dof_idx],
                            ),
                        );
                        step.get_profile(&mut p)
                    }
                }
                ControlMode::Velocity => {
                    if !spec.max_jerk[dof_idx].is_infinite() {
                        let step = vel_math::third_order::StepB::new(
                            t_section,
                            KinThirdVel::new(p.v[0], p.a[0]),
                            KinThirdVel::new(p.vf, p.af),
                            LimitsThirdVel::new(
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                                spec.max_jerk[dof_idx],
                            ),
                        );
                        step.get_profile(&mut p)
                    } else {
                        let step = vel_math::second_order::StepB::new(
                            t_section,
                            KinSecondVel::new(p.v[0]),
                            KinSecondVel::new(p.vf),
                            LimitsSecondVel::new(
                                spec.max_accel[dof_idx],
                                self.inp_min_acceleration[dof_idx],
                            ),
                        );
                        step.get_profile(&mut p)
                    }
                }
            };
            plan.profiles[0][dof_idx] = p;
            if !step2_ok {
                return StepStatus::StepTwoFailed;
            }
        }
        StepStatus::InProgress
    }
}

impl<const N: usize, F: FKScalar> Default for TargetSolver<N, F> {
    fn default() -> Self {
        Self::new()
    }
}
