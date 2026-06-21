//! Private time-parametrised trajectory representation.
//!
//! A [`Plan`] is the analytical, multi-section, jerk-limited trajectory that
//! the solvers produce. It is a sequence of "sections" (one per inter-waypoint
//! interval); each section holds one [`Segment`] per joint. Together with the
//! cumulative intermediate-section durations, this is enough to evaluate
//! position/velocity/acceleration/jerk at any time `t` along the trajectory.
//!
//! The type is internal to the crate. The public surface (the `ToppSolver`,
//! the `Pursuer`) projects samples out as [`SRobotQ`] tuples or as packaged
//! [`crate::sample::MotionSample`]s.

use deke_types::{DekeError, DekeResult, KinScalar, SRobotQ};

use crate::extent::Extent;
use crate::kin_state::KinThirdPose;
use crate::segment::Segment;

/// Analytical multi-section trajectory: one [`Segment`] per joint per section,
/// plus the cumulative end-times of every section.
///
/// `profiles` always has at least one section; `intermediate_durations[i]`
/// holds the trajectory time at which section `i` ends, so
/// `intermediate_durations.last()` equals [`Plan::duration`].
#[derive(Debug, Clone)]
pub(crate) struct Plan<const N: usize, F: KinScalar = f32> {
    /// One [`Segment`] per joint per section.
    pub(crate) profiles: Vec<[Segment<F>; N]>,
    /// Cumulative end-time of each section, in trajectory time.
    pub(crate) intermediate_durations: Vec<F>,
    /// Minimum per-joint single-axis durations from the last solve.
    pub(crate) independent_min_durations: SRobotQ<N, F>,
    /// Per-joint position extrema cache, populated by [`Plan::position_extrema`].
    pub(crate) position_extrema: [Extent<F>; N],
    /// Per-joint velocity extrema cache, populated by [`Plan::velocity_extrema`].
    #[allow(dead_code)]
    pub(crate) velocity_extrema: [Extent<F>; N],
    /// Iteration count consumed by the last waypoint solve.
    pub(crate) waypoint_iterations: usize,
    /// Total trajectory duration.
    pub(crate) duration: F,
}

#[allow(dead_code)] // inspection / public-API surface kept for parity even when unused internally
impl<const N: usize, F: KinScalar> Plan<N, F> {
    /// Construct an empty plan: one zero-length section, all states at the
    /// origin.
    pub(crate) fn empty() -> Self {
        Self {
            profiles: vec![[Segment::empty(); N]],
            intermediate_durations: vec![F::zero()],
            independent_min_durations: SRobotQ::zeros(),
            position_extrema: [Extent::point(F::zero()); N],
            velocity_extrema: [Extent::point(F::zero()); N],
            waypoint_iterations: 0,
            duration: F::zero(),
        }
    }

    /// Total trajectory duration.
    pub(crate) fn duration(&self) -> F {
        self.duration
    }

    /// Cumulative end-time of each section, in trajectory time.
    pub(crate) fn intermediate_durations(&self) -> &[F] {
        &self.intermediate_durations
    }

    /// Minimum per-joint single-axis durations from the last solve.
    pub(crate) fn independent_min_durations(&self) -> SRobotQ<N, F> {
        self.independent_min_durations
    }

    /// Locate which interior step of a single-joint segment contains the
    /// given local time. On entry `time_ref` is the section-local time; on
    /// return `time_ref` has been rebased to the start of the step. The
    /// return value is the step index (0..7) or `7` when the requested time
    /// falls past the last step.
    fn locate_segment(segment: &Segment<F>, time_ref: &mut F) -> usize {
        let mut cumulative = F::zero();
        for step in 0..segment.t.len() {
            let next = cumulative + segment.t[step];
            if *time_ref < next {
                *time_ref = *time_ref - cumulative;
                return step;
            }
            cumulative = next;
        }
        *time_ref = *time_ref - cumulative;
        segment.t.len()
    }

    /// Copy one joint's `(p, v, a)` plus its `jerk` into the per-joint output
    /// arrays at index `dof_idx`.
    fn write_state(
        dof_idx: usize,
        state: &KinThirdPose<F>,
        jerk: F,
        out_pose: &mut SRobotQ<N, F>,
        out_vel: &mut SRobotQ<N, F>,
        out_accel: &mut SRobotQ<N, F>,
        out_jerk: &mut SRobotQ<N, F>,
    ) {
        out_pose[dof_idx] = state.p;
        out_vel[dof_idx] = state.v;
        out_accel[dof_idx] = state.a;
        out_jerk[dof_idx] = jerk;
    }

    /// Sample the trajectory at trajectory time `t`. Returns
    /// `(pose, velocity, acceleration, jerk, section_idx)`. When `t` is past
    /// the end the trajectory is extrapolated at zero jerk from the final
    /// state and `section_idx` is the number of sections.
    pub(crate) fn sample_at(
        &self,
        t: F,
    ) -> (
        SRobotQ<N, F>,
        SRobotQ<N, F>,
        SRobotQ<N, F>,
        SRobotQ<N, F>,
        usize,
    ) {
        let mut pose = SRobotQ::zeros();
        let mut vel = SRobotQ::zeros();
        let mut accel = SRobotQ::zeros();
        let mut jerk = SRobotQ::zeros();

        // Past the end: extrapolate at zero jerk from the final state of the
        // last section.
        if t >= self.duration {
            let new_section = self.profiles.len();
            let last = self
                .profiles
                .last()
                .expect("Plan always has at least one section");
            for (dof_idx, seg) in last.iter().enumerate() {
                let halt_dur = seg.halt.duration;
                let t_offset = if self.profiles.len() > 1 {
                    // The end of the section before the last.
                    self.intermediate_durations[self.intermediate_durations.len() - 2]
                } else {
                    halt_dur
                };
                let dt = t - (t_offset + seg.duration);
                let end_state = KinThirdPose::new(seg.p[7], seg.v[7], seg.a[7]);
                let final_state = end_state.next(dt, F::zero());
                Self::write_state(
                    dof_idx,
                    &final_state,
                    F::zero(),
                    &mut pose,
                    &mut vel,
                    &mut accel,
                    &mut jerk,
                );
            }
            return (pose, vel, accel, jerk, new_section);
        }

        // Find the first section whose end-time is strictly greater than `t`.
        let mut new_section = 0usize;
        for (idx, &end_t) in self.intermediate_durations.iter().enumerate() {
            if end_t > t {
                new_section = idx;
                break;
            }
            new_section = idx + 1;
        }

        let mut t_local = t;
        if new_section > 0 {
            t_local = t_local - self.intermediate_durations[new_section - 1];
        }

        for dof_idx in 0..N {
            let segment = &self.profiles[new_section][dof_idx];
            let mut t_in_segment = t_local;

            // Section 0 may carry a pre-shape halt ramp whose duration sits
            // before the seven-step body.
            if new_section == 0 && segment.halt.duration > F::zero() {
                if t_in_segment < segment.halt.duration {
                    let brake_idx = if t_in_segment < segment.halt.t[0] {
                        0
                    } else {
                        1
                    };
                    if brake_idx > 0 {
                        t_in_segment = t_in_segment - segment.halt.t[brake_idx - 1];
                    }
                    let brake_state = KinThirdPose::new(
                        segment.halt.p[brake_idx],
                        segment.halt.v[brake_idx],
                        segment.halt.a[brake_idx],
                    );
                    let brake_final = brake_state.next(t_in_segment, segment.halt.j[brake_idx]);
                    Self::write_state(
                        dof_idx,
                        &brake_final,
                        segment.halt.j[brake_idx],
                        &mut pose,
                        &mut vel,
                        &mut accel,
                        &mut jerk,
                    );
                    continue;
                } else {
                    t_in_segment = t_in_segment - segment.halt.duration;
                }
            }

            // Past the seven-step body: hold the final state.
            if t_in_segment >= segment.duration {
                let end_state = KinThirdPose::new(segment.p[7], segment.v[7], segment.a[7]);
                let final_state = end_state.next(t_in_segment - segment.duration, F::zero());
                Self::write_state(
                    dof_idx,
                    &final_state,
                    F::zero(),
                    &mut pose,
                    &mut vel,
                    &mut accel,
                    &mut jerk,
                );
                continue;
            }

            // Inside the seven-step body.
            let step_idx = Self::locate_segment(segment, &mut t_in_segment);
            let step_idx = step_idx.min(6);
            let step_state = KinThirdPose::new(
                segment.p[step_idx],
                segment.v[step_idx],
                segment.a[step_idx],
            );
            let stepped = step_state.next(t_in_segment, segment.j[step_idx]);
            Self::write_state(
                dof_idx,
                &stepped,
                segment.j[step_idx],
                &mut pose,
                &mut vel,
                &mut accel,
                &mut jerk,
            );
        }

        (pose, vel, accel, jerk, new_section)
    }

    /// Fast position-only resample on a uniform `dt` grid.
    ///
    /// Walks each axis with a forward-only cursor over its halt ramp and
    /// seven-step body, amortising the section/step search across all
    /// `total` samples. This replaces the per-sample dispatch loop in
    /// [`Self::sample_at`] for callers that only need the position track.
    ///
    /// Samples are written into `out[i][dof_idx]` for `i in 0..total`,
    /// `dof_idx in 0..N`. Samples past the trajectory's duration are
    /// extrapolated at zero jerk from the final state of the last section.
    pub(crate) fn resample_positions(&self, dt: F, total: usize, out: &mut [SRobotQ<N, F>]) {
        debug_assert_eq!(out.len(), total);
        let zero = F::zero();
        let n_sections = self.profiles.len();
        let traj_duration = self.duration;
        let half = F::from(0.5).unwrap();
        let six_inv = F::one() / F::from(6.0).unwrap();

        for dof_idx in 0..N {
            // Persistent cursors across samples — samples are monotonically
            // increasing in time, so the section/step cursor only ever
            // advances. This collapses the inner 7-step search to amortized
            // O(1) per sample (and to fully constant time once we are past
            // the halt phase).
            let mut section_idx: usize = 0;
            let mut section_t0: F = zero;
            // 0..6 inside body, 7 means "after body", -2/-1 mean halt sub 0/1
            let mut step_idx: i32 =
                if n_sections > 0 && self.profiles[0][dof_idx].halt.duration > zero {
                    -2
                } else {
                    0
                };
            let mut elem_t0: F = zero; // cumulative time at start of current element

            let mut sample_t: F = zero;
            let mut sample_idx: usize = 0;

            // Main body: while we still have sections to cover.
            'outer: while sample_idx < total && section_idx < n_sections {
                let segment = &self.profiles[section_idx][dof_idx];
                let halt = &segment.halt;
                let halt_end = section_t0 + halt.duration;

                if step_idx < 0 {
                    let sub = if step_idx == -2 { 0usize } else { 1usize };
                    let halt_p = halt.p[sub];
                    let halt_v = halt.v[sub];
                    let halt_a_half = halt.a[sub] * half;
                    let halt_j_sixth = halt.j[sub] * six_inv;
                    let sub_end = section_t0 + if sub == 0 { halt.t[0] } else { halt.duration };
                    while sample_idx < total && sample_t < sub_end {
                        let dt_into = sample_t - elem_t0;
                        let p = halt_p
                            + dt_into * (halt_v + dt_into * (halt_a_half + dt_into * halt_j_sixth));
                        // SAFETY: `sample_idx < total = out.len()`.
                        let slot = unsafe { out.get_unchecked_mut(sample_idx) };
                        slot[dof_idx] = p;
                        sample_idx += 1;
                        sample_t = sample_t + dt;
                    }
                    if sample_idx >= total {
                        break 'outer;
                    }
                    // Advance to next halt sub or into body.
                    if step_idx == -2 {
                        step_idx = -1;
                        elem_t0 = section_t0 + halt.t[0];
                    } else {
                        step_idx = 0;
                        elem_t0 = halt_end;
                    }
                    continue;
                }

                if step_idx >= 7 {
                    // Past body in this section — move on.
                    section_idx += 1;
                    section_t0 =
                        if section_idx > 0 && section_idx - 1 < self.intermediate_durations.len() {
                            self.intermediate_durations[section_idx - 1]
                        } else {
                            section_t0
                        };
                    step_idx = 0;
                    elem_t0 = section_t0;
                    continue;
                }
                let si = step_idx as usize;
                let body_p = segment.p[si];
                let body_v = segment.v[si];
                let body_a_half = segment.a[si] * half;
                let body_j_sixth = segment.j[si] * six_inv;

                // End time of the current body step (absolute traj time).
                let step_end_abs = elem_t0 + segment.t[si];

                while sample_idx < total && sample_t < step_end_abs {
                    let dt_into = sample_t - elem_t0;
                    let p = body_p
                        + dt_into * (body_v + dt_into * (body_a_half + dt_into * body_j_sixth));
                    let slot = unsafe { out.get_unchecked_mut(sample_idx) };
                    slot[dof_idx] = p;
                    sample_idx += 1;
                    sample_t = sample_t + dt;
                }
                if sample_idx >= total {
                    break 'outer;
                }
                // Step exhausted — advance.
                elem_t0 = step_end_abs;
                step_idx += 1;
                if step_idx >= 7 {
                    // Section's body done — go to next section.
                    section_idx += 1;
                    section_t0 =
                        if section_idx > 0 && section_idx - 1 < self.intermediate_durations.len() {
                            self.intermediate_durations[section_idx - 1]
                        } else {
                            section_t0
                        };
                    step_idx = 0;
                    elem_t0 = section_t0;
                }
            }

            // Tail: extrapolate at zero jerk from the end of the last section.
            if sample_idx < total {
                let last = self.profiles.last().expect("Plan has at least one section");
                let end_p = last[dof_idx].p[7];
                let end_v = last[dof_idx].v[7];
                let end_a_half = last[dof_idx].a[7] * half;
                while sample_idx < total {
                    let dt_past = sample_t - traj_duration;
                    let p = end_p + dt_past * (end_v + dt_past * end_a_half);
                    let slot = unsafe { out.get_unchecked_mut(sample_idx) };
                    slot[dof_idx] = p;
                    sample_idx += 1;
                    sample_t = sample_t + dt;
                }
            }
        }
    }

    #[inline]
    fn section_end_time(&self, idx: usize) -> F {
        if idx < self.intermediate_durations.len() {
            self.intermediate_durations[idx]
        } else {
            self.duration
        }
    }

    /// Recompute and return the per-joint position extrema across all
    /// sections. Times are expressed in trajectory time.
    pub(crate) fn position_extrema(&mut self) -> [Extent<F>; N] {
        for dof_idx in 0..N {
            self.position_extrema[dof_idx] = self.profiles[0][dof_idx].get_position_extrema();
        }
        for section_idx in 1..self.profiles.len() {
            let section_offset = self.intermediate_durations[section_idx - 1];
            for dof_idx in 0..N {
                let bound = self.profiles[section_idx][dof_idx].get_position_extrema();
                if bound.max > self.position_extrema[dof_idx].max {
                    self.position_extrema[dof_idx].max = bound.max;
                    self.position_extrema[dof_idx].t_max = section_offset + bound.t_max;
                }
                if bound.min < self.position_extrema[dof_idx].min {
                    self.position_extrema[dof_idx].min = bound.min;
                    self.position_extrema[dof_idx].t_min = section_offset + bound.t_min;
                }
            }
        }
        self.position_extrema
    }

    /// Recompute and return the per-joint velocity extrema across all
    /// sections. Times are expressed in trajectory time.
    pub(crate) fn velocity_extrema(&mut self) -> [Extent<F>; N] {
        for dof_idx in 0..N {
            self.velocity_extrema[dof_idx] = self.profiles[0][dof_idx].get_velocity_extrema();
        }
        for section_idx in 1..self.profiles.len() {
            let section_offset = self.intermediate_durations[section_idx - 1];
            for dof_idx in 0..N {
                let bound = self.profiles[section_idx][dof_idx].get_velocity_extrema();
                if bound.max > self.velocity_extrema[dof_idx].max {
                    self.velocity_extrema[dof_idx].max = bound.max;
                    self.velocity_extrema[dof_idx].t_max = section_offset + bound.t_max;
                }
                if bound.min < self.velocity_extrema[dof_idx].min {
                    self.velocity_extrema[dof_idx].min = bound.min;
                    self.velocity_extrema[dof_idx].t_min = section_offset + bound.t_min;
                }
            }
        }
        self.velocity_extrema
    }

    /// First trajectory time at which joint `dof` reaches position `value`,
    /// restricted to `t >= t_min`. Returns `None` when the value is never
    /// reached or `dof` is out of range.
    pub(crate) fn first_time_at_pose(&self, dof: usize, value: F, t_min: F) -> Option<F> {
        if dof >= N {
            return None;
        }
        let mut time_local = F::zero();
        for (section_idx, section) in self.profiles.iter().enumerate() {
            if section[dof].find_first_time_at_pose(value, &mut time_local, t_min) {
                let offset = if section_idx > 0 {
                    self.intermediate_durations[section_idx - 1]
                } else {
                    F::zero()
                };
                return Some(offset + time_local);
            }
        }
        None
    }

    /// First trajectory time at which joint `dof` reaches velocity `value`,
    /// restricted to `t >= t_min`. Returns `None` when the value is never
    /// reached or `dof` is out of range.
    pub(crate) fn first_time_at_vel(&self, dof: usize, value: F, t_min: F) -> Option<F> {
        if dof >= N {
            return None;
        }
        let mut time_local = F::zero();
        for (section_idx, section) in self.profiles.iter().enumerate() {
            if section[dof].find_first_time_at_vel(value, &mut time_local, t_min) {
                let offset = if section_idx > 0 {
                    self.intermediate_durations[section_idx - 1]
                } else {
                    F::zero()
                };
                return Some(offset + time_local);
            }
        }
        None
    }

    /// Scale the trajectory in time and position. `position_scale` rescales
    /// the position-domain quantities (positions, velocities, accelerations,
    /// jerks via their derivative powers of `time_scale`).
    pub(crate) fn scale(&mut self, position_scale: F, time_scale: F) {
        self.duration = self.duration * time_scale;
        for end_t in self.intermediate_durations.iter_mut() {
            *end_t = *end_t * time_scale;
        }
        for section in self.profiles.iter_mut() {
            for seg in section.iter_mut() {
                seg.scale(position_scale, time_scale);
            }
        }
        for dof_idx in 0..N {
            self.independent_min_durations[dof_idx] =
                self.independent_min_durations[dof_idx] * time_scale;
        }
    }

    /// Append `other` onto the end of `self`. The sections of `other` are
    /// concatenated, with intermediate-duration timestamps shifted by the
    /// current total duration.
    pub(crate) fn append(&mut self, other: &Self) {
        let extra_sections = other.profiles.len();
        let base_duration = self.duration;
        for section_idx in 0..extra_sections {
            self.profiles.push(other.profiles[section_idx]);
            self.intermediate_durations
                .push(base_duration + other.intermediate_durations[section_idx]);
        }
        self.duration = self.duration + other.duration;
    }

    /// Concatenate a sequence of plans in order. Returns an error if the
    /// input slice is empty.
    pub(crate) fn merge(plans: &[Self]) -> DekeResult<Self> {
        let (first, rest) = plans.split_first().ok_or_else(|| {
            DekeError::RetimerFailed(String::from("merge requires at least one input"))
        })?;
        let mut combined = first.clone();
        for next in rest {
            combined.append(next);
        }
        Ok(combined)
    }
}

impl<const N: usize, F: KinScalar> Default for Plan<N, F> {
    fn default() -> Self {
        Self::empty()
    }
}
