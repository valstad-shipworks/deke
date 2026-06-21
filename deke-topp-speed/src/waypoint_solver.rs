//! Multi-waypoint trajectory solver.
//!
//! Produces a jerk-limited, multi-section trajectory that passes through a
//! sequence of intermediate poses. The solve has four phases:
//!
//! 1. Initial Step-A pass for every (section, joint) and a per-section feasible
//!    time pick.
//! 2. Global optimisation pass: gradient-descent on per-section velocities and
//!    accelerations to shrink the total duration.
//! 3. Three-segment sliding-window relaxation.
//! 4. Two-segment sliding-window relaxation.

use deke_types::{KinScalar, SRobotQ};
use num_traits::Float;

use crate::feasible::Feasible;
use crate::halt_segment;
use crate::jacobian::Jacobian;
use crate::kin_state::{KinThirdPose, LimitsThirdPose};
use crate::modes::GoalOutOfBounds;
use crate::plan::Plan;
use crate::pose_math;
use crate::segment::Segment;
use crate::spec::MotionSpec;
use crate::status::StepStatus;

#[inline]
fn from_f<F: Float>(x: f64) -> F {
    F::from(x).unwrap()
}

#[inline]
fn fmin<F: Float>(a: F, b: F) -> F {
    if a < b { a } else { b }
}

#[inline]
fn fmax<F: Float>(a: F, b: F) -> F {
    if a > b { a } else { b }
}

#[inline]
fn fabs<F: Float>(x: F) -> F {
    if x < F::zero() { F::zero() - x } else { x }
}

/// Per-section, per-joint kinematic limits including an optional pose
/// envelope.
#[derive(Debug, Clone, Copy)]
struct FullLimits<F: Float> {
    base: LimitsThirdPose<F>,
    min_pose: Option<F>,
    max_pose: Option<F>,
}

impl<F: Float> FullLimits<F> {
    fn new() -> Self {
        Self {
            base: LimitsThirdPose::new(F::zero(), F::zero(), F::zero(), F::zero(), F::zero()),
            min_pose: None,
            max_pose: None,
        }
    }

    #[inline]
    fn max_vel(&self) -> F {
        self.base.max_vel
    }
    #[inline]
    fn min_vel(&self) -> F {
        self.base.min_vel
    }
    #[inline]
    fn max_accel(&self) -> F {
        self.base.max_accel
    }
    #[inline]
    fn min_accel(&self) -> F {
        self.base.min_accel
    }
    #[inline]
    fn jerk(&self) -> F {
        self.base.jerk
    }
}

#[derive(Debug, Clone, Copy)]
struct GradientContribution<F: Float> {
    delta_v: F,
    delta_a: F,
    scale: F,
}

impl<F: Float> GradientContribution<F> {
    fn new() -> Self {
        Self {
            delta_v: F::zero(),
            delta_a: F::zero(),
            scale: F::one(),
        }
    }

    fn reset(&mut self) {
        self.delta_v = F::zero();
        self.delta_a = F::zero();
        self.scale = F::one();
    }
}

/// Per-section bookkeeping for synchronising the per-joint single-axis
/// profiles to a common section duration. Holds one [`Feasible`] per DoF, the
/// chosen synchronisation time, the joint that selected it, and a buffer of
/// candidate sync times sorted across joints.
#[derive(Debug, Clone)]
struct Segment2Segment<const N: usize, F: Float> {
    dof_blocks: [Feasible<F>; N],
    /// Per-joint candidate sync times, in flat layout:
    /// `[t_min × N, blocked_a_right × N, blocked_b_right × N, t_min_user]`.
    candidate_times: Vec<F>,
    candidate_indices: Vec<usize>,
    sync_time: F,
    /// Index of the DoF whose profile is controlling, or `None` when the
    /// section is governed by the user-supplied `t_min`.
    sync_dof_index: Option<usize>,
    /// Which of the three time options (`p_min`, `blocked_interval_a`,
    /// `blocked_interval_b`) was selected on the controlling DoF.
    block_selector: usize,
    /// User-imposed minimum section duration, if any.
    t_min: Option<F>,
}

impl<const N: usize, F: Float> Segment2Segment<N, F> {
    fn new() -> Self {
        Self {
            dof_blocks: core::array::from_fn(|_| Feasible::empty()),
            candidate_times: vec![F::zero(); 3 * N + 1],
            candidate_indices: (0..3 * N + 1).collect(),
            sync_time: F::zero(),
            sync_dof_index: None,
            block_selector: 0,
            t_min: None,
        }
    }

    /// Pick the smallest sync time that is consistent with every joint's
    /// feasibility constraints and the user-imposed `t_min`. Updates the
    /// synced-time, controlling joint and block selector. Returns `false` if
    /// no time is feasible.
    #[inline]
    fn find_feasible_time(&mut self) -> bool {
        // Fast path: a single DoF and no user-imposed minimum.
        if N == 1 && self.t_min.is_none() {
            self.sync_dof_index = Some(0);
            self.block_selector = 0;
            self.sync_time = self.dof_blocks[0].t_min;
            return true;
        }

        // Pre-check the most common case: no blocked intervals and no user
        // t_min. The sync time is simply max(t_min) across DoFs, picked by
        // the DoF that owns that max. This skips the sort + has_extra +
        // is_blocked machinery below.
        if self.t_min.is_none() {
            let mut any_blocked = false;
            for dof_idx in 0..N {
                if self.dof_blocks[dof_idx].blocked_interval_a.is_some()
                    || self.dof_blocks[dof_idx].blocked_interval_b.is_some()
                {
                    any_blocked = true;
                    break;
                }
            }
            if !any_blocked {
                let mut argmax = 0usize;
                let mut tmax = self.dof_blocks[0].t_min;
                for dof_idx in 1..N {
                    let t = self.dof_blocks[dof_idx].t_min;
                    if t > tmax {
                        tmax = t;
                        argmax = dof_idx;
                    }
                }
                self.sync_time = tmax;
                self.sync_dof_index = Some(argmax);
                self.block_selector = 0;
                return true;
            }
        }

        let inf = F::infinity();
        let mut has_extra = false;
        for dof_idx in 0..N {
            self.candidate_times[dof_idx] = self.dof_blocks[dof_idx].t_min;
            self.candidate_times[N + dof_idx] = self.dof_blocks[dof_idx]
                .blocked_interval_a
                .map(|s| s.right_time)
                .unwrap_or(inf);
            self.candidate_times[2 * N + dof_idx] = self.dof_blocks[dof_idx]
                .blocked_interval_b
                .map(|s| s.right_time)
                .unwrap_or(inf);
            has_extra |= self.dof_blocks[dof_idx].blocked_interval_a.is_some()
                || self.dof_blocks[dof_idx].blocked_interval_b.is_some();
        }
        self.candidate_times[3 * N] = self.t_min.unwrap_or(inf);
        has_extra |= self.t_min.is_some();

        // Reset indices and sort prefix.
        let end_idx = if has_extra { 3 * N + 1 } else { N };
        for i in 0..end_idx {
            self.candidate_indices[i] = i;
        }
        // Sort the leading `end_idx` indices by their candidate time.
        let times = &self.candidate_times;
        let sub = &mut self.candidate_indices[..end_idx];
        sub.sort_by(|&l, &r| {
            times[l]
                .partial_cmp(&times[r])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Walk from index N-1 onward (in sorted order): the first feasible
        // time wins.
        let t_min_val = self.t_min.unwrap_or(F::zero());
        for i in (N - 1)..end_idx {
            let idx = self.candidate_indices[i];
            let t = self.candidate_times[idx];
            let mut time_blocked = false;
            for dof_idx in 0..N {
                if self.dof_blocks[dof_idx].is_blocked(t) {
                    time_blocked = true;
                    break;
                }
            }
            if t >= t_min_val && !time_blocked {
                self.sync_time = t;
                if idx == 3 * N {
                    // The user-supplied `t_min` itself was chosen.
                    self.sync_dof_index = None;
                    return true;
                }
                let quot = idx / N;
                let rem = idx % N;
                self.sync_dof_index = Some(rem);
                self.block_selector = quot;
                return true;
            }
        }
        false
    }

    /// Return the profile on the controlling DoF that produces the selected
    /// sync time.
    fn get_controlling_profile(&self) -> Segment<F> {
        let dof = self.sync_dof_index.unwrap_or(0);
        match self.block_selector {
            0 => self.dof_blocks[dof].p_min,
            1 => self.dof_blocks[dof]
                .blocked_interval_a
                .map(|s| s.profile_at_right)
                .unwrap_or(self.dof_blocks[dof].p_min),
            2 => self.dof_blocks[dof]
                .blocked_interval_b
                .map(|s| s.profile_at_right)
                .unwrap_or(self.dof_blocks[dof].p_min),
            _ => self.dof_blocks[dof].p_min,
        }
    }
}

#[derive(Debug, Clone)]
struct SearchState<F: Float> {
    global_step_index: usize,
    local_step_index: usize,
    smoothing_step_index: usize,
    step_sizes: Vec<F>,
    duration_cap: F,
}

impl<F: Float> SearchState<F> {
    fn new() -> Self {
        Self {
            global_step_index: 0,
            local_step_index: 0,
            smoothing_step_index: 0,
            step_sizes: Vec::new(),
            duration_cap: F::zero(),
        }
    }

    fn reset(&mut self, duration: F) {
        self.global_step_index = 0;
        self.local_step_index = 0;
        self.smoothing_step_index = 0;
        self.duration_cap = duration;
        let init = from_f::<F>(1e-6);
        for ss in self.step_sizes.iter_mut() {
            *ss = init;
        }
    }
}

/// Multi-waypoint trajectory solver. See module documentation for the
/// optimisation strategy.
#[derive(Debug)]
pub(crate) struct WaypointSolver<const N: usize, F: KinScalar> {
    number_global_steps: usize,
    number_local_steps: usize,
    number_smoothing_steps: usize,
    number_acceleration_smoothing_steps: usize,
    min_global_steps: usize,
    duration_break_eps: F,
    search_state: SearchState<F>,
    section_positions: Vec<SRobotQ<N, F>>,
    segments: Vec<Segment2Segment<N, F>>,
    segments_tmp: Vec<Segment2Segment<N, F>>,
    waypoint_states: Vec<[KinThirdPose<F>; N]>,
    states_buffer: Vec<[KinThirdPose<F>; N]>,
    gradient_buffer: Vec<[GradientContribution<F>; N]>,
    segment_limits: Vec<[FullLimits<F>; N]>,
    step_divider: Vec<F>,
    step_divider_tmp: Vec<F>,
    cost_estimate: F,
    /// Scratch buffer reused across every `StepA::get_profile_with_scratch`
    /// call so each call doesn't burn ~3 KB of stack zero-init. Slot 0 is
    /// reseeded by `set_boundary` at entry; the rest are filled by
    /// `advance` before they are read by `pick_from_candidates`.
    step_a_scratch: [Segment<F>; 6],
}

impl<const N: usize, F: KinScalar> WaypointSolver<N, F> {
    pub fn new() -> Self {
        let mut s = Self {
            number_global_steps: 0,
            number_local_steps: 0,
            number_smoothing_steps: 0,
            number_acceleration_smoothing_steps: 0,
            min_global_steps: 0,
            duration_break_eps: F::zero(),
            search_state: SearchState::new(),
            section_positions: Vec::new(),
            segments: Vec::new(),
            segments_tmp: Vec::new(),
            waypoint_states: Vec::new(),
            states_buffer: Vec::new(),
            gradient_buffer: Vec::new(),
            segment_limits: Vec::new(),
            step_divider: Vec::new(),
            step_divider_tmp: Vec::new(),
            cost_estimate: F::zero(),
            step_a_scratch: [Segment::empty(); 6],
        };
        s.set_default_settings();
        s
    }

    fn set_default_settings(&mut self) {
        self.number_global_steps = 96;
        self.number_local_steps = 16;
        self.number_smoothing_steps = 0;
        self.number_acceleration_smoothing_steps = 0;
        self.min_global_steps = 8;
        self.duration_break_eps = from_f::<F>(1e-6);
    }

    /// Resize the per-section buffers to hold `waypoint_count` intermediate
    /// waypoints (`waypoint_count + 1` sections, `waypoint_count + 2`
    /// waypoint states).
    fn resize(&mut self, waypoint_count: usize) {
        let zero_state = KinThirdPose::zero();
        let zero_grad = GradientContribution::<F>::new();
        let zero_limits = FullLimits::<F>::new();

        self.section_positions
            .resize(waypoint_count, SRobotQ::zeros());
        self.gradient_buffer.resize(waypoint_count, [zero_grad; N]);
        self.search_state
            .step_sizes
            .resize(waypoint_count, F::zero());
        self.step_divider.resize(waypoint_count, F::zero());
        self.step_divider_tmp.resize(waypoint_count, F::zero());

        let sections = waypoint_count + 1;
        self.segments.resize_with(sections, Segment2Segment::new);
        self.segments_tmp
            .resize_with(sections, Segment2Segment::new);
        self.segment_limits.resize(sections, [zero_limits; N]);
        self.waypoint_states
            .resize(waypoint_count + 2, [zero_state; N]);
        // states_buffer mirrors waypoint_states' shape; sized at first use.
        if self.states_buffer.len() < self.waypoint_states.len() {
            self.states_buffer
                .resize(self.waypoint_states.len(), [zero_state; N]);
        }
    }

    #[inline]
    fn copy_boundary_state(
        profile: &mut Segment<F>,
        from_state: &KinThirdPose<F>,
        to_state: &KinThirdPose<F>,
    ) {
        profile.v[0] = from_state.v;
        profile.a[0] = from_state.a;
        profile.vf = to_state.v;
        profile.af = to_state.a;
    }

    #[inline]
    fn profile_within_position_limits(profile: &Segment<F>, limits: &FullLimits<F>) -> bool {
        if limits.max_pose.is_none() && limits.min_pose.is_none() {
            return true;
        }
        let bound = profile.get_position_extrema();
        if let Some(mx) = limits.max_pose
            && bound.max > mx
        {
            return false;
        }
        if let Some(mn) = limits.min_pose
            && bound.min < mn
        {
            return false;
        }
        true
    }

    #[inline]
    fn clamp(value: F, lo: F, hi: F) -> F {
        fmax(fmin(value, hi), lo)
    }

    #[inline]
    fn max_v_at_zero_a(v: F, a: F, jerk: F) -> F {
        let two = from_f::<F>(2.0);
        v + a * a / (two * jerk)
    }

    /// Clamp the kinematic state of a waypoint after taking a gradient step.
    /// The state's velocity and acceleration must respect both the section
    /// preceding the waypoint and the one following it; the lookahead also
    /// guards against the velocity envelope being violated by the next
    /// constant-acceleration ramp.
    fn clamp_state_with_gradient(
        state: &mut KinThirdPose<F>,
        reference: &KinThirdPose<F>,
        step: F,
        gradient: &GradientContribution<F>,
        lower_limits: &FullLimits<F>,
        upper_limits: &FullLimits<F>,
    ) {
        let eps_14 = from_f::<F>(1e-14);
        let two = from_f::<F>(2.0);
        let min_velocity = fmax(lower_limits.min_vel(), upper_limits.min_vel());
        let max_velocity = fmin(lower_limits.max_vel(), upper_limits.max_vel());
        let min_accel = fmax(lower_limits.min_accel(), upper_limits.min_accel());
        let max_accel = fmin(lower_limits.max_accel(), upper_limits.max_accel());
        state.v = Self::clamp(
            reference.v - step * gradient.scale * gradient.delta_v,
            min_velocity + eps_14,
            max_velocity - eps_14,
        );
        state.a = Self::clamp(
            reference.a - step * gradient.scale * gradient.delta_a,
            min_accel + eps_14,
            max_accel - eps_14,
        );
        if state.a > F::zero()
            && Self::max_v_at_zero_a(state.v, state.a, upper_limits.jerk())
                > upper_limits.max_vel() - eps_14
        {
            state.a = (two * upper_limits.jerk() * (upper_limits.max_vel() - state.v)).sqrt();
        }
        if state.a < F::zero()
            && Self::max_v_at_zero_a(state.v, state.a, -upper_limits.jerk())
                < upper_limits.min_vel() + eps_14
        {
            state.a = -((-two * upper_limits.jerk() * (upper_limits.min_vel() - state.v)).sqrt());
        }
        if state.a < F::zero()
            && Self::max_v_at_zero_a(state.v, state.a, lower_limits.jerk())
                > lower_limits.max_vel() - eps_14
        {
            state.a = -((two * lower_limits.jerk() * (lower_limits.max_vel() - state.v)).sqrt());
        }
        if state.a > F::zero()
            && Self::max_v_at_zero_a(state.v, state.a, -lower_limits.jerk())
                < lower_limits.min_vel() + eps_14
        {
            state.a = (-two * lower_limits.jerk() * (lower_limits.min_vel() - state.v)).sqrt();
        }
    }

    /// Outer gradient-descent loop. For each global iteration:
    /// - reset gradient accumulators
    /// - sum each section's analytic Jacobian into the bracketing waypoint
    ///   gradient buffers
    /// - call [`apply_step_length_update`](Self::apply_step_length_update) to
    ///   pick a step size that shrinks the total duration without breaking
    ///   feasibility
    ///
    /// Stops when the duration plateau is hit (and the synchronising joint
    /// stays put) or when the per-solve time budget is exceeded.
    fn run_global_optimization_pass(&mut self, profile_buffer: &mut [[Segment<F>; N]]) {
        // Each trial rewrites every `segments_tmp` cell before reading it:
        // non-zero-gradient (seg, dof) cells via StepA, zero-gradient cells
        // via the explicit refresh inside the trial loop, and
        // sync_time/sync_dof_index/block_selector via find_feasible_time.
        // The prev_sync_time tracker in apply_step_length_update reads
        // from `segments[]` rather than `segments_tmp[]` to avoid a stale
        // value.
        if self.segments_tmp.len() != self.segments.len() {
            self.segments_tmp
                .resize_with(self.segments.len(), Segment2Segment::new);
        }
        // `states_buffer` needs an initial sync because
        // `clamp_state_with_gradient` only writes the interior states (indices
        // 1..=n_segments−1); the start (index 0) and goal (last) entries are
        // read by StepA but never written by the optimizer. The buffer is tiny
        // (one `[KinThirdPose; N]` per waypoint, ~72 B/entry) so a full clone
        // is cheap.
        self.states_buffer.clone_from(&self.waypoint_states);

        while self.search_state.global_step_index < self.number_global_steps {
            // Reset per-section gradient accumulators.
            for seg_idx in 0..self.gradient_buffer.len() {
                for dof_idx in 0..N {
                    self.gradient_buffer[seg_idx][dof_idx].reset();
                }
            }

            let mut current_total_duration = F::zero();
            for segment_idx in 0..self.segments.len() {
                let sync_time = self.segments[segment_idx].sync_time;
                for dof_idx in 0..N {
                    // Borrow the profile rather than copying the whole 544 B
                    // Segment — Jacobian::compute only reads a handful of
                    // fields and never mutates.
                    let profile =
                        self.segments[segment_idx].dof_blocks[dof_idx].get_profile(sync_time);
                    let total_time =
                        profile.duration + profile.halt.duration + profile.accel_halt.duration;
                    if total_time < sync_time {
                        continue;
                    }
                    let jac = Jacobian::new(
                        self.waypoint_states[segment_idx][dof_idx],
                        self.waypoint_states[segment_idx + 1][dof_idx],
                        self.segment_limits[segment_idx][dof_idx].base,
                    );
                    let gradient = jac.compute(profile);
                    let jerk_over_8 =
                        self.segment_limits[segment_idx][dof_idx].jerk() / from_f::<F>(8.0);
                    if segment_idx > 0 {
                        self.gradient_buffer[segment_idx - 1][dof_idx].scale =
                            self.gradient_buffer[segment_idx - 1][dof_idx].scale
                                * gradient.scale_left;
                        self.gradient_buffer[segment_idx - 1][dof_idx].delta_v =
                            self.gradient_buffer[segment_idx - 1][dof_idx].delta_v + gradient.cv;
                        self.gradient_buffer[segment_idx - 1][dof_idx].delta_a =
                            self.gradient_buffer[segment_idx - 1][dof_idx].delta_a
                                + jerk_over_8 * gradient.ca;
                    }
                    if segment_idx < self.segments.len() - 1 {
                        self.gradient_buffer[segment_idx][dof_idx].scale =
                            self.gradient_buffer[segment_idx][dof_idx].scale * gradient.scale_right;
                        self.gradient_buffer[segment_idx][dof_idx].delta_v =
                            self.gradient_buffer[segment_idx][dof_idx].delta_v + gradient.vf;
                        self.gradient_buffer[segment_idx][dof_idx].delta_a =
                            self.gradient_buffer[segment_idx][dof_idx].delta_a
                                + jerk_over_8 * gradient.af;
                    }
                }
                current_total_duration = current_total_duration + sync_time;
            }

            let mut new_duration = F::zero();
            let mut sync_axis_changed = false;
            let step_accepted = self.apply_step_length_update(
                profile_buffer,
                &mut new_duration,
                current_total_duration,
                &mut sync_axis_changed,
            );

            let plateau = !step_accepted
                || fabs(current_total_duration - new_duration)
                    < self.duration_break_eps * current_total_duration;
            if plateau
                && !sync_axis_changed
                && self.search_state.global_step_index > self.min_global_steps
            {
                self.search_state.global_step_index += 1;
                break;
            }

            self.search_state.global_step_index += 1;
        }
    }

    /// Try up to eight gradient-descent step sizes per section. On success
    /// writes the new total duration into `new_total_duration`, sets
    /// `sync_axis_changed` if any controlling joint switched, and stores the
    /// updated profiles into `profile_trial`. On failure the step lengths are
    /// shrunk via a per-section diffusion smoothing and the trial is
    /// reattempted.
    fn apply_step_length_update(
        &mut self,
        profile_trial: &mut [[Segment<F>; N]],
        new_total_duration: &mut F,
        current_duration: F,
        sync_axis_changed: &mut bool,
    ) -> bool {
        let dbl_eps = from_f::<F>(f64::EPSILON);
        let duration_cap = fmin(
            current_duration * from_f::<F>(1.05),
            self.search_state.duration_cap,
        );
        for _iteration in 0..8usize {
            self.cost_estimate = F::zero();
            for seg_idx in 0..self.gradient_buffer.len() {
                for dof_idx in 0..N {
                    let lower = self.segment_limits[seg_idx][dof_idx];
                    let upper = self.segment_limits[seg_idx + 1][dof_idx];
                    Self::clamp_state_with_gradient(
                        &mut self.states_buffer[seg_idx + 1][dof_idx],
                        &self.waypoint_states[seg_idx + 1][dof_idx],
                        self.search_state.step_sizes[seg_idx],
                        &self.gradient_buffer[seg_idx][dof_idx],
                        &lower,
                        &upper,
                    );
                }
                if let Some(sync_dof) = self.segments[seg_idx].sync_dof_index {
                    let grad = self.gradient_buffer[seg_idx][sync_dof];
                    let jerk_over_8 =
                        self.segment_limits[seg_idx][sync_dof].jerk() / from_f::<F>(8.0);
                    self.cost_estimate = self.cost_estimate
                        + self.search_state.step_sizes[seg_idx]
                            * grad.scale
                            * (grad.delta_v * grad.delta_v
                                + grad.delta_a * grad.delta_a / jerk_over_8);
                }
            }

            let duration_target = duration_cap - from_f::<F>(0.5) * self.cost_estimate;
            let mut total_duration = F::zero();
            *sync_axis_changed = false;
            let mut profiles_valid = true;
            let mut step_length_consistent = true;

            #[allow(clippy::needless_range_loop)]
            for seg_idx in 0..self.segments.len() {
                #[allow(clippy::needless_range_loop)]
                for dof_idx in 0..N {
                    let left_zero = seg_idx == 0
                        || (fabs(self.gradient_buffer[seg_idx - 1][dof_idx].delta_v) < dbl_eps
                            && fabs(self.gradient_buffer[seg_idx - 1][dof_idx].delta_a) < dbl_eps);
                    let right_zero = seg_idx == self.segments.len() - 1
                        || (fabs(self.gradient_buffer[seg_idx][dof_idx].delta_v) < dbl_eps
                            && fabs(self.gradient_buffer[seg_idx][dof_idx].delta_a) < dbl_eps);
                    if left_zero && right_zero {
                        // No StepA recompute needed for this DoF, but the
                        // scratch buffer might still hold a stale value from a
                        // previous accepted swap; refresh it from `segments`
                        // so the upcoming `find_feasible_time` sees consistent
                        // state across all DoFs.
                        self.segments_tmp[seg_idx].dof_blocks[dof_idx] =
                            self.segments[seg_idx].dof_blocks[dof_idx];
                        continue;
                    }
                    let limits = self.segment_limits[seg_idx][dof_idx];
                    let step1 = pose_math::third_order::StepA::new(
                        self.states_buffer[seg_idx][dof_idx],
                        self.states_buffer[seg_idx + 1][dof_idx],
                        limits.base,
                    );
                    Self::copy_boundary_state(
                        &mut profile_trial[seg_idx][dof_idx],
                        &self.states_buffer[seg_idx][dof_idx],
                        &self.states_buffer[seg_idx + 1][dof_idx],
                    );
                    let input_profile = profile_trial[seg_idx][dof_idx];
                    let ok = step1.get_profile_with_scratch(
                        &input_profile,
                        &mut self.segments_tmp[seg_idx].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                    profiles_valid = profiles_valid && ok;
                    if !ok {
                        break;
                    }
                    // Sync the modified p_min back into the trial buffer.
                    profile_trial[seg_idx][dof_idx] =
                        self.segments_tmp[seg_idx].dof_blocks[dof_idx].p_min;
                    profiles_valid = profiles_valid
                        && Self::profile_within_position_limits(
                            &self.segments_tmp[seg_idx].dof_blocks[dof_idx].p_min,
                            &limits,
                        );
                    if !profiles_valid {
                        break;
                    }
                }
                // The "did this step push the section's sync_time past
                // 2× of where we started" check compares against the
                // currently-committed sync_time, not the prior scratch
                // value — which post-swap may be stale.
                let prev_sync_time = self.segments[seg_idx].sync_time;
                self.segments_tmp[seg_idx].find_feasible_time();
                total_duration = total_duration + self.segments_tmp[seg_idx].sync_time;
                if self.segments_tmp[seg_idx].sync_dof_index
                    != self.segments[seg_idx].sync_dof_index
                {
                    *sync_axis_changed = true;
                }
                if self.segments_tmp[seg_idx].sync_time >= from_f::<F>(2.0) * prev_sync_time {
                    step_length_consistent = false;
                }
            }

            if total_duration <= duration_target
                && total_duration > F::zero()
                && profiles_valid
                && step_length_consistent
            {
                let sixteen = from_f::<F>(16.0);
                for step_idx in 0..self.search_state.step_sizes.len() {
                    self.search_state.step_sizes[step_idx] =
                        self.search_state.step_sizes[step_idx] * sixteen;
                }
                *new_total_duration = total_duration;
                // Swap rather than clone: `segments` now owns the new state
                // and `segments_tmp` has the prior one (used as the next
                // trial's starting point). Cells in `segments_tmp` that the
                // next trial would skip (zero gradient) are refreshed
                // explicitly from `segments` in the trial loop, so the
                // overall optimisation stays consistent.
                core::mem::swap(&mut self.waypoint_states, &mut self.states_buffer);
                core::mem::swap(&mut self.segments, &mut self.segments_tmp);
                return true;
            }

            // Failed: tighten the per-section step lengths.
            let len_steps = self.search_state.step_sizes.len();
            for seg_idx in 0..len_steps {
                let left_grew =
                    self.segments_tmp[seg_idx].sync_time > self.segments[seg_idx].sync_time;
                let right_grew =
                    self.segments_tmp[seg_idx + 1].sync_time > self.segments[seg_idx + 1].sync_time;
                self.step_divider[seg_idx] = if left_grew && right_grew {
                    from_f::<F>(16.0)
                } else if left_grew || right_grew {
                    from_f::<F>(8.0)
                } else {
                    from_f::<F>(4.0)
                };
            }
            // Diffuse step dividers across the section ring.
            for _diffuse_iter in 0..8usize {
                if len_steps == 0 {
                    break;
                }
                self.step_divider_tmp[0] = (self.step_divider[0]
                    + if len_steps > 1 {
                        self.step_divider[1]
                    } else {
                        self.step_divider[0]
                    })
                    / from_f::<F>(2.0);
                for seg_idx in 1..len_steps.saturating_sub(1) {
                    self.step_divider_tmp[seg_idx] = (self.step_divider[seg_idx - 1]
                        + self.step_divider[seg_idx]
                        + self.step_divider[seg_idx + 1])
                        / from_f::<F>(3.0);
                }
                if len_steps >= 2 {
                    self.step_divider_tmp[len_steps - 1] = (self.step_divider[len_steps - 2]
                        + self.step_divider[len_steps - 1])
                        / from_f::<F>(2.0);
                }
                self.step_divider.clone_from(&self.step_divider_tmp);
            }
            for step_idx in 0..len_steps {
                self.search_state.step_sizes[step_idx] =
                    self.search_state.step_sizes[step_idx] / self.step_divider[step_idx];
            }
        }
        self.segments_tmp = self.segments.clone();
        self.states_buffer = self.waypoint_states.clone();
        false
    }

    /// Three-segment sliding-window relaxation. Walks `[start, start+1,
    /// start+2]` triples through the segment array in alternating phase and
    /// runs a small inner gradient-descent on the two interior waypoints.
    fn optimize_three_segment_window(&mut self, profile_buffer: &mut [[Segment<F>; N]]) {
        let max_relax_iterations = 4usize;
        if self.segments.len() < 3 {
            return;
        }
        let mut prev_total_duration = F::zero();
        while self.search_state.local_step_index < self.number_local_steps / 2 {
            let local_step = self.search_state.local_step_index;
            let mut window_start = (local_step % 2) as i64;
            while window_start + 2 < self.segments.len() as i64 {
                let ws = window_start as usize;
                let mut step_scale_primary = F::one();
                let mut step_scale_backup = F::one();
                for relax_iteration in 0..max_relax_iterations {
                    let is_backup_pass = relax_iteration == 0;
                    for dof_idx in 0..N {
                        self.gradient_buffer[ws][dof_idx].reset();
                        self.gradient_buffer[ws + 1][dof_idx].reset();

                        // Left segment contributes to gradient_buffer[ws].
                        let left_sync = self.segments[ws].sync_time;
                        let left_profile =
                            self.segments[ws].dof_blocks[dof_idx].get_profile(left_sync);
                        let left_total = left_profile.duration
                            + left_profile.halt.duration
                            + left_profile.accel_halt.duration;
                        if left_total >= left_sync {
                            let jac = Jacobian::new(
                                self.waypoint_states[ws][dof_idx],
                                self.waypoint_states[ws + 1][dof_idx],
                                self.segment_limits[ws][dof_idx].base,
                            );
                            let grad = jac.compute(left_profile);
                            if !is_backup_pass {
                                self.gradient_buffer[ws][dof_idx].delta_v =
                                    self.gradient_buffer[ws][dof_idx].delta_v
                                        + grad.vf / from_f::<F>(8.0);
                            }
                            self.gradient_buffer[ws][dof_idx].delta_a =
                                self.gradient_buffer[ws][dof_idx].delta_a
                                    + self.segment_limits[ws][dof_idx].jerk() * grad.af
                                        / from_f::<F>(8.0);
                        }

                        // Middle segment contributes to gradient_buffer[ws]
                        // and [ws+1].
                        let mid_sync = self.segments[ws + 1].sync_time;
                        let mid_profile =
                            self.segments[ws + 1].dof_blocks[dof_idx].get_profile(mid_sync);
                        let mid_total = mid_profile.duration
                            + mid_profile.halt.duration
                            + mid_profile.accel_halt.duration;
                        if mid_total >= mid_sync {
                            let jac = Jacobian::new(
                                self.waypoint_states[ws + 1][dof_idx],
                                self.waypoint_states[ws + 2][dof_idx],
                                self.segment_limits[ws + 1][dof_idx].base,
                            );
                            let grad = jac.compute(mid_profile);
                            if !is_backup_pass {
                                self.gradient_buffer[ws][dof_idx].delta_v =
                                    self.gradient_buffer[ws][dof_idx].delta_v
                                        + grad.cv * from_f::<F>(2.0);
                            }
                            self.gradient_buffer[ws][dof_idx].delta_a =
                                self.gradient_buffer[ws][dof_idx].delta_a
                                    + self.segment_limits[ws + 1][dof_idx].jerk()
                                        * grad.ca
                                        * from_f::<F>(2.0);
                            if !is_backup_pass {
                                self.gradient_buffer[ws + 1][dof_idx].delta_v =
                                    self.gradient_buffer[ws + 1][dof_idx].delta_v
                                        + grad.vf * from_f::<F>(2.0);
                            }
                            self.gradient_buffer[ws + 1][dof_idx].delta_a =
                                self.gradient_buffer[ws + 1][dof_idx].delta_a
                                    + self.segment_limits[ws + 1][dof_idx].jerk()
                                        * grad.af
                                        * from_f::<F>(2.0);
                        }

                        // Right segment contributes to gradient_buffer[ws+1].
                        let right_sync = self.segments[ws + 2].sync_time;
                        let right_profile =
                            self.segments[ws + 2].dof_blocks[dof_idx].get_profile(right_sync);
                        let right_total = right_profile.duration
                            + right_profile.halt.duration
                            + right_profile.accel_halt.duration;
                        if right_total >= right_sync {
                            let jac = Jacobian::new(
                                self.waypoint_states[ws + 2][dof_idx],
                                self.waypoint_states[ws + 3][dof_idx],
                                self.segment_limits[ws + 2][dof_idx].base,
                            );
                            let grad = jac.compute(right_profile);
                            if !is_backup_pass {
                                self.gradient_buffer[ws + 1][dof_idx].delta_v =
                                    self.gradient_buffer[ws + 1][dof_idx].delta_v
                                        + grad.cv / from_f::<F>(8.0);
                            }
                            self.gradient_buffer[ws + 1][dof_idx].delta_a =
                                self.gradient_buffer[ws + 1][dof_idx].delta_a
                                    + self.segment_limits[ws + 2][dof_idx].jerk() * grad.ca
                                        / from_f::<F>(8.0);
                        }
                    }
                    let mut new_window_duration = F::zero();
                    let window_duration_cap = self.segments[ws].sync_time
                        + self.segments[ws + 1].sync_time
                        + self.segments[ws + 2].sync_time;
                    let step_scale_ref = if is_backup_pass {
                        &mut step_scale_backup
                    } else {
                        &mut step_scale_primary
                    };
                    let window_relaxed = self.try_three_segment_window(
                        profile_buffer,
                        step_scale_ref,
                        ws,
                        &mut new_window_duration,
                        window_duration_cap,
                    );
                    if (!window_relaxed
                        || new_window_duration > window_duration_cap * from_f::<F>(0.9999))
                        && !is_backup_pass
                    {
                        break;
                    }
                }
                window_start += 2;
            }

            let mut current_total_duration = F::zero();
            for seg_idx in 0..self.segments.len() {
                current_total_duration = current_total_duration + self.segments[seg_idx].sync_time;
            }
            if fabs(prev_total_duration - current_total_duration)
                < self.duration_break_eps * current_total_duration
            {
                self.search_state.local_step_index += 1;
                break;
            }
            prev_total_duration = current_total_duration;
            self.search_state.local_step_index += 1;
        }
    }

    /// Inner relaxation step for [`optimize_three_segment_window`].
    fn try_three_segment_window(
        &mut self,
        profile_buffer: &mut [[Segment<F>; N]],
        step_scale: &mut F,
        window_start: usize,
        new_duration: &mut F,
        duration_cap: F,
    ) -> bool {
        let dbl_eps = from_f::<F>(f64::EPSILON);
        for _iteration in 0..8usize {
            let mut window_valid = true;
            #[allow(clippy::needless_range_loop)]
            for dof_idx in 0..N {
                let limits_left = self.segment_limits[window_start][dof_idx];
                let limits_mid = self.segment_limits[window_start + 1][dof_idx];
                let limits_right = self.segment_limits[window_start + 2][dof_idx];
                let grad_left = self.gradient_buffer[window_start][dof_idx];
                let grad_mid = self.gradient_buffer[window_start + 1][dof_idx];
                let wp1 = self.waypoint_states[window_start + 1][dof_idx];
                let wp2 = self.waypoint_states[window_start + 2][dof_idx];
                Self::clamp_state_with_gradient(
                    &mut self.states_buffer[window_start + 1][dof_idx],
                    &wp1,
                    *step_scale,
                    &grad_left,
                    &limits_left,
                    &limits_mid,
                );
                Self::clamp_state_with_gradient(
                    &mut self.states_buffer[window_start + 2][dof_idx],
                    &wp2,
                    *step_scale,
                    &grad_mid,
                    &limits_mid,
                    &limits_right,
                );

                // Skip when nothing measurably changed.
                let p0 = profile_buffer[window_start][dof_idx];
                let p1 = profile_buffer[window_start + 1][dof_idx];
                let p2 = profile_buffer[window_start + 2][dof_idx];
                let s0 = self.states_buffer[window_start][dof_idx];
                let s1 = self.states_buffer[window_start + 1][dof_idx];
                let s2 = self.states_buffer[window_start + 2][dof_idx];
                let s3 = self.states_buffer[window_start + 3][dof_idx];
                if fabs(p0.v[0] - s0.v) < dbl_eps
                    && fabs(p0.a[0] - s0.a) < dbl_eps
                    && fabs(p1.v[0] - s1.v) < dbl_eps
                    && fabs(p1.a[0] - s1.a) < dbl_eps
                    && fabs(p2.v[0] - s2.v) < dbl_eps
                    && fabs(p2.a[0] - s2.a) < dbl_eps
                    && fabs(p0.vf - s1.v) < dbl_eps
                    && fabs(p0.af - s1.a) < dbl_eps
                    && fabs(p1.vf - s2.v) < dbl_eps
                    && fabs(p1.af - s2.a) < dbl_eps
                    && fabs(p2.vf - s3.v) < dbl_eps
                    && fabs(p2.af - s3.a) < dbl_eps
                {
                    continue;
                }

                let step_left = pose_math::third_order::StepA::new(s0, s1, limits_left.base);
                let step_mid = pose_math::third_order::StepA::new(s1, s2, limits_mid.base);
                let step_right = pose_math::third_order::StepA::new(s2, s3, limits_right.base);
                Self::copy_boundary_state(&mut profile_buffer[window_start][dof_idx], &s0, &s1);
                Self::copy_boundary_state(&mut profile_buffer[window_start + 1][dof_idx], &s1, &s2);
                Self::copy_boundary_state(&mut profile_buffer[window_start + 2][dof_idx], &s2, &s3);
                let pb0 = profile_buffer[window_start][dof_idx];
                let pb1 = profile_buffer[window_start + 1][dof_idx];
                let pb2 = profile_buffer[window_start + 2][dof_idx];
                window_valid = window_valid
                    && step_left.get_profile_with_scratch(
                        &pb0,
                        &mut self.segments_tmp[window_start].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                window_valid = window_valid
                    && step_mid.get_profile_with_scratch(
                        &pb1,
                        &mut self.segments_tmp[window_start + 1].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                window_valid = window_valid
                    && step_right.get_profile_with_scratch(
                        &pb2,
                        &mut self.segments_tmp[window_start + 2].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                window_valid = window_valid
                    && Self::profile_within_position_limits(
                        &self.segments_tmp[window_start].dof_blocks[dof_idx].p_min,
                        &limits_left,
                    );
                window_valid = window_valid
                    && Self::profile_within_position_limits(
                        &self.segments_tmp[window_start + 1].dof_blocks[dof_idx].p_min,
                        &limits_mid,
                    );
                window_valid = window_valid
                    && Self::profile_within_position_limits(
                        &self.segments_tmp[window_start + 2].dof_blocks[dof_idx].p_min,
                        &limits_right,
                    );
                if !window_valid {
                    break;
                }
                // Push updated p_min into the trial profiles.
                profile_buffer[window_start][dof_idx] =
                    self.segments_tmp[window_start].dof_blocks[dof_idx].p_min;
                profile_buffer[window_start + 1][dof_idx] =
                    self.segments_tmp[window_start + 1].dof_blocks[dof_idx].p_min;
                profile_buffer[window_start + 2][dof_idx] =
                    self.segments_tmp[window_start + 2].dof_blocks[dof_idx].p_min;
            }
            self.segments_tmp[window_start].find_feasible_time();
            self.segments_tmp[window_start + 1].find_feasible_time();
            self.segments_tmp[window_start + 2].find_feasible_time();
            let window_duration = self.segments_tmp[window_start].sync_time
                + self.segments_tmp[window_start + 1].sync_time
                + self.segments_tmp[window_start + 2].sync_time;
            if window_duration < duration_cap && window_duration > F::zero() && window_valid {
                *new_duration = window_duration;
                *step_scale = *step_scale * from_f::<F>(4.0);
                self.waypoint_states[window_start + 1] = self.states_buffer[window_start + 1];
                self.waypoint_states[window_start + 2] = self.states_buffer[window_start + 2];
                for off in 0..3usize {
                    self.segments[window_start + off].dof_blocks =
                        self.segments_tmp[window_start + off].dof_blocks;
                    self.segments[window_start + off].sync_time =
                        self.segments_tmp[window_start + off].sync_time;
                    self.segments[window_start + off].sync_dof_index =
                        self.segments_tmp[window_start + off].sync_dof_index;
                    self.segments[window_start + off].block_selector =
                        self.segments_tmp[window_start + off].block_selector;
                }
                return true;
            }
            *step_scale = *step_scale / from_f::<F>(8.0);
        }
        // Restore state buffer on failure.
        self.states_buffer[window_start + 1] = self.waypoint_states[window_start + 1];
        self.states_buffer[window_start + 2] = self.waypoint_states[window_start + 2];
        false
    }

    /// Two-segment sliding-window relaxation. Same structure as the
    /// three-segment pass but with one degree of freedom: only the waypoint
    /// at `window_start + 1` is moved.
    fn optimize_two_segment_window(&mut self, profile_buffer: &mut [[Segment<F>; N]]) {
        let step_repeats = 4usize;
        let grid_stride = 2usize;
        let mut prev_total_duration = F::zero();

        while self.search_state.smoothing_step_index < self.number_local_steps / 2 {
            let smoothing_idx = self.search_state.smoothing_step_index;
            for phase_offset in 0..grid_stride {
                let mut window_start = ((smoothing_idx + phase_offset) % grid_stride) as i64;
                while window_start + 1 < self.segments.len() as i64 {
                    let ws = window_start as usize;
                    let mut step_scale_primary = F::one();
                    let mut step_scale_backup = from_f::<F>(16.0);
                    for iteration in 0..step_repeats {
                        let is_backup_pass = iteration == 0;
                        for dof_idx in 0..N {
                            self.gradient_buffer[ws][dof_idx].reset();

                            // Left segment.
                            let left_sync = self.segments[ws].sync_time;
                            let left_profile =
                                self.segments[ws].dof_blocks[dof_idx].get_profile(left_sync);
                            let left_total = left_profile.duration
                                + left_profile.halt.duration
                                + left_profile.accel_halt.duration;
                            if left_total >= left_sync {
                                let jac = Jacobian::new(
                                    self.waypoint_states[ws][dof_idx],
                                    self.waypoint_states[ws + 1][dof_idx],
                                    self.segment_limits[ws][dof_idx].base,
                                );
                                let grad = jac.compute(left_profile);
                                if !is_backup_pass {
                                    self.gradient_buffer[ws][dof_idx].delta_v =
                                        self.gradient_buffer[ws][dof_idx].delta_v + grad.vf;
                                }
                                self.gradient_buffer[ws][dof_idx].delta_a =
                                    self.gradient_buffer[ws][dof_idx].delta_a
                                        + self.segment_limits[ws][dof_idx].jerk() * grad.af
                                            / from_f::<F>(8.0);
                            }

                            // Right segment.
                            let right_sync = self.segments[ws + 1].sync_time;
                            let right_profile =
                                self.segments[ws + 1].dof_blocks[dof_idx].get_profile(right_sync);
                            let right_total = right_profile.duration
                                + right_profile.halt.duration
                                + right_profile.accel_halt.duration;
                            if right_total >= right_sync {
                                let jac = Jacobian::new(
                                    self.waypoint_states[ws + 1][dof_idx],
                                    self.waypoint_states[ws + 2][dof_idx],
                                    self.segment_limits[ws + 1][dof_idx].base,
                                );
                                let grad = jac.compute(right_profile);
                                if !is_backup_pass {
                                    self.gradient_buffer[ws][dof_idx].delta_v =
                                        self.gradient_buffer[ws][dof_idx].delta_v + grad.cv;
                                }
                                self.gradient_buffer[ws][dof_idx].delta_a =
                                    self.gradient_buffer[ws][dof_idx].delta_a
                                        + self.segment_limits[ws + 1][dof_idx].jerk() * grad.ca
                                            / from_f::<F>(8.0);
                            }
                        }
                        let mut new_window_duration = F::zero();
                        let duration_cap =
                            self.segments[ws].sync_time + self.segments[ws + 1].sync_time;
                        let step_scale_ref = if is_backup_pass {
                            &mut step_scale_backup
                        } else {
                            &mut step_scale_primary
                        };
                        let window_relaxed = self.try_two_segment_window(
                            profile_buffer,
                            step_scale_ref,
                            ws,
                            &mut new_window_duration,
                            duration_cap,
                        );
                        if (!window_relaxed
                            || new_window_duration > duration_cap * from_f::<F>(0.9999))
                            && !is_backup_pass
                        {
                            break;
                        }
                    }
                    window_start += grid_stride as i64;
                }
            }

            let mut current_total_duration = F::zero();
            for segment_idx in 0..self.segments.len() {
                current_total_duration =
                    current_total_duration + self.segments[segment_idx].sync_time;
            }
            if fabs(prev_total_duration - current_total_duration)
                < self.duration_break_eps * current_total_duration
            {
                self.search_state.smoothing_step_index += 1;
                break;
            }
            prev_total_duration = current_total_duration;
            self.search_state.smoothing_step_index += 1;
        }
    }

    /// Inner relaxation step for [`optimize_two_segment_window`].
    fn try_two_segment_window(
        &mut self,
        profile_buffer: &mut [[Segment<F>; N]],
        step_scale: &mut F,
        window_start: usize,
        new_duration: &mut F,
        duration_cap: F,
    ) -> bool {
        let dbl_eps = from_f::<F>(f64::EPSILON);
        for _iteration in 0..8usize {
            let mut window_valid = true;
            #[allow(clippy::needless_range_loop)]
            for dof_idx in 0..N {
                let limits_left = self.segment_limits[window_start][dof_idx];
                let limits_right = self.segment_limits[window_start + 1][dof_idx];
                let grad_left = self.gradient_buffer[window_start][dof_idx];
                let wp1 = self.waypoint_states[window_start + 1][dof_idx];
                Self::clamp_state_with_gradient(
                    &mut self.states_buffer[window_start + 1][dof_idx],
                    &wp1,
                    *step_scale,
                    &grad_left,
                    &limits_left,
                    &limits_right,
                );

                let p0 = profile_buffer[window_start][dof_idx];
                let p1 = profile_buffer[window_start + 1][dof_idx];
                let s0 = self.states_buffer[window_start][dof_idx];
                let s1 = self.states_buffer[window_start + 1][dof_idx];
                let s2 = self.states_buffer[window_start + 2][dof_idx];
                if fabs(p0.v[0] - s0.v) < dbl_eps
                    && fabs(p0.a[0] - s0.a) < dbl_eps
                    && fabs(p1.v[0] - s1.v) < dbl_eps
                    && fabs(p1.a[0] - s1.a) < dbl_eps
                    && fabs(p0.vf - s1.v) < dbl_eps
                    && fabs(p0.af - s1.a) < dbl_eps
                    && fabs(p1.vf - s2.v) < dbl_eps
                    && fabs(p1.af - s2.a) < dbl_eps
                {
                    continue;
                }

                let step_left = pose_math::third_order::StepA::new(s0, s1, limits_left.base);
                let step_right = pose_math::third_order::StepA::new(s1, s2, limits_right.base);
                Self::copy_boundary_state(&mut profile_buffer[window_start][dof_idx], &s0, &s1);
                Self::copy_boundary_state(&mut profile_buffer[window_start + 1][dof_idx], &s1, &s2);
                let pb0 = profile_buffer[window_start][dof_idx];
                let pb1 = profile_buffer[window_start + 1][dof_idx];
                window_valid = window_valid
                    && step_left.get_profile_with_scratch(
                        &pb0,
                        &mut self.segments_tmp[window_start].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                window_valid = window_valid
                    && step_right.get_profile_with_scratch(
                        &pb1,
                        &mut self.segments_tmp[window_start + 1].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                window_valid = window_valid
                    && Self::profile_within_position_limits(
                        &self.segments_tmp[window_start].dof_blocks[dof_idx].p_min,
                        &limits_left,
                    );
                window_valid = window_valid
                    && Self::profile_within_position_limits(
                        &self.segments_tmp[window_start + 1].dof_blocks[dof_idx].p_min,
                        &limits_right,
                    );
                if !window_valid {
                    break;
                }
                profile_buffer[window_start][dof_idx] =
                    self.segments_tmp[window_start].dof_blocks[dof_idx].p_min;
                profile_buffer[window_start + 1][dof_idx] =
                    self.segments_tmp[window_start + 1].dof_blocks[dof_idx].p_min;
            }
            self.segments_tmp[window_start].find_feasible_time();
            self.segments_tmp[window_start + 1].find_feasible_time();
            let window_duration = self.segments_tmp[window_start].sync_time
                + self.segments_tmp[window_start + 1].sync_time;
            if window_duration < duration_cap && window_duration > F::zero() && window_valid {
                *new_duration = window_duration;
                *step_scale = *step_scale * from_f::<F>(4.0);
                self.waypoint_states[window_start + 1] = self.states_buffer[window_start + 1];
                for off in 0..2usize {
                    self.segments[window_start + off].dof_blocks =
                        self.segments_tmp[window_start + off].dof_blocks;
                    self.segments[window_start + off].sync_time =
                        self.segments_tmp[window_start + off].sync_time;
                    self.segments[window_start + off].sync_dof_index =
                        self.segments_tmp[window_start + off].sync_dof_index;
                    self.segments[window_start + off].block_selector =
                        self.segments_tmp[window_start + off].block_selector;
                }
                return true;
            }
            *step_scale = *step_scale / from_f::<F>(8.0);
        }
        self.states_buffer[window_start + 1] = self.waypoint_states[window_start + 1];
        false
    }

    /// Per-non-controlling-DoF smoothing of waypoint velocity/acceleration
    /// states using finite differences. Accepts a tentative update only when
    /// every section's profile stays feasible and unblocked.
    fn smooth_waypoint_states(&mut self, iterations: usize, plan: &mut Plan<N, F>) {
        if iterations == 0 || plan.profiles.len() < 3 {
            return;
        }
        let smoothing_weight = (F::from(iterations).unwrap() / from_f::<F>(8.0)) + from_f::<F>(8.0);
        for _iter in 0..iterations {
            for seg_idx in 0..plan.profiles.len() - 2 {
                for dof_idx in 0..N {
                    let sync_left = self.segments[seg_idx].sync_dof_index;
                    let sync_mid = self.segments[seg_idx + 1].sync_dof_index;
                    let sync_right = self.segments[seg_idx + 2].sync_dof_index;
                    if sync_left == Some(dof_idx)
                        || sync_mid == Some(dof_idx)
                        || sync_right == Some(dof_idx)
                    {
                        continue;
                    }
                    self.states_buffer[seg_idx + 1][dof_idx] =
                        self.waypoint_states[seg_idx + 1][dof_idx];
                    self.states_buffer[seg_idx + 2][dof_idx] =
                        self.waypoint_states[seg_idx + 2][dof_idx];
                    let v1 = (self.waypoint_states[seg_idx + 1][dof_idx].p
                        - self.waypoint_states[seg_idx][dof_idx].p)
                        / self.segments[seg_idx].sync_time;
                    let v2 = (self.waypoint_states[seg_idx + 2][dof_idx].p
                        - self.waypoint_states[seg_idx + 1][dof_idx].p)
                        / self.segments[seg_idx + 1].sync_time;
                    let v3 = (self.waypoint_states[seg_idx + 3][dof_idx].p
                        - self.waypoint_states[seg_idx + 2][dof_idx].p)
                        / self.segments[seg_idx + 2].sync_time;
                    self.states_buffer[seg_idx + 1][dof_idx].v =
                        (v1 + smoothing_weight * self.states_buffer[seg_idx + 1][dof_idx].v + v2)
                            / (smoothing_weight + from_f::<F>(2.0));
                    self.states_buffer[seg_idx + 2][dof_idx].v =
                        (v2 + smoothing_weight * self.states_buffer[seg_idx + 2][dof_idx].v + v3)
                            / (smoothing_weight + from_f::<F>(2.0));
                    let a1 = (self.states_buffer[seg_idx + 1][dof_idx].v
                        - self.waypoint_states[seg_idx][dof_idx].v)
                        / self.segments[seg_idx].sync_time;
                    let a2 = (self.states_buffer[seg_idx + 2][dof_idx].v
                        - self.states_buffer[seg_idx + 1][dof_idx].v)
                        / self.segments[seg_idx + 1].sync_time;
                    let a3 = (self.waypoint_states[seg_idx + 3][dof_idx].v
                        - self.states_buffer[seg_idx + 2][dof_idx].v)
                        / self.segments[seg_idx + 2].sync_time;
                    self.states_buffer[seg_idx + 1][dof_idx].a =
                        (a1 + smoothing_weight * self.states_buffer[seg_idx + 1][dof_idx].a + a2)
                            / (smoothing_weight + from_f::<F>(2.0));
                    self.states_buffer[seg_idx + 2][dof_idx].a =
                        (a2 + smoothing_weight * self.states_buffer[seg_idx + 2][dof_idx].a + a3)
                            / (smoothing_weight + from_f::<F>(2.0));

                    let tol = from_f::<F>(1e-5);
                    if fabs(
                        self.waypoint_states[seg_idx + 1][dof_idx].v
                            - self.states_buffer[seg_idx + 1][dof_idx].v,
                    ) < tol
                        && fabs(
                            self.waypoint_states[seg_idx + 2][dof_idx].v
                                - self.states_buffer[seg_idx + 2][dof_idx].v,
                        ) < tol
                        && fabs(
                            self.waypoint_states[seg_idx + 1][dof_idx].a
                                - self.states_buffer[seg_idx + 1][dof_idx].a,
                        ) < tol
                        && fabs(
                            self.waypoint_states[seg_idx + 2][dof_idx].a
                                - self.states_buffer[seg_idx + 2][dof_idx].a,
                        ) < tol
                    {
                        continue;
                    }

                    let step_left = pose_math::third_order::StepA::new(
                        self.waypoint_states[seg_idx][dof_idx],
                        self.states_buffer[seg_idx + 1][dof_idx],
                        self.segment_limits[seg_idx][dof_idx].base,
                    );
                    let step_mid = pose_math::third_order::StepA::new(
                        self.states_buffer[seg_idx + 1][dof_idx],
                        self.states_buffer[seg_idx + 2][dof_idx],
                        self.segment_limits[seg_idx + 1][dof_idx].base,
                    );
                    let step_right = pose_math::third_order::StepA::new(
                        self.states_buffer[seg_idx + 2][dof_idx],
                        self.waypoint_states[seg_idx + 3][dof_idx],
                        self.segment_limits[seg_idx + 2][dof_idx].base,
                    );
                    let mut profile_left = Segment::<F>::empty();
                    let mut profile_mid = Segment::<F>::empty();
                    let mut profile_right = Segment::<F>::empty();
                    profile_left.set_boundary_explicit(
                        self.waypoint_states[seg_idx][dof_idx].p,
                        self.waypoint_states[seg_idx][dof_idx].v,
                        self.waypoint_states[seg_idx][dof_idx].a,
                        self.waypoint_states[seg_idx + 1][dof_idx].p,
                        self.states_buffer[seg_idx + 1][dof_idx].v,
                        self.states_buffer[seg_idx + 1][dof_idx].a,
                    );
                    profile_mid.set_boundary_explicit(
                        self.waypoint_states[seg_idx + 1][dof_idx].p,
                        self.states_buffer[seg_idx + 1][dof_idx].v,
                        self.states_buffer[seg_idx + 1][dof_idx].a,
                        self.waypoint_states[seg_idx + 2][dof_idx].p,
                        self.states_buffer[seg_idx + 2][dof_idx].v,
                        self.states_buffer[seg_idx + 2][dof_idx].a,
                    );
                    profile_right.set_boundary_explicit(
                        self.waypoint_states[seg_idx + 2][dof_idx].p,
                        self.states_buffer[seg_idx + 2][dof_idx].v,
                        self.states_buffer[seg_idx + 2][dof_idx].a,
                        self.waypoint_states[seg_idx + 3][dof_idx].p,
                        self.waypoint_states[seg_idx + 3][dof_idx].v,
                        self.waypoint_states[seg_idx + 3][dof_idx].a,
                    );
                    let ok_left = step_left.get_profile_with_scratch(
                        &profile_left,
                        &mut self.segments_tmp[seg_idx].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                    let ok_mid = step_mid.get_profile_with_scratch(
                        &profile_mid,
                        &mut self.segments_tmp[seg_idx + 1].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                    let ok_right = step_right.get_profile_with_scratch(
                        &profile_right,
                        &mut self.segments_tmp[seg_idx + 2].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                    let ok = ok_left && ok_mid && ok_right;
                    if ok
                        && !self.segments_tmp[seg_idx].dof_blocks[dof_idx]
                            .is_blocked(self.segments[seg_idx].sync_time)
                        && !self.segments_tmp[seg_idx + 1].dof_blocks[dof_idx]
                            .is_blocked(self.segments[seg_idx + 1].sync_time)
                        && !self.segments_tmp[seg_idx + 2].dof_blocks[dof_idx]
                            .is_blocked(self.segments[seg_idx + 2].sync_time)
                    {
                        self.waypoint_states[seg_idx + 1][dof_idx] =
                            self.states_buffer[seg_idx + 1][dof_idx];
                        self.waypoint_states[seg_idx + 2][dof_idx] =
                            self.states_buffer[seg_idx + 2][dof_idx];
                    }
                }
            }
        }
    }

    /// Per-non-controlling-DoF smoothing of the waypoint accelerations
    /// based on the neighbouring segment plateau values. Accepts updates that
    /// remain feasible and unblocked.
    fn smooth_waypoint_acceleration(&mut self, iterations: usize, plan: &mut Plan<N, F>) {
        if iterations == 0 || plan.profiles.len() < 2 {
            return;
        }
        for seg_idx in 0..plan.profiles.len() - 1 {
            for dof_idx in 0..N {
                let sync_left = self.segments[seg_idx].sync_dof_index;
                let sync_right = self.segments[seg_idx + 1].sync_dof_index;
                if sync_left == Some(dof_idx) || sync_right == Some(dof_idx) {
                    continue;
                }
                let mut profile_left_acc = plan.profiles[seg_idx][dof_idx];
                let mut profile_right_acc = plan.profiles[seg_idx + 1][dof_idx];
                for _iter in 0..iterations {
                    let middle_a = self.waypoint_states[seg_idx + 1][dof_idx].a;
                    let both_above =
                        profile_left_acc.a[6] > middle_a && profile_right_acc.a[1] > middle_a;
                    let both_below =
                        profile_left_acc.a[6] < middle_a && profile_right_acc.a[1] < middle_a;
                    if !both_above && !both_below {
                        continue;
                    }
                    self.states_buffer[seg_idx + 1][dof_idx].a = if both_above {
                        fmin(profile_left_acc.a[6], profile_right_acc.a[1])
                    } else {
                        fmax(profile_left_acc.a[6], profile_right_acc.a[1])
                    };
                    let step_left = pose_math::third_order::StepA::new(
                        self.states_buffer[seg_idx][dof_idx],
                        self.states_buffer[seg_idx + 1][dof_idx],
                        self.segment_limits[seg_idx][dof_idx].base,
                    );
                    let step_right = pose_math::third_order::StepA::new(
                        self.states_buffer[seg_idx + 1][dof_idx],
                        self.states_buffer[seg_idx + 2][dof_idx],
                        self.segment_limits[seg_idx + 1][dof_idx].base,
                    );
                    let mut profile_left_step = Segment::<F>::empty();
                    let mut profile_right_step = Segment::<F>::empty();
                    profile_left_step.set_boundary_explicit(
                        self.states_buffer[seg_idx][dof_idx].p,
                        self.states_buffer[seg_idx][dof_idx].v,
                        self.states_buffer[seg_idx][dof_idx].a,
                        self.states_buffer[seg_idx + 1][dof_idx].p,
                        self.states_buffer[seg_idx + 1][dof_idx].v,
                        self.states_buffer[seg_idx + 1][dof_idx].a,
                    );
                    profile_right_step.set_boundary_explicit(
                        self.states_buffer[seg_idx + 1][dof_idx].p,
                        self.states_buffer[seg_idx + 1][dof_idx].v,
                        self.states_buffer[seg_idx + 1][dof_idx].a,
                        self.states_buffer[seg_idx + 2][dof_idx].p,
                        self.states_buffer[seg_idx + 2][dof_idx].v,
                        self.states_buffer[seg_idx + 2][dof_idx].a,
                    );
                    let ok_left = step_left.get_profile_with_scratch(
                        &profile_left_step,
                        &mut self.segments_tmp[seg_idx].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                    let ok_right = step_right.get_profile_with_scratch(
                        &profile_right_step,
                        &mut self.segments_tmp[seg_idx + 1].dof_blocks[dof_idx],
                        &mut self.step_a_scratch,
                    );
                    if !ok_left
                        || !ok_right
                        || self.segments_tmp[seg_idx].dof_blocks[dof_idx]
                            .is_blocked(self.segments[seg_idx].sync_time)
                        || self.segments_tmp[seg_idx + 1].dof_blocks[dof_idx]
                            .is_blocked(self.segments[seg_idx + 1].sync_time)
                    {
                        continue;
                    }
                    self.waypoint_states[seg_idx + 1][dof_idx] =
                        self.states_buffer[seg_idx + 1][dof_idx];
                    let t_section_left = self.segments[seg_idx].sync_time
                        - profile_left_acc.halt.duration
                        - profile_left_acc.accel_halt.duration;
                    let t_section_right = self.segments[seg_idx + 1].sync_time
                        - profile_right_acc.halt.duration
                        - profile_right_acc.accel_halt.duration;
                    let step_left2 = pose_math::third_order::StepB::new(
                        t_section_left,
                        self.waypoint_states[seg_idx][dof_idx],
                        self.waypoint_states[seg_idx + 1][dof_idx],
                        self.segment_limits[seg_idx][dof_idx].base,
                    );
                    let mut step_left2 = step_left2;
                    let mut step_right2 = pose_math::third_order::StepB::new(
                        t_section_right,
                        self.waypoint_states[seg_idx + 1][dof_idx],
                        self.waypoint_states[seg_idx + 2][dof_idx],
                        self.segment_limits[seg_idx + 1][dof_idx].base,
                    );
                    Self::copy_boundary_state(
                        &mut profile_left_acc,
                        &self.waypoint_states[seg_idx][dof_idx],
                        &self.waypoint_states[seg_idx + 1][dof_idx],
                    );
                    Self::copy_boundary_state(
                        &mut profile_right_acc,
                        &self.waypoint_states[seg_idx + 1][dof_idx],
                        &self.waypoint_states[seg_idx + 2][dof_idx],
                    );
                    if step_b_get_profile(&mut step_left2, &mut profile_left_acc) {
                        plan.profiles[seg_idx][dof_idx] = profile_left_acc;
                    }
                    if step_b_get_profile(&mut step_right2, &mut profile_right_acc) {
                        plan.profiles[seg_idx + 1][dof_idx] = profile_right_acc;
                    }
                }
            }
        }
    }

    fn finalize_second_order(
        &mut self,
        spec: &MotionSpec<N, F>,
        plan: &mut Plan<N, F>,
        with_smoothing: bool,
    ) -> StepStatus {
        if with_smoothing && self.number_smoothing_steps > 0 {
            self.smooth_waypoint_states(self.number_smoothing_steps, plan);
        }
        let dbl_eps = from_f::<F>(f64::EPSILON);
        let tol = from_f::<F>(4.0) * dbl_eps;
        for seg_idx in 0..plan.profiles.len() {
            for dof_idx in 0..N {
                let enabled = spec.axis_active[dof_idx];
                if !enabled {
                    if seg_idx > 0 {
                        let prev = plan.profiles[seg_idx - 1][dof_idx];
                        let prev_state = KinThirdPose::new(prev.pf, prev.vf, prev.af);
                        let ext_state =
                            prev_state.next(self.segments[seg_idx - 1].sync_time, F::zero());
                        plan.profiles[seg_idx][dof_idx].pf = ext_state.p;
                        plan.profiles[seg_idx][dof_idx].vf = ext_state.v;
                        plan.profiles[seg_idx][dof_idx].af = ext_state.a;
                    }
                    continue;
                }
                if Some(dof_idx) == self.segments[seg_idx].sync_dof_index {
                    plan.profiles[seg_idx][dof_idx] =
                        self.segments[seg_idx].get_controlling_profile();
                    continue;
                }
                let profile_now = plan.profiles[seg_idx][dof_idx];
                let t_section = self.segments[seg_idx].sync_time
                    - profile_now.halt.duration
                    - profile_now.accel_halt.duration;
                let block = &self.segments[seg_idx].dof_blocks[dof_idx];
                if fabs(t_section - block.t_min) < tol {
                    plan.profiles[seg_idx][dof_idx] = block.p_min;
                    continue;
                }
                if let Some(span) = block.blocked_interval_a
                    && fabs(t_section - span.right_time) < tol
                {
                    plan.profiles[seg_idx][dof_idx] = span.profile_at_right;
                    continue;
                }
                if let Some(span) = block.blocked_interval_b
                    && fabs(t_section - span.right_time) < tol
                {
                    plan.profiles[seg_idx][dof_idx] = span.profile_at_right;
                    continue;
                }
                let mut step2 = pose_math::third_order::StepB::new(
                    t_section,
                    self.waypoint_states[seg_idx][dof_idx],
                    self.waypoint_states[seg_idx + 1][dof_idx],
                    self.segment_limits[seg_idx][dof_idx].base,
                );
                step2.single_inflection_enabled = self.number_smoothing_steps > 0;
                Self::copy_boundary_state(
                    &mut plan.profiles[seg_idx][dof_idx],
                    &self.waypoint_states[seg_idx][dof_idx],
                    &self.waypoint_states[seg_idx + 1][dof_idx],
                );
                if !step_b_get_profile(&mut step2, &mut plan.profiles[seg_idx][dof_idx]) {
                    return StepStatus::StepTwoFailed;
                }
            }
        }
        if with_smoothing && self.number_acceleration_smoothing_steps > 0 {
            self.smooth_waypoint_acceleration(self.number_acceleration_smoothing_steps, plan);
        }
        if plan.profiles.is_empty() {
            plan.duration = F::zero();
            return StepStatus::Done;
        }
        plan.intermediate_durations
            .resize(plan.profiles.len(), F::zero());
        plan.intermediate_durations[0] = self.segments[0].sync_time;
        for seg_idx in 1..plan.profiles.len() {
            plan.intermediate_durations[seg_idx] =
                plan.intermediate_durations[seg_idx - 1] + self.segments[seg_idx].sync_time;
        }
        plan.duration = *plan.intermediate_durations.last().unwrap_or(&F::zero());
        StepStatus::Done
    }

    /// Compress consecutive identical waypoints into a single section. Mirrors
    /// the per-DoF "no-movement" filter on the C++ side (the special-cased 1-DoF
    /// monotonicity collapse is skipped here for simplicity).
    fn collect_section_positions(&mut self, spec: &MotionSpec<N, F>) -> usize {
        let waypoint_count = spec.waypoint_poses.len();
        let dbl_eps = from_f::<F>(f64::EPSILON);
        let mut section_marker = 0usize;
        for inter_pos_idx in 0..waypoint_count {
            let mut no_movement = true;
            for dof_idx in 0..N {
                let prev = if inter_pos_idx > 0 {
                    spec.waypoint_poses[inter_pos_idx - 1][dof_idx]
                } else {
                    spec.current_pose[dof_idx]
                };
                let cur = spec.waypoint_poses[inter_pos_idx][dof_idx];
                if fabs(prev - cur) > dbl_eps {
                    no_movement = false;
                    break;
                }
            }
            if !no_movement {
                self.section_positions[section_marker] = spec.waypoint_poses[inter_pos_idx];
                section_marker += 1;
            }
        }
        section_marker
    }

    /// Populate `waypoint_states` and per-section `segment_limits` from the
    /// MotionSpec. Number of intermediate waypoints is `section_marker`, so
    /// there are `section_marker + 1` sections in total.
    fn fill_states_and_limits(&mut self, spec: &MotionSpec<N, F>, section_marker: usize) {
        let section_cnt = section_marker + 1;
        for dof_idx in 0..N {
            // Initial state.
            self.waypoint_states[0][dof_idx].p = spec.current_pose[dof_idx];
            self.waypoint_states[0][dof_idx].v = spec.current_vel[dof_idx];
            self.waypoint_states[0][dof_idx].a = spec.current_accel[dof_idx];
            // Intermediate states with zero v/a.
            for section_idx in 1..section_cnt {
                self.waypoint_states[section_idx][dof_idx].p =
                    self.section_positions[section_idx - 1][dof_idx];
                self.waypoint_states[section_idx][dof_idx].v = F::zero();
                self.waypoint_states[section_idx][dof_idx].a = F::zero();
            }
            // Per-section limits.
            for seg_idx in 0..self.segment_limits.len() {
                let max_vel = spec
                    .per_section_max_vel
                    .as_ref()
                    .map(|v| v[seg_idx][dof_idx])
                    .unwrap_or(spec.max_vel[dof_idx]);
                let max_accel = spec
                    .per_section_max_accel
                    .as_ref()
                    .map(|v| v[seg_idx][dof_idx])
                    .unwrap_or(spec.max_accel[dof_idx]);
                let max_jerk = spec
                    .per_section_max_jerk
                    .as_ref()
                    .map(|v| v[seg_idx][dof_idx])
                    .unwrap_or(spec.max_jerk[dof_idx]);
                let min_vel = if let Some(ps) = spec.per_section_min_vel.as_ref() {
                    ps[seg_idx][dof_idx]
                } else if let Some(ps) = spec.per_section_max_vel.as_ref() {
                    -ps[seg_idx][dof_idx]
                } else if let Some(mv) = spec.min_vel.as_ref() {
                    mv[dof_idx]
                } else {
                    -spec.max_vel[dof_idx]
                };
                let min_accel = if let Some(ps) = spec.per_section_min_accel.as_ref() {
                    ps[seg_idx][dof_idx]
                } else if let Some(ps) = spec.per_section_max_accel.as_ref() {
                    -ps[seg_idx][dof_idx]
                } else if let Some(mv) = spec.min_accel.as_ref() {
                    mv[dof_idx]
                } else {
                    -spec.max_accel[dof_idx]
                };
                let min_pose = if let Some(ps) = spec.per_section_min_pose.as_ref() {
                    Some(ps[seg_idx][dof_idx])
                } else {
                    spec.min_pose.as_ref().map(|v| v[dof_idx])
                };
                let max_pose = if let Some(ps) = spec.per_section_max_pose.as_ref() {
                    Some(ps[seg_idx][dof_idx])
                } else {
                    spec.max_pose.as_ref().map(|v| v[dof_idx])
                };
                self.segment_limits[seg_idx][dof_idx].base =
                    LimitsThirdPose::new(max_vel, min_vel, max_accel, min_accel, max_jerk);
                self.segment_limits[seg_idx][dof_idx].min_pose = min_pose;
                self.segment_limits[seg_idx][dof_idx].max_pose = max_pose;
            }
            // Final/target state with optional clipping.
            let mut tgt_pos = spec.goal_pose[dof_idx];
            let mut tgt_vel = spec.goal_vel[dof_idx];
            let mut tgt_accel = spec.goal_accel[dof_idx];
            if spec.goal_overflow == GoalOutOfBounds::Clip {
                let limits_back = self.segment_limits[section_cnt - 1][dof_idx];
                tgt_vel = fmin(fmax(tgt_vel, limits_back.min_vel()), limits_back.max_vel());
                tgt_accel = fmin(
                    fmax(tgt_accel, limits_back.min_accel()),
                    limits_back.max_accel(),
                );
                if let Some(mp) = limits_back.min_pose {
                    tgt_pos = fmax(tgt_pos, mp);
                }
                if let Some(mp) = limits_back.max_pose {
                    tgt_pos = fmin(tgt_pos, mp);
                }
            }
            self.waypoint_states[section_cnt][dof_idx].p = tgt_pos;
            self.waypoint_states[section_cnt][dof_idx].v = tgt_vel;
            self.waypoint_states[section_cnt][dof_idx].a = tgt_accel;
        }
        // Per-section minimum durations.
        if let Some(mins) = spec.per_section_min_duration.as_ref() {
            for seg_idx in 0..self.segments.len() {
                if seg_idx < mins.len() {
                    self.segments[seg_idx].t_min = Some(mins[seg_idx]);
                    self.segments_tmp[seg_idx].t_min = Some(mins[seg_idx]);
                }
            }
        }
    }

    /// Resize the plan's profile/duration buffers to `section_cnt` sections.
    fn resize_plan(plan: &mut Plan<N, F>, section_cnt: usize) {
        let segment_proto = [Segment::<F>::empty(); N];
        plan.profiles.resize(section_cnt, segment_proto);
        plan.intermediate_durations.resize(section_cnt, F::zero());
    }

    /// Solve the full multi-waypoint trajectory in one shot.
    pub fn solve(&mut self, spec: &MotionSpec<N, F>, plan: &mut Plan<N, F>) -> StepStatus {
        plan.waypoint_iterations = 0;

        let inter_pos_cnt = spec.waypoint_poses.len();
        // Initial section sizing (allow space; will be trimmed after collapse).
        self.resize(inter_pos_cnt);
        Self::resize_plan(plan, inter_pos_cnt + 1);

        // Compress identical waypoints.
        let section_marker = self.collect_section_positions(spec);
        let section_cnt = section_marker + 1;
        self.resize(section_marker);
        Self::resize_plan(plan, section_cnt);

        // Populate states and limits.
        self.fill_states_and_limits(spec, section_marker);

        // Seed the initial halt ramp on section 0 for every active DoF.
        for dof_idx in 0..N {
            let state = self.waypoint_states[0][dof_idx];
            let profile = &mut plan.profiles[0][dof_idx];
            if !spec.axis_active[dof_idx] {
                profile.p[7] = state.p;
                profile.v[7] = state.v;
                profile.a[7] = state.a;
                continue;
            }
            profile.halt = halt_segment::third_order_pose::get_profile(
                state.v,
                state.a,
                self.segment_limits[0][dof_idx].base,
            );
            profile
                .halt
                .finalize_second_order(state.p, state.v, state.a);
        }

        // Initial Step-A pass for every section and joint, with per-section
        // feasibility/synchronisation.
        let mut total_duration = F::zero();
        for seg_idx in 0..section_cnt {
            for dof_idx in 0..N {
                if !spec.axis_active[dof_idx] {
                    plan.profiles[seg_idx][dof_idx].duration = F::zero();
                    continue;
                }
                let state_start = self.waypoint_states[seg_idx][dof_idx];
                let state_end = self.waypoint_states[seg_idx + 1][dof_idx];
                let step1 = pose_math::third_order::StepA::new(
                    state_start,
                    state_end,
                    self.segment_limits[seg_idx][dof_idx].base,
                );
                plan.profiles[seg_idx][dof_idx].set_boundary_explicit(
                    state_start.p,
                    state_start.v,
                    state_start.a,
                    state_end.p,
                    state_end.v,
                    state_end.a,
                );
                let prof = plan.profiles[seg_idx][dof_idx];
                let ok = step1.get_profile_with_scratch(
                    &prof,
                    &mut self.segments[seg_idx].dof_blocks[dof_idx],
                    &mut self.step_a_scratch,
                );
                if !ok {
                    return StepStatus::StepOneFailed;
                }
                plan.profiles[seg_idx][dof_idx] = self.segments[seg_idx].dof_blocks[dof_idx].p_min;
            }
            self.segments[seg_idx].find_feasible_time();
            total_duration = total_duration + self.segments[seg_idx].sync_time;
        }

        self.search_state.reset(total_duration);

        // Drive the three optimisation passes in sequence — this is a single
        // shot offline solver, no interruption.
        self.run_global_optimization_pass(&mut plan.profiles);
        self.optimize_three_segment_window(&mut plan.profiles);
        self.optimize_two_segment_window(&mut plan.profiles);
        self.finalize_second_order(spec, plan, true)
    }
}

impl<const N: usize, F: KinScalar> Default for WaypointSolver<N, F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrap the third-order Step-B in a uniform success/failure call. The
/// underlying [`pose_math::third_order::StepB::get_profile`] takes `&mut self`;
/// this helper forwards through.
fn step_b_get_profile<F: KinScalar>(
    step: &mut pose_math::third_order::StepB<F>,
    profile: &mut Segment<F>,
) -> bool {
    step.get_profile(profile)
}
