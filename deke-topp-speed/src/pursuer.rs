//! Real-time live tracker that adapts the goal each tick to follow a moving
//! [`PursuitTarget`].
//!
//! The [`Pursuer`] mirrors the offline solver's plumbing but produces one
//! [`MotionSample`] per call to [`Pursuer::tick`], driven by a (possibly
//! changing) goal state supplied at every cycle. Two follow strategies are
//! exposed:
//!
//! - [`FollowMode::Tuned`]: iteratively re-tunes a lookahead horizon so the
//!   synthesised trajectory's duration matches the requested horizon. This
//!   minimises overshoot and matches the slow-and-careful trade-off.
//! - [`FollowMode::Quick`]: a closed-form per-axis correction. Cheap on CPU
//!   but treats axes independently and can lag during high-bandwidth chases.

use std::time::Duration;

use deke_types::{KinScalar, SRobotQ};

use crate::kin_state::KinThirdPose;
use crate::modes::{Coordination, DurationGrid, FollowMode};
use crate::plan::Plan;
use crate::sample::{MotionSample, PursuitTarget};
use crate::solver::Solver;
use crate::spec::MotionSpec;
use crate::status::StepStatus;

/// Optional user-supplied target-state predictor.
///
/// Given a lookahead `dt` (seconds) and a reference target state, returns the
/// predicted state at `dt` in the future. The boolean out-parameter signals
/// "this prediction was interrupted" (e.g. the offline targets list ran out);
/// when set, the pursuer treats the target as at-rest for the duration of the
/// current tick.
///
/// A plain function pointer keeps the [`Pursuer`] auto-`Send + Sync`.
pub type PredictionModel<const N: usize, F> =
    fn(F, &PursuitTarget<N, F>, &mut bool) -> PursuitTarget<N, F>;

/// Real-time follower.
///
/// The follower keeps its own scratch trajectory and the time position at
/// which it is currently sampling that trajectory; mutating [`Pursuer::tick`]
/// advances both. Call [`Pursuer::reset`] to force a fresh solve on the next
/// tick.
#[derive(Debug)]
pub struct Pursuer<const N: usize, F: KinScalar = f32> {
    /// Control-cycle interval. Stored as [`Duration`] for ergonomic use; the
    /// math internally converts to `F`.
    pub dt: Duration,
    /// Follow strategy.
    pub mode: FollowMode,
    /// Pursuit gain in `[0, 1]`. Lower values produce smoother but laggier
    /// tracking; higher values chase harder.
    pub reactiveness: F,
    /// Number of cycles ahead of the current state to plan toward in
    /// [`FollowMode::Tuned`]. One is the minimum useful value.
    pub look_ahead_cycles: usize,
    /// Iteration cap for the lookahead-time tuning loop.
    pub max_iterations: usize,
    /// Optional user-supplied target-state predictor. See [`PredictionModel`].
    pub prediction_model: Option<PredictionModel<N, F>>,

    last_iterations_counter: usize,
    last_status: StepStatus,

    /// Last solve's input snapshot. When the next tick's input matches this,
    /// the cached trajectory can be re-sampled instead of re-solved.
    inp2: MotionSpec<N, F>,
    /// Last produced output. Holds the most recent motion sample plus the
    /// time at which it was sampled from `trajectory`.
    out: MotionSample<N, F>,
    /// Current sampling time along [`Self::trajectory`].
    out_time: F,
    /// Cached output trajectory: re-sampled until input changes.
    trajectory: Plan<N, F>,
    /// Scratch trajectory used during the optimise-duration loop.
    traj: Plan<N, F>,
    /// Predicted target state (filled by [`Self::predict_target_state`]).
    predicted_target_state: PursuitTarget<N, F>,
    /// After-synchronisation target (filled by [`Self::step_optimized_mode`]).
    synchronized_target_state: PursuitTarget<N, F>,
    /// Whether the cached `trajectory` reflects the latest input.
    started: bool,
    /// Most recent step result.
    res: StepStatus,

    solver: Solver<N, F>,
}

impl<const N: usize, F: KinScalar> Pursuer<N, F> {
    /// Create a new pursuer with default tracking gains. The default
    /// [`FollowMode`] is [`FollowMode::Tuned`].
    pub fn new(dt: Duration) -> Self {
        Self {
            dt,
            mode: FollowMode::Tuned,
            reactiveness: F::one(),
            look_ahead_cycles: 1,
            max_iterations: 64,
            prediction_model: None,
            last_iterations_counter: 0,
            last_status: StepStatus::InProgress,
            inp2: MotionSpec::new(),
            out: MotionSample::zero(),
            out_time: F::zero(),
            trajectory: Plan::empty(),
            traj: Plan::empty(),
            predicted_target_state: PursuitTarget::zero(),
            synchronized_target_state: PursuitTarget::zero(),
            started: false,
            res: StepStatus::InProgress,
            solver: Solver::new(),
        }
    }

    /// Forget any cached internal state. The next tick will start a fresh
    /// solve.
    pub fn reset(&mut self) {
        self.started = false;
        self.last_iterations_counter = 0;
        self.last_status = StepStatus::InProgress;
        self.res = StepStatus::InProgress;
        self.out_time = F::zero();
    }

    pub fn mode(&self) -> FollowMode {
        self.mode
    }
    pub fn set_mode(&mut self, m: FollowMode) {
        self.mode = m;
    }
    pub fn reactiveness(&self) -> F {
        self.reactiveness
    }
    pub fn set_reactiveness(&mut self, r: F) {
        self.reactiveness = r;
    }
    pub fn look_ahead_cycles(&self) -> usize {
        self.look_ahead_cycles
    }
    pub fn set_look_ahead_cycles(&mut self, n: usize) {
        self.look_ahead_cycles = n;
    }
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }
    pub fn set_max_iterations(&mut self, n: usize) {
        self.max_iterations = n;
    }
    /// Iteration count consumed by the most recent [`FollowMode::Tuned`] tick.
    pub fn last_iteration_count(&self) -> usize {
        self.last_iterations_counter
    }

    /// Advance the pursuer by one control cycle.
    ///
    /// Reads `target`, mutates `spec.current_*` to reflect the post-cycle
    /// state, and returns `(status, motion_sample)` for the produced sample.
    pub fn tick(
        &mut self,
        target: &PursuitTarget<N, F>,
        spec: &mut MotionSpec<N, F>,
    ) -> (StepStatus, MotionSample<N, F>) {
        // Only pay for `Instant::now`/`elapsed` when the caller actually
        // wired up an inner step (which is the only path that mutates
        // `out.solve_micros` away from the cached value). For the common
        // steady-state pursuit tick this skips the ~25 ns syscall.
        let mut out = self.out;

        let t_start = std::time::Instant::now();
        self.res = match self.mode {
            FollowMode::Tuned => self.step_optimized_mode(target, spec, &mut out),
            FollowMode::Quick => self.step_fast(target, spec, &mut out),
        };
        out.solve_micros = t_start.elapsed().as_secs_f64() * 1e6;

        // Echo the sample back into the spec's current state so the next
        // tick starts where this one ended.
        spec.current_pose = out.pose;
        spec.current_vel = out.vel;
        spec.current_accel = out.accel;

        out.fresh_solve = !self.started || out.fresh_solve;
        self.out = out;
        self.last_status = self.res;
        self.started = true;
        (self.res, out)
    }

    /// `delta_time` as the math scalar.
    #[inline]
    fn dt_f(&self) -> F {
        F::from(self.dt.as_secs_f64()).unwrap_or_else(F::zero)
    }

    /// `true` when both velocity and acceleration are within an epsilon of
    /// zero across every joint. With a user-supplied `prediction_model`, the
    /// predictor's interrupted-flag substitutes for the velocity/accel check.
    #[inline]
    fn is_at_rest(&self, target: &PursuitTarget<N, F>) -> bool {
        if let Some(predictor) = self.prediction_model {
            let mut interrupted = false;
            let _ = predictor(F::zero(), target, &mut interrupted);
            return interrupted;
        }
        let eps = F::from(1e-12).unwrap_or_else(F::zero);
        for i in 0..N {
            if target.vel[i].abs() > eps || target.accel[i].abs() > eps {
                return false;
            }
        }
        true
    }

    /// Populate [`Self::predicted_target_state`] with `target` integrated
    /// forward by `dt` seconds. If a `prediction_model` is set it is invoked
    /// instead; otherwise each axis is integrated under zero jerk.
    fn predict_target_state(&mut self, target: &PursuitTarget<N, F>, dt: F) {
        if let Some(predictor) = self.prediction_model {
            let mut interrupted = false;
            self.predicted_target_state = predictor(dt, target, &mut interrupted);
            return;
        }
        let mut pose = SRobotQ::<N, F>::zeros();
        let mut vel = SRobotQ::<N, F>::zeros();
        let mut accel = SRobotQ::<N, F>::zeros();
        for i in 0..N {
            let state = KinThirdPose::new(target.pose[i], target.vel[i], target.accel[i]);
            let next = state.next(dt, F::zero());
            pose[i] = next.p;
            vel[i] = next.v;
            accel[i] = next.a;
        }
        self.predicted_target_state = PursuitTarget::new(pose, vel, accel);
    }

    /// Predicted velocity at the moment a constant-jerk ramp brings the
    /// acceleration to zero.
    #[inline]
    fn v_at_zero_a(v: F, a: F, j: F) -> F {
        v + (a * a) / (F::from(2.0).unwrap() * j)
    }

    #[inline]
    fn square(x: F) -> F {
        x * x
    }

    #[inline]
    fn cube(x: F) -> F {
        x * x * x
    }

    #[inline]
    fn sign(x: F) -> F {
        if x > F::zero() {
            F::one()
        } else if x < F::zero() {
            -F::one()
        } else {
            F::zero()
        }
    }

    /// Copy a target state into the spec's goal fields, clamping into the
    /// kinematic envelope so the offline solve never sees an infeasible goal.
    #[inline]
    fn apply_target_state_to_input(spec: &mut MotionSpec<N, F>, target: &PursuitTarget<N, F>) {
        let eps = F::from(1e-14).unwrap_or_else(F::zero);
        let two = F::from(2.0).unwrap();
        for i in 0..N {
            if !spec.axis_active[i] {
                continue;
            }
            spec.goal_pose[i] = target.pose[i];

            // Velocity clamp into [min_vel, max_vel] minus a sliver, where
            // min_vel defaults to -max_vel when unset.
            let min_vel = match spec.min_vel {
                Some(v) => v[i],
                None => -spec.max_vel[i],
            };
            let max_vel = spec.max_vel[i];
            let mut v = target.vel[i];
            if v < min_vel + eps {
                v = min_vel + eps;
            }
            if v > max_vel - eps {
                v = max_vel - eps;
            }
            spec.goal_vel[i] = v;

            // Acceleration clamp into [min_acc, max_acc] minus a sliver.
            let min_acc = match spec.min_accel {
                Some(a) => a[i],
                None => -spec.max_accel[i],
            };
            let max_acc = spec.max_accel[i];
            let mut a = target.accel[i];
            if a < min_acc + eps {
                a = min_acc + eps;
            }
            if a > max_acc - eps {
                a = max_acc - eps;
            }
            spec.goal_accel[i] = a;

            // If the requested acceleration would, when ramped to zero under
            // max jerk, push the velocity past the velocity envelope, clip
            // the acceleration so the at-zero-a velocity sits on the bound.
            let acc = spec.goal_accel[i];
            let max_jerk = spec.max_jerk[i];
            let vel = spec.goal_vel[i];
            if acc > F::zero() && Self::v_at_zero_a(vel, acc, -max_jerk) < -max_vel {
                let rad = -two * max_jerk * (-max_vel - vel);
                if rad >= F::zero() {
                    spec.goal_accel[i] = rad.sqrt() - eps;
                }
            }
            if acc < F::zero() && Self::v_at_zero_a(vel, acc, max_jerk) > max_vel {
                let rad = two * max_jerk * (max_vel - vel);
                if rad >= F::zero() {
                    spec.goal_accel[i] = -(rad.sqrt()) + eps;
                }
            }

            // Position envelope: if the target position pierces the ceiling
            // (or floor) and the velocity points further into the violation,
            // clamp the entire goal to the bound at rest.
            if let Some(max_pose) = spec.max_pose {
                let limit = max_pose[i];
                if spec.goal_pose[i] > limit + eps && (spec.goal_vel[i] >= F::zero() || N > 1) {
                    spec.goal_pose[i] = limit;
                    spec.goal_vel[i] = F::zero();
                    spec.goal_accel[i] = F::zero();
                    continue;
                }
            }
            if let Some(min_pose) = spec.min_pose {
                let limit = min_pose[i];
                if spec.goal_pose[i] < limit - eps && (spec.goal_vel[i] <= F::zero() || N > 1) {
                    spec.goal_pose[i] = limit;
                    spec.goal_vel[i] = F::zero();
                    spec.goal_accel[i] = F::zero();
                }
            }
        }
    }

    /// Iteratively tune the lookahead horizon `dt` so that the resulting
    /// trajectory duration matches the requested horizon. Returns the final
    /// (reactiveness-scaled) prediction interval.
    fn optimize_duration(
        &mut self,
        target: &PursuitTarget<N, F>,
        spec: &mut MotionSpec<N, F>,
    ) -> F {
        let dt_f = self.dt_f();
        let eps = F::from(1e-10).unwrap_or_else(F::zero);
        let two = F::from(2.0).unwrap();
        let half = F::from(0.5).unwrap();
        let look_ahead_f = F::from(self.look_ahead_cycles as f64).unwrap_or_else(F::one);

        let mut dt_curr = look_ahead_f * dt_f;
        let mut dt_low = dt_curr;
        let mut dt_best = dt_curr;
        let mut best_diff = F::infinity();
        let mut recover = false;
        let mut converged = false;
        let mut iter = 0usize;

        while iter < self.max_iterations {
            iter += 1;

            let dt_used = self.reactiveness * (dt_curr - dt_f);
            self.predict_target_state(target, dt_used);
            let predicted = self.predicted_target_state;
            Self::apply_target_state_to_input(spec, &predicted);

            let probe_status = self.solver.solve(spec, &mut self.traj);

            // Map StepStatus back into the C++ flow:
            // - DurationInfeasible  -> tighten and probe halfway down.
            // - PoseOverrun         -> tolerate (matches ErrorPositionalLimits).
            // - any other failure   -> widen and retry.
            if probe_status == StepStatus::DurationInfeasible {
                dt_low = dt_best * half;
                dt_curr = dt_best;
                recover = true;
                continue;
            } else if probe_status == StepStatus::PoseOverrun {
                // tolerated; fall through.
            } else if probe_status.is_err() {
                dt_curr = dt_curr + dt_f;
                continue;
            }

            let traj_duration = self.traj.duration();
            let diff = (dt_curr - traj_duration).abs();
            if dt_curr < traj_duration + dt_f
                && (diff < best_diff || (diff == best_diff && dt_curr < dt_best))
            {
                dt_best = dt_curr;
                best_diff = diff;
                if diff < eps {
                    if dt_curr <= dt_f {
                        break;
                    }
                    dt_curr = dt_curr - dt_f;
                    converged = true;
                    continue;
                }
            }

            let mut dt_next;
            if recover {
                dt_next = (dt_curr + dt_low) * half;
                dt_low = dt_best;
            } else if dt_curr < dt_low && dt_curr < traj_duration {
                dt_next = (dt_curr + dt_low) * half;
                if converged || ((dt_curr - dt_next).abs() < eps && (dt_curr - dt_low).abs() < eps)
                {
                    break;
                }
            } else {
                dt_next = (dt_curr + traj_duration) * half;
                if (dt_curr - dt_next).abs() < eps && (dt_curr - dt_low).abs() < eps {
                    break;
                }
                dt_low = dt_curr;
            }

            // Clamp the step to a (1/2, 2) growth band around the current dt.
            let lower_band = dt_curr * half;
            let upper_band = dt_curr * two;
            if dt_next < lower_band {
                dt_next = lower_band;
            } else if dt_next > upper_band {
                dt_next = upper_band;
            }
            dt_curr = dt_next;
        }

        self.last_iterations_counter = iter;
        self.reactiveness * (dt_best - dt_f)
    }

    /// Solve (or re-sample) the cached trajectory under [`Self::dt`], handle
    /// position-envelope violations, and emit the next sample.
    fn update_optimized(
        &mut self,
        spec: &mut MotionSpec<N, F>,
        out: &mut MotionSample<N, F>,
    ) -> StepStatus {
        let eps = F::from(1e-14).unwrap_or_else(F::zero);
        let dt_f = self.dt_f();
        let mut res = StepStatus::InProgress;

        let input_changed = !self.started || !specs_match(&self.inp2, spec);
        if input_changed {
            res = self.solver.solve(spec, &mut self.trajectory);

            // If the produced trajectory would breach a position envelope,
            // clamp the goal to the bound at rest and re-solve once.
            if spec.max_pose.is_some() || spec.min_pose.is_some() {
                let extrema = self.trajectory.position_extrema();
                let mut clamped = false;
                if let Some(ref max_pose) = spec.max_pose {
                    for i in 0..N {
                        if extrema[i].max > max_pose[i] + eps && extrema[i].t_max > F::zero() {
                            spec.goal_pose[i] = max_pose[i];
                            spec.goal_vel[i] = F::zero();
                            spec.goal_accel[i] = F::zero();
                            clamped = true;
                        }
                    }
                }
                if let Some(ref min_pose) = spec.min_pose {
                    for i in 0..N {
                        if extrema[i].min < min_pose[i] - eps && extrema[i].t_min > F::zero() {
                            spec.goal_pose[i] = min_pose[i];
                            spec.goal_vel[i] = F::zero();
                            spec.goal_accel[i] = F::zero();
                            clamped = true;
                        }
                    }
                }
                if clamped {
                    res = self.solver.solve(spec, &mut self.trajectory);
                }
            }

            self.inp2 = spec.clone();
            self.out_time = F::zero();
            out.fresh_solve = true;
        } else {
            out.fresh_solve = false;
        }

        // Advance and sample.
        self.out_time = self.out_time + dt_f;
        let (pose, vel, accel, jerk, section_idx) = self.trajectory.sample_at(self.out_time);
        let prev_section = out.section_idx;
        out.pose = pose;
        out.vel = vel;
        out.accel = accel;
        out.jerk = jerk;
        out.t = self.out_time;
        out.crossed_section = section_idx > prev_section;
        out.section_idx = section_idx;

        // Pass new state back into the cached input so the next tick starts
        // from the freshly-emitted sample.
        self.inp2.current_pose = pose;
        self.inp2.current_vel = vel;
        self.inp2.current_accel = accel;

        if self.out_time > self.trajectory.duration() {
            return StepStatus::Done;
        }
        res
    }

    /// [`FollowMode::Tuned`] dispatch.
    ///
    /// Mirrors the compiled-upstream `Trackig::step_optimized_mode`: every
    /// tick the spec is forced into `(min_duration=dt, Quantized,
    /// coordination=TimeLockedSoft)` before either running the lookahead-time
    /// tuner (`optimize_duration`) or — when the target is at rest — taking a
    /// fast path that skips the tuner but keeps the same sync/duration
    /// overrides so the produced trajectory matches upstream bit-for-bit.
    fn step_optimized_mode(
        &mut self,
        target: &PursuitTarget<N, F>,
        spec: &mut MotionSpec<N, F>,
        out: &mut MotionSample<N, F>,
    ) -> StepStatus {
        let dt_f = self.dt_f();
        spec.min_duration = Some(dt_f);
        spec.duration_grid = DurationGrid::Quantized;
        let do_global = spec.coordination != Coordination::Independent;

        if self.is_at_rest(target) {
            // For an at-rest target the lookahead-time tuner converges in
            // one step (dt_used = 0 → predicted = current target), so we
            // skip the iterative loop but keep the same flags as the
            // non-rest path so `update_optimized` sees an equivalent spec.
            if do_global {
                spec.coordination = Coordination::TimeLockedSoft;
            }
            self.last_iterations_counter = 1;
            Self::apply_target_state_to_input(spec, target);
            return self.update_optimized(spec, out);
        }

        if do_global {
            spec.coordination = Coordination::TimeLockedSoft;
            let optimised_dt = self.optimize_duration(target, spec);
            self.predict_target_state(target, optimised_dt);
            self.synchronized_target_state = self.predicted_target_state;
        } else {
            // Independent: optimise each axis in isolation, then collect the
            // per-axis predicted states into the synchronised target.
            spec.coordination = Coordination::TimeLockedSoft;
            let saved_active = spec.axis_active;
            let mut synced = self.synchronized_target_state;
            for outer in 0..N {
                let mut mask = [false; N];
                mask[outer] = true;
                spec.axis_active = mask;
                let per_axis_dt = self.optimize_duration(target, spec);
                self.predict_target_state(target, per_axis_dt);
                synced.pose[outer] = self.predicted_target_state.pose[outer];
                synced.vel[outer] = self.predicted_target_state.vel[outer];
                synced.accel[outer] = self.predicted_target_state.accel[outer];
            }
            spec.axis_active = saved_active;
            spec.coordination = Coordination::Independent;
            self.synchronized_target_state = synced;
        }

        Self::apply_target_state_to_input(spec, &self.synchronized_target_state);
        // Note: spec.coordination is intentionally left at TimeLockedSoft
        // (or Independent in the rare `do_global == false` path) so that
        // subsequent ticks see the same coordination upstream `Trackig`
        // does — restoring it would cause `update_optimized`'s cached
        // `inp2 != spec` check to mis-fire and force spurious re-solves.
        self.update_optimized(spec, out)
    }

    /// [`FollowMode::Quick`] dispatch. Closed-form per-axis correction.
    fn step_fast(
        &mut self,
        target: &PursuitTarget<N, F>,
        spec: &mut MotionSpec<N, F>,
        out: &mut MotionSample<N, F>,
    ) -> StepStatus {
        let dt_f = self.dt_f();
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let eight = F::from(8.0).unwrap();
        let six_inv = F::one() / F::from(6.0).unwrap();

        let mut pose_out = SRobotQ::<N, F>::zeros();
        let mut vel_out = SRobotQ::<N, F>::zeros();
        let mut accel_out = SRobotQ::<N, F>::zeros();
        let mut jerk_out = SRobotQ::<N, F>::zeros();

        for i in 0..N {
            if !spec.axis_active[i] {
                pose_out[i] = spec.current_pose[i];
                vel_out[i] = spec.current_vel[i];
                accel_out[i] = spec.current_accel[i];
                continue;
            }

            let target_p = target.pose[i];
            let target_v = target.vel[i];
            let target_a = target.accel[i];
            let curr_p = spec.current_pose[i];
            let curr_v = spec.current_vel[i];
            let curr_a = spec.current_accel[i];
            let max_jerk = spec.max_jerk[i];
            let max_v = spec.max_vel[i];
            let max_a = spec.max_accel[i];
            let min_v = match spec.min_vel {
                Some(v) => v[i],
                None => -max_v,
            };
            let min_a = match spec.min_accel {
                Some(a) => a[i],
                None => -max_a,
            };

            let inv_jerk = F::one() / max_jerk;
            let dp = (curr_p - target_p) * inv_jerk;
            let dv = (curr_v - target_v) * inv_jerk;
            let da = (curr_a - target_a) * inv_jerk;
            let dvmin = (min_v - target_v) * inv_jerk;
            let dvmax = (max_v - target_v) * inv_jerk;
            let damin = (min_a - target_a) * inv_jerk;
            let damax = (max_a - target_a) * inv_jerk;

            let pred_dv = dv + (da * da.abs()) * F::from(0.5).unwrap();
            let sign_pred = Self::sign(pred_dv);

            // Closed-form position residual.
            let da_sq = Self::square(da);
            let dp_unclamped =
                if da <= damax && dv <= da_sq * F::from(0.5).unwrap() - Self::square(damax) {
                    dp - (damax * (da_sq - two * dv)) / four
                        - Self::square(da_sq - two * dv) / (eight * damax)
                        - da * (dv - da_sq / three)
                } else if da >= damin && dv >= Self::square(damin) - da_sq * F::from(0.5).unwrap() {
                    dp - (damin * (da_sq + two * dv)) / four
                        - Self::square(da_sq + two * dv) / (eight * damin)
                        + da * (dv + da_sq / three)
                } else {
                    dp + dv * da * sign_pred
                        - Self::cube(da) * six_inv * (F::one() - three * sign_pred.abs())
                        + sign_pred / four * (two * Self::cube(da_sq + two * dv * sign_pred)).sqrt()
                };

            // Position-trigger jerk, with fall-back to lower-order triggers
            // when the position term is at rest.
            let sign_dp = Self::sign(dp_unclamped);
            let fall_back =
                (F::one() - sign_dp.abs()) * (pred_dv + (F::one() - sign_pred.abs()) * da);
            let jerk_unclamped = -max_jerk * Self::sign(dp_unclamped + fall_back);

            // Per-bound jerk magnitudes that hold each constraint at the
            // edge of feasibility.
            let jerk_at_amin = -max_jerk * Self::sign(da - damin);
            let jerk_at_amax = -max_jerk * Self::sign(da - damax);

            let pred_dvmin = da * da.abs() + two * (dv - dvmin);
            let pred_dvmax = da * da.abs() + two * (dv - dvmax);
            let jerk_at_vmin =
                -max_jerk * Self::sign(pred_dvmin + (F::one() - Self::sign(pred_dvmin).abs()) * da);
            let jerk_at_vmax =
                -max_jerk * Self::sign(pred_dvmax + (F::one() - Self::sign(pred_dvmax).abs()) * da);

            let jerk_v_low = fmax(jerk_at_amin, fmin(jerk_at_vmin, jerk_at_amax));
            let jerk_v_high = fmax(jerk_at_amin, fmin(jerk_at_vmax, jerk_at_amax));
            let jerk_final = fmax(jerk_v_low, fmin(jerk_unclamped, jerk_v_high));

            // Integrate one cycle forward at the selected jerk.
            let state = KinThirdPose::new(curr_p, curr_v, curr_a);
            let next = state.next(dt_f, jerk_final);
            pose_out[i] = next.p;
            vel_out[i] = next.v;
            accel_out[i] = next.a;
            jerk_out[i] = jerk_final;
        }

        out.pose = pose_out;
        out.vel = vel_out;
        out.accel = accel_out;
        out.jerk = jerk_out;
        out.t = self.out_time + dt_f;
        out.section_idx = 0;
        out.crossed_section = false;
        out.fresh_solve = !self.started;
        self.out_time = out.t;
        StepStatus::InProgress
    }
}

impl<const N: usize, F: KinScalar> Clone for Pursuer<N, F> {
    fn clone(&self) -> Self {
        Self {
            dt: self.dt,
            mode: self.mode,
            reactiveness: self.reactiveness,
            look_ahead_cycles: self.look_ahead_cycles,
            max_iterations: self.max_iterations,
            prediction_model: self.prediction_model,
            last_iterations_counter: self.last_iterations_counter,
            last_status: self.last_status,
            inp2: self.inp2.clone(),
            out: self.out,
            out_time: self.out_time,
            trajectory: self.trajectory.clone(),
            traj: self.traj.clone(),
            predicted_target_state: self.predicted_target_state,
            synchronized_target_state: self.synchronized_target_state,
            started: self.started,
            res: self.res,
            solver: Solver::new(),
        }
    }
}

// Generic min/max for `F: KinScalar` (which doesn't bring `Ord`).
#[inline]
fn fmin<F: KinScalar>(a: F, b: F) -> F {
    if a < b { a } else { b }
}

#[inline]
fn fmax<F: KinScalar>(a: F, b: F) -> F {
    if a > b { a } else { b }
}

/// Coarse equality for `MotionSpec`: returns `true` when the two specs share
/// goal/current/kinematic fields within machine epsilon. Used by
/// [`Pursuer::update_optimized`] to decide whether to re-solve or just
/// resample the cached trajectory.
#[inline]
fn specs_match<const N: usize, F: KinScalar>(a: &MotionSpec<N, F>, b: &MotionSpec<N, F>) -> bool {
    let eps = F::from(1e-12).unwrap_or_else(F::zero);
    fn q_close<const N: usize, F: KinScalar>(a: &SRobotQ<N, F>, b: &SRobotQ<N, F>, eps: F) -> bool {
        for i in 0..N {
            if (a[i] - b[i]).abs() > eps {
                return false;
            }
        }
        true
    }
    q_close::<N, F>(&a.goal_pose, &b.goal_pose, eps)
        && q_close::<N, F>(&a.goal_vel, &b.goal_vel, eps)
        && q_close::<N, F>(&a.goal_accel, &b.goal_accel, eps)
        && q_close::<N, F>(&a.current_pose, &b.current_pose, eps)
        && q_close::<N, F>(&a.current_vel, &b.current_vel, eps)
        && q_close::<N, F>(&a.current_accel, &b.current_accel, eps)
        && q_close::<N, F>(&a.max_vel, &b.max_vel, eps)
        && q_close::<N, F>(&a.max_accel, &b.max_accel, eps)
        && q_close::<N, F>(&a.max_jerk, &b.max_jerk, eps)
        && a.axis_active == b.axis_active
        && a.coordination == b.coordination
        && a.duration_grid == b.duration_grid
        && a.control_mode == b.control_mode
}
