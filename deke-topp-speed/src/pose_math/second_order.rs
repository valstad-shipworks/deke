//! Second-order (pose + velocity) shaping math.
//!
//! At second order each segment is composed of up to three sections:
//! an accelerating ramp, a constant-velocity cruise, and a decelerating
//! ramp. The shaping math treats the acceleration extremes as fixed
//! constants (the limits) and solves analytically for the section times.
//!
//! Two solver shapes are exposed:
//!
//! - [`StepA`]: minimum-duration profile, enumerating up to four candidate
//!   shapes (the "up" and "down" branches under the nominal and inverted
//!   limits) and folding them into a [`Feasible`] window.
//! - [`StepB`]: shapes a profile that completes in a fixed total time `tf`.

// StepA/StepB retain `pub current`, `target`, and `limits` for API parity
// with the other step classes even though the shaping math doesn't read them.
#![allow(dead_code)]

use num_traits::Float;

use crate::check::second_order_pose;
use crate::feasible::{Feasible, Span};
use crate::kin_state::{KinSecondPose, LimitsSecondPose};
use crate::segment::{Segment, SignBlock, Touched};

// ---------------------------------------------------------------------------
// StepA
// ---------------------------------------------------------------------------

/// Boundary-time step (Step-A): build the minimum-duration profile that
/// drives `current` to `target` while respecting `limits`.
pub struct StepA<F: Float> {
    /// Initial velocity (current state).
    cv: F,
    /// Final velocity (target state).
    vf: F,
    /// Kinematic limits used for shaping.
    reached_limits: LimitsSecondPose<F>,
    /// Signed pose offset from current to target.
    position_from_target: F,
    /// Public mirrors of the constructor inputs.
    pub current: KinSecondPose<F>,
    pub target: KinSecondPose<F>,
    pub limits: LimitsSecondPose<F>,
}

impl<F: Float> StepA<F> {
    /// Cache the per-step constants used by the shaping math.
    pub fn new(
        current: KinSecondPose<F>,
        target: KinSecondPose<F>,
        limits: LimitsSecondPose<F>,
    ) -> Self {
        let position_from_target = target.p - current.p;
        Self {
            cv: current.v,
            vf: target.v,
            reached_limits: limits,
            position_from_target,
            current,
            target,
            limits,
        }
    }

    /// Compute the "acc0" branch: accelerate from `cv` to the cruise
    /// velocity, cruise, then decelerate to `vf`. On success the candidate
    /// profile is committed via `advance` and the iterator is bumped to the
    /// next slot.
    fn calculate_up(
        &self,
        profiles: &mut [Segment<F>; 3],
        cursor: &mut usize,
        limits: &LimitsSecondPose<F>,
        _is_single_path: bool,
    ) {
        if *cursor >= profiles.len() {
            return;
        }
        let two = F::from(2.0).unwrap();
        let idx = *cursor;
        {
            let profile = &mut profiles[idx];
            profile.t[0] = (-self.cv + limits.max_vel) / limits.max_accel;
            profile.t[1] = (limits.min_accel * self.cv * self.cv
                - limits.max_accel * self.vf * self.vf)
                / (two * limits.max_accel * limits.min_accel * limits.max_vel)
                + limits.max_vel * (limits.max_accel - limits.min_accel)
                    / (two * limits.max_accel * limits.min_accel)
                + self.position_from_target / limits.max_vel;
            profile.t[2] = (self.vf - limits.max_vel) / limits.min_accel;
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = F::zero();
        }
        if second_order_pose::check_profile(
            &mut profiles[idx],
            SignBlock::Uddu,
            Touched::Acc0,
            limits.max_accel,
            limits.min_accel,
            limits,
        ) {
            self.advance_profile_iter(profiles, cursor);
        }
    }

    /// Compute the "none" branch: a pair of accelerate / decelerate ramps with
    /// no cruise plateau. Two roots `±disc` produce up to two candidates per
    /// call.
    fn calculate_down(
        &self,
        profiles: &mut [Segment<F>; 3],
        cursor: &mut usize,
        limits: &LimitsSecondPose<F>,
        is_single_path: bool,
    ) {
        if *cursor >= profiles.len() {
            return;
        }
        let two = F::from(2.0).unwrap();
        let mut disc = (limits.max_accel * self.vf * self.vf
            - limits.min_accel * self.cv * self.cv
            - two * limits.max_accel * limits.min_accel * self.position_from_target)
            / (limits.max_accel - limits.min_accel);
        if disc >= F::zero() {
            disc = disc.sqrt();
            {
                let idx = *cursor;
                let profile = &mut profiles[idx];
                profile.t[0] = -(self.cv + disc) / limits.max_accel;
                profile.t[1] = F::zero();
                profile.t[2] = (self.vf + disc) / limits.min_accel;
                profile.t[3] = F::zero();
                profile.t[4] = F::zero();
                profile.t[5] = F::zero();
                profile.t[6] = F::zero();
                if second_order_pose::check_profile(
                    &mut profiles[idx],
                    SignBlock::Uddu,
                    Touched::None,
                    limits.max_accel,
                    limits.min_accel,
                    limits,
                ) {
                    self.advance_profile_iter(profiles, cursor);
                    if is_single_path {
                        return;
                    }
                }
            }
            if *cursor >= profiles.len() {
                return;
            }
            {
                let idx = *cursor;
                let profile = &mut profiles[idx];
                profile.t[0] = (-self.cv + disc) / limits.max_accel;
                profile.t[1] = F::zero();
                profile.t[2] = (self.vf - disc) / limits.min_accel;
                profile.t[3] = F::zero();
                profile.t[4] = F::zero();
                profile.t[5] = F::zero();
                profile.t[6] = F::zero();
                if second_order_pose::check_profile(
                    &mut profiles[idx],
                    SignBlock::Uddu,
                    Touched::None,
                    limits.max_accel,
                    limits.min_accel,
                    limits,
                ) {
                    self.advance_profile_iter(profiles, cursor);
                }
            }
        }
    }

    /// Degenerate-limit fallback: when one of the velocity/acceleration
    /// extremes is exactly zero, the only feasible profile is a pure cruise
    /// at `cv` (provided `vf == cv`). Returns `true` when the cruise is
    /// validated.
    fn check_profile(&self, profile: &mut Segment<F>, limits: &LimitsSecondPose<F>) -> bool {
        let eps = F::epsilon();
        if (self.vf - self.cv).abs() > eps {
            return false;
        }
        profile.t[0] = F::zero();
        profile.t[1] = F::zero();
        profile.t[2] = F::zero();
        profile.t[3] = F::zero();
        profile.t[4] = F::zero();
        profile.t[5] = F::zero();
        profile.t[6] = F::zero();
        if self.cv.abs() > eps {
            profile.t[3] = self.position_from_target / self.cv;
            if second_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::None,
                F::zero(),
                limits,
            ) {
                return true;
            }
        } else if self.position_from_target.abs() < eps
            && second_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::None,
                F::zero(),
                limits,
            )
        {
            return true;
        }
        false
    }

    /// Carry the section-0 boundary state forward into the next candidate
    /// slot.
    fn advance_profile_iter(&self, profiles: &mut [Segment<F>; 3], cursor: &mut usize) {
        let prev = *cursor;
        *cursor += 1;
        if *cursor < profiles.len() {
            let prev_profile = profiles[prev];
            profiles[*cursor].set_boundary(&prev_profile);
        }
    }

    /// Enumerate candidate profiles and fold them into `block`. The shape of
    /// the search depends on whether the final velocity `vf` is zero (single
    /// path, four sequential attempts) or non-zero (parallel attempts on both
    /// directions of the limits). Returns whether any candidate is feasible.
    pub fn get_profile(&self, input_profile: &Segment<F>, block: &mut Feasible<F>) -> bool {
        let limit_is_zero = self.reached_limits.max_accel == F::zero()
            || self.reached_limits.min_accel == F::zero()
            || self.reached_limits.max_vel == F::zero()
            || self.reached_limits.min_vel == F::zero();
        if limit_is_zero {
            let profile = &mut block.p_min;
            profile.set_boundary(input_profile);
            if self.check_profile(profile, &self.reached_limits) {
                block.t_min =
                    profile.duration + profile.halt.duration + profile.accel_halt.duration;
                if self.cv.abs() > F::epsilon() {
                    block.blocked_interval_a = Some(Span::from_times(block.t_min, F::infinity()));
                }
                return true;
            }
            return false;
        }

        let mut profiles: [Segment<F>; 3] = [Segment::empty(); 3];
        profiles[0].set_boundary(input_profile);
        let mut cursor: usize = 0;
        let inv_limits = self.reached_limits.inverse();

        if self.vf.abs() < F::epsilon() {
            let limits_for_direction = if self.position_from_target >= F::zero() {
                self.reached_limits
            } else {
                inv_limits
            };
            self.calculate_down(&mut profiles, &mut cursor, &limits_for_direction, true);
            if cursor > 0 {
                return block.pick_from_candidates(&mut profiles[..], cursor);
            }
            self.calculate_up(&mut profiles, &mut cursor, &limits_for_direction, true);
            if cursor > 0 {
                return block.pick_from_candidates(&mut profiles[..], cursor);
            }
            let other_limits = limits_for_direction.inverse();
            self.calculate_down(&mut profiles, &mut cursor, &other_limits, true);
            if cursor > 0 {
                return block.pick_from_candidates(&mut profiles[..], cursor);
            }
            self.calculate_up(&mut profiles, &mut cursor, &other_limits, true);
        } else {
            self.calculate_down(&mut profiles, &mut cursor, &self.reached_limits, false);
            self.calculate_down(&mut profiles, &mut cursor, &inv_limits, false);
            self.calculate_up(&mut profiles, &mut cursor, &self.reached_limits, false);
            self.calculate_up(&mut profiles, &mut cursor, &inv_limits, false);
        }
        block.pick_from_candidates(&mut profiles[..], cursor)
    }
}

// ---------------------------------------------------------------------------
// StepB
// ---------------------------------------------------------------------------

/// Timed step (Step-B): build a profile that completes in exactly `tf`.
pub struct StepB<F: Float> {
    /// Initial velocity (current state).
    cv: F,
    /// Fixed total time the profile must span.
    tf: F,
    /// Final velocity (target state).
    vf: F,
    /// Kinematic limits used for shaping and validation.
    reached_limits: LimitsSecondPose<F>,
    /// Signed pose offset from current to target.
    position_from_target: F,
    /// Signed velocity offset from current to target.
    velocity_from_target: F,
    /// Public mirrors of the constructor inputs.
    pub current: KinSecondPose<F>,
    pub target: KinSecondPose<F>,
    pub limits: LimitsSecondPose<F>,
}

impl<F: Float> StepB<F> {
    /// Cache the per-step constants used by the shaping math.
    pub fn new(
        tf: F,
        current: KinSecondPose<F>,
        target: KinSecondPose<F>,
        limits: LimitsSecondPose<F>,
    ) -> Self {
        let position_from_target = target.p - current.p;
        let velocity_from_target = target.v - current.v;
        Self {
            cv: current.v,
            tf,
            vf: target.v,
            reached_limits: limits,
            position_from_target,
            velocity_from_target,
            current,
            target,
            limits,
        }
    }

    /// Try the three "up" branches (accelerate-cruise-decelerate, two
    /// reduced shapes). Returns `true` and finalises the segment on the
    /// first branch that validates.
    fn calculate_up(&self, profile: &mut Segment<F>, limits: &LimitsSecondPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        {
            let t_branch =
                ((two * limits.max_accel * (self.position_from_target - self.tf * self.vf)
                    - two * limits.min_accel * (self.position_from_target - self.tf * self.cv)
                    + self.velocity_from_target * self.velocity_from_target)
                    / (limits.max_accel * limits.min_accel)
                    + self.tf * self.tf)
                    .sqrt();
            profile.t[0] = (limits.max_accel * self.velocity_from_target
                - limits.max_accel * limits.min_accel * (self.tf - t_branch))
                / (limits.max_accel * (limits.max_accel - limits.min_accel));
            profile.t[1] = t_branch;
            profile.t[2] = self.tf - (profile.t[0] + t_branch);
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = F::zero();
            if second_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::Acc0,
                self.tf,
                limits,
            ) {
                profile.pf = profile.p[7];
                return true;
            }
        }
        {
            let denom = -self.velocity_from_target + limits.max_accel * self.tf;
            profile.t[0] = -self.velocity_from_target * self.velocity_from_target
                / (two * limits.max_accel * denom)
                + (self.position_from_target - self.cv * self.tf) / denom;
            profile.t[1] = -self.velocity_from_target / limits.max_accel + self.tf;
            profile.t[2] = F::zero();
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = self.tf - (profile.t[0] + profile.t[1]);
            if second_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::Acc0,
                self.tf,
                limits,
            ) {
                profile.pf = profile.p[7];
                return true;
            }
        }
        {
            profile.t[0] = F::zero();
            profile.t[1] = -self.velocity_from_target / limits.max_accel + self.tf;
            profile.t[2] = F::zero();
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = self.velocity_from_target / limits.max_accel;
            if second_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::Acc0,
                self.tf,
                limits,
            ) {
                profile.pf = profile.p[7];
                return true;
            }
        }
        false
    }

    /// Try the "down" branches. The first branch handles the fully-zero
    /// boundary case (no displacement, no velocities). The second branch
    /// shapes a single accelerate / decelerate pair without a cruise plateau
    /// and validates that the implied acceleration sits inside the envelope.
    fn calculate_down(&self, profile: &mut Segment<F>, limits: &LimitsSecondPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        let eps = F::epsilon();
        let eps_lim = F::from(1e-12).unwrap();
        if self.cv.abs() < eps && self.vf.abs() < eps && self.position_from_target.abs() < eps {
            profile.t[0] = F::zero();
            profile.t[1] = self.tf;
            profile.t[2] = F::zero();
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = F::zero();
            if second_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::None,
                self.tf,
                limits,
            ) {
                profile.pf = profile.p[7];
                return true;
            }
        }
        {
            let tmp = two * (self.vf * self.tf - self.position_from_target);
            profile.t[0] = tmp / self.velocity_from_target;
            profile.t[1] = self.tf - profile.t[0];
            profile.t[2] = F::zero();
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = F::zero();
            let a_calc = self.velocity_from_target * self.velocity_from_target / tmp;
            if (limits.min_accel - eps_lim < a_calc)
                && (a_calc < limits.max_accel + eps_lim)
                && second_order_pose::check_profile2_accels(
                    profile,
                    SignBlock::Uddu,
                    Touched::None,
                    self.tf,
                    a_calc,
                    -a_calc,
                    limits,
                )
            {
                profile.pf = profile.p[7];
                return true;
            }
        }
        false
    }

    /// Calculate under the given limits via either the up or down branches.
    fn calculate(&self, profile: &mut Segment<F>, limits: &LimitsSecondPose<F>) -> bool {
        self.calculate_up(profile, limits) || self.calculate_down(profile, limits)
    }

    /// Materialise a timed profile. When the pose is moving in the positive
    /// direction we try the nominal limits first; otherwise we start from the
    /// inverted limits.
    pub fn get_profile(&self, profile: &mut Segment<F>) -> bool {
        if self.position_from_target > F::zero() {
            return self.calculate(profile, &self.reached_limits)
                || self.calculate(profile, &self.reached_limits.inverse());
        }
        self.calculate(profile, &self.reached_limits.inverse())
            || self.calculate(profile, &self.reached_limits)
    }
}
