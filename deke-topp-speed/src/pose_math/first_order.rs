//! First-order (pose-only) shaping math.
//!
//! At first order each segment is a constant-velocity cruise: the trajectory
//! drives from the current pose to the target pose along the single
//! interior cruise section, with no acceleration or jerk shaping. The
//! reachable cruise velocity is bracketed by the per-axis `[min_vel, max_vel]`
//! envelope.

// StepA/StepB intentionally retain `pub current`, `target`, and `limits`
// fields for API parity with higher-order step classes even though the
// shaping math here doesn't read them back out.
#![allow(dead_code)]

use num_traits::Float;

use crate::check::first_order_pose;
use crate::feasible::Feasible;
use crate::kin_state::{KinFirstPose, LimitsFirstPose};
use crate::segment::{Segment, SignBlock, Touched};

/// Boundary-time step (Step-A): computes the minimum-duration cruise profile
/// that connects `current` to `target` while respecting the velocity envelope.
pub struct StepA<F: Float> {
    /// Velocity envelope this step shapes within. The min/max bounds choose
    /// the cruise velocity according to the sign of `position_from_target`.
    reached_limits: LimitsFirstPose<F>,
    /// Signed pose offset from the current pose to the target pose.
    position_from_target: F,
    /// Original current state (retained so the public surface keeps the same
    /// shape as the higher-order step classes).
    pub current: KinFirstPose<F>,
    /// Original target state.
    pub target: KinFirstPose<F>,
    /// Original limits (mirror of `reached_limits`).
    pub limits: LimitsFirstPose<F>,
}

impl<F: Float> StepA<F> {
    /// Capture the pose offset and stash the velocity envelope. No shaping
    /// happens here; the time profile is materialised by [`Self::get_profile`].
    pub fn new(
        current: KinFirstPose<F>,
        target: KinFirstPose<F>,
        limits: LimitsFirstPose<F>,
    ) -> Self {
        let position_from_target = target.p - current.p;
        Self {
            reached_limits: limits,
            position_from_target,
            current,
            target,
            limits,
        }
    }

    /// Assemble the minimum-duration cruise profile and copy it into the
    /// `p_min` slot of `block`. The cruise direction follows the sign of the
    /// pose offset: positive offsets cruise at `max_vel`, negative at
    /// `min_vel`. Returns `true` when the resulting profile passes
    /// validation.
    pub fn get_profile(&self, input_profile: &Segment<F>, block: &mut Feasible<F>) -> bool {
        let profile = &mut block.p_min;
        profile.set_boundary(input_profile);
        let v_lim = if self.position_from_target > F::zero() {
            self.reached_limits.max_vel
        } else {
            self.reached_limits.min_vel
        };
        profile.t[0] = F::zero();
        profile.t[1] = F::zero();
        profile.t[2] = F::zero();
        profile.t[3] = self.position_from_target / v_lim;
        profile.t[4] = F::zero();
        profile.t[5] = F::zero();
        profile.t[6] = F::zero();
        if first_order_pose::check_profile(profile, SignBlock::Uddu, Touched::Vel, v_lim) {
            block.t_min = profile.duration + profile.halt.duration + profile.accel_halt.duration;
            return true;
        }
        false
    }
}

/// Timed step (Step-B): assembles a cruise profile that completes within a
/// fixed target duration `tf`. The implied cruise velocity is `dp / tf` and
/// is checked against the envelope rather than chosen from it.
pub struct StepB<F: Float> {
    /// Fixed total time the profile must span.
    tf: F,
    /// Velocity envelope used to validate the implied cruise velocity.
    reached_limits: LimitsFirstPose<F>,
    /// Signed pose offset from the current pose to the target pose.
    position_from_target: F,
    /// Public mirrors of the constructor inputs.
    pub current: KinFirstPose<F>,
    pub target: KinFirstPose<F>,
    pub limits: LimitsFirstPose<F>,
}

impl<F: Float> StepB<F> {
    /// Capture the target time and pose offset.
    pub fn new(
        tf: F,
        current: KinFirstPose<F>,
        target: KinFirstPose<F>,
        limits: LimitsFirstPose<F>,
    ) -> Self {
        let position_from_target = target.p - current.p;
        Self {
            tf,
            reached_limits: limits,
            position_from_target,
            current,
            target,
            limits,
        }
    }

    /// Materialise the timed cruise into `profile` and validate that the
    /// implied cruise velocity sits within the envelope.
    pub fn get_profile(&self, profile: &mut Segment<F>) -> bool {
        let v_avg = self.position_from_target / self.tf;
        profile.t[0] = F::zero();
        profile.t[1] = F::zero();
        profile.t[2] = F::zero();
        profile.t[3] = self.tf;
        profile.t[4] = F::zero();
        profile.t[5] = F::zero();
        profile.t[6] = F::zero();
        first_order_pose::check_profile2(
            profile,
            SignBlock::Uddu,
            Touched::None,
            self.tf,
            v_avg,
            &self.reached_limits,
        )
    }
}
