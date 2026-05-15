//! Second-order (velocity-only) shaping math.

// StepA/StepB retain `pub current`, `target`, and `limits` for API parity.
#![allow(dead_code)]

use num_traits::Float;

use crate::check::second_order_vel as checker;
use crate::feasible::Feasible;
use crate::kin_state::{KinSecondVel, LimitsSecondVel};
use crate::segment::{Segment, SignBlock, Touched};

/// Boundary-time step (Step-A): computes the minimum-duration profile.
pub struct StepA<F: Float> {
    pub current: KinSecondVel<F>,
    pub target: KinSecondVel<F>,
    pub limits: LimitsSecondVel<F>,
    velocity_from_target: F,
    reached_limits: LimitsSecondVel<F>,
}

impl<F: Float> StepA<F> {
    pub fn new(
        current: KinSecondVel<F>,
        target: KinSecondVel<F>,
        limits: LimitsSecondVel<F>,
    ) -> Self {
        let velocity_from_target = target.v - current.v;
        Self {
            current,
            target,
            limits,
            velocity_from_target,
            reached_limits: limits,
        }
    }

    pub fn get_profile(&self, input_profile: &Segment<F>, block: &mut Feasible<F>) -> bool {
        let profile = &mut block.p_min;
        profile.set_boundary(input_profile);
        let a_lim = if self.velocity_from_target > F::zero() {
            self.reached_limits.max_accel
        } else {
            self.reached_limits.min_accel
        };
        profile.t[0] = F::zero();
        profile.t[1] = self.velocity_from_target / a_lim;
        profile.t[2] = F::zero();
        profile.t[3] = F::zero();
        profile.t[4] = F::zero();
        profile.t[5] = F::zero();
        profile.t[6] = F::zero();
        if checker::check_profile(profile, SignBlock::Uddu, Touched::Acc0, a_lim) {
            block.t_min = profile.duration + profile.halt.duration + profile.accel_halt.duration;
            return true;
        }
        false
    }
}

/// Timed step (Step-B): assembles a profile that completes within target duration `tf`.
pub struct StepB<F: Float> {
    pub tf: F,
    pub current: KinSecondVel<F>,
    pub target: KinSecondVel<F>,
    pub limits: LimitsSecondVel<F>,
    velocity_from_target: F,
    reached_limits: LimitsSecondVel<F>,
}

impl<F: Float> StepB<F> {
    pub fn new(
        tf: F,
        current: KinSecondVel<F>,
        target: KinSecondVel<F>,
        limits: LimitsSecondVel<F>,
    ) -> Self {
        let velocity_from_target = target.v - current.v;
        Self {
            tf,
            current,
            target,
            limits,
            velocity_from_target,
            reached_limits: limits,
        }
    }

    pub fn get_profile(&self, profile: &mut Segment<F>) -> bool {
        let a_avg = self.velocity_from_target / self.tf;
        profile.t[0] = F::zero();
        profile.t[1] = self.tf;
        profile.t[2] = F::zero();
        profile.t[3] = F::zero();
        profile.t[4] = F::zero();
        profile.t[5] = F::zero();
        profile.t[6] = F::zero();
        if checker::check_profile2(
            profile,
            SignBlock::Uddu,
            Touched::None,
            self.tf,
            a_avg,
            &self.reached_limits,
        ) {
            profile.pf = profile.p[7];
            return true;
        }
        false
    }
}
