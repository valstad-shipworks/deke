//! Third-order (velocity + acceleration) shaping math.

// StepA/StepB retain `pub current`, `target`, and `limits` for API parity.
#![allow(dead_code)]

use num_traits::Float;

use crate::check::third_order_vel as checker;
use crate::feasible::{Feasible, Span};
use crate::kin_state::{KinThirdVel, LimitsThirdVel};
use crate::segment::{Segment, SignBlock, Touched};

/// Boundary-time step (Step-A): computes the minimum-duration profile.
pub struct StepA<F: Float> {
    pub current: KinThirdVel<F>,
    pub target: KinThirdVel<F>,
    pub limits: LimitsThirdVel<F>,
    ca: F,
    af: F,
    velocity_from_target: F,
    reached_limits: LimitsThirdVel<F>,
}

impl<F: Float> StepA<F> {
    pub fn new(current: KinThirdVel<F>, target: KinThirdVel<F>, limits: LimitsThirdVel<F>) -> Self {
        let velocity_from_target = target.v - current.v;
        Self {
            current,
            target,
            limits,
            ca: current.a,
            af: target.a,
            velocity_from_target,
            reached_limits: limits,
        }
    }

    /// `ACC0` branch: the segment touches the acceleration ceiling.
    fn calculate_up(
        &self,
        profiles: &mut [Segment<F>; 3],
        iter: &mut usize,
        limits: &LimitsThirdVel<F>,
        _is_single_path: bool,
    ) {
        let two = F::from(2.0).unwrap();
        let prof = &mut profiles[*iter];
        prof.t[0] = (-self.ca + limits.max_accel) / limits.jerk;
        prof.t[1] = (self.ca * self.ca + self.af * self.af)
            / (two * limits.max_accel * limits.jerk)
            - limits.max_accel / limits.jerk
            + self.velocity_from_target / limits.max_accel;
        prof.t[2] = (-self.af + limits.max_accel) / limits.jerk;
        prof.t[3] = F::zero();
        prof.t[4] = F::zero();
        prof.t[5] = F::zero();
        prof.t[6] = F::zero();
        if checker::check_profile(prof, SignBlock::Uddu, Touched::Acc0, limits.jerk, limits) {
            advance_profile_iter(profiles, iter);
        }
    }

    /// `NONE` branch: the segment does not reach the acceleration ceiling.
    fn calculate_down(
        &self,
        profiles: &mut [Segment<F>; 3],
        iter: &mut usize,
        limits: &LimitsThirdVel<F>,
        is_single_path: bool,
    ) {
        let two = F::from(2.0).unwrap();
        let mut disc =
            (self.ca * self.ca + self.af * self.af) / two + limits.jerk * self.velocity_from_target;
        if disc >= F::zero() {
            disc = disc.sqrt();
            {
                let prof = &mut profiles[*iter];
                prof.t[0] = -(self.ca + disc) / limits.jerk;
                prof.t[1] = F::zero();
                prof.t[2] = -(self.af + disc) / limits.jerk;
                prof.t[3] = F::zero();
                prof.t[4] = F::zero();
                prof.t[5] = F::zero();
                prof.t[6] = F::zero();
                if checker::check_profile(prof, SignBlock::Uddu, Touched::None, limits.jerk, limits)
                {
                    advance_profile_iter(profiles, iter);
                    if is_single_path {
                        return;
                    }
                }
            }
            {
                if *iter >= profiles.len() {
                    return;
                }
                let prof = &mut profiles[*iter];
                prof.t[0] = (-self.ca + disc) / limits.jerk;
                prof.t[1] = F::zero();
                prof.t[2] = (-self.af + disc) / limits.jerk;
                prof.t[3] = F::zero();
                prof.t[4] = F::zero();
                prof.t[5] = F::zero();
                prof.t[6] = F::zero();
                if checker::check_profile(prof, SignBlock::Uddu, Touched::None, limits.jerk, limits)
                {
                    advance_profile_iter(profiles, iter);
                }
            }
        }
    }

    /// Degenerate-jerk branch: constant-acceleration coast.
    fn check_profile(&self, profile: &mut Segment<F>, limits: &LimitsThirdVel<F>) -> bool {
        let eps = F::epsilon();
        if (self.af - self.ca).abs() > eps {
            return false;
        }
        profile.t[0] = F::zero();
        profile.t[1] = F::zero();
        profile.t[2] = F::zero();
        profile.t[3] = F::zero();
        profile.t[4] = F::zero();
        profile.t[5] = F::zero();
        profile.t[6] = F::zero();
        if self.ca.abs() > eps {
            profile.t[3] = self.velocity_from_target / self.ca;
            if checker::check_profile(profile, SignBlock::Uddu, Touched::None, F::zero(), limits) {
                return true;
            }
        } else if self.velocity_from_target.abs() < eps
            && checker::check_profile(profile, SignBlock::Uddu, Touched::None, F::zero(), limits)
        {
            return true;
        }
        false
    }

    pub fn get_profile(&self, input_profile: &Segment<F>, block: &mut Feasible<F>) -> bool {
        let eps = F::epsilon();
        if self.reached_limits.jerk == F::zero() {
            let profile = &mut block.p_min;
            profile.set_boundary(input_profile);
            if self.check_profile(profile, &self.reached_limits) {
                block.t_min =
                    profile.duration + profile.halt.duration + profile.accel_halt.duration;
                if self.ca.abs() > eps {
                    let p_min_copy = *profile;
                    block.blocked_interval_a = Some(Span {
                        left_time: block.t_min,
                        right_time: F::infinity(),
                        profile_at_right: p_min_copy,
                    });
                }
                return true;
            }
            return false;
        }
        let mut profiles: [Segment<F>; 3] = [Segment::empty(); 3];
        let mut iter: usize = 0;
        profiles[0].set_boundary(input_profile);
        let inv_limits = self.reached_limits.inverse();
        if self.af.abs() < eps {
            let limits_for_direction = if self.velocity_from_target >= F::zero() {
                self.reached_limits
            } else {
                inv_limits
            };
            self.calculate_down(&mut profiles, &mut iter, &limits_for_direction, true);
            if iter > 0 {
                return calculate_feasible(block, &mut profiles, iter);
            }
            self.calculate_up(&mut profiles, &mut iter, &limits_for_direction, true);
            if iter > 0 {
                return calculate_feasible(block, &mut profiles, iter);
            }
            let other_limits = limits_for_direction.inverse();
            self.calculate_down(&mut profiles, &mut iter, &other_limits, true);
            if iter > 0 {
                return calculate_feasible(block, &mut profiles, iter);
            }
            self.calculate_up(&mut profiles, &mut iter, &other_limits, true);
        } else {
            self.calculate_down(&mut profiles, &mut iter, &self.reached_limits, false);
            self.calculate_down(&mut profiles, &mut iter, &inv_limits, false);
            self.calculate_up(&mut profiles, &mut iter, &self.reached_limits, false);
            self.calculate_up(&mut profiles, &mut iter, &inv_limits, false);
        }
        calculate_feasible(block, &mut profiles, iter)
    }
}

/// Timed step (Step-B): assembles a profile that completes within target duration `tf`.
pub struct StepB<F: Float> {
    pub tf: F,
    pub current: KinThirdVel<F>,
    pub target: KinThirdVel<F>,
    pub limits: LimitsThirdVel<F>,
    ca: F,
    af: F,
    velocity_from_target: F,
    acceleration_from_target: F,
    reached_limits: LimitsThirdVel<F>,
}

impl<F: Float> StepB<F> {
    pub fn new(
        tf: F,
        current: KinThirdVel<F>,
        target: KinThirdVel<F>,
        limits: LimitsThirdVel<F>,
    ) -> Self {
        let velocity_from_target = target.v - current.v;
        let acceleration_from_target = target.a - current.a;
        Self {
            tf,
            current,
            target,
            limits,
            ca: current.a,
            af: target.a,
            velocity_from_target,
            acceleration_from_target,
            reached_limits: limits,
        }
    }

    fn calculate_up(&self, profile: &mut Segment<F>, limits: &LimitsThirdVel<F>) -> bool {
        let two = F::from(2.0).unwrap();
        {
            let t_branch = ((-self.acceleration_from_target * self.acceleration_from_target
                + two
                    * limits.jerk
                    * ((self.ca + self.af) * self.tf - two * self.velocity_from_target))
                / (limits.jerk * limits.jerk)
                + self.tf * self.tf)
                .sqrt();
            profile.t[0] =
                self.acceleration_from_target / (two * limits.jerk) + (self.tf - t_branch) / two;
            profile.t[1] = t_branch;
            profile.t[2] = self.tf - (profile.t[0] + t_branch);
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = F::zero();
            if checker::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::Acc0,
                self.tf,
                limits.jerk,
                limits,
            ) {
                profile.pf = profile.p[7];
                return true;
            }
        }
        {
            let denom = -self.acceleration_from_target + limits.jerk * self.tf;
            profile.t[0] = -self.acceleration_from_target * self.acceleration_from_target
                / (two * limits.jerk * denom)
                + (self.velocity_from_target - self.ca * self.tf) / denom;
            profile.t[1] = -self.acceleration_from_target / limits.jerk + self.tf;
            profile.t[2] = F::zero();
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = self.tf - (profile.t[0] + profile.t[1]);
            if checker::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::Acc0,
                self.tf,
                limits.jerk,
                limits,
            ) {
                profile.pf = profile.p[7];
                return true;
            }
        }
        {
            profile.t[0] = F::zero();
            profile.t[1] = -self.acceleration_from_target / limits.jerk + self.tf;
            profile.t[2] = F::zero();
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = self.acceleration_from_target / limits.jerk;
            if checker::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::Acc0,
                self.tf,
                limits.jerk,
                limits,
            ) {
                profile.pf = profile.p[7];
                return true;
            }
        }
        false
    }

    fn calculate_down(&self, profile: &mut Segment<F>, limits: &LimitsThirdVel<F>) -> bool {
        let eps = F::epsilon();
        let two = F::from(2.0).unwrap();
        if self.ca.abs() < eps && self.af.abs() < eps && self.velocity_from_target.abs() < eps {
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
                limits.jerk,
                limits,
            ) {
                profile.pf = profile.p[7];
                return true;
            }
        }
        {
            let tmp = two * (self.af * self.tf - self.velocity_from_target);
            profile.t[0] = tmp / self.acceleration_from_target;
            profile.t[1] = self.tf - profile.t[0];
            profile.t[2] = F::zero();
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = F::zero();
            let j_calc = self.acceleration_from_target * self.acceleration_from_target / tmp;
            let eps12 = F::from(1e-12).unwrap();
            if j_calc.abs() < limits.jerk.abs() + eps12
                && checker::check_profile2(
                    profile,
                    SignBlock::Uddu,
                    Touched::None,
                    self.tf,
                    j_calc,
                    limits,
                )
            {
                profile.pf = profile.p[7];
                return true;
            }
        }
        false
    }

    #[inline]
    fn calculate(&self, profile: &mut Segment<F>, limits: &LimitsThirdVel<F>) -> bool {
        self.calculate_up(profile, limits) || self.calculate_down(profile, limits)
    }

    pub fn get_profile(&self, profile: &mut Segment<F>) -> bool {
        if self.velocity_from_target > F::zero() {
            return self.calculate(profile, &self.reached_limits)
                || self.calculate(profile, &self.reached_limits.inverse());
        }
        self.calculate(profile, &self.reached_limits.inverse())
            || self.calculate(profile, &self.reached_limits)
    }
}

/// Advance the iterator after a profile has been accepted. The next slot's
/// boundary conditions are seeded from the just-accepted profile.
fn advance_profile_iter<F: Float>(profiles: &mut [Segment<F>; 3], iter: &mut usize) {
    let prev_iter = *iter;
    *iter += 1;
    if *iter < profiles.len() {
        let prev = profiles[prev_iter];
        profiles[*iter].set_boundary(&prev);
    }
}

/// Per-axis block aggregator: pick the minimum-duration profile and, when
/// multiple are valid, surface the resulting blocked spans.
fn calculate_feasible<F: Float>(
    block: &mut Feasible<F>,
    valid_profiles: &mut [Segment<F>],
    mut valid_profile_cnt: usize,
) -> bool {
    let dbl_eps = F::epsilon();
    let eps_8 = F::from(8.0).unwrap() * dbl_eps;
    let eps_32 = F::from(32.0).unwrap() * dbl_eps;
    let eps_256 = F::from(256.0).unwrap() * dbl_eps;

    if valid_profile_cnt == 1 {
        block.set_min_profile(valid_profiles[0]);
        return true;
    } else if valid_profile_cnt == 2 {
        if (valid_profiles[0].duration - valid_profiles[1].duration).abs() < eps_8 {
            block.set_min_profile(valid_profiles[0]);
            return true;
        }
        let idx_min = if valid_profiles[0].duration < valid_profiles[1].duration {
            0
        } else {
            1
        };
        let idx_other = (idx_min + 1) % 2;
        block.set_min_profile(valid_profiles[idx_min]);
        block.blocked_interval_a = Some(Span::new(
            valid_profiles[idx_min],
            valid_profiles[idx_other],
        ));
        return true;
    } else if valid_profile_cnt == 4 {
        if (valid_profiles[0].duration - valid_profiles[1].duration).abs() < eps_32
            && valid_profiles[0].sweep != valid_profiles[1].sweep
        {
            remove_profile(valid_profiles, &mut valid_profile_cnt, 1);
        } else if (valid_profiles[2].duration - valid_profiles[3].duration).abs() < eps_256
            && valid_profiles[2].sweep != valid_profiles[3].sweep
        {
            remove_profile(valid_profiles, &mut valid_profile_cnt, 3);
        } else if (valid_profiles[0].duration - valid_profiles[3].duration).abs() < eps_256
            && valid_profiles[0].sweep != valid_profiles[3].sweep
        {
            remove_profile(valid_profiles, &mut valid_profile_cnt, 3);
        } else {
            return false;
        }
    } else if valid_profile_cnt.is_multiple_of(2) {
        return false;
    }
    let mut idx_fastest = 0usize;
    let mut t_fastest = valid_profiles[0].duration;
    for i in 1..valid_profile_cnt {
        let t_current = valid_profiles[i].duration;
        if t_current < t_fastest {
            t_fastest = t_current;
            idx_fastest = i;
        }
    }
    block.set_min_profile(valid_profiles[idx_fastest]);
    if valid_profile_cnt == 3 {
        let idx_a = (idx_fastest + 1) % 3;
        let idx_b = (idx_fastest + 2) % 3;
        block.blocked_interval_a = Some(Span::new(valid_profiles[idx_a], valid_profiles[idx_b]));
        return true;
    } else if valid_profile_cnt == 5 {
        let idx_0 = (idx_fastest + 1) % 5;
        let idx_1 = (idx_fastest + 2) % 5;
        let idx_2 = (idx_fastest + 3) % 5;
        let idx_3 = (idx_fastest + 4) % 5;
        if valid_profiles[idx_0].sweep == valid_profiles[idx_1].sweep {
            block.blocked_interval_a =
                Some(Span::new(valid_profiles[idx_0], valid_profiles[idx_1]));
            block.blocked_interval_b =
                Some(Span::new(valid_profiles[idx_2], valid_profiles[idx_3]));
        } else {
            block.blocked_interval_a =
                Some(Span::new(valid_profiles[idx_0], valid_profiles[idx_3]));
            block.blocked_interval_b =
                Some(Span::new(valid_profiles[idx_1], valid_profiles[idx_2]));
        }
        return true;
    }
    false
}

fn remove_profile<F: Float>(
    profiles: &mut [Segment<F>],
    profile_cnt: &mut usize,
    remove_idx: usize,
) {
    for i in remove_idx..(*profile_cnt - 1) {
        profiles[i] = profiles[i + 1];
    }
    *profile_cnt -= 1;
}
