//! Third-order (pose + velocity + acceleration) shaping math.

// StepA/StepB retain `pub current`, `target`, and `cv_sq` for API parity
// across step classes. The `get_profile` thin wrapper exists for callers
// that don't have a scratch buffer.
#![allow(dead_code)]

use core::f64::{EPSILON, MAX as F64_MAX};
use num_traits::Float;

use crate::check::third_order_pose;
use crate::feasible::{Feasible, Span};
use crate::kin_state::{KinThirdPose, LimitsThirdPose};
use crate::roots::{
    PositiveSet, ROOT_TOLERANCE, poly_derivative, poly_eval, poly_monic_derivative,
    poly_root_newton, solve_cubic, solve_quartic, solve_quartic_arr,
};
use crate::segment::{Segment, SignBlock, Touched};

#[inline]
fn square<F: Float>(x: F) -> F {
    x * x
}

#[inline]
fn copysign_one<F: Float>(x: F) -> F {
    if x >= F::zero() { F::one() } else { -F::one() }
}

#[inline]
fn next_after_zero<F: Float>(x: F) -> F {
    let eps = F::from(EPSILON).unwrap_or_else(F::epsilon);
    if x > F::zero() {
        x - eps * x.abs()
    } else if x < F::zero() {
        x + eps * x.abs()
    } else {
        F::zero()
    }
}

#[inline]
fn next_after_max<F: Float>(x: F) -> F {
    let eps = F::from(EPSILON).unwrap_or_else(F::epsilon);
    let big = F::from(F64_MAX).unwrap_or_else(F::max_value);
    if x >= big {
        x
    } else {
        x + eps * (x.abs() + F::one())
    }
}

#[inline]
fn fmin<F: Float>(a: F, b: F) -> F {
    if a < b { a } else { b }
}

#[inline]
fn fmax<F: Float>(a: F, b: F) -> F {
    if a > b { a } else { b }
}

pub struct StepA<F: Float> {
    pub current: KinThirdPose<F>,
    pub target: KinThirdPose<F>,
    pub limits: LimitsThirdPose<F>,
    pub cv: F,
    pub ca: F,
    pub vf: F,
    pub af: F,
    pub position_from_target: F,
    pub cv_sq: F,
    pub vf_sq: F,
    pub ca_sq: F,
    pub ca_cu: F,
    pub ca_pow4: F,
    pub af_sq: F,
    pub af_cu: F,
    pub af_pow4: F,
    pub j_sq: F,
}

impl<F: Float> StepA<F> {
    #[inline]
    pub fn new(
        current: KinThirdPose<F>,
        target: KinThirdPose<F>,
        limits: LimitsThirdPose<F>,
    ) -> Self {
        let cv = current.v;
        let ca = current.a;
        let vf = target.v;
        let af = target.a;
        let position_from_target = target.p - current.p;
        let cv_sq = cv * cv;
        let vf_sq = vf * vf;
        let ca_sq = ca * ca;
        let af_sq = af * af;
        let ca_cu = ca * ca_sq;
        let ca_pow4 = ca_sq * ca_sq;
        let af_cu = af * af_sq;
        let af_pow4 = af_sq * af_sq;
        let j_sq = limits.jerk * limits.jerk;
        Self {
            current,
            target,
            limits,
            cv,
            ca,
            vf,
            af,
            position_from_target,
            cv_sq,
            vf_sq,
            ca_sq,
            ca_cu,
            ca_pow4,
            af_sq,
            af_cu,
            af_pow4,
            j_sq,
        }
    }

    #[inline]
    fn advance(profiles: &mut [Segment<F>], iter: &mut usize) {
        let prev = profiles[*iter];
        *iter += 1;
        profiles[*iter].set_boundary(&prev);
    }

    fn vel_cases(
        &self,
        profiles: &mut [Segment<F>; 6],
        iter: &mut usize,
        limits: &LimitsThirdPose<F>,
        _is_single_path: bool,
    ) {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let eight = F::from(8.0).unwrap();
        let one = F::one();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        let min_a = limits.min_accel;
        let max_v = limits.max_vel;
        {
            let it = &mut profiles[*iter];
            it.t[0] = (-self.ca + max_a) / jerk;
            it.t[1] = (self.ca_sq / (two * max_a) - max_a) / jerk - (self.cv - max_v) / max_a;
            it.t[2] = max_a / jerk;
            it.t[3] = (((self.ca_pow4 / max_a - self.af_pow4 / min_a) / eight
                + (self.af_cu - self.ca_cu) / three
                + (self.ca_sq * max_a - self.af_sq * min_a) / four)
                / self.j_sq
                + self.vf * (square(self.af - min_a) / jerk - self.vf) / (two * min_a)
                - self.cv * (square(self.ca - max_a) / jerk - self.cv) / (two * max_a)
                + self.position_from_target)
                / max_v
                + ((min_a - max_a) / jerk + max_v / min_a - max_v / max_a) / two;
            it.t[4] = -min_a / jerk;
            it.t[5] = (-self.af_sq / (two * min_a) + min_a) / jerk + (self.vf - max_v) / min_a;
            it.t[6] = it.t[4] + self.af / jerk;
            if third_order_pose::check_profile(
                it,
                SignBlock::Uddu,
                Touched::Acc0Acc1Vel,
                false,
                limits,
            ) {
                Self::advance(profiles, iter);
                return;
            }
        }
        let sqrt_term_ca = (self.ca_sq / (two * self.j_sq) + (max_v - self.cv) / jerk).sqrt();
        {
            let it = &mut profiles[*iter];
            it.t[0] = sqrt_term_ca - self.ca / jerk;
            it.t[1] = F::zero();
            it.t[2] = sqrt_term_ca;
            it.t[3] = ((-self.af_pow4 / (eight * min_a) + (self.af_cu - self.ca_cu) / three
                - self.af_sq * min_a / four)
                / self.j_sq
                + (self.vf * (self.af - min_a) * (self.af / min_a - one) / two
                    + self.ca * self.cv
                    + sqrt_term_ca * self.ca_sq / two)
                    / jerk
                + self.position_from_target
                - self.vf_sq / (two * min_a))
                / max_v
                + max_v / (two * min_a)
                - (self.cv / max_v + one) * sqrt_term_ca
                + min_a / (two * jerk);
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::Acc1Vel, false, limits)
            {
                Self::advance(profiles, iter);
                return;
            }
        }
        let sqrt_term_af = (self.af_sq / (two * self.j_sq) + (max_v - self.vf) / jerk).sqrt();
        {
            let it = &mut profiles[*iter];
            it.t[0] = (-self.ca + max_a) / jerk;
            it.t[1] = (self.ca_sq / (two * max_a) - max_a) / jerk - (self.cv - max_v) / max_a;
            it.t[2] = max_a / jerk;
            it.t[3] = ((self.ca_pow4 / (eight * max_a)
                + (self.af_cu - self.ca_cu) / three
                + self.ca_sq * max_a / four)
                / self.j_sq
                + (self.af_sq * sqrt_term_af / two
                    - self.cv * (self.ca - max_a) * (self.ca / max_a - one) / two
                    - self.af * self.vf)
                    / jerk
                + self.position_from_target)
                / max_v
                - (max_v - self.cv_sq / max_v) / (two * max_a)
                - (self.vf / max_v + one) * sqrt_term_af
                - max_a / (two * jerk);
            it.t[4] = sqrt_term_af;
            it.t[5] = F::zero();
            it.t[6] = sqrt_term_af + self.af / jerk;
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::Acc0Vel, false, limits)
            {
                Self::advance(profiles, iter);
                return;
            }
        }
        {
            let it = &mut profiles[*iter];
            it.t[0] = sqrt_term_ca - self.ca / jerk;
            it.t[1] = F::zero();
            it.t[2] = sqrt_term_ca;
            it.t[3] = ((self.af_cu - self.ca_cu) / (three * self.j_sq)
                + (self.ca * self.cv - self.af * self.vf
                    + (self.af_sq * sqrt_term_af + self.ca_sq * sqrt_term_ca) / two)
                    / jerk
                + self.position_from_target)
                / max_v
                - (self.cv / max_v + one) * sqrt_term_ca
                - (self.vf / max_v + one) * sqrt_term_af;
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::Vel, false, limits) {
                Self::advance(profiles, iter);
            }
        }
    }

    fn acc0_acc1_cases(
        &self,
        profiles: &mut [Segment<F>; 6],
        iter: &mut usize,
        limits: &LimitsThirdPose<F>,
        is_single_path: bool,
    ) {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let eight = F::from(8.0).unwrap();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        let min_a = limits.min_accel;
        let mut disc = ((self.af_pow4 * max_a - self.ca_pow4 * min_a
            + two
                * max_a
                * min_a
                * (four * (self.ca_cu - self.af_cu) / three + min_a * self.af_sq
                    - max_a * self.ca_sq))
            / self.j_sq
            + four
                * (min_a * self.cv * square(max_a - self.ca)
                    - max_a * self.vf * square(min_a - self.af))
                / jerk
            + four
                * (max_a * self.vf_sq
                    - min_a * self.cv_sq
                    - two * min_a * max_a * self.position_from_target))
            / (max_a - min_a)
            + square(max_a * min_a / jerk);
        if disc >= F::zero() {
            disc = disc.sqrt() / two;
            let t_acc0 =
                (self.ca_sq / (two * max_a) + min_a / two - max_a) / jerk - self.cv / max_a;
            let t_acc1 =
                -(self.af_sq / (two * min_a) + max_a / two - min_a) / jerk + self.vf / min_a;
            if t_acc0 > disc / max_a && t_acc1 > -disc / min_a {
                {
                    let it = &mut profiles[*iter];
                    it.t[0] = (-self.ca + max_a) / jerk;
                    it.t[1] = t_acc0 - disc / max_a;
                    it.t[2] = max_a / jerk;
                    it.t[3] = F::zero();
                    it.t[4] = -min_a / jerk;
                    it.t[5] = t_acc1 + disc / min_a;
                    it.t[6] = it.t[4] + self.af / jerk;
                    let ok = third_order_pose::check_profile(
                        it,
                        SignBlock::Uddu,
                        Touched::Acc0Acc1,
                        false,
                        limits,
                    );
                    if ok {
                        Self::advance(profiles, iter);
                        if is_single_path {
                            return;
                        }
                    }
                }
            }
            if t_acc0 > -disc / max_a && t_acc1 > disc / min_a {
                let it = &mut profiles[*iter];
                it.t[0] = (-self.ca + max_a) / jerk;
                it.t[1] = t_acc0 + disc / max_a;
                it.t[2] = max_a / jerk;
                it.t[3] = F::zero();
                it.t[4] = -min_a / jerk;
                it.t[5] = t_acc1 - disc / min_a;
                it.t[6] = it.t[4] + self.af / jerk;
                if third_order_pose::check_profile(
                    it,
                    SignBlock::Uddu,
                    Touched::Acc0Acc1,
                    false,
                    limits,
                ) {
                    Self::advance(profiles, iter);
                }
            }
        }
        let _ = (three, four, eight);
    }

    fn acc_polynomial_cases(
        &self,
        profiles: &mut [Segment<F>; 6],
        iter: &mut usize,
        limits: &LimitsThirdPose<F>,
        is_single_path: bool,
    ) {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        let min_a = limits.min_accel;
        let eps = F::from(EPSILON).unwrap_or_else(F::epsilon);
        let eps_tol = F::from(1e-9).unwrap();
        let vel_offset = (self.ca_sq - self.af_sq) / (two * jerk) + (self.vf - self.cv);
        let vel_offset_sq = vel_offset * vel_offset;
        let t_lower_none = next_after_zero((self.ca - self.af) / jerk);
        let t_upper_none = next_after_max((max_a - min_a) / jerk);
        let mut poly_none = [F::zero(); 4];
        poly_none[0] = F::zero();
        poly_none[1] = -two * ((self.ca_sq + self.af_sq) / jerk - two * (self.cv + self.vf)) / jerk;
        poly_none[2] = four
            * ((self.ca_cu - self.af_cu) / (three * self.j_sq)
                + (self.af * self.vf - self.ca * self.cv) / jerk
                - self.position_from_target)
            / jerk;
        poly_none[3] = -vel_offset_sq / self.j_sq;
        let vel_offset_acc0 =
            ((self.ca_sq - self.af_sq) / (two * jerk) + self.vf - self.cv) / max_a;
        let t_lower_acc0 = next_after_zero((max_a - self.af) / jerk);
        let t_upper_acc0 = next_after_max((max_a - min_a) / jerk);
        let acc0_const = ((self.af_pow4 - self.ca_pow4) / four
            + two * (self.ca_cu - self.af_cu) * max_a / three
            + (self.af_sq - self.ca_sq) * max_a * max_a / two)
            / self.j_sq
            + (self.cv * square(self.ca - max_a) - self.vf * square(self.af - max_a)) / jerk
            + self.vf_sq
            - self.cv_sq
            - two * max_a * self.position_from_target;
        let acc0_quad = -self.af_sq + max_a * max_a + two * jerk * self.vf;
        let mut poly_acc0 = [F::zero(); 4];
        poly_acc0[0] = -two * max_a / jerk;
        poly_acc0[1] = acc0_quad / self.j_sq;
        poly_acc0[2] = F::zero();
        poly_acc0[3] = acc0_const / self.j_sq;
        let vel_offset_acc1 = -(self.ca_sq + self.af_sq) / (two * jerk * min_a)
            + min_a / jerk
            + (self.vf - self.cv) / min_a;
        let t_lower_acc1 = next_after_zero((min_a - self.ca) / jerk);
        let t_upper_acc1 = next_after_max((max_a - self.ca) / jerk);
        let acc1_const = ((self.ca_pow4 - self.af_pow4) / four
            + two * (self.af_cu - self.ca_cu) * min_a / three
            + (self.ca_sq - self.af_sq) * min_a * min_a / two)
            / self.j_sq
            + (self.cv * square(self.ca - min_a) + self.vf * square(self.af - min_a)) / jerk
            + self.cv_sq
            - self.vf_sq
            + two * min_a * self.position_from_target;
        let acc1_inner = self.ca * (self.ca - min_a) / jerk + two * self.cv;
        let mut poly_acc1 = [F::zero(); 4];
        poly_acc1[0] = two * (two * self.ca - min_a) / jerk;
        poly_acc1[1] = (square(three * self.ca - min_a) - four * self.ca_sq) / self.j_sq
            + two * self.cv / jerk;
        poly_acc1[2] = two * (self.ca - min_a) * acc1_inner / self.j_sq;
        poly_acc1[3] = acc1_const / self.j_sq;
        let mut poly_acc0_shifted = poly_acc0;
        poly_acc0_shifted[0] = poly_acc0_shifted[0] + four * t_lower_acc0;
        poly_acc0_shifted[1] =
            poly_acc0_shifted[1] + (three * poly_acc0[0] + six * t_lower_acc0) * t_lower_acc0;
        poly_acc0_shifted[2] = poly_acc0_shifted[2]
            + (two * poly_acc0[1] + (three * poly_acc0[0] + four * t_lower_acc0) * t_lower_acc0)
                * t_lower_acc0;
        poly_acc0_shifted[3] = poly_acc0_shifted[3]
            + (poly_acc0[2]
                + (poly_acc0[1] + (poly_acc0[0] + t_lower_acc0) * t_lower_acc0) * t_lower_acc0)
                * t_lower_acc0;
        let has_acc0_roots = poly_acc0_shifted[0] < F::zero()
            || poly_acc0_shifted[1] < F::zero()
            || poly_acc0_shifted[2] < F::zero()
            || poly_acc0_shifted[3] <= F::zero();
        let has_acc1_roots = poly_acc1[0] < F::zero()
            || poly_acc1[1] < F::zero()
            || poly_acc1[2] < F::zero()
            || poly_acc1[3] <= F::zero();
        let roots_none = solve_quartic_arr(&poly_none);
        let roots_acc0 = if has_acc0_roots {
            solve_quartic_arr(&poly_acc0)
        } else {
            PositiveSet::new()
        };
        let roots_acc1 = if has_acc1_roots {
            solve_quartic_arr(&poly_acc1)
        } else {
            PositiveSet::new()
        };
        for &raw in roots_none.as_slice() {
            let mut t = raw;
            if t < t_lower_none || t > t_upper_none {
                continue;
            }
            if t > eps {
                let j_t_sq = jerk * t * t;
                let residual = (two * self.ca_cu + self.af_cu - three * self.ca_sq * self.af)
                    / (six * self.j_sq)
                    + ((self.af - self.ca) * self.cv - self.ca_sq * t - vel_offset_sq / (four * t)
                        + vel_offset * self.af)
                        / jerk
                    - self.position_from_target
                    + (j_t_sq / four + two * self.cv + vel_offset) * t;
                let derivative = vel_offset + two * self.cv - self.ca_sq / jerk
                    + vel_offset_sq / (four * j_t_sq)
                    + (three * j_t_sq) / four;
                t = t - residual / derivative;
            }
            let t_shift = vel_offset / (two * jerk * t);
            let it = &mut profiles[*iter];
            it.t[0] = t_shift + t / two - self.ca / jerk;
            it.t[1] = F::zero();
            it.t[2] = t;
            it.t[3] = F::zero();
            it.t[4] = F::zero();
            it.t[5] = F::zero();
            it.t[6] = -t_shift + t / two + self.af / jerk;
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::None, false, limits) {
                it.polynomial_root = t;
                Self::advance(profiles, iter);
                if is_single_path {
                    return;
                }
            }
        }
        let _ = eps_tol;
        for &raw in roots_acc0.as_slice() {
            let mut t_acc0_root = raw;
            if t_acc0_root < t_lower_acc0 || t_acc0_root > t_upper_acc0 {
                continue;
            }
            if t_acc0_root > eps {
                let j_t = jerk * t_acc0_root;
                let residual_acc0 = acc0_const / t_acc0_root
                    + t_acc0_root * (acc0_quad + j_t * (j_t - two * max_a));
                let derivative_acc0 = two * (acc0_quad + j_t * (two * j_t - three * max_a));
                t_acc0_root = t_acc0_root - residual_acc0 / derivative_acc0;
            }
            let it = &mut profiles[*iter];
            it.t[0] = (-self.ca + max_a) / jerk;
            it.t[1] =
                vel_offset_acc0 - two * t_acc0_root + jerk / max_a * t_acc0_root * t_acc0_root;
            it.t[2] = t_acc0_root;
            it.t[3] = F::zero();
            it.t[4] = F::zero();
            it.t[5] = F::zero();
            it.t[6] = (self.af - max_a) / jerk + t_acc0_root;
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::Acc0, false, limits) {
                it.polynomial_root = t_acc0_root;
                Self::advance(profiles, iter);
                if is_single_path {
                    return;
                }
            }
        }
        for &raw in roots_acc1.as_slice() {
            let mut t_acc1_root = raw;
            if t_acc1_root < t_lower_acc1 || t_acc1_root > t_upper_acc1 {
                continue;
            }
            if t_acc1_root > eps {
                for _ in 0..2usize {
                    let j_t_acc1 = jerk * t_acc1_root;
                    let residual_acc1 = -acc1_const / two
                        - t_acc1_root
                            * ((self.ca / jerk + t_acc1_root / two) * square(self.ca - min_a)
                                + two * self.ca * t_acc1_root * (self.ca - min_a + j_t_acc1)
                                + two * self.ca * self.cv
                                + (j_t_acc1 / two - min_a)
                                    * (j_t_acc1 * t_acc1_root + two * self.cv));
                    if residual_acc1.abs() < eps_tol {
                        break;
                    }
                    let derivative_acc1 = (min_a - self.ca - j_t_acc1)
                        * (acc1_inner + t_acc1_root * (four * self.ca - min_a + two * j_t_acc1));
                    t_acc1_root = t_acc1_root - residual_acc1 / derivative_acc1;
                }
            }
            let it = &mut profiles[*iter];
            it.t[0] = t_acc1_root;
            it.t[1] = F::zero();
            it.t[2] = (self.ca - min_a) / jerk + t_acc1_root;
            it.t[3] = F::zero();
            it.t[4] = F::zero();
            it.t[5] = vel_offset_acc1 - (two * self.ca + jerk * t_acc1_root) * t_acc1_root / min_a;
            it.t[6] = (self.af - min_a) / jerk;
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::Acc1, false, limits) {
                it.polynomial_root = t_acc1_root;
                Self::advance(profiles, iter);
                if is_single_path {
                    return;
                }
            }
        }
    }

    fn acc1_case(
        &self,
        profiles: &mut [Segment<F>; 6],
        iter: &mut usize,
        limits: &LimitsThirdPose<F>,
    ) {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let jerk = limits.jerk;
        let min_a = limits.min_accel;
        let max_v = limits.max_vel;
        let it = &mut profiles[*iter];
        it.t[0] = F::zero();
        it.t[1] = F::zero();
        it.t[2] = self.ca / jerk;
        it.t[3] = ((min_a * (two * self.af_cu + self.ca_cu) / three
            - self.af_sq * (self.af_sq / two + min_a * min_a) / two
            + jerk
                * (self.vf * square(self.af - min_a)
                    + min_a * min_a * max_v
                    + jerk * (max_v * max_v - self.vf_sq)))
            / (two * min_a * self.j_sq)
            + self.position_from_target)
            / max_v
            - self.ca / jerk;
        it.t[4] = -min_a / jerk;
        it.t[5] = -(self.af_sq / two - min_a * min_a + jerk * (max_v - self.vf)) / (min_a * jerk);
        it.t[6] = it.t[4] + self.af / jerk;
        if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::Acc1, false, limits) {
            Self::advance(profiles, iter);
        }
    }

    fn acc0_case(
        &self,
        profiles: &mut [Segment<F>; 6],
        iter: &mut usize,
        limits: &LimitsThirdPose<F>,
    ) {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let six = F::from(6.0).unwrap();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        let min_a = limits.min_accel;
        {
            let it = &mut profiles[*iter];
            it.t[0] = F::zero();
            it.t[1] = ((self.af_sq - self.ca_sq) / (two * jerk) + self.vf - self.cv) / self.ca;
            it.t[2] = (self.ca - self.af) / jerk;
            it.t[3] = F::zero();
            it.t[4] = F::zero();
            it.t[5] = F::zero();
            it.t[6] = F::zero();
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::None, false, limits) {
                Self::advance(profiles, iter);
                return;
            }
        }
        {
            let it = &mut profiles[*iter];
            it.t[0] = (-self.ca + max_a) / jerk;
            it.t[1] = ((self.ca_sq + self.af_sq) / (two * max_a) - max_a) / jerk
                + (self.vf - self.cv) / max_a;
            it.t[2] = (-self.af + max_a) / jerk;
            it.t[3] = F::zero();
            it.t[4] = F::zero();
            it.t[5] = F::zero();
            it.t[6] = F::zero();
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::Acc0, false, limits) {
                Self::advance(profiles, iter);
                return;
            }
        }
        {
            let a_coef = three * ((self.af_sq - self.ca_sq) / jerk + two * (self.cv + self.vf));
            let b_coef = (two * self.af_cu + self.ca_cu - three * self.ca * self.af_sq) / jerk
                + six * jerk * self.position_from_target
                + six * (self.af - self.ca) * self.vf;
            let c_coef = square(self.af - self.ca)
                * ((self.ca - self.af) * (three * self.af + self.ca) / (two * jerk)
                    - six * self.vf)
                + six
                    * jerk
                    * (two * self.ca * self.position_from_target - self.vf_sq + self.cv_sq);
            let disc_sqrt = (b_coef * b_coef + a_coef * c_coef).sqrt();
            let it = &mut profiles[*iter];
            it.t[0] = (b_coef + disc_sqrt) / (jerk * a_coef);
            it.t[1] = -two * disc_sqrt / (jerk * a_coef);
            it.t[2] = ((-two * self.ca_cu - self.af_cu + three * self.ca_sq * self.af) / jerk
                + six * jerk * self.position_from_target
                - six * (self.af - self.ca) * self.cv
                + disc_sqrt)
                / (jerk * a_coef);
            it.t[3] = F::zero();
            it.t[4] = F::zero();
            it.t[5] = F::zero();
            it.t[6] = F::zero();
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::None, true, limits) {
                Self::advance(profiles, iter);
                return;
            }
        }
        {
            let t_const = (max_a - min_a) / jerk;
            let it = &mut profiles[*iter];
            it.t[0] = (-self.ca + max_a) / jerk;
            it.t[1] = (self.ca_sq - self.af_sq) / (two * max_a * jerk)
                + (self.vf - self.cv + jerk * t_const * t_const) / max_a
                - two * t_const;
            it.t[2] = t_const;
            it.t[3] = F::zero();
            it.t[4] = F::zero();
            it.t[5] = F::zero();
            it.t[6] = (self.af - min_a) / jerk;
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::Acc0, false, limits) {
                it.polynomial_root = t_const;
                Self::advance(profiles, iter);
            }
        }
    }

    fn none_case_a(
        &self,
        profiles: &mut [Segment<F>; 6],
        iter: &mut usize,
        limits: &LimitsThirdPose<F>,
    ) {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let one = F::one();
        let jerk = limits.jerk;
        let max_v = limits.max_vel;
        let sqrt_term = (self.af_sq / (two * self.j_sq) + (max_v - self.vf) / jerk).sqrt();
        {
            let it = &mut profiles[*iter];
            it.t[0] = -self.ca / jerk;
            it.t[1] = F::zero();
            it.t[2] = F::zero();
            it.t[3] = (self.af_cu - self.ca_cu) / (three * self.j_sq * max_v)
                + (self.ca * self.cv - self.af * self.vf + (self.af_sq * sqrt_term) / two)
                    / (jerk * max_v)
                - (self.vf / max_v + one) * sqrt_term
                + self.position_from_target / max_v;
            it.t[4] = sqrt_term;
            it.t[5] = F::zero();
            it.t[6] = sqrt_term + self.af / jerk;
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::None, false, limits) {
                Self::advance(profiles, iter);
                return;
            }
        }
        {
            let it = &mut profiles[*iter];
            it.t[0] = F::zero();
            it.t[1] = F::zero();
            it.t[2] = self.ca / jerk;
            it.t[3] = (self.af_cu - self.ca_cu) / (three * self.j_sq * max_v)
                + (self.ca * self.cv - self.af * self.vf
                    + (self.af_sq * sqrt_term + self.ca_cu / jerk) / two)
                    / (jerk * max_v)
                - (self.cv / max_v + one) * self.ca / jerk
                - (self.vf / max_v + one) * sqrt_term
                + self.position_from_target / max_v;
            it.t[4] = sqrt_term;
            it.t[5] = F::zero();
            it.t[6] = sqrt_term + self.af / jerk;
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::None, false, limits) {
                Self::advance(profiles, iter);
            }
        }
    }

    fn none_case_b(
        &self,
        profiles: &mut [Segment<F>; 6],
        iter: &mut usize,
        limits: &LimitsThirdPose<F>,
    ) {
        let two = F::from(2.0).unwrap();
        let jerk = limits.jerk;
        {
            let a_meet = ((self.ca_sq + self.af_sq) / two + jerk * (self.vf - self.cv)).sqrt()
                * copysign_one(jerk);
            let it = &mut profiles[*iter];
            it.t[0] = (a_meet - self.ca) / jerk;
            it.t[1] = F::zero();
            it.t[2] = (a_meet - self.af) / jerk;
            it.t[3] = F::zero();
            it.t[4] = F::zero();
            it.t[5] = F::zero();
            it.t[6] = F::zero();
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::None, false, limits) {
                it.polynomial_root = it.t[2];
                Self::advance(profiles, iter);
                return;
            }
        }
        {
            let it = &mut profiles[*iter];
            it.t[0] = (self.af - self.ca) / jerk;
            it.t[1] = F::zero();
            it.t[2] = F::zero();
            it.t[3] = F::zero();
            it.t[4] = F::zero();
            it.t[5] = F::zero();
            it.t[6] = F::zero();
            if third_order_pose::check_profile(it, SignBlock::Uddu, Touched::None, false, limits) {
                it.polynomial_root = F::zero();
                Self::advance(profiles, iter);
            }
        }
    }

    fn check_profile_zero(&self, profile: &mut Segment<F>, limits: &LimitsThirdPose<F>) -> bool {
        let eps = F::from(EPSILON).unwrap_or_else(F::epsilon);
        let two = F::from(2.0).unwrap();
        if (self.af - self.ca).abs() > eps {
            return false;
        }
        for i in 0..7 {
            profile.t[i] = F::zero();
        }
        if self.ca.abs() > eps {
            let disc_sqrt = (two * self.ca * self.position_from_target + self.cv_sq).sqrt();
            profile.t[3] = (-self.cv + disc_sqrt) / self.ca;
            if profile.t[3] >= F::zero()
                && third_order_pose::check_profile_jerk(
                    profile,
                    SignBlock::Uddu,
                    Touched::None,
                    false,
                    F::zero(),
                    limits,
                )
            {
                return true;
            }
            profile.t[3] = -(self.cv + disc_sqrt) / self.ca;
            if profile.t[3] >= F::zero()
                && third_order_pose::check_profile_jerk(
                    profile,
                    SignBlock::Uddu,
                    Touched::None,
                    false,
                    F::zero(),
                    limits,
                )
            {
                return true;
            }
        } else if self.cv.abs() > eps {
            profile.t[3] = self.position_from_target / self.cv;
            if third_order_pose::check_profile_jerk(
                profile,
                SignBlock::Uddu,
                Touched::None,
                false,
                F::zero(),
                limits,
            ) {
                return true;
            }
        } else if self.position_from_target.abs() < eps
            && third_order_pose::check_profile_jerk(
                profile,
                SignBlock::Uddu,
                Touched::None,
                false,
                F::zero(),
                limits,
            )
        {
            return true;
        }
        false
    }

    pub fn get_profile(&self, input_profile: &Segment<F>, block: &mut Feasible<F>) -> bool {
        // Convenience wrapper that allocates a fresh scratch on the stack.
        // Hot-path callers should use `get_profile_with_scratch` and pass a
        // long-lived buffer to amortise the zero-init.
        let mut profiles: [Segment<F>; 6] = [Segment::empty(); 6];
        self.get_profile_with_scratch(input_profile, block, &mut profiles)
    }

    pub fn get_profile_with_scratch(
        &self,
        input_profile: &Segment<F>,
        block: &mut Feasible<F>,
        profiles: &mut [Segment<F>; 6],
    ) -> bool {
        let eps = F::from(EPSILON).unwrap_or_else(F::epsilon);
        let limit_is_zero = self.limits.jerk == F::zero()
            || self.limits.max_accel == F::zero()
            || self.limits.min_accel == F::zero()
            || self.limits.max_vel == F::zero();
        if limit_is_zero {
            let mut profile = block.p_min;
            profile.set_boundary(input_profile);
            if self.check_profile_zero(&mut profile, &self.limits) {
                block.p_min = profile;
                block.t_min =
                    profile.duration + profile.halt.duration + profile.accel_halt.duration;
                if self.cv.abs() > eps || self.ca.abs() > eps {
                    let big = F::from(F64_MAX).unwrap_or_else(F::max_value);
                    block.blocked_interval_a = Some(Span::from_times(block.t_min, big));
                }
                return true;
            }
            return false;
        }
        // The caller owns the scratch. We only need slot 0 freshly seeded;
        // every subsequent slot is set_boundary'd via `advance` before its
        // case function writes `t[..]` and check_profile fills the rest.
        // Slots beyond `profile_iter` are not read by `pick_from_candidates`.
        let mut profile_iter = 0usize;
        profiles[0].set_boundary(input_profile);
        let inv_limits = self.limits.inverse();
        if self.vf.abs() < eps && self.af.abs() < eps {
            let limits_for_direction = if self.position_from_target >= F::zero() {
                self.limits
            } else {
                inv_limits
            };
            if self.cv.abs() < eps && self.ca.abs() < eps && self.position_from_target.abs() < eps {
                self.acc_polynomial_cases(
                    &mut *profiles,
                    &mut profile_iter,
                    &limits_for_direction,
                    true,
                );
            } else {
                self.vel_cases(
                    &mut *profiles,
                    &mut profile_iter,
                    &limits_for_direction,
                    true,
                );
                if profile_iter == 0 {
                    self.acc_polynomial_cases(
                        profiles,
                        &mut profile_iter,
                        &limits_for_direction,
                        true,
                    );
                }
                if profile_iter == 0 {
                    self.acc0_acc1_cases(profiles, &mut profile_iter, &limits_for_direction, true);
                }
                if profile_iter == 0 {
                    let other = limits_for_direction.inverse();
                    self.vel_cases(&mut *profiles, &mut profile_iter, &other, true);
                    if profile_iter == 0 {
                        self.acc_polynomial_cases(&mut *profiles, &mut profile_iter, &other, true);
                    }
                    if profile_iter == 0 {
                        self.acc0_acc1_cases(&mut *profiles, &mut profile_iter, &other, true);
                    }
                }
            }
        } else {
            self.vel_cases(&mut *profiles, &mut profile_iter, &self.limits, false);
            self.vel_cases(&mut *profiles, &mut profile_iter, &inv_limits, false);
            self.acc_polynomial_cases(&mut *profiles, &mut profile_iter, &self.limits, false);
            self.acc_polynomial_cases(&mut *profiles, &mut profile_iter, &inv_limits, false);
            self.acc0_acc1_cases(&mut *profiles, &mut profile_iter, &self.limits, false);
            self.acc0_acc1_cases(&mut *profiles, &mut profile_iter, &inv_limits, false);
        }
        if profile_iter == 0 {
            self.none_case_b(&mut *profiles, &mut profile_iter, &self.limits);
            if profile_iter == 0 {
                self.none_case_b(&mut *profiles, &mut profile_iter, &inv_limits);
            }
            if profile_iter == 0 {
                self.acc0_case(&mut *profiles, &mut profile_iter, &self.limits);
            }
            if profile_iter == 0 {
                self.acc0_case(&mut *profiles, &mut profile_iter, &inv_limits);
            }
            if profile_iter == 0 {
                self.none_case_a(&mut *profiles, &mut profile_iter, &self.limits);
            }
            if profile_iter == 0 {
                self.none_case_a(&mut *profiles, &mut profile_iter, &inv_limits);
            }
            if profile_iter == 0 {
                self.acc1_case(&mut *profiles, &mut profile_iter, &self.limits);
            }
            if profile_iter == 0 {
                self.acc1_case(&mut *profiles, &mut profile_iter, &inv_limits);
            }
        }
        block.pick_from_candidates(&mut profiles.as_mut_slice()[..], profile_iter)
    }
}

pub struct StepB<F: Float> {
    pub current: KinThirdPose<F>,
    pub target: KinThirdPose<F>,
    pub limits: LimitsThirdPose<F>,
    pub tf: F,
    pub cv: F,
    pub vf: F,
    pub ca: F,
    pub af: F,
    pub position_from_target: F,
    pub velocity_from_target: F,
    pub acceleration_from_target: F,
    pub single_inflection_enabled: bool,
    pub time_scale: F,
    pub tf_sq: F,
    pub tf_cu: F,
    pub tf_pow4: F,
    pub velocity_from_target_sq: F,
    pub cv_sq: F,
    pub vf_sq: F,
    pub acceleration_from_target_sq: F,
    pub ca_sq: F,
    pub af_sq: F,
    pub ca_cu: F,
    pub ca_pow4: F,
    pub ca_pow5: F,
    pub ca_pow6: F,
    pub af_cu: F,
    pub af_pow4: F,
    pub af_pow5: F,
    pub af_pow6: F,
    pub j_sq: F,
    pub cv_tf_offset: F,
    pub cv_vf_tf_offset: F,
}

impl<F: Float> StepB<F> {
    pub fn new(
        tf_in: F,
        current: KinThirdPose<F>,
        target: KinThirdPose<F>,
        limits: LimitsThirdPose<F>,
    ) -> Self {
        let two = F::from(2.0).unwrap();
        let mut cv = current.v;
        let mut ca = current.a;
        let mut vf = target.v;
        let mut af = target.a;
        let mut tf = tf_in;
        let position_from_target = target.p - current.p;
        let mut limits = limits;
        // `cbrt` is the analytic cube root — strictly faster than `powf(1/3)`
        // which goes through the generic `pow` exp/log path.
        let time_scale = if limits.jerk != F::zero() {
            (limits.jerk / tf).cbrt()
        } else {
            F::one()
        };
        cv = cv / time_scale;
        ca = ca / (time_scale * time_scale);
        tf = tf * time_scale;
        vf = vf / time_scale;
        af = af / (time_scale * time_scale);
        limits.min_vel = limits.min_vel / time_scale;
        limits.max_vel = limits.max_vel / time_scale;
        limits.min_accel = limits.min_accel / (time_scale * time_scale);
        limits.max_accel = limits.max_accel / (time_scale * time_scale);
        limits.jerk = limits.jerk / (time_scale * time_scale * time_scale);
        let tf_sq = tf * tf;
        let tf_cu = tf_sq * tf;
        let tf_pow4 = tf_sq * tf_sq;
        let velocity_from_target = vf - cv;
        let velocity_from_target_sq = velocity_from_target * velocity_from_target;
        let cv_sq = cv * cv;
        let vf_sq = vf * vf;
        let acceleration_from_target = af - ca;
        let acceleration_from_target_sq = acceleration_from_target * acceleration_from_target;
        let ca_sq = ca * ca;
        let af_sq = af * af;
        let ca_cu = ca * ca_sq;
        let ca_pow4 = ca_sq * ca_sq;
        let ca_pow5 = ca_cu * ca_sq;
        let ca_pow6 = ca_pow4 * ca_sq;
        let af_cu = af * af_sq;
        let af_pow4 = af_sq * af_sq;
        let af_pow5 = af_cu * af_sq;
        let af_pow6 = af_pow4 * af_sq;
        let j_sq = limits.jerk * limits.jerk;
        let cv_tf_offset = -position_from_target + tf * cv;
        let cv_vf_tf_offset = -two * position_from_target + tf * (cv + vf);
        Self {
            current,
            target,
            limits,
            tf,
            cv,
            vf,
            ca,
            af,
            position_from_target,
            velocity_from_target,
            acceleration_from_target,
            single_inflection_enabled: false,
            time_scale,
            tf_sq,
            tf_cu,
            tf_pow4,
            velocity_from_target_sq,
            cv_sq,
            vf_sq,
            acceleration_from_target_sq,
            ca_sq,
            af_sq,
            ca_cu,
            ca_pow4,
            ca_pow5,
            ca_pow6,
            af_cu,
            af_pow4,
            af_pow5,
            af_pow6,
            j_sq,
            cv_tf_offset,
            cv_vf_tf_offset,
        }
    }

    fn time_acc0_acc1_a(&self, profile: &mut Segment<F>, limits: &LimitsThirdPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let eight = F::from(8.0).unwrap();
        let twelve = F::from(12.0).unwrap();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        let min_a = limits.min_accel;
        if (two * (max_a - min_a) + self.acceleration_from_target) / jerk < self.tf {
            let disc_sqrt = (((self.ca_pow4 + self.af_pow4
                - four * self.af_cu * (max_a + two * min_a) / three
                - four * self.ca_cu * (two * max_a + min_a) / three
                + two * self.af_sq * ((min_a - max_a) * (max_a + min_a) + two * self.ca * max_a)
                - two * self.ca_sq * (square(self.af - min_a) - max_a * max_a))
                / (four * self.j_sq)
                + (self.tf * (self.af_sq * max_a - self.ca_sq * min_a)
                    - self.velocity_from_target
                        * (square(self.af - min_a) - square(self.ca - max_a)))
                    / jerk
                + (two * (max_a - min_a) * self.position_from_target
                    + two * self.tf * (min_a * self.cv - max_a * self.vf)
                    + self.velocity_from_target_sq))
                / (max_a * min_a)
                + square(((max_a - min_a) / two + self.acceleration_from_target) / jerk - self.tf))
            .sqrt();
            profile.t[0] = (-self.ca + max_a) / jerk;
            profile.t[1] = ((square(self.ca - min_a) - square(self.af - min_a))
                / (two * (max_a - min_a))
                + (min_a / two - max_a))
                / jerk
                + (min_a * (disc_sqrt - self.tf) + self.velocity_from_target) / (max_a - min_a);
            profile.t[2] = max_a / jerk;
            profile.t[3] = (min_a - max_a) / (two * jerk) + disc_sqrt;
            profile.t[4] = -min_a / jerk;
            profile.t[5] = self.tf
                - (profile.t[0]
                    + profile.t[1]
                    + profile.t[2]
                    + profile.t[3]
                    + two * profile.t[4]
                    + self.af / jerk);
            profile.t[6] = profile.t[4] + self.af / jerk;
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::Acc0Acc1,
                false,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        if (-self.ca + four * max_a - self.af) / jerk < self.tf {
            profile.t[0] = (-self.ca + max_a) / jerk;
            profile.t[1] = (three * (self.ca_pow4 + self.af_pow4)
                - four * self.ca_cu * max_a
                - eight * self.af_cu * max_a
                - six
                    * max_a
                    * max_a
                    * (two * square(self.ca - max_a) + two * square(self.af - max_a) - self.af_sq)
                + six * self.ca_sq * square(self.af - max_a)
                + twelve
                    * jerk
                    * ((self.af_sq + self.ca_sq - max_a * (max_a + two * self.af))
                        * self.velocity_from_target
                        + max_a * (two * max_a * max_a - self.ca_sq) * self.tf
                        + jerk * (two * max_a * self.cv_tf_offset + self.velocity_from_target_sq)))
                / (twelve
                    * max_a
                    * jerk
                    * (square(self.ca - max_a) + square(self.af - max_a)
                        - two * jerk * (max_a * self.tf - self.velocity_from_target)));
            profile.t[2] = max_a / jerk;
            profile.t[3] = -(square(self.ca - max_a) + square(self.af - max_a))
                / (two * max_a * jerk)
                - max_a / jerk
                - self.velocity_from_target / max_a
                + self.tf;
            profile.t[4] = profile.t[2];
            profile.t[5] = self.tf
                - (profile.t[0] + profile.t[1] + profile.t[2] + profile.t[3] + two * profile.t[4]
                    - self.af / jerk);
            profile.t[6] = profile.t[4] - self.af / jerk;
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Udud,
                Touched::Acc0Acc1,
                false,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        false
    }

    fn time_acc1_vel(&self, profile: &mut Segment<F>, limits: &LimitsThirdPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let eight = F::from(8.0).unwrap();
        let twelve = F::from(12.0).unwrap();
        let sixteen = F::from(16.0).unwrap();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        let min_a = limits.min_accel;
        let eps = F::from(EPSILON).unwrap_or_else(F::epsilon);
        {
            let poly_b = self.ca * (self.ca - min_a) + square(self.af - min_a)
                - two * jerk * (self.velocity_from_target - min_a * self.tf);
            let poly_d_part =
                two * min_a * (jerk * self.cv_tf_offset + self.af * self.velocity_from_target)
                    - min_a * min_a * self.velocity_from_target
                    + jerk * self.velocity_from_target_sq;
            let poly_c = self.af_sq + min_a * (min_a - two * self.af)
                - two * jerk * (self.velocity_from_target - min_a * self.tf);
            let mut poly1 = [F::zero(); 4];
            poly1[0] = (two * (two * self.ca - min_a)) / jerk;
            poly1[1] = (four * self.ca_sq + poly_b - three * self.ca * min_a) / self.j_sq;
            poly1[2] = (two * self.ca * poly_b) / (self.j_sq * jerk);
            poly1[3] = (three * (self.ca_pow4 + self.af_pow4)
                - four * (self.ca_cu + two * self.af_cu) * min_a
                + six * self.af_sq * (min_a * min_a - two * jerk * self.velocity_from_target)
                + twelve * jerk * poly_d_part
                + six * self.ca_sq * poly_c)
                / (twelve * self.j_sq * self.j_sq);
            let t_lower1 = next_after_zero(-self.ca / jerk);
            let t_upper1 = next_after_max(fmin(
                (self.tf + two * min_a / jerk - (self.ca + self.af) / jerk) / two,
                (max_a - self.ca) / jerk,
            ));
            let roots1 = solve_quartic_arr(&poly1);
            for &raw in roots1.as_slice() {
                let mut t_root1 = raw;
                if t_root1 < t_lower1 || t_root1 > t_upper1 {
                    continue;
                }
                if (self.ca + jerk * t_root1).abs() > sixteen * eps {
                    let j_t_sq = jerk * t_root1 * t_root1;
                    let residual1 = -self.position_from_target
                        + ((self.ca_sq + self.af_sq)
                            * (three * (self.ca_sq + self.af_sq) - eight * self.af * min_a
                                + six * min_a * min_a)
                            - four * self.ca_sq * (self.ca + self.af) * min_a
                            + twelve
                                * jerk
                                * (self.ca_sq
                                    * (min_a * self.tf + two * (self.ca - min_a) * t_root1
                                        - self.velocity_from_target)
                                    + (two * self.ca * t_root1 - self.velocity_from_target)
                                        * square(self.af - min_a)
                                    + jerk
                                        * ((four * self.ca * t_root1
                                            - self.velocity_from_target
                                            + j_t_sq)
                                            * (j_t_sq - self.velocity_from_target)
                                            + t_root1
                                                * t_root1
                                                * (self.af_sq
                                                    + F::from(5.0).unwrap() * self.ca_sq
                                                    + min_a * min_a))))
                            / (F::from(24.0).unwrap() * min_a * self.j_sq)
                        + (two * self.ca * t_root1 + j_t_sq) * (self.tf - t_root1)
                        + self.tf * self.cv
                        - self.af * t_root1 * t_root1;
                    let derivative1 = (self.ca + jerk * t_root1)
                        * ((self.ca_sq + self.af_sq) / (min_a * jerk)
                            + (min_a - self.ca - two * self.af) / jerk
                            + (four * self.ca * t_root1 + two * j_t_sq
                                - two * self.velocity_from_target)
                                / min_a
                            + two * self.tf
                            - three * t_root1);
                    t_root1 = t_root1 - residual1 / derivative1;
                }
                let plateau1 = ((self.ca_sq - self.af_sq) / two - square(self.ca + jerk * t_root1)
                    + jerk * self.velocity_from_target)
                    / min_a;
                profile.t[0] = t_root1;
                profile.t[1] = F::zero();
                profile.t[2] = self.ca / jerk + t_root1;
                profile.t[3] =
                    self.tf - (plateau1 - min_a + self.ca + self.af) / jerk - two * t_root1;
                profile.t[4] = -min_a / jerk;
                profile.t[5] = (plateau1 + min_a) / jerk;
                profile.t[6] = profile.t[4] + self.af / jerk;
                if third_order_pose::check_profile2(
                    profile,
                    SignBlock::Uddu,
                    Touched::Acc1Vel,
                    false,
                    self.tf,
                    limits,
                ) {
                    return true;
                }
            }
        }
        {
            let part_a = square(self.af - max_a)
                + two * jerk * (self.velocity_from_target - max_a * self.tf);
            let poly_b2 = self.ca * (self.ca - max_a) - part_a;
            let part_f = max_a * max_a + two * jerk * self.velocity_from_target;
            let poly_d_part2 = two * max_a * jerk * self.cv_tf_offset
                + max_a * max_a * self.velocity_from_target
                + jerk * self.velocity_from_target_sq;
            let mut poly2 = [F::zero(); 4];
            poly2[0] = (four * self.ca - two * max_a) / jerk;
            poly2[1] =
                (F::from(5.0).unwrap() * self.ca_sq - four * self.ca * max_a - part_a) / self.j_sq;
            poly2[2] = (two * self.ca * poly_b2) / (self.j_sq * jerk);
            poly2[3] = (three * (self.ca_pow4 + self.af_pow4)
                - four * (self.ca_cu + two * self.af_cu) * max_a
                - F::from(24.0).unwrap() * self.af * max_a * jerk * self.velocity_from_target
                + twelve * jerk * poly_d_part2
                - six * self.ca_sq * part_a
                + six * self.af_sq * part_f)
                / (twelve * self.j_sq * self.j_sq);
            let t_lower2 = next_after_zero(-self.ca / jerk);
            let t_upper2 = next_after_max(fmin(
                (self.tf + self.acceleration_from_target / jerk - two * max_a / jerk) / two,
                (max_a - self.ca) / jerk,
            ));
            let roots2 = solve_quartic_arr(&poly2);
            for &raw in roots2.as_slice() {
                let t_root2 = raw;
                if t_root2 > t_upper2 || t_root2 < t_lower2 {
                    continue;
                }
                let plateau2 = (-(self.ca_sq + self.af_sq) / two
                    + square(self.ca + jerk * t_root2)
                    - jerk * self.velocity_from_target)
                    / max_a;
                profile.t[0] = t_root2;
                profile.t[1] = F::zero();
                profile.t[2] = t_root2 + self.ca / jerk;
                profile.t[3] = self.tf + (plateau2 + self.acceleration_from_target - max_a) / jerk
                    - two * t_root2;
                profile.t[4] = max_a / jerk;
                profile.t[5] = -(plateau2 + max_a) / jerk;
                profile.t[6] = profile.t[4] - self.af / jerk;
                if third_order_pose::check_profile2(
                    profile,
                    SignBlock::Udud,
                    Touched::Acc1Vel,
                    false,
                    self.tf,
                    limits,
                ) {
                    return true;
                }
            }
        }
        false
    }

    fn time_acc0(&self, profile: &mut Segment<F>, limits: &LimitsThirdPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let eight = F::from(8.0).unwrap();
        let twelve = F::from(12.0).unwrap();
        let twenty_four = F::from(24.0).unwrap();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        let min_a = limits.min_accel;
        let eps = F::from(EPSILON).unwrap_or_else(F::epsilon);
        if self.tf < fmax((-self.ca + max_a) / jerk, F::zero()) + fmax(max_a / jerk, F::zero()) {
            return false;
        }
        {
            let mut poly1 = [F::zero(); 4];
            poly1[0] = two * max_a / jerk;
            poly1[1] = (square(self.acceleration_from_target + max_a)
                - two * self.acceleration_from_target * self.af)
                / self.j_sq
                + two * (self.velocity_from_target - max_a * self.tf) / jerk;
            poly1[2] = F::zero();
            poly1[3] = (three
                * (self.af_sq - self.ca_sq)
                * (self.af_sq - self.ca_sq - two * max_a * max_a)
                - four * (self.af_cu + two * self.ca_cu - three * self.af_sq * self.ca) * max_a
                + twelve
                    * jerk
                    * (self.af_sq * (max_a * self.tf - self.velocity_from_target)
                        + square(self.ca - max_a) * self.velocity_from_target
                        + jerk
                            * (self.velocity_from_target_sq
                                - two * max_a * (self.tf * self.vf - self.position_from_target))))
                / (twelve * self.j_sq * self.j_sq);
            let t_lower1 = next_after_zero(-self.af / jerk);
            let t_upper1 = next_after_max(fmin(
                self.tf - (two * max_a - self.ca) / jerk,
                -min_a / jerk,
            ));
            let roots1 = solve_quartic_arr(&poly1);
            for &raw in roots1.as_slice() {
                let mut t_root1 = raw;
                if t_root1 < t_lower1 || t_root1 > t_upper1 {
                    continue;
                }
                if t_root1 > eps {
                    let v_inner = jerk * t_root1 * t_root1 + self.velocity_from_target;
                    let residual1 = (-(self.af_sq - self.ca_sq)
                        * (three * (self.af_sq - self.ca_sq)
                            + (eight * self.ca - six * max_a) * max_a)
                        + four * (self.af - self.ca) * self.af_sq * max_a
                        + twelve
                            * jerk
                            * (self.af_sq * (v_inner - max_a * self.tf)
                                - v_inner * (square(self.ca - max_a) + jerk * v_inner)))
                        / (twenty_four * max_a * self.j_sq)
                        - self.position_from_target
                        + t_root1 * t_root1 * (jerk * (self.tf - t_root1) - self.af)
                        + self.tf * self.vf;
                    let derivative1 = -t_root1 * (self.ca_sq - self.af_sq + two * jerk * v_inner)
                        / max_a
                        + two * (self.acceleration_from_target - jerk * self.tf)
                        + max_a
                        + three * jerk * t_root1;
                    t_root1 = t_root1 - residual1 / derivative1;
                }
                let plateau1 = ((self.ca_sq - self.af_sq) / two
                    + jerk * (jerk * t_root1 * t_root1 + self.velocity_from_target))
                    / max_a;
                profile.t[0] = (-self.ca + max_a) / jerk;
                profile.t[1] = (plateau1 - max_a) / jerk;
                profile.t[2] = max_a / jerk;
                profile.t[3] = self.tf
                    - (plateau1 + self.acceleration_from_target + max_a) / jerk
                    - two * t_root1;
                profile.t[4] = t_root1;
                profile.t[5] = F::zero();
                profile.t[6] = self.af / jerk + t_root1;
                if third_order_pose::check_profile2(
                    profile,
                    SignBlock::Uddu,
                    Touched::Acc0,
                    false,
                    self.tf,
                    limits,
                ) {
                    return true;
                }
            }
        }
        {
            let part_u = twelve
                * jerk
                * (-max_a * max_a * self.velocity_from_target
                    - jerk * self.velocity_from_target_sq
                    + two * max_a * jerk * (-self.position_from_target + self.tf * self.vf));
            let mut poly2 = [F::zero(); 4];
            poly2[0] = (-two * max_a) / jerk;
            poly2[1] = -((square(self.ca + self.af - max_a) - two * self.ca * self.af) / jerk
                + two * (self.velocity_from_target - max_a * self.tf))
                / jerk;
            poly2[2] = F::zero();
            poly2[3] = (three * (self.ca_pow4 + self.af_pow4)
                - four * (self.af_cu + two * self.ca_cu) * max_a
                + six
                    * self.ca_sq
                    * (self.af_sq + max_a * max_a + two * jerk * self.velocity_from_target)
                - twelve * self.ca * max_a * (self.af_sq + two * jerk * self.velocity_from_target)
                + six
                    * self.af_sq
                    * (max_a * max_a - two * max_a * jerk * self.tf
                        + two * jerk * self.velocity_from_target)
                - part_u)
                / (twelve * self.j_sq * self.j_sq);
            let t_lower2 = next_after_zero(self.af / jerk);
            let t_upper2 = next_after_max(fmin(self.tf - max_a / jerk, max_a / jerk));
            let roots2 = solve_quartic_arr(&poly2);
            for &raw in roots2.as_slice() {
                let mut t_root2 = raw;
                if t_root2 < t_lower2 || t_root2 > t_upper2 {
                    continue;
                }
                {
                    let v_inner2 = jerk * t_root2 * t_root2 - self.velocity_from_target;
                    let residual2 = ((self.ca_sq + self.af_sq)
                        * ((eight * self.ca - six * max_a) * max_a
                            - three * (self.ca_sq + self.af_sq))
                        + four * self.af_sq * (self.af + self.ca) * max_a
                        + twelve
                            * jerk
                            * (v_inner2 * (square(self.ca - max_a) + self.af_sq)
                                + self.af_sq * self.tf * max_a
                                - jerk * v_inner2 * v_inner2))
                        / (twenty_four * max_a * self.j_sq)
                        - self.position_from_target
                        + self.tf * self.vf
                        + jerk * t_root2 * t_root2 * (t_root2 - self.tf)
                        - self.af * t_root2 * t_root2;
                    let derivative2 = t_root2
                        * ((self.ca_sq + self.af_sq - two * jerk * v_inner2) / max_a
                            + max_a
                            + three * jerk * t_root2
                            - two * (self.ca + self.af + jerk * self.tf));
                    t_root2 = t_root2 - residual2 / derivative2;
                }
                let plateau2 = ((self.ca_sq + self.af_sq) / two
                    + jerk * (self.velocity_from_target - jerk * t_root2 * t_root2))
                    / max_a;
                profile.t[0] = (-self.ca + max_a) / jerk;
                profile.t[1] = (plateau2 - max_a) / jerk;
                profile.t[2] = max_a / jerk;
                profile.t[3] =
                    self.tf - (plateau2 - self.ca - self.af + max_a) / jerk - two * t_root2;
                profile.t[4] = t_root2;
                profile.t[5] = F::zero();
                profile.t[6] = -(self.af / jerk) + t_root2;
                if third_order_pose::check_profile2(
                    profile,
                    SignBlock::Udud,
                    Touched::Acc0,
                    false,
                    self.tf,
                    limits,
                ) {
                    return true;
                }
            }
        }
        false
    }

    fn time_vel_at_root(
        &self,
        mut polynomial_root: F,
        profile: &mut Segment<F>,
        limits: &LimitsThirdPose<F>,
    ) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let six = F::from(6.0).unwrap();
        let eps_tol = F::from(1e-9).unwrap();
        let jerk = limits.jerk;
        for _ in 0..2usize {
            let t_offset_sq_arg = (self.ca_sq + self.af_sq) / (two * self.j_sq)
                + (two * self.ca * polynomial_root - self.velocity_from_target) / jerk
                + polynomial_root * polynomial_root;
            let t_offset = t_offset_sq_arg.sqrt();
            let residual = -self.position_from_target
                + self.tf * self.cv
                + t_offset * self.velocity_from_target
                - self.af * polynomial_root * polynomial_root
                + (jerk * polynomial_root * polynomial_root + two * self.ca * polynomial_root)
                    * (self.tf - polynomial_root - t_offset)
                - (self.ca_cu + two * self.af_cu + three * self.ca_sq * self.af)
                    / (six * self.j_sq)
                - (self.ca_sq * (t_offset + two * polynomial_root - self.tf)
                    + self.af_sq * t_offset
                    + two
                        * self.af
                        * (two * self.ca * polynomial_root - self.velocity_from_target))
                    / (two * jerk);
            if residual.is_nan() || residual.abs() < eps_tol {
                break;
            }
            let derivative = (self.ca + jerk * polynomial_root)
                * (two * self.tf
                    - three * (t_offset + polynomial_root)
                    - (self.ca + two * self.af) / jerk);
            polynomial_root = polynomial_root - residual / derivative;
        }
        if polynomial_root > self.tf || polynomial_root.is_nan() {
            return false;
        }
        let t_offset_final = ((self.ca_sq + self.af_sq) / (two * self.j_sq)
            + (two * self.ca * polynomial_root - self.velocity_from_target) / jerk
            + polynomial_root * polynomial_root)
            .sqrt();
        profile.t[0] = polynomial_root;
        profile.t[1] = F::zero();
        profile.t[2] = polynomial_root + self.ca / jerk;
        profile.t[3] =
            self.tf - two * (polynomial_root + t_offset_final) - (self.ca + self.af) / jerk;
        profile.t[4] = t_offset_final;
        profile.t[5] = F::zero();
        profile.t[6] = t_offset_final + self.af / jerk;
        third_order_pose::check_profile2(
            profile,
            SignBlock::Uddu,
            Touched::Vel,
            false,
            self.tf,
            limits,
        )
    }

    fn time_vel_at_root_alt(
        &self,
        mut polynomial_root: F,
        profile: &mut Segment<F>,
        limits: &LimitsThirdPose<F>,
    ) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let six = F::from(6.0).unwrap();
        let eps_tol = F::from(1e-9).unwrap();
        let jerk = limits.jerk;
        for _ in 0..3usize {
            let t_offset = ((self.af_sq - self.ca_sq) / (two * self.j_sq)
                + (self.velocity_from_target - two * self.ca * polynomial_root) / jerk
                - polynomial_root * polynomial_root)
                .sqrt();
            let residual = -self.position_from_target
                + (self.af_cu - self.ca_cu) / (six * self.j_sq)
                + self.ca_sq * (self.tf - two * polynomial_root) / (two * jerk)
                + (two * self.ca + jerk * polynomial_root)
                    * polynomial_root
                    * (self.tf - polynomial_root)
                + (jerk * t_offset - self.af) * t_offset * t_offset
                + self.tf * self.cv;
            if residual.abs() < eps_tol {
                break;
            }
            let derivative = (self.ca + jerk * polynomial_root)
                * ((two * self.af - self.ca) / jerk + two * self.tf
                    - three * (t_offset + polynomial_root));
            polynomial_root = polynomial_root - residual / derivative;
        }
        let t_offset_final = ((self.af_sq - self.ca_sq) / (two * self.j_sq)
            + (self.velocity_from_target - two * self.ca * polynomial_root) / jerk
            - polynomial_root * polynomial_root)
            .sqrt();
        profile.t[0] = polynomial_root;
        profile.t[1] = F::zero();
        profile.t[2] = polynomial_root + self.ca / jerk;
        profile.t[3] = self.tf - two * (polynomial_root + t_offset_final)
            + self.acceleration_from_target / jerk;
        profile.t[4] = t_offset_final;
        profile.t[5] = F::zero();
        profile.t[6] = t_offset_final - self.af / jerk;
        third_order_pose::check_profile2(
            profile,
            SignBlock::Udud,
            Touched::Vel,
            false,
            self.tf,
            limits,
        )
    }

    fn time_vel(&mut self, profile: &mut Segment<F>, limits: &LimitsThirdPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let eight = F::from(8.0).unwrap();
        let twelve = F::from(12.0).unwrap();
        let twenty_four = F::from(24.0).unwrap();
        let one_forty_four = F::from(144.0).unwrap();
        let sixty_four = F::from(64.0).unwrap();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        let eps = F::from(EPSILON).unwrap_or_else(F::epsilon);
        let root_tol = F::from(ROOT_TOLERANCE).unwrap();
        let t_lower = fmax(F::zero(), -self.ca / jerk);
        let t_upper = fmin((self.tf - self.ca / jerk) / two, (max_a - self.ca) / jerk);
        if self.cv.abs() < eps && self.ca.abs() < eps && self.vf.abs() < eps && self.af.abs() < eps
        {
            let mut poly_zero = [F::zero(); 4];
            poly_zero[0] = F::one();
            poly_zero[1] = -self.tf / two;
            poly_zero[2] = F::zero();
            poly_zero[3] = self.position_from_target / (two * jerk);
            let roots_zero = solve_cubic(poly_zero[0], poly_zero[1], poly_zero[2], poly_zero[3]);
            for &raw in roots_zero.as_slice() {
                let mut t_root_zero = raw;
                if t_root_zero > self.tf / four {
                    continue;
                }
                if t_root_zero > eps {
                    let residual_zero = -self.position_from_target / (jerk * t_root_zero)
                        + t_root_zero * (self.tf - two * t_root_zero);
                    let derivative_zero = two * (self.tf - three * t_root_zero);
                    t_root_zero = t_root_zero - residual_zero / derivative_zero;
                }
                profile.t[0] = t_root_zero;
                profile.t[1] = F::zero();
                profile.t[2] = t_root_zero;
                profile.t[3] = self.tf - four * t_root_zero;
                profile.t[4] = t_root_zero;
                profile.t[5] = F::zero();
                profile.t[6] = t_root_zero;
                if third_order_pose::check_profile2(
                    profile,
                    SignBlock::Uddu,
                    Touched::Vel,
                    false,
                    self.tf,
                    limits,
                ) {
                    return true;
                }
            }
        } else {
            let part_v = self.af_sq
                - two
                    * jerk
                    * (-two * self.af * self.tf
                        + jerk * self.tf_sq
                        + three * self.velocity_from_target);
            let part_w = self.af_cu
                - three * self.j_sq * self.cv_tf_offset
                - three * self.af * jerk * self.velocity_from_target;
            let part_x = self.af_pow4
                + eight * self.af_cu * jerk * self.tf
                + twelve
                    * jerk
                    * (three * jerk * self.velocity_from_target_sq
                        - self.af_sq * self.velocity_from_target
                        + two
                            * self.af
                            * jerk
                            * (self.cv_tf_offset - self.tf * self.velocity_from_target)
                        - two * self.j_sq * self.tf * self.cv_tf_offset);
            let part_y = self.ca * (self.af - jerk * self.tf);
            let part_z = jerk * (-self.acceleration_from_target + jerk * self.tf);
            let mut poly5 = [F::zero(); 6];
            poly5[0] = F::one();
            poly5[1] = (F::from(15.0).unwrap() * self.ca_sq + self.af_sq
                - F::from(16.0).unwrap() * part_y
                + two
                    * jerk
                    * (two * self.af * self.tf
                        - jerk * self.tf_sq
                        - three * self.velocity_from_target))
                / (four * part_z);
            poly5[2] = (F::from(29.0).unwrap() * self.ca_cu - two * self.af_cu
                + self.ca * (six * part_v - F::from(33.0).unwrap() * part_y))
                / (six * jerk * part_z)
                + (jerk * self.cv_tf_offset + self.af * self.velocity_from_target) / part_z;
            poly5[3] = (F::from(61.0).unwrap() * self.ca_pow4
                + two
                    * self.ca_sq
                    * (F::from(15.0).unwrap() * part_v - F::from(38.0).unwrap() * part_y)
                - F::from(16.0).unwrap() * self.ca * part_w
                + part_x)
                / (twenty_four * self.j_sq * part_z);
            poly5[4] = self.ca
                * (F::from(7.0).unwrap() * self.ca_pow4
                    + two * self.ca_sq * (three * part_v - F::from(5.0).unwrap() * part_y)
                    - four * self.ca * part_w
                    + part_x)
                / (twelve * self.j_sq * jerk * part_z);
            poly5[5] = ((self.ca_sq
                * (F::from(7.0).unwrap() * self.ca_pow4
                    + three * self.ca_sq * (three * part_v - four * part_y)
                    - eight * self.ca * part_w
                    + three * part_x)
                + self.af_pow4 * (self.af_sq - six * jerk * self.velocity_from_target))
                / (one_forty_four * self.j_sq * self.j_sq)
                + self.af_sq
                    * (three * self.velocity_from_target_sq + four * self.af * self.cv_tf_offset)
                    / (twelve * self.j_sq)
                - self.velocity_from_target
                    * (self.velocity_from_target_sq + two * self.af * self.cv_tf_offset)
                    / (two * jerk)
                - self.cv_tf_offset * self.cv_tf_offset / two)
                / part_z;
            let poly5_d1: [F; 5] = poly_monic_derivative::<F, 6, 5>(&poly5);
            let poly5_d2: [F; 4] = poly_derivative::<F, 5, 4>(&poly5_d1);
            let roots_poly5 = solve_quartic(poly5_d1[1], poly5_d1[2], poly5_d1[3], poly5_d1[4]);
            let mut t_prev = t_lower;
            for &raw in roots_poly5.as_slice() {
                let mut t_extremum = raw;
                if t_extremum >= t_upper {
                    continue;
                }
                let poly5_d1_val = poly_eval(&poly5_d1, t_extremum);
                if poly5_d1_val.abs() > root_tol {
                    t_extremum = t_extremum - poly5_d1_val / poly_eval(&poly5_d2, t_extremum);
                }
                let poly5_val = poly_eval(&poly5, t_extremum);
                if poly5_val.abs() < sixty_four * poly_eval(&poly5_d2, t_extremum).abs() * root_tol
                {
                    if self.time_vel_at_root(t_extremum, profile, limits) {
                        return true;
                    }
                } else if poly_eval(&poly5, t_prev) * poly5_val < F::zero() {
                    let r = poly_root_newton::<F, 6, 5>(&poly5, t_prev, t_extremum);
                    if self.time_vel_at_root(r, profile, limits) {
                        return true;
                    }
                }
                t_prev = t_extremum;
            }
            let poly5_at_upper = poly_eval(&poly5, t_upper);
            if poly_eval(&poly5, t_prev) * poly5_at_upper < F::zero() {
                let r = poly_root_newton::<F, 6, 5>(&poly5, t_prev, t_upper);
                if self.time_vel_at_root(r, profile, limits) {
                    return true;
                }
            } else if poly5_at_upper.abs() < eight * eps
                && self.time_vel_at_root(t_upper, profile, limits)
            {
                return true;
            }
        }
        {
            let part_v_alt = self.af_sq
                - two
                    * jerk
                    * (two * self.af * self.tf + jerk * self.tf_sq
                        - three * self.velocity_from_target);
            let part_w_alt = self.af_cu - three * self.j_sq * self.cv_tf_offset
                + three * self.af * jerk * self.velocity_from_target;
            let part_x_alt =
                two * jerk * self.tf * self.cv_tf_offset + three * self.velocity_from_target_sq;
            let part_y_alt = self.af_pow4 - eight * self.af_cu * jerk * self.tf
                + twelve
                    * jerk
                    * (jerk * part_x_alt
                        + self.af_sq * self.velocity_from_target
                        + two
                            * self.af
                            * jerk
                            * (self.cv_tf_offset - self.tf * self.velocity_from_target));
            let part_z_alt = self.af + jerk * self.tf;
            let mut poly6 = [F::zero(); 7];
            poly6[0] = F::one();
            poly6[1] = (F::from(5.0).unwrap() * self.ca - self.af) / jerk - self.tf;
            poly6[2] = (F::from(39.0).unwrap() * self.ca_sq
                - part_v_alt
                - F::from(16.0).unwrap() * self.ca * part_z_alt)
                / (four * self.j_sq);
            poly6[3] = (F::from(55.0).unwrap() * self.ca_cu
                - F::from(33.0).unwrap() * self.ca_sq * part_z_alt
                - six * self.ca * part_v_alt
                + two * part_w_alt)
                / (six * self.j_sq * jerk);
            poly6[4] = (F::from(101.0).unwrap() * self.ca_pow4
                - F::from(76.0).unwrap() * self.ca_cu * part_z_alt
                - F::from(30.0).unwrap() * self.ca_sq * part_v_alt
                + F::from(16.0).unwrap() * self.ca * part_w_alt
                + part_y_alt)
                / (twenty_four * self.j_sq * self.j_sq);
            poly6[5] = self.ca
                * (F::from(11.0).unwrap() * self.ca_pow4
                    - F::from(10.0).unwrap() * self.ca_cu * part_z_alt
                    - six * self.ca_sq * part_v_alt
                    + four * self.ca * part_w_alt
                    + part_y_alt)
                / (twelve * self.j_sq * self.j_sq * jerk);
            poly6[6] = ((F::from(11.0).unwrap() * self.ca_pow6
                - self.af_pow6
                - twelve * self.ca_pow5 * part_z_alt
                - F::from(9.0).unwrap() * self.ca_pow4 * part_v_alt
                + eight * self.ca_cu * part_w_alt
                + three * self.ca_sq * part_y_alt
                - six * self.af_pow4 * jerk * self.velocity_from_target)
                / (one_forty_four * self.j_sq * self.j_sq)
                - self.af_sq
                    * (three * self.velocity_from_target_sq + four * self.af * self.cv_tf_offset)
                    / (twelve * self.j_sq)
                - self.velocity_from_target
                    * (self.velocity_from_target_sq / two + self.af * self.cv_tf_offset)
                    / jerk
                + self.cv_tf_offset * self.cv_tf_offset / two)
                / self.j_sq;
            let poly6_d1: [F; 6] = poly_monic_derivative::<F, 7, 6>(&poly6);
            let poly6_d2: [F; 5] = poly_monic_derivative::<F, 6, 5>(&poly6_d1);
            let mut t_prev_alt = t_lower;
            let mut intervals: [(F, F); 6] = [(F::zero(), F::zero()); 6];
            let mut interval_count = 0usize;
            let roots_poly6 = solve_quartic(poly6_d2[1], poly6_d2[2], poly6_d2[3], poly6_d2[4]);
            for &raw in roots_poly6.as_slice() {
                let mut t_extremum_alt = raw;
                if t_extremum_alt >= t_upper {
                    continue;
                }
                let poly6_d2_val = poly_eval(&poly6_d2, t_extremum_alt);
                if poly6_d2_val.abs() > root_tol {
                    let poly6_d3: [F; 4] = poly_derivative::<F, 5, 4>(&poly6_d2);
                    t_extremum_alt =
                        t_extremum_alt - poly6_d2_val / poly_eval(&poly6_d3, t_extremum_alt);
                }
                if poly_eval(&poly6_d1, t_prev_alt) * poly_eval(&poly6_d1, t_extremum_alt)
                    < F::zero()
                    && interval_count < intervals.len()
                {
                    intervals[interval_count] = (t_prev_alt, t_extremum_alt);
                    interval_count += 1;
                }
                t_prev_alt = t_extremum_alt;
            }
            if poly_eval(&poly6_d1, t_prev_alt) * poly_eval(&poly6_d1, t_upper) < F::zero()
                && interval_count < intervals.len()
            {
                intervals[interval_count] = (t_prev_alt, t_upper);
                interval_count += 1;
            }
            let mut t_root_alt = t_lower;
            for i in 0..interval_count {
                let interval = intervals[i];
                let t_root_inner = poly_root_newton::<F, 6, 5>(&poly6_d1, interval.0, interval.1);
                if t_root_inner >= t_upper {
                    continue;
                }
                let poly6_val = poly_eval(&poly6, t_root_inner);
                if poly6_val.abs()
                    < sixty_four * poly_eval(&poly6_d2, t_root_inner).abs() * root_tol
                {
                    if self.time_vel_at_root_alt(t_root_inner, profile, limits) {
                        return true;
                    }
                } else if poly_eval(&poly6, t_root_alt) * poly6_val < F::zero() {
                    let r = poly_root_newton::<F, 7, 6>(&poly6, t_root_alt, t_root_inner);
                    if self.time_vel_at_root_alt(r, profile, limits) {
                        return true;
                    }
                }
                t_root_alt = t_root_inner;
            }
            if poly_eval(&poly6, t_root_alt) * poly_eval(&poly6, t_upper) < F::zero() {
                let r = poly_root_newton::<F, 7, 6>(&poly6, t_root_alt, t_upper);
                if self.time_vel_at_root_alt(r, profile, limits) {
                    return true;
                }
            }
        }
        false
    }

    fn acc0_acc1_cases(&self, profile: &mut Segment<F>, limits: &LimitsThirdPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let nine = F::from(9.0).unwrap();
        let twelve = F::from(12.0).unwrap();
        let max_a = limits.max_accel;
        let min_a = limits.min_accel;
        let eps = F::from(EPSILON).unwrap_or_else(F::epsilon);
        if self.ca.abs() < eps && self.af.abs() < eps {
            let disc_num = two * min_a * self.cv_tf_offset
                + self.velocity_from_target_sq
                + max_a
                    * (two * self.position_from_target + min_a * self.tf_sq
                        - two * self.tf * self.vf);
            let disc_den = (max_a - min_a)
                * (-min_a * self.velocity_from_target
                    + max_a * (min_a * self.tf - self.velocity_from_target));
            let j_calc = disc_den / disc_num;
            profile.t[0] = max_a / j_calc;
            profile.t[1] =
                (-two * max_a * disc_num + min_a * min_a * self.cv_vf_tf_offset) / disc_den;
            profile.t[2] = profile.t[0];
            profile.t[3] = F::zero();
            profile.t[4] = -min_a / j_calc;
            profile.t[5] = self.tf - (two * profile.t[0] + profile.t[1] + two * profile.t[4]);
            profile.t[6] = profile.t[4];
            return third_order_pose::check_profile2_jerk(
                profile,
                SignBlock::Uddu,
                Touched::Acc0Acc1,
                false,
                self.tf,
                j_calc,
                limits,
            );
        }
        {
            let disc_sqrt = four
                * (nine
                    * square(
                        (two * self.af * min_a - self.af_sq)
                            * (max_a * self.tf - self.velocity_from_target)
                            + (square(max_a - self.ca) - min_a * max_a)
                                * (min_a * self.tf - self.velocity_from_target)
                            - min_a * (max_a - min_a) * self.velocity_from_target,
                    )
                    + three
                        * self.acceleration_from_target
                        * (three
                            * (self.af + self.ca)
                            * (two * max_a * max_a - two * min_a * min_a - self.af_sq
                                + self.ca_sq)
                            + twelve * max_a * min_a * (min_a - max_a - self.af + self.ca)
                            + four
                                * (two * self.ca * self.af + two * self.af_sq - self.ca_sq)
                                * min_a
                            - four
                                * (two * self.ca * self.af + two * self.ca_sq - self.af_sq)
                                * max_a)
                        * (two * min_a * self.cv_tf_offset
                            + self.velocity_from_target * self.velocity_from_target
                            + max_a
                                * (two * self.position_from_target
                                    + self.tf * (min_a * self.tf - two * self.vf))))
                    .sqrt();
            let j_calc = (three
                * square(self.ca - max_a)
                * (self.tf * min_a - self.velocity_from_target)
                - three * square(self.af - min_a) * (self.tf * max_a - self.velocity_from_target)
                - disc_sqrt / four)
                / (six
                    * (two * self.position_from_target * (max_a - min_a)
                        + two * self.tf * (min_a * self.cv - max_a * self.vf)
                        + self.velocity_from_target * self.velocity_from_target
                        + max_a * min_a * self.tf_sq));
            profile.t[0] = (max_a - self.ca) / j_calc;
            profile.t[1] = ((square(self.ca - min_a) - square(self.af - min_a))
                / (two * (max_a - min_a))
                - (max_a - min_a))
                / j_calc
                - (self.tf * min_a - self.velocity_from_target) / (max_a - min_a);
            profile.t[2] = max_a / j_calc;
            profile.t[3] = F::zero();
            profile.t[4] = -min_a / j_calc;
            profile.t[5] = self.tf
                - (profile.t[0]
                    + profile.t[1]
                    + profile.t[2]
                    + two * profile.t[4]
                    + self.af / j_calc);
            profile.t[6] = profile.t[4] + self.af / j_calc;
            if third_order_pose::check_profile2_jerk(
                profile,
                SignBlock::Uddu,
                Touched::Acc0Acc1,
                false,
                self.tf,
                j_calc,
                limits,
            ) {
                return true;
            }
        }
        false
    }

    fn time_acc1(&self, profile: &mut Segment<F>, limits: &LimitsThirdPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let twelve = F::from(12.0).unwrap();
        let twenty_four = F::from(24.0).unwrap();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        let min_a = limits.min_accel;
        {
            let mut disc = (self.ca_pow4 + self.af_pow4 - four * self.af_cu * jerk * self.tf
                + six * self.af_sq * self.j_sq * self.tf_sq
                - four * self.ca_cu * (self.af - jerk * self.tf)
                + six * self.ca_sq * (self.af - jerk * self.tf) * (self.af - jerk * self.tf)
                + twenty_four * self.af * self.j_sq * self.cv_tf_offset
                - four
                    * self.ca
                    * (self.af_cu - three * self.af_sq * jerk * self.tf
                        + six * self.j_sq * (-self.position_from_target + self.tf * self.vf))
                - twelve
                    * self.j_sq
                    * (-self.velocity_from_target_sq + jerk * self.tf * self.cv_vf_tf_offset))
                / three;
            if disc >= F::zero() {
                disc = disc.sqrt() * copysign_one(jerk);
                let t_branch = ((self.acceleration_from_target_sq
                    - two * self.acceleration_from_target * jerk * self.tf
                    + two * disc)
                    / self.j_sq
                    + self.tf_sq)
                    .sqrt();
                profile.t[0] = -(self.acceleration_from_target_sq
                    + two * jerk * (self.ca * self.tf - self.velocity_from_target)
                    + disc)
                    / (two * jerk * (-self.acceleration_from_target + jerk * self.tf));
                profile.t[1] = F::zero();
                profile.t[2] =
                    (self.tf - t_branch) / two - self.acceleration_from_target / (two * jerk);
                profile.t[3] = F::zero();
                profile.t[4] = F::zero();
                profile.t[5] = t_branch;
                profile.t[6] = self.tf - (profile.t[0] + profile.t[2] + profile.t[5]);
                if third_order_pose::check_profile2(
                    profile,
                    SignBlock::Uddu,
                    Touched::None,
                    true,
                    self.tf,
                    limits,
                ) {
                    return true;
                }
            }
        }
        {
            let disc_sqrt = ((self.ca_pow4
                + self.af_pow4
                + four * (self.af_cu - self.ca_cu) * jerk * self.tf
                + six * self.af_sq * self.j_sq * self.tf_sq
                + six * self.ca_sq * (self.af + jerk * self.tf) * (self.af + jerk * self.tf)
                + twenty_four * self.af * self.j_sq * self.cv_tf_offset
                - four
                    * self.ca
                    * (self.ca_sq * self.af
                        + self.af_cu
                        + three * self.af_sq * jerk * self.tf
                        + six * self.j_sq * (-self.position_from_target + self.tf * self.vf))
                + twelve
                    * self.j_sq
                    * (self.velocity_from_target_sq + jerk * self.tf * self.cv_vf_tf_offset))
                / three)
                .sqrt()
                * copysign_one(jerk);
            let t_branch = ((self.acceleration_from_target_sq
                + two * self.acceleration_from_target * jerk * self.tf
                + two * disc_sqrt)
                / self.j_sq
                + self.tf_sq)
                .sqrt();
            profile.t[0] = F::zero();
            profile.t[1] = F::zero();
            profile.t[2] = -(self.acceleration_from_target_sq
                + two * jerk * (self.velocity_from_target - self.ca * self.tf)
                + disc_sqrt)
                / (two * jerk * (self.acceleration_from_target + jerk * self.tf));
            profile.t[3] = F::zero();
            profile.t[4] =
                self.acceleration_from_target / (two * jerk) + (self.tf - t_branch) / two;
            profile.t[5] = t_branch;
            profile.t[6] = self.tf - (profile.t[5] + profile.t[4] + profile.t[2]);
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Udud,
                Touched::None,
                false,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        {
            let poly_b = self.ca_cu
                - self.af_cu
                - three
                    * min_a
                    * (self.ca_sq + (self.af - self.ca) * min_a - square(self.af - jerk * self.tf))
                + three * (self.af_sq + min_a * min_a) * jerk * self.tf
                + six * self.j_sq * (self.position_from_target - self.tf * self.vf);
            let disc_inner = square(self.ca - min_a)
                + square(self.af - min_a)
                + two * jerk * (min_a * self.tf - self.velocity_from_target);
            let poly_c = self.ca_pow4 + three * self.af_pow4 - four * self.ca * self.af_cu
                + two
                    * (three * min_a - two * self.ca - four * self.af)
                    * square(self.af - self.ca)
                    * min_a
                + twelve
                    * jerk
                    * ((self.ca * self.tf - self.velocity_from_target) * square(min_a - self.af)
                        + jerk
                            * (two * (self.ca - min_a) * self.position_from_target
                                + self.ca * min_a * self.tf_sq
                                + two * self.tf * (min_a * self.cv - self.ca * self.vf)
                                + self.velocity_from_target_sq));
            let disc_sqrt2 =
                (four * poly_b * poly_b - six * disc_inner * poly_c).sqrt() * copysign_one(jerk);
            let denom_acc1 = six * jerk * disc_inner;
            profile.t[0] = F::zero();
            profile.t[1] = F::zero();
            profile.t[2] = (two * poly_b + disc_sqrt2) / denom_acc1;
            profile.t[3] = ((square(self.ca - min_a) + square(self.af - min_a)) / (two * jerk)
                + min_a * self.tf
                - self.velocity_from_target)
                / (min_a - self.ca + jerk * profile.t[2]);
            profile.t[4] = (self.ca - min_a) / jerk - profile.t[2];
            profile.t[5] =
                self.tf - (profile.t[2] + profile.t[3] + profile.t[4] + (self.af - min_a) / jerk);
            profile.t[6] = (self.af - min_a) / jerk;
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::Acc1,
                false,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        {
            let poly_b_max = two
                * (-self.ca_cu + self.af_cu + three * (self.ca_sq - self.af_sq) * max_a
                    - three * self.acceleration_from_target * max_a * max_a
                    - six * self.af * max_a * jerk * self.tf
                    + three * self.af_sq * jerk * self.tf
                    + three
                        * jerk
                        * (max_a * max_a * self.tf
                            + jerk
                                * (-two * self.position_from_target - max_a * self.tf_sq
                                    + two * self.tf * self.vf)))
                / jerk;
            let disc_inner_max = six
                * (self.ca_sq - self.af_sq
                    + two * self.acceleration_from_target * max_a
                    + two * jerk * (max_a * self.tf - self.velocity_from_target));
            let poly_c_max = (self.ca_pow4 + three * self.af_pow4
                - four * (self.ca_cu + two * self.af_cu) * max_a
                + six * self.ca_sq * max_a * max_a
                - twenty_four * self.af * max_a * jerk * self.velocity_from_target
                + twelve
                    * jerk
                    * (two * max_a * jerk * self.cv_tf_offset
                        + jerk * self.velocity_from_target_sq
                        + max_a * max_a * self.velocity_from_target)
                + six * self.af_sq * (max_a * max_a + two * jerk * self.velocity_from_target)
                - four
                    * self.ca
                    * (self.af_cu + three * self.af * max_a * (max_a - two * jerk * self.tf)
                        - three * self.af_sq * (max_a - jerk * self.tf)
                        + three
                            * jerk
                            * (max_a * max_a * self.tf
                                + jerk
                                    * (-two * self.position_from_target - max_a * self.tf_sq
                                        + two * self.tf * self.vf))))
                / self.j_sq;
            let disc_sqrt_max = (poly_b_max * poly_b_max - disc_inner_max * poly_c_max).sqrt();
            profile.t[0] = F::zero();
            profile.t[1] = F::zero();
            profile.t[2] = -(poly_b_max + disc_sqrt_max) / disc_inner_max;
            profile.t[3] = two * disc_sqrt_max / disc_inner_max;
            profile.t[4] = (max_a - self.ca) / jerk + profile.t[2];
            profile.t[5] =
                self.tf - (profile.t[2] + profile.t[3] + profile.t[4] + (-self.af + max_a) / jerk);
            profile.t[6] = (-self.af + max_a) / jerk;
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Udud,
                Touched::Acc1,
                false,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        false
    }

    fn calculate_up(&self, profile: &mut Segment<F>, limits: &LimitsThirdPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let nine = F::from(9.0).unwrap();
        let eighteen = F::from(18.0).unwrap();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        {
            let t_branch = ((self.acceleration_from_target_sq / two
                - self.acceleration_from_target * (max_a - self.ca))
                / self.j_sq
                + (max_a * self.tf - self.velocity_from_target) / jerk)
                .sqrt();
            profile.t[0] = (-self.ca + max_a) / jerk;
            profile.t[1] = self.tf - self.acceleration_from_target / jerk - two * t_branch;
            profile.t[2] = t_branch;
            profile.t[3] = F::zero();
            profile.t[4] = (self.af - max_a) / jerk + t_branch;
            profile.t[5] = F::zero();
            profile.t[6] = F::zero();
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Udud,
                Touched::Acc0,
                false,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        {
            let inner = square(self.af - max_a) - square(self.ca - max_a)
                + two * jerk * (max_a * self.tf - self.velocity_from_target);
            let outer = self.ca_cu + two * self.af_cu
                - six * self.af_sq * max_a
                - three * self.ca_sq * (self.af - jerk * self.tf)
                - three * self.ca * max_a * (max_a - two * self.af + two * jerk * self.tf)
                - three
                    * jerk
                    * (jerk
                        * (-two * self.position_from_target
                            + max_a * self.tf_sq
                            + two * self.tf * self.cv)
                        + max_a * (max_a * self.tf - two * self.velocity_from_target))
                + three
                    * self.af
                    * (max_a * max_a + two * max_a * jerk * self.tf
                        - two * jerk * self.velocity_from_target);
            let disc_sqrt =
                (four * outer * outer - eighteen * inner * inner * inner).sqrt() * jerk.abs();
            let denom = three * jerk * inner;
            profile.t[0] = (-self.ca + max_a) / jerk;
            profile.t[1] = (-self.ca_cu
                + self.af_cu
                + self.af_sq * (-six * max_a + three * jerk * self.tf)
                + self.ca_sq * (-three * self.af + six * max_a + three * jerk * self.tf)
                + six * self.af * (max_a * max_a - jerk * self.velocity_from_target)
                + three
                    * self.ca
                    * (self.af_sq - two * (max_a * max_a + jerk * self.velocity_from_target))
                - six
                    * jerk
                    * (max_a * (max_a * self.tf - two * self.velocity_from_target)
                        + jerk * self.cv_vf_tf_offset))
                / denom;
            profile.t[2] = -(self.acceleration_from_target + disc_sqrt / denom) / (two * jerk)
                + self.tf / two
                - profile.t[1] / two;
            profile.t[3] = disc_sqrt / (jerk * denom);
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = self.tf - (profile.t[0] + profile.t[1] + profile.t[2] + profile.t[3]);
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::Acc0,
                true,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        {
            let outer3 = self.ca_cu + two * self.af_cu
                - six * (self.af_sq + max_a * max_a) * max_a
                - six * (self.ca + self.af) * max_a * jerk * self.tf
                + nine * max_a * max_a * (self.af + jerk * self.tf)
                + three * self.ca * max_a * (-two * self.af + three * max_a)
                + three * self.ca_sq * (self.af - two * max_a + jerk * self.tf)
                - six * self.j_sq * self.cv_tf_offset
                + six * (self.af - max_a) * jerk * self.velocity_from_target
                - three * max_a * self.j_sq * self.tf_sq;
            let inner3 = square(self.ca - max_a)
                + square(self.af - max_a)
                + two * jerk * (self.velocity_from_target - max_a * self.tf);
            let disc_sqrt3 = (four * outer3 * outer3 - eighteen * inner3 * inner3 * inner3).sqrt()
                * copysign_one(jerk);
            let denom3 = six * jerk * inner3;
            profile.t[0] = (-self.ca + max_a) / jerk;
            profile.t[1] = self.acceleration_from_target / jerk
                - two * profile.t[0]
                - (two * outer3 - disc_sqrt3) / denom3
                + self.tf;
            profile.t[2] = -(two * outer3 + disc_sqrt3) / denom3;
            profile.t[3] = (two * outer3 - disc_sqrt3) / denom3;
            profile.t[4] = self.tf - (profile.t[0] + profile.t[1] + profile.t[2] + profile.t[3]);
            profile.t[5] = F::zero();
            profile.t[6] = F::zero();
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::Acc0,
                true,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        false
    }

    fn calculate_down(&self, profile: &mut Segment<F>, limits: &LimitsThirdPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let eight = F::from(8.0).unwrap();
        let nine = F::from(9.0).unwrap();
        let twelve = F::from(12.0).unwrap();
        let eighteen = F::from(18.0).unwrap();
        let twenty_four = F::from(24.0).unwrap();
        let thirty_six = F::from(36.0).unwrap();
        let forty_eight = F::from(48.0).unwrap();
        let seventy_two = F::from(72.0).unwrap();
        let jerk = limits.jerk;
        let max_a = limits.max_accel;
        let min_a = limits.min_accel;
        let eps = F::from(EPSILON).unwrap_or_else(F::epsilon);
        if self.ca.abs() < eps && self.af.abs() < eps {
            if self.cv.abs() < eps {
                let disc_sqrt = (self.tf_sq * self.vf_sq
                    + square(four * self.position_from_target - self.tf * self.vf))
                .sqrt();
                let j_calc = four
                    * (four * self.position_from_target - two * self.tf * self.vf + disc_sqrt)
                    / self.tf_cu;
                profile.t[0] = self.tf / four;
                profile.t[1] = F::zero();
                profile.t[2] = two * profile.t[0];
                profile.t[3] = F::zero();
                profile.t[4] = F::zero();
                profile.t[5] = F::zero();
                profile.t[6] = profile.t[0];
                if third_order_pose::check_profile2_jerk(
                    profile,
                    SignBlock::Uddu,
                    Touched::None,
                    false,
                    self.tf,
                    j_calc,
                    limits,
                ) {
                    return true;
                }
            }
            {
                let mut poly_z = [F::zero(); 4];
                poly_z[0] = -two * self.tf;
                poly_z[1] = two * self.velocity_from_target / jerk + self.tf_sq;
                poly_z[2] = four * (self.position_from_target - self.tf * self.vf) / jerk;
                poly_z[3] = (self.velocity_from_target_sq + jerk * self.tf * self.cv_vf_tf_offset)
                    / self.j_sq;
                let roots_z = solve_quartic_arr(&poly_z);
                for &raw in roots_z.as_slice() {
                    let mut t_root_z = raw;
                    if t_root_z > self.tf / two || t_root_z > (max_a - self.ca) / jerk {
                        continue;
                    }
                    {
                        let plateau_z = (jerk * t_root_z * (t_root_z - self.tf)
                            + self.velocity_from_target)
                            / (jerk * (two * t_root_z - self.tf));
                        let d_plateau = (two * jerk * t_root_z * (t_root_z - self.tf)
                            + jerk * self.tf_sq
                            - two * self.velocity_from_target)
                            / (jerk * (two * t_root_z - self.tf) * (two * t_root_z - self.tf));
                        let residual_z = (-two * self.position_from_target
                            + two * self.tf * self.cv
                            + plateau_z * plateau_z * jerk * (self.tf - two * t_root_z)
                            + jerk
                                * self.tf
                                * (two * plateau_z * t_root_z
                                    - t_root_z * t_root_z
                                    - (plateau_z - t_root_z) * self.tf))
                            / two;
                        let derivative_z = (jerk
                            * self.tf
                            * (two * t_root_z - self.tf)
                            * (d_plateau - F::one()))
                            / two
                            + plateau_z
                                * jerk
                                * (self.tf - (two * t_root_z - self.tf) * d_plateau - plateau_z);
                        t_root_z = t_root_z - residual_z / derivative_z;
                    }
                    profile.t[0] = t_root_z;
                    profile.t[1] = F::zero();
                    profile.t[2] = (jerk * t_root_z * (t_root_z - self.tf)
                        + self.velocity_from_target)
                        / (jerk * (two * t_root_z - self.tf));
                    profile.t[3] = self.tf - two * t_root_z;
                    profile.t[4] = t_root_z - profile.t[2];
                    profile.t[5] = F::zero();
                    profile.t[6] = F::zero();
                    if third_order_pose::check_profile2(
                        profile,
                        SignBlock::Uddu,
                        Touched::None,
                        false,
                        self.tf,
                        limits,
                    ) {
                        return true;
                    }
                }
            }
        }
        {
            let outer1 = self.ca_cu - self.af_cu
                + three * self.ca * self.af * self.acceleration_from_target
                - three * self.acceleration_from_target_sq * jerk * self.tf
                + three
                    * self.j_sq
                    * (eight * (self.position_from_target - self.tf * self.vf)
                        + (self.ca + three * self.af + jerk * self.tf) * self.tf_sq);
            let disc_sqrt1 = (two
                * (two * outer1 * outer1
                    - three
                        * (self.acceleration_from_target_sq
                            - two * (self.ca + self.af) * jerk * self.tf
                            - jerk * (jerk * self.tf_sq - four * self.velocity_from_target))
                        * (self.ca_pow4
                            + self.af_pow4
                            + four * self.af_cu * jerk * self.tf
                            + six * self.af_sq * self.j_sq * self.tf_sq
                            - three * self.j_sq * self.j_sq * self.tf_sq * self.tf_sq
                            - four * self.ca_cu * (self.af + jerk * self.tf)
                            + six * self.ca_sq * square(self.af + jerk * self.tf)
                            - twelve
                                * self.af
                                * self.j_sq
                                * (eight * self.position_from_target
                                    + jerk * self.tf_sq * self.tf
                                    - eight * self.tf * self.cv)
                            + forty_eight * self.j_sq * self.velocity_from_target_sq
                            + forty_eight * self.j_sq * jerk * self.tf * self.cv_vf_tf_offset
                            - four
                                * self.ca
                                * (self.af_cu + three * self.af_sq * jerk * self.tf
                                    - nine * self.af * self.j_sq * self.tf_sq
                                    - three
                                        * self.j_sq
                                        * (eight * self.position_from_target
                                            + jerk * self.tf_sq * self.tf
                                            - eight * self.tf * self.vf)))))
                .sqrt()
                * copysign_one(jerk);
            let denom1 = twelve
                * jerk
                * (-self.acceleration_from_target_sq
                    + two * (self.ca + self.af) * jerk * self.tf
                    + self.j_sq * self.tf_sq
                    - four * self.velocity_from_target * jerk);
            let part1 = four
                * (self.acceleration_from_target * self.acceleration_from_target_sq
                    + three
                        * jerk
                        * self.acceleration_from_target
                        * (two * self.velocity_from_target - self.tf * (self.af + self.ca))
                    + six
                        * self.j_sq
                        * (two * self.position_from_target - self.tf * (self.vf + self.cv)));
            profile.t[0] = (-two
                * self.acceleration_from_target_sq
                * (self.acceleration_from_target + three * jerk * self.tf)
                + six
                    * self.j_sq
                    * ((self.ca + three * self.af + jerk * self.tf) * self.tf_sq
                        + eight * (self.position_from_target - self.tf * self.vf))
                - disc_sqrt1)
                / denom1;
            profile.t[1] = F::zero();
            profile.t[2] = (part1 + disc_sqrt1) / denom1;
            profile.t[3] = F::zero();
            profile.t[4] = (-part1 + disc_sqrt1) / denom1;
            profile.t[5] = F::zero();
            profile.t[6] = self.tf - (profile.t[0] + profile.t[2] + profile.t[4]);
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Udud,
                Touched::None,
                true,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        {
            {
                let a_meet = self.af + jerk * self.tf;
                let mut poly_a = [F::zero(); 4];
                poly_a[0] = -two * (self.acceleration_from_target / jerk + self.tf);
                poly_a[1] = two
                    * (self.ca_sq
                        + self.af_sq
                        + jerk * (self.af * self.tf + self.velocity_from_target)
                        - two * self.ca * a_meet)
                    / self.j_sq
                    + self.tf_sq;
                poly_a[2] = two
                    * (self.ca_cu - self.af_cu - three * self.af_sq * jerk * self.tf
                        + three * self.ca * a_meet * (a_meet - self.ca)
                        - six * self.j_sq * (-self.position_from_target + self.tf * self.vf))
                    / (three * self.j_sq * jerk);
                poly_a[3] = (self.ca_pow4 + self.af_pow4 + four * self.af_cu * jerk * self.tf
                    - four * self.ca_cu * a_meet
                    + six * self.ca_sq * a_meet * a_meet
                    + twenty_four * self.j_sq * self.af * self.cv_tf_offset
                    - four
                        * self.ca
                        * (self.af_cu
                            + three * self.af_sq * jerk * self.tf
                            + six * self.j_sq * (-self.position_from_target + self.tf * self.vf))
                    + six * self.j_sq * self.af_sq * self.tf_sq
                    + twelve
                        * self.j_sq
                        * (self.velocity_from_target_sq + jerk * self.tf * self.cv_vf_tf_offset))
                    / (twelve * self.j_sq * self.j_sq);
                let t_lower_a = next_after_zero(self.acceleration_from_target / jerk);
                let t_upper_a = next_after_max(fmin(
                    (max_a - self.ca) / jerk,
                    (self.acceleration_from_target / jerk + self.tf) / two,
                ));
                let roots_a = solve_quartic_arr(&poly_a);
                for &raw in roots_a.as_slice() {
                    let mut t_root_a = raw;
                    if t_root_a < t_lower_a || t_root_a > t_upper_a {
                        continue;
                    }
                    {
                        let delta_j =
                            jerk * (two * t_root_a - self.tf) - self.acceleration_from_target;
                        let plateau_a = (self.acceleration_from_target_sq
                            - two * self.af * jerk * t_root_a
                            + two * self.ca * jerk * (t_root_a - self.tf)
                            + two
                                * jerk
                                * (jerk * t_root_a * (t_root_a - self.tf)
                                    + self.velocity_from_target))
                            / (two * jerk * delta_j);
                        let d_plateau_a = (-self.acceleration_from_target_sq
                            + two * self.j_sq * (self.tf_sq + t_root_a * (t_root_a - self.tf))
                            + (self.ca + self.af) * jerk * self.tf
                            - self.acceleration_from_target * delta_j
                            - two * jerk * self.velocity_from_target)
                            / (delta_j * delta_j);
                        let residual_a = (-self.ca_cu
                            + self.af_cu
                            + three
                                * self.acceleration_from_target_sq
                                * jerk
                                * (plateau_a - t_root_a)
                            + three
                                * self.acceleration_from_target
                                * self.j_sq
                                * (plateau_a - t_root_a)
                                * (plateau_a - t_root_a)
                            - three * self.ca * self.af * self.acceleration_from_target
                            + three
                                * self.j_sq
                                * (self.ca * self.tf_sq - two * self.position_from_target
                                    + two * self.tf * self.cv
                                    + plateau_a * plateau_a * jerk * (self.tf - two * t_root_a)
                                    + jerk
                                        * self.tf
                                        * (two * plateau_a * t_root_a
                                            - t_root_a * t_root_a
                                            - (plateau_a - t_root_a) * self.tf)))
                            / (six * self.j_sq);
                        let derivative_a = (delta_j
                            * (-self.acceleration_from_target + jerk * self.tf)
                            * (d_plateau_a - F::one()))
                            / (two * jerk)
                            + plateau_a
                                * (-self.acceleration_from_target + jerk * (self.tf - plateau_a)
                                    - delta_j * d_plateau_a);
                        t_root_a = t_root_a - residual_a / derivative_a;
                    }
                    profile.t[0] = t_root_a;
                    profile.t[1] = F::zero();
                    profile.t[2] = (self.acceleration_from_target_sq
                        + two
                            * jerk
                            * (-self.ca * self.tf - self.acceleration_from_target * t_root_a
                                + jerk * t_root_a * (t_root_a - self.tf)
                                + self.velocity_from_target))
                        / (two
                            * jerk
                            * (-self.acceleration_from_target + jerk * (two * t_root_a - self.tf)));
                    profile.t[3] = self.acceleration_from_target / jerk + self.tf - two * t_root_a;
                    profile.t[4] = self.tf - (t_root_a + profile.t[2] + profile.t[3]);
                    profile.t[5] = F::zero();
                    profile.t[6] = F::zero();
                    if third_order_pose::check_profile2(
                        profile,
                        SignBlock::Uddu,
                        Touched::None,
                        true,
                        self.tf,
                        limits,
                    ) {
                        return true;
                    }
                }
            }
            {
                let denom_b = three
                    * jerk
                    * (self.acceleration_from_target_sq
                        + two * jerk * (self.ca * self.tf - self.velocity_from_target));
                let inner_b = self.acceleration_from_target_sq
                    + two * jerk * (self.ca * self.tf - self.velocity_from_target);
                let disc_sqrt_b = (four
                    * square(
                        two * (self.ca_cu - self.af_cu)
                            - six * self.ca_sq * (self.af - jerk * self.tf)
                            + six * self.j_sq * self.cv_tf_offset
                            + three
                                * self.ca
                                * (two * self.af_sq - two * jerk * self.af * self.tf
                                    + self.j_sq * self.tf_sq)
                            + six
                                * self.acceleration_from_target
                                * jerk
                                * self.velocity_from_target,
                    )
                    - eighteen * inner_b * inner_b * inner_b)
                    .sqrt()
                    / denom_b
                    * copysign_one(jerk);
                profile.t[0] = F::zero();
                profile.t[1] = F::zero();
                profile.t[2] = F::zero();
                profile.t[3] = (self.af_cu - self.ca_cu
                    + three * (self.af_sq - self.ca_sq) * jerk * self.tf
                    - three
                        * self.acceleration_from_target
                        * (self.ca * self.af + two * jerk * self.velocity_from_target)
                    - six * self.j_sq * self.cv_vf_tf_offset)
                    / denom_b;
                profile.t[4] = (self.tf - profile.t[3] - disc_sqrt_b) / two
                    - self.acceleration_from_target / (two * jerk);
                profile.t[5] = disc_sqrt_b;
                profile.t[6] = (self.tf - profile.t[3] + self.acceleration_from_target / jerk
                    - disc_sqrt_b)
                    / two;
                if third_order_pose::check_profile2(
                    profile,
                    SignBlock::Uddu,
                    Touched::None,
                    true,
                    self.tf,
                    limits,
                ) {
                    return true;
                }
            }
            {
                let inner_c = self.acceleration_from_target_sq
                    + two * (self.af + self.ca) * jerk * self.tf
                    - jerk * (jerk * self.tf_sq + four * self.velocity_from_target);
                let part_c1 = jerk * self.tf_sq * self.cv_tf_offset
                    - self.velocity_from_target
                        * (-two * self.position_from_target - self.tf * self.cv
                            + three * self.tf * self.vf);
                let part_c2 = F::from(5.0).unwrap() * self.af_sq - eight * self.af * jerk * self.tf
                    + two * jerk * (two * jerk * self.tf_sq - self.velocity_from_target);
                let part_c3 = self.j_sq * self.tf_pow4 - two * self.velocity_from_target_sq
                    + eight * jerk * self.tf * (-self.position_from_target + self.tf * self.vf);
                let part_c4 = F::from(5.0).unwrap() * self.af_pow4
                    - eight * self.af_cu * jerk * self.tf
                    - twelve * self.af_sq * jerk * (jerk * self.tf_sq + self.velocity_from_target)
                    + twenty_four
                        * self.af
                        * self.j_sq
                        * (-two * self.position_from_target
                            + jerk * self.tf_cu
                            + two * self.tf * self.vf)
                    - six * self.j_sq * part_c3;
                let part_c5 = -self.velocity_from_target_sq
                    + jerk
                        * self.tf
                        * (-two * self.position_from_target + three * self.tf * self.cv
                            - self.tf * self.vf)
                    - self.af * self.cv_vf_tf_offset;
                let mut poly_c = [F::zero(); 4];
                poly_c[0] = -(four * (self.ca_cu - self.af_cu)
                    + twelve
                        * self.acceleration_from_target
                        * (self.ca * self.af
                            + jerk * (self.velocity_from_target - self.ca * self.tf))
                    + six
                        * self.j_sq
                        * (self.tf_sq * (self.ca + three * self.af)
                            + self.tf * (two * self.cv - six * self.vf)
                            + four * self.position_from_target
                            - jerk * self.tf_cu))
                    / (three * jerk * inner_c);
                poly_c[1] = -(-self.ca_pow4 - self.af_pow4
                    + four * self.ca_cu * (self.af - jerk * self.tf)
                    + self.ca_sq
                        * (-six * self.af_sq + eight * self.af * jerk * self.tf
                            - four * jerk * (jerk * self.tf_sq - self.velocity_from_target))
                    + two
                        * self.af_sq
                        * jerk
                        * (jerk * self.tf_sq + two * self.velocity_from_target)
                    - four
                        * self.af
                        * self.j_sq
                        * (-three * self.position_from_target
                            + jerk * self.tf_cu
                            + two * self.tf * self.cv
                            + self.tf * self.vf)
                    + self.j_sq
                        * (self.j_sq * self.tf_pow4 - eight * self.velocity_from_target_sq
                            + four
                                * jerk
                                * self.tf
                                * (-three * self.position_from_target
                                    + self.tf * self.cv
                                    + two * self.tf * self.vf))
                    + two
                        * self.ca
                        * (two * self.af_cu - two * self.af_sq * jerk * self.tf
                            + self.af
                                * jerk
                                * (-three * jerk * self.tf_sq - four * self.velocity_from_target)
                            + self.j_sq
                                * (-six * self.position_from_target + jerk * self.tf_cu
                                    - four * self.tf * self.cv
                                    + F::from(10.0).unwrap() * self.tf * self.vf)))
                    / (self.j_sq * inner_c);
                poly_c[2] = -(self.ca_pow5 - self.af_pow5 + self.af_pow4 * jerk * self.tf
                    - F::from(5.0).unwrap() * self.ca_pow4 * (self.af - jerk * self.tf)
                    + two * self.ca_cu * part_c2
                    + four * self.af_cu * jerk * (jerk * self.tf_sq + self.velocity_from_target)
                    + twelve * self.j_sq * self.af * part_c5
                    - two
                        * self.ca_sq
                        * (F::from(5.0).unwrap() * self.af_cu
                            - nine * self.af_sq * jerk * self.tf
                            - six * self.af * jerk * self.velocity_from_target
                            + six
                                * self.j_sq
                                * (-two * self.position_from_target - self.tf * self.cv
                                    + three * self.tf * self.vf))
                    - twelve * self.j_sq * jerk * part_c1
                    + self.ca * part_c4)
                    / (three * self.j_sq * jerk * inner_c);
                poly_c[3] = -(-self.ca_pow6 - self.af_pow6
                    + six * self.ca_pow5 * (self.af - jerk * self.tf)
                    - forty_eight * self.af_cu * self.j_sq * self.cv_tf_offset
                    + seventy_two
                        * self.j_sq
                        * jerk
                        * (jerk * self.cv_tf_offset * self.cv_tf_offset
                            + self.velocity_from_target_sq * self.velocity_from_target
                            + two * self.af * self.cv_tf_offset * self.velocity_from_target)
                    - three * self.ca_pow4 * part_c2
                    - thirty_six * self.af_sq * self.j_sq * self.velocity_from_target_sq
                    + six * self.af_pow4 * jerk * self.velocity_from_target
                    + four
                        * self.ca_cu
                        * (F::from(5.0).unwrap() * self.af_cu
                            - nine * self.af_sq * jerk * self.tf
                            - six * self.af * jerk * self.velocity_from_target
                            + six
                                * self.j_sq
                                * (-two * self.position_from_target - self.tf * self.cv
                                    + three * self.tf * self.vf))
                    - three * self.ca_sq * part_c4
                    + six
                        * self.ca
                        * (self.af_pow5
                            - self.af_pow4 * jerk * self.tf
                            - four
                                * self.af_cu
                                * jerk
                                * (jerk * self.tf_sq + self.velocity_from_target)
                            + twelve * self.j_sq * (-self.af * part_c5 + jerk * part_c1)))
                    / (eighteen * self.j_sq * self.j_sq * inner_c);
                let t_upper_c = next_after_max((self.ca - min_a) / jerk);
                let roots_c = solve_quartic_arr(&poly_c);
                for &raw in roots_c.as_slice() {
                    let mut t_root_c = raw;
                    if t_root_c > t_upper_c {
                        continue;
                    }
                    {
                        let inner_root = self.acceleration_from_target_sq / two
                            + jerk
                                * (self.af * t_root_c
                                    + (jerk * t_root_c - self.ca) * (t_root_c - self.tf)
                                    - self.velocity_from_target);
                        let delta_j_c =
                            -self.acceleration_from_target + jerk * (self.tf - two * t_root_c);
                        let g_sqrt = inner_root.sqrt();
                        let residual_c = (self.af_cu
                            - self.ca_cu
                            - three
                                * self.acceleration_from_target
                                * (self.ca * self.af + two * inner_root)
                            + three
                                * jerk
                                * t_root_c
                                * (self.acceleration_from_target_sq - two * inner_root)
                            - six * g_sqrt * g_sqrt * g_sqrt * copysign_one(jerk))
                            / (six * self.j_sq)
                            - self.position_from_target
                            + (jerk * t_root_c * (t_root_c - self.tf) * self.tf
                                + two * self.tf * self.cv
                                - self.ca * (t_root_c * t_root_c - self.tf_sq)
                                + self.af * t_root_c * t_root_c)
                                / two;
                        let derivative_c = (six * delta_j_c * g_sqrt * jerk / jerk.abs()
                            + two * (-self.acceleration_from_target - jerk * self.tf) * delta_j_c
                            - two
                                * (three * self.acceleration_from_target_sq
                                    + self.af * jerk * (eight * t_root_c - two * self.tf)
                                    + four * self.ca * jerk * (-two * t_root_c + self.tf)
                                    + two
                                        * jerk
                                        * (jerk * t_root_c * (three * t_root_c - two * self.tf)
                                            - self.velocity_from_target)))
                            / (four * jerk);
                        t_root_c = t_root_c - residual_c / derivative_c;
                    }
                    let t_offset_c = (two
                        * square(self.acceleration_from_target / jerk + t_root_c)
                        + two * t_root_c * (t_root_c - two * self.tf)
                        + four * (self.ca * self.tf - self.velocity_from_target) / jerk)
                        .sqrt();
                    profile.t[0] = F::zero();
                    profile.t[1] = F::zero();
                    profile.t[2] = t_root_c;
                    profile.t[3] = self.tf
                        - two * t_root_c
                        - self.acceleration_from_target / jerk
                        - t_offset_c;
                    profile.t[4] = t_offset_c / two;
                    profile.t[5] = F::zero();
                    profile.t[6] = self.tf - (t_root_c + profile.t[3] + profile.t[4]);
                    if third_order_pose::check_profile2(
                        profile,
                        SignBlock::Uddu,
                        Touched::None,
                        true,
                        self.tf,
                        limits,
                    ) {
                        return true;
                    }
                }
            }
            {
                let part_d1 = -two * self.position_from_target - self.tf * self.cv
                    + three * self.tf * self.vf;
                let delta_d = -self.acceleration_from_target + jerk * self.tf;
                let part_d2 =
                    jerk * self.tf_sq * self.cv_tf_offset - self.velocity_from_target * part_d1;
                let part_d3 = F::from(5.0).unwrap() * self.af_sq
                    + two
                        * jerk
                        * (two * jerk * self.tf_sq
                            - self.velocity_from_target
                            - four * self.af * self.tf);
                let part_d4 = self.j_sq * self.tf_pow4 - two * self.velocity_from_target_sq
                    + eight * jerk * self.tf * (-self.position_from_target + self.tf * self.vf);
                let part_d5 = F::from(5.0).unwrap() * self.af_pow4
                    - eight * self.af_cu * jerk * self.tf
                    - twelve * self.af_sq * jerk * (jerk * self.tf_sq + self.velocity_from_target)
                    + twenty_four
                        * self.af
                        * self.j_sq
                        * (-two * self.position_from_target
                            + jerk * self.tf_cu
                            + two * self.tf * self.vf)
                    - six * self.j_sq * part_d4;
                let part_d6 = -self.velocity_from_target_sq
                    + jerk
                        * self.tf
                        * (-two * self.position_from_target + three * self.tf * self.cv
                            - self.tf * self.vf);
                let denom_d = three * self.j_sq * delta_d * delta_d;
                let mut poly_d = [F::zero(); 4];
                poly_d[0] = (four * self.af * self.tf
                    - two * jerk * self.tf_sq
                    - four * self.velocity_from_target)
                    / delta_d;
                poly_d[1] = (-two * (self.ca_pow4 + self.af_pow4)
                    + eight * self.af_cu * jerk * self.tf
                    + six * self.af_sq * self.j_sq * self.tf_sq
                    + eight * self.ca_cu * (self.af - jerk * self.tf)
                    - twelve
                        * self.ca_sq
                        * (self.af - jerk * self.tf)
                        * (self.af - jerk * self.tf)
                    - twelve
                        * self.af
                        * self.j_sq
                        * (-self.position_from_target + jerk * self.tf_cu
                            - two * self.tf * self.cv
                            + three * self.tf * self.vf)
                    + two
                        * self.ca
                        * (four * self.af_cu - twelve * self.af_sq * jerk * self.tf
                            + nine * self.af * self.j_sq * self.tf_sq
                            - three
                                * self.j_sq
                                * (two * self.position_from_target + jerk * self.tf_cu
                                    - two * self.tf * self.vf))
                    + three
                        * self.j_sq
                        * (self.j_sq * self.tf_pow4 + four * self.velocity_from_target_sq
                            - four
                                * jerk
                                * self.tf
                                * (self.position_from_target + self.tf * self.cv
                                    - two * self.tf * self.vf)))
                    / denom_d;
                poly_d[2] = (-self.ca_pow5 + self.af_pow5 - self.af_pow4 * jerk * self.tf
                    + F::from(5.0).unwrap() * self.ca_pow4 * (self.af - jerk * self.tf)
                    - two * self.ca_cu * part_d3
                    - four * self.af_cu * jerk * (jerk * self.tf_sq + self.velocity_from_target)
                    + twelve * self.af_sq * self.j_sq * self.cv_vf_tf_offset
                    - twelve * self.af * self.j_sq * part_d6
                    + two
                        * self.ca_sq
                        * (F::from(5.0).unwrap() * self.af_cu
                            - nine * self.af_sq * jerk * self.tf
                            - six * self.af * jerk * self.velocity_from_target
                            + six * self.j_sq * part_d1)
                    + twelve * self.j_sq * jerk * part_d2
                    + self.ca
                        * (-F::from(5.0).unwrap() * self.af_pow4
                            + eight * self.af_cu * jerk * self.tf
                            + twelve
                                * self.af_sq
                                * jerk
                                * (jerk * self.tf_sq + self.velocity_from_target)
                            - twenty_four
                                * self.af
                                * self.j_sq
                                * (-two * self.position_from_target
                                    + jerk * self.tf_cu
                                    + two * self.tf * self.vf)
                            + six * self.j_sq * part_d4))
                    / (jerk * denom_d);
                poly_d[3] = -(self.ca_pow6 + self.af_pow6
                    - six * self.ca_pow5 * (self.af - jerk * self.tf)
                    + forty_eight * self.af_cu * self.j_sq * self.cv_tf_offset
                    - seventy_two
                        * self.j_sq
                        * jerk
                        * (jerk * self.cv_tf_offset * self.cv_tf_offset
                            + self.velocity_from_target_sq * self.velocity_from_target
                            + two * self.af * self.cv_tf_offset * self.velocity_from_target)
                    + three * self.ca_pow4 * part_d3
                    - six * self.af_pow4 * jerk * self.velocity_from_target
                    + thirty_six * self.af_sq * self.j_sq * self.velocity_from_target_sq
                    - four
                        * self.ca_cu
                        * (F::from(5.0).unwrap() * self.af_cu
                            - nine * self.af_sq * jerk * self.tf
                            - six * self.af * jerk * self.velocity_from_target
                            + six * self.j_sq * part_d1)
                    + three * self.ca_sq * part_d5
                    - six
                        * self.ca
                        * (self.af_pow5
                            - self.af_pow4 * jerk * self.tf
                            - four
                                * self.af_cu
                                * jerk
                                * (jerk * self.tf_sq + self.velocity_from_target)
                            + twelve
                                * self.j_sq
                                * (self.af_sq * self.cv_vf_tf_offset - self.af * part_d6
                                    + jerk * part_d2)))
                    / (six * self.j_sq * denom_d);
                let roots_d = solve_quartic_arr(&poly_d);
                for &raw in roots_d.as_slice() {
                    let t_root_d = raw;
                    if t_root_d > self.tf || t_root_d > (max_a - self.ca) / jerk {
                        continue;
                    }
                    let t_offset_d = (self.acceleration_from_target_sq / (two * self.j_sq)
                        + (self.ca * (t_root_d + self.tf) - self.af * t_root_d
                            + jerk * t_root_d * self.tf
                            - self.velocity_from_target)
                            / jerk)
                        .sqrt();
                    profile.t[0] = t_root_d;
                    profile.t[1] =
                        self.tf - self.acceleration_from_target / jerk - two * t_offset_d;
                    profile.t[2] = t_offset_d;
                    profile.t[3] = F::zero();
                    profile.t[4] = self.acceleration_from_target / jerk + t_offset_d - t_root_d;
                    profile.t[5] = F::zero();
                    profile.t[6] = F::zero();
                    if third_order_pose::check_profile2(
                        profile,
                        SignBlock::Udud,
                        Touched::None,
                        true,
                        self.tf,
                        limits,
                    ) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn time_none(&self, profile: &mut Segment<F>, limits: &LimitsThirdPose<F>) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let ten = F::from(10.0).unwrap();
        let twelve = F::from(12.0).unwrap();
        let twenty_four = F::from(24.0).unwrap();
        let jerk = limits.jerk;
        {
            let inner_arg = -self.acceleration_from_target_sq
                + jerk
                    * (two * (self.ca + self.af) * self.tf - four * self.velocity_from_target
                        + jerk * self.tf_sq);
            let inner_clamped = fmax(inner_arg, F::zero());
            let t_inner = inner_clamped.sqrt() / jerk.abs();
            profile.t[0] = (self.tf - t_inner + self.acceleration_from_target / jerk) / two;
            profile.t[1] = t_inner;
            profile.t[2] = (self.tf - t_inner - self.acceleration_from_target / jerk) / two;
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = F::zero();
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::None,
                true,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        {
            let mut poly_cubic = [F::zero(); 4];
            poly_cubic[0] = self.acceleration_from_target_sq;
            poly_cubic[1] = self.acceleration_from_target_sq * self.tf;
            poly_cubic[2] = (self.ca_sq + self.af_sq + ten * self.ca * self.af) * self.tf_sq
                + twenty_four
                    * (self.tf * (self.af * self.cv - self.ca * self.vf)
                        - self.position_from_target * self.acceleration_from_target)
                + twelve * self.velocity_from_target_sq;
            poly_cubic[3] = -three
                * self.tf
                * ((self.ca_sq + self.af_sq + two * self.ca * self.af) * self.tf_sq
                    - four * self.velocity_from_target * (self.ca + self.af) * self.tf
                    + four * self.velocity_from_target_sq);
            let roots_cubic =
                solve_cubic(poly_cubic[0], poly_cubic[1], poly_cubic[2], poly_cubic[3]);
            for &raw in roots_cubic.as_slice() {
                let t_root_cubic = raw;
                if t_root_cubic > self.tf {
                    continue;
                }
                let a_calc = self.acceleration_from_target / (self.tf - t_root_cubic);
                profile.t[0] = (two * (self.velocity_from_target - self.ca * self.tf)
                    + self.acceleration_from_target * (t_root_cubic - self.tf))
                    / (two * a_calc * t_root_cubic);
                profile.t[1] = t_root_cubic;
                profile.t[2] = F::zero();
                profile.t[3] = F::zero();
                profile.t[4] = F::zero();
                profile.t[5] = F::zero();
                profile.t[6] = self.tf - (profile.t[0] + profile.t[1]);
                if third_order_pose::check_profile2_jerk(
                    profile,
                    SignBlock::Uddu,
                    Touched::None,
                    true,
                    self.tf,
                    a_calc,
                    limits,
                ) {
                    return true;
                }
            }
        }
        {
            profile.t[0] = (self.acceleration_from_target_sq / jerk
                + two * (self.ca + self.af) * self.tf
                - jerk * self.tf_sq
                - four * self.velocity_from_target)
                / (four * (self.acceleration_from_target - jerk * self.tf));
            profile.t[1] = F::zero();
            profile.t[2] = -self.acceleration_from_target / (two * jerk) + self.tf / two;
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = self.tf - (profile.t[0] + profile.t[2]);
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::None,
                true,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        false
    }

    fn time_single_inflection(
        &self,
        profile: &mut Segment<F>,
        limits: &LimitsThirdPose<F>,
    ) -> bool {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let twelve = F::from(12.0).unwrap();
        let twenty_four = F::from(24.0).unwrap();
        let eighteen = F::from(18.0).unwrap();
        let jerk = limits.jerk;
        {
            let inner1 = self.acceleration_from_target_sq
                + two * jerk * (self.ca * self.tf - self.velocity_from_target);
            let outer1 = two * (self.ca_cu - self.af_cu)
                - six * self.ca_sq * (self.af - jerk * self.tf)
                + six * self.j_sq * (-self.position_from_target + self.tf * self.cv)
                + six * self.ca * self.af_sq
                + three * self.ca * jerk * (jerk * self.tf_sq - two * self.velocity_from_target)
                + six * self.af * jerk * (self.velocity_from_target - self.tf * self.ca);
            let disc_sqrt1 = (four * outer1 * outer1 - eighteen * inner1 * inner1 * inner1).sqrt()
                * copysign_one(jerk);
            profile.t[0] = F::zero();
            profile.t[1] =
                (-self.ca_cu + self.af_cu + three * (self.af_sq - self.ca_sq) * jerk * self.tf
                    - three * self.ca * self.af * self.acceleration_from_target
                    - six * jerk * self.acceleration_from_target * self.velocity_from_target
                    - six
                        * self.j_sq
                        * (-two * self.position_from_target + self.tf * (self.cv + self.vf)))
                    / (three * jerk * inner1);
            profile.t[2] = (four * (self.ca_cu - self.af_cu)
                + six * self.j_sq * self.ca * self.tf_sq
                + twelve * self.ca * self.af * self.acceleration_from_target
                + twelve
                    * jerk
                    * (jerk * (self.tf * self.cv - self.position_from_target)
                        + self.acceleration_from_target
                            * (self.velocity_from_target - self.ca * self.tf))
                - disc_sqrt1)
                / (six * jerk * inner1);
            profile.t[3] = disc_sqrt1 / (three * jerk * inner1);
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = self.tf - (profile.t[1] + profile.t[2] + profile.t[3]);
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::None,
                true,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        {
            let inner2 = self.acceleration_from_target_sq
                + two * jerk * (self.velocity_from_target - self.af * self.tf);
            let part_w2 = self.af_cu
                - three
                    * self.j_sq
                    * (self.af * self.tf_sq
                        + two * (self.position_from_target - self.tf * self.vf));
            let outer2 =
                self.ca_cu + three * self.ca * self.af * self.acceleration_from_target - part_w2;
            let disc_sqrt2 = (four * outer2 * outer2
                - six
                    * inner2
                    * (self.ca_pow4 + self.af_pow4 - four * self.ca_cu * self.af
                        + six * self.ca_sq * self.af_sq
                        + twelve
                            * self.j_sq
                            * (self.velocity_from_target_sq
                                - two
                                    * self.af
                                    * (self.position_from_target - self.tf * self.cv))
                        - four * self.ca * part_w2))
                .sqrt()
                * copysign_one(jerk);
            profile.t[0] = -(two * outer2 + disc_sqrt2) / (six * jerk * inner2);
            profile.t[1] = disc_sqrt2 / (three * jerk * inner2);
            profile.t[2] = profile.t[0] - (self.af - self.ca) / jerk;
            profile.t[3] = F::zero();
            profile.t[4] = F::zero();
            profile.t[5] = self.tf - (profile.t[0] + profile.t[1] + profile.t[2]);
            profile.t[6] = F::zero();
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::None,
                true,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        {
            let disc_sqrt3 = ((self.ca_pow4 + self.af_pow4 - four * self.af_cu * jerk * self.tf
                + six * self.af_sq * self.j_sq * self.tf_sq
                - four * self.ca_cu * (self.af - jerk * self.tf)
                + six * self.ca_sq * (self.af - jerk * self.tf) * (self.af - jerk * self.tf)
                + twenty_four
                    * self.af
                    * self.j_sq
                    * (-self.position_from_target + self.tf * self.cv)
                - four
                    * self.ca
                    * (self.af_cu - three * self.af_sq * jerk * self.tf
                        + six * self.j_sq * (-self.position_from_target + self.tf * self.vf))
                - twelve
                    * self.j_sq
                    * (-self.velocity_from_target_sq
                        + jerk
                            * self.tf
                            * (-two * self.position_from_target + self.tf * (self.cv + self.vf))))
                / three)
                .sqrt()
                * copysign_one(jerk);
            let t_branch3 = (square(self.acceleration_from_target / jerk - self.tf)
                - two * disc_sqrt3 / self.j_sq)
                .sqrt();
            profile.t[0] = (-self.acceleration_from_target_sq
                + two * jerk * (self.velocity_from_target - self.ca * self.tf)
                + disc_sqrt3)
                / (two * jerk * (-self.acceleration_from_target + jerk * self.tf));
            profile.t[1] = F::zero();
            profile.t[2] = (self.tf - self.acceleration_from_target / jerk - t_branch3) / two;
            profile.t[3] = t_branch3;
            profile.t[4] = F::zero();
            profile.t[5] = F::zero();
            profile.t[6] = self.tf - (profile.t[0] + profile.t[2] + profile.t[3]);
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::None,
                true,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        {
            let outer4 = six
                * (self.acceleration_from_target_sq + two * self.af * jerk * self.tf
                    - two * jerk * self.velocity_from_target);
            let inner4 = two
                * (self.ca_cu - self.af_cu
                    + three * self.ca * self.af * self.acceleration_from_target
                    + six * self.j_sq * (self.position_from_target - self.tf * self.vf)
                    + three * self.j_sq * self.af * self.tf_sq);
            let disc_sqrt4 = (inner4 * inner4
                - outer4
                    * (self.ca_pow4 - four * self.ca_cu * self.af
                        + six * self.ca_sq * self.af_sq
                        + self.af_pow4
                        + twenty_four
                            * self.af
                            * self.j_sq
                            * (-self.position_from_target + self.tf * self.cv)
                        + twelve * self.j_sq * self.velocity_from_target_sq
                        - four
                            * self.ca
                            * (self.af_cu - three * self.af * self.j_sq * self.tf_sq
                                + six
                                    * self.j_sq
                                    * (-self.position_from_target + self.tf * self.vf))))
                .sqrt()
                * copysign_one(jerk);
            let part_h4 = four * self.ca_cu - four * self.af_cu
                + twelve * self.ca * self.af * self.acceleration_from_target
                - twelve * self.j_sq * (self.position_from_target - self.tf * self.vf)
                - six * self.j_sq * self.af * self.tf_sq
                + twelve
                    * self.acceleration_from_target
                    * jerk
                    * (self.velocity_from_target - self.af * self.tf);
            let denom4 = jerk * outer4;
            profile.t[0] = F::zero();
            profile.t[1] = F::zero();
            profile.t[2] = (inner4 + disc_sqrt4) / denom4;
            profile.t[3] = -(part_h4 + disc_sqrt4) / denom4;
            profile.t[4] = (part_h4 - disc_sqrt4) / denom4;
            profile.t[5] = self.tf - (profile.t[2] + profile.t[3] + profile.t[4]);
            profile.t[6] = F::zero();
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::None,
                true,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        {
            let disc_sqrt5 = ((self.ca_pow4 + self.af_pow4 - four * self.af_cu * jerk * self.tf
                + six * self.af_sq * self.j_sq * self.tf_sq
                - four * self.ca_cu * (self.af - jerk * self.tf)
                + six * self.ca_sq * (self.af - jerk * self.tf) * (self.af - jerk * self.tf)
                + twenty_four
                    * self.af
                    * self.j_sq
                    * (-self.position_from_target + self.tf * self.cv)
                - four
                    * self.ca
                    * (self.af_cu - three * self.af_sq * jerk * self.tf
                        + six * self.j_sq * (-self.position_from_target + self.tf * self.vf))
                - twelve
                    * self.j_sq
                    * (-self.velocity_from_target_sq
                        + jerk
                            * self.tf
                            * (-two * self.position_from_target + self.tf * (self.cv + self.vf))))
                / three)
                .sqrt()
                * copysign_one(jerk);
            let t_branch5 = (self.acceleration_from_target_sq
                - two * self.acceleration_from_target * jerk * self.tf
                + self.j_sq * self.tf_sq
                + two * disc_sqrt5)
                .sqrt()
                * copysign_one(jerk);
            profile.t[0] = -(self.acceleration_from_target_sq
                + two * jerk * (self.ca * self.tf - self.velocity_from_target)
                + disc_sqrt5)
                / (two * jerk * (-self.acceleration_from_target + jerk * self.tf));
            profile.t[1] = F::zero();
            profile.t[2] = F::zero();
            profile.t[3] = F::zero();
            profile.t[4] =
                (-self.acceleration_from_target + jerk * self.tf - t_branch5) / (two * jerk);
            profile.t[5] = t_branch5 / jerk;
            profile.t[6] = self.tf - (profile.t[0] + profile.t[4] + profile.t[5]);
            if third_order_pose::check_profile2(
                profile,
                SignBlock::Uddu,
                Touched::None,
                true,
                self.tf,
                limits,
            ) {
                return true;
            }
        }
        false
    }

    pub fn get_profile(&mut self, profile: &mut Segment<F>) -> bool {
        let limits = if self.position_from_target > self.tf * self.cv {
            self.limits
        } else {
            self.limits.inverse()
        };
        let inv_limits = limits.inverse();
        profile.scale_initial(F::one(), self.time_scale);
        if self.single_inflection_enabled
            && (self.time_single_inflection(profile, &limits)
                || self.time_single_inflection(profile, &inv_limits))
        {
            profile.scale(F::one(), F::one() / self.time_scale);
            return true;
        }
        let found = self.time_acc0_acc1_a(profile, &limits)
            || self.time_vel(profile, &limits)
            || self.time_acc0(profile, &limits)
            || self.time_acc1_vel(profile, &limits)
            || self.time_acc0_acc1_a(profile, &inv_limits)
            || self.time_vel(profile, &inv_limits)
            || self.time_acc0(profile, &inv_limits)
            || self.time_acc1_vel(profile, &inv_limits)
            || self.acc0_acc1_cases(profile, &limits)
            || self.calculate_up(profile, &limits)
            || self.time_acc1(profile, &limits)
            || self.calculate_down(profile, &limits)
            || self.acc0_acc1_cases(profile, &inv_limits)
            || self.calculate_up(profile, &inv_limits)
            || self.time_acc1(profile, &inv_limits)
            || self.calculate_down(profile, &inv_limits)
            || self.time_none(profile, &limits)
            || self.time_none(profile, &inv_limits);
        profile.scale(F::one(), F::one() / self.time_scale);
        found
    }
}
