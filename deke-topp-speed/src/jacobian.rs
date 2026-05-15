//! Analytic Jacobian / gradient computation used by the multi-waypoint
//! optimiser.

use num_traits::Float;

use crate::kin_state::{KinThirdPose, LimitsThirdPose};
use crate::segment::{Segment, Sweep, Touched};

/// Per-waypoint partial derivatives of segment duration w.r.t. boundary
/// velocity and acceleration on both ends.
#[derive(Debug, Clone, Copy)]
pub struct Gradient<F: Float> {
    pub cv: F,
    pub vf: F,
    pub ca: F,
    pub af: F,
    pub scale_left: F,
    pub scale_right: F,
}

impl<F: Float> Gradient<F> {
    pub fn zero() -> Self {
        Self {
            cv: F::zero(),
            vf: F::zero(),
            ca: F::zero(),
            af: F::zero(),
            scale_left: F::one(),
            scale_right: F::one(),
        }
    }
}

impl<F: Float> Default for Gradient<F> {
    fn default() -> Self {
        Self::zero()
    }
}

#[inline]
fn square<F: Float>(x: F) -> F {
    x * x
}

#[inline]
fn copysign_one<F: Float>(x: F) -> F {
    F::one().copysign(x)
}

/// Analytic Jacobian for a single waypoint segment.
pub struct Jacobian<F: Float> {
    pub cv: F,
    pub ca: F,
    pub vf: F,
    pub af: F,
    pub reached_limits: LimitsThirdPose<F>,
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

impl<F: Float> Jacobian<F> {
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
        let cv_sq = cv * cv;
        let vf_sq = vf * vf;
        let ca_sq = ca * ca;
        let af_sq = af * af;
        let ca_cu = ca * ca_sq;
        let ca_pow4 = ca_sq * ca_sq;
        let af_cu = af * af_sq;
        let af_pow4 = af_sq * af_sq;
        let j_sq = limits.jerk * limits.jerk;
        let position_from_target = target.p - current.p;
        Self {
            cv,
            ca,
            vf,
            af,
            reached_limits: limits,
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

    pub(crate) fn for_acc0_acc1_vel(&self, limits: &LimitsThirdPose<F>) -> Gradient<F> {
        let two = F::from(2.0).unwrap();
        let denom = two * limits.jerk * limits.max_vel;
        let mut g: Gradient<F> = Gradient::zero();
        g.cv = -(square(self.ca - limits.max_accel)
            + two * limits.jerk * (limits.max_vel - self.cv))
            / (denom * limits.max_accel);
        g.vf = (square(self.af - limits.min_accel)
            + two * limits.jerk * (limits.max_vel - self.vf))
            / (denom * limits.min_accel);
        g.ca = ((self.ca - limits.max_accel)
            * (self.ca * (self.ca - limits.max_accel)
                + two * limits.jerk * (limits.max_vel - self.cv)))
            / (denom * limits.max_accel * limits.jerk);
        g.af = -((self.af - limits.min_accel)
            * (self.af * (self.af - limits.min_accel)
                + two * limits.jerk * (limits.max_vel - self.vf)))
            / (denom * limits.min_accel * limits.jerk);
        g
    }

    pub(crate) fn for_acc1_vel(&self, limits: &LimitsThirdPose<F>) -> Gradient<F> {
        let two = F::from(2.0).unwrap();
        let three_halves = F::from(1.5).unwrap();
        let half_sqrt = three_halves
            * (self.ca_sq / two + limits.jerk * (limits.max_vel - self.cv)).sqrt()
            * copysign_one(limits.jerk);
        let mut g: Gradient<F> = Gradient::zero();
        g.cv = (self.ca - half_sqrt) / (limits.jerk * limits.max_vel);
        g.vf = (square(self.af - limits.min_accel)
            + two * limits.jerk * (limits.max_vel - self.vf))
            / (two * limits.min_accel * limits.jerk * limits.max_vel);
        g.ca = (-self.ca_sq + self.ca * half_sqrt + limits.jerk * (self.cv - limits.max_vel))
            / (self.j_sq * limits.max_vel);
        g.af = -((self.af - limits.min_accel)
            * (self.af * (self.af - limits.min_accel)
                + two * limits.jerk * (limits.max_vel - self.vf)))
            / (two * limits.min_accel * self.j_sq * limits.max_vel);
        g
    }

    pub(crate) fn for_acc0_vel(&self, limits: &LimitsThirdPose<F>) -> Gradient<F> {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let sqrt_term = four * (self.af_sq / two + limits.jerk * (limits.max_vel - self.vf)).sqrt();
        let mut g: Gradient<F> = Gradient::zero();
        g.cv = -(square(self.ca - limits.max_accel)
            + two * limits.jerk * (limits.max_vel - self.cv))
            / (two * limits.max_accel * limits.jerk * limits.max_vel);
        g.vf = (three * self.af_sq + six * limits.jerk * (limits.max_vel - self.vf)
            - self.af * sqrt_term)
            / (limits.jerk * limits.max_vel * sqrt_term);
        g.ca = ((self.ca - limits.max_accel)
            * (self.ca * (self.ca - limits.max_accel)
                + two * limits.jerk * (limits.max_vel - self.cv)))
            / (two * limits.max_accel * self.j_sq * limits.max_vel);
        g.af = (-three * self.af_cu - six * self.af * limits.jerk * (limits.max_vel - self.vf)
            + self.af_sq * sqrt_term
            + limits.jerk * (limits.max_vel - self.vf) * sqrt_term)
            / (self.j_sq * limits.max_vel * sqrt_term);
        g
    }

    pub(crate) fn for_vel(&self, limits: &LimitsThirdPose<F>) -> Gradient<F> {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let three_halves = F::from(1.5).unwrap();
        let eps = F::from(1e-14).unwrap();
        let sqrt_term_af = four
            * (self.af_sq / two + limits.jerk * (-self.vf + limits.max_vel)).sqrt()
            * copysign_one(limits.jerk)
            + eps;
        let sqrt_term_ca = three_halves
            * (self.ca_sq / two + limits.jerk * (-self.cv + limits.max_vel)).sqrt()
            * copysign_one(limits.jerk);
        let mut g: Gradient<F> = Gradient::zero();
        g.cv = (self.ca - sqrt_term_ca) / (limits.jerk * limits.max_vel);
        g.vf = (-three * self.af_sq + six * limits.jerk * (self.vf - limits.max_vel))
            / (limits.jerk * limits.max_vel * sqrt_term_af)
            - self.af / (limits.jerk * limits.max_vel);
        g.ca = (-self.ca_sq + limits.jerk * (self.cv - limits.max_vel) + self.ca * sqrt_term_ca)
            / (self.j_sq * limits.max_vel);
        g.af = (three * self.af_cu - six * self.af * limits.jerk * (self.vf - limits.max_vel)
            + self.af_sq * sqrt_term_af
            + limits.jerk * (-self.vf + limits.max_vel) * sqrt_term_af)
            / (self.j_sq * limits.max_vel * sqrt_term_af);
        g
    }

    pub(crate) fn for_acc0_acc1(&self, limits: &LimitsThirdPose<F>) -> Gradient<F> {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let eight = F::from(8.0).unwrap();
        let twenty_four = F::from(24.0).unwrap();
        let inner = three
            * (limits.max_accel - limits.min_accel)
            * (three * (self.af_pow4 * limits.max_accel - self.ca_pow4 * limits.min_accel)
                + eight * (self.ca_cu - self.af_cu) * limits.max_accel * limits.min_accel
                + twenty_four
                    * limits.max_accel
                    * limits.min_accel
                    * limits.jerk
                    * (self.af * self.vf - self.ca * self.cv)
                + six
                    * self.af_sq
                    * limits.max_accel
                    * (limits.min_accel * limits.min_accel - two * limits.jerk * self.vf)
                - six
                    * self.ca_sq
                    * limits.min_accel
                    * (limits.max_accel * limits.max_accel - two * limits.jerk * self.cv)
                + three
                    * (limits.max_accel
                        * limits.max_accel
                        * limits.max_accel
                        * limits.min_accel
                        * limits.min_accel
                        - four * limits.min_accel * self.j_sq * self.cv_sq
                        - limits.max_accel
                            * limits.max_accel
                            * limits.min_accel
                            * (limits.min_accel * limits.min_accel
                                - four * limits.jerk * self.cv)
                        + four
                            * limits.max_accel
                            * limits.jerk
                            * (-two * limits.min_accel * limits.jerk * self.position_from_target
                                - limits.min_accel * limits.min_accel * self.vf
                                + limits.jerk * self.vf_sq)));
        let sqrt_term = inner.sqrt() * limits.jerk.abs()
            / (three * (limits.max_accel - limits.min_accel) * limits.jerk);
        let mut g: Gradient<F> = Gradient::zero();
        g.cv = -(F::one() / limits.max_accel)
            - (square(self.ca - limits.max_accel) - two * limits.jerk * self.cv)
                / (limits.max_accel * sqrt_term);
        g.vf = F::one() / limits.min_accel
            + (square(self.af - limits.min_accel) - two * limits.jerk * self.vf)
                / (limits.min_accel * sqrt_term);
        g.ca = -(F::one() / limits.jerk)
            + self.ca / (limits.max_accel * limits.jerk)
            + ((self.ca - limits.max_accel)
                * (self.ca * (self.ca - limits.max_accel) - two * limits.jerk * self.cv))
                / (limits.max_accel * limits.jerk * sqrt_term);
        g.af = F::one() / limits.jerk
            - self.af / (limits.min_accel * limits.jerk)
            - ((self.af - limits.min_accel)
                * (self.af * (self.af - limits.min_accel) - two * limits.jerk * self.vf))
                / (limits.min_accel * limits.jerk * sqrt_term);
        g
    }

    pub(crate) fn for_acc1(&self, polynomial_root: F, limits: &LimitsThirdPose<F>) -> Gradient<F> {
        let two = F::from(2.0).unwrap();
        let four = F::from(4.0).unwrap();
        let denom = two
            * (self.ca - limits.min_accel + limits.jerk * polynomial_root)
            * (self.ca_sq
                - limits.min_accel * limits.jerk * polynomial_root
                - self.ca * (limits.min_accel - four * limits.jerk * polynomial_root)
                + two * limits.jerk * (limits.jerk * polynomial_root * polynomial_root + self.cv));
        let d_cv = -(square(self.ca - limits.min_accel)
            + two
                * limits.jerk
                * (two * (self.ca - limits.min_accel) * polynomial_root
                    + limits.jerk * polynomial_root * polynomial_root
                    + self.cv))
            / denom;
        let d_vf = -(square(self.af - limits.min_accel) - two * limits.jerk * self.vf) / denom;
        let d_ca = -((self.ca - limits.min_accel + two * limits.jerk * polynomial_root)
            * (self.ca_sq - self.ca * (limits.min_accel - four * limits.jerk * polynomial_root)
                + two
                    * limits.jerk
                    * (-limits.min_accel * polynomial_root
                        + limits.jerk * polynomial_root * polynomial_root
                        + self.cv)))
            / denom;
        let d_af = ((self.af - limits.min_accel)
            * (self.af * (self.af - limits.min_accel) - two * limits.jerk * self.vf))
            / denom;
        let mut g: Gradient<F> = Gradient::zero();
        g.cv = (-F::one()
            + two * d_cv * (limits.min_accel - self.ca - limits.jerk * polynomial_root))
            / limits.min_accel;
        g.vf = (F::one()
            + two * d_vf * (limits.min_accel - self.ca - limits.jerk * polynomial_root))
            / limits.min_accel;
        g.ca = -(two * limits.jerk * polynomial_root * (F::one() + d_ca)
            + (self.ca - limits.min_accel) * (F::one() + two * d_ca))
            / (limits.min_accel * limits.jerk);
        g.af = (-self.af + limits.min_accel
            - two * (self.ca - limits.min_accel + limits.jerk * polynomial_root) * d_af)
            / (limits.min_accel * limits.jerk);
        g
    }

    pub(crate) fn for_acc0(&self, polynomial_root: F, limits: &LimitsThirdPose<F>) -> Gradient<F> {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let denom = -self.af_sq + limits.max_accel * limits.max_accel
            - three * limits.max_accel * limits.jerk * polynomial_root
            + two * limits.jerk * (limits.jerk * polynomial_root * polynomial_root + self.vf);
        let d_cv = -(square(self.ca - limits.max_accel) - two * limits.jerk * self.cv) / denom;
        let d_vf = (square(self.af - limits.max_accel)
            - two * limits.jerk * (limits.jerk * polynomial_root * polynomial_root + self.vf))
            / denom;
        let d_ca = ((self.ca - limits.max_accel)
            * (self.ca * (self.ca - limits.max_accel) - two * limits.jerk * self.cv))
            / denom;
        let d_af = -(self.af_cu - two * self.af_sq * limits.max_accel
            + two * limits.max_accel * limits.jerk * self.vf
            + self.af
                * (limits.max_accel * limits.max_accel
                    - two
                        * limits.jerk
                        * (limits.jerk * polynomial_root * polynomial_root + self.vf)))
            / denom;
        let mut g: Gradient<F> = Gradient::zero();
        g.cv = (-F::one() + d_cv) / limits.max_accel;
        g.vf = (F::one() + d_vf) / limits.max_accel;
        g.ca = (self.ca - limits.max_accel + d_ca) / (limits.max_accel * limits.jerk);
        g.af = (-self.af + limits.max_accel + d_af) / (limits.max_accel * limits.jerk);
        g
    }

    pub(crate) fn for_none(&self, polynomial_root: F, limits: &LimitsThirdPose<F>) -> Gradient<F> {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let eight = F::from(8.0).unwrap();
        let half = F::from(0.5).unwrap();
        let lower_clamp = F::from(0.0001).unwrap();
        let dbl_epsilon = F::from(f64::EPSILON).unwrap();
        let denom = four
            * ((self.ca_cu - self.af_cu) / three
                + limits.jerk
                    * (self.af * self.vf
                        - self.ca * self.cv
                        - (self.ca_sq + self.af_sq) * polynomial_root)
                + self.j_sq
                    * (-self.position_from_target
                        + limits.jerk * polynomial_root * polynomial_root * polynomial_root
                        + two * polynomial_root * (self.cv + self.vf)));
        if denom.abs() < dbl_epsilon {
            let mut g_default: Gradient<F> = Gradient::zero();
            g_default.cv = F::zero();
            g_default.vf = F::zero();
            g_default.ca = -F::one() / limits.jerk;
            g_default.af = F::one() / limits.jerk;
            return g_default;
        }
        let d_cv = (-self.ca_sq + self.af_sq
            - two
                * limits.jerk
                * (two * polynomial_root * (limits.jerk * polynomial_root - self.ca) - self.cv
                    + self.vf))
            / denom;
        let d_vf = (self.ca_sq
            - self.af_sq
            - two
                * limits.jerk
                * (two * polynomial_root * (limits.jerk * polynomial_root + self.af) + self.cv
                    - self.vf))
            / denom;
        let d_ca =
            -self.ca * d_cv / limits.jerk + four * limits.jerk * polynomial_root * self.cv / denom;
        let d_af =
            -self.af * d_vf / limits.jerk - four * limits.jerk * polynomial_root * self.vf / denom;
        let mut g: Gradient<F> = Gradient::zero();
        g.cv = two * d_cv;
        g.vf = two * d_vf;
        g.ca = two * d_ca - F::one() / limits.jerk;
        g.af = two * d_af + F::one() / limits.jerk;
        let t_shift = ((self.ca_sq - self.af_sq) / (two * limits.jerk) + (self.vf - self.cv))
            / (two * limits.jerk * polynomial_root);
        let t_left = t_shift + polynomial_root * half - self.ca / limits.jerk;
        let t_right = -t_shift + polynomial_root * half + self.af / limits.jerk;
        let t_total = two * polynomial_root + (self.af - self.ca) / limits.jerk;
        let t_quarter = t_total / eight;
        let _ = three;
        if t_left < t_quarter {
            let ratio = t_left / t_quarter;
            let clamped_high = if ratio > F::one() { F::one() } else { ratio };
            let clamped = if clamped_high < lower_clamp {
                lower_clamp
            } else {
                clamped_high
            };
            g.scale_left = clamped;
        }
        if t_right < t_quarter {
            let ratio = t_right / t_quarter;
            let clamped_high = if ratio > F::one() { F::one() } else { ratio };
            let clamped = if clamped_high < lower_clamp {
                lower_clamp
            } else {
                clamped_high
            };
            g.scale_right = clamped;
        }
        g
    }

    /// Dispatch on the segment's touched-limits profile.
    #[inline]
    pub fn compute(&self, profile: &Segment<F>) -> Gradient<F> {
        let limits = match profile.sweep {
            Sweep::Up => self.reached_limits,
            Sweep::Down => self.reached_limits.inverse(),
        };
        match profile.touched {
            Touched::Acc0Acc1Vel => self.for_acc0_acc1_vel(&limits),
            Touched::Acc1Vel => self.for_acc1_vel(&limits),
            Touched::Acc0Vel => self.for_acc0_vel(&limits),
            Touched::Vel => self.for_vel(&limits),
            Touched::Acc0Acc1 => self.for_acc0_acc1(&limits),
            Touched::Acc1 => self.for_acc1(profile.polynomial_root, &limits),
            Touched::Acc0 => self.for_acc0(profile.polynomial_root, &limits),
            Touched::None => self.for_none(profile.polynomial_root, &limits),
        }
    }
}
