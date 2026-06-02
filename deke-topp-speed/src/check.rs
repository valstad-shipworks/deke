//! Per-order segment validators.
//!
//! Each validator confirms that an analytically constructed segment satisfies
//! the kinematic limits and boundary conditions to within numerical tolerance.
//! The segment is integrated forward from its initial state, the section-wise
//! jerk/acceleration values are recomputed, and the resulting trajectory is
//! compared against the supplied limits and the prescribed final state.
//!
//! In the original C++ formulation each validator was templated on the sign
//! pattern (`ControlSigns`), the touched-limit pattern (`ReachedLimits`) and a
//! boolean (`check_pf`). Here those template parameters are accepted as
//! runtime arguments so the call sites can dispatch dynamically.

use num_traits::Float;

use crate::kin_state::{
    LimitsFirstPose, LimitsSecondPose, LimitsSecondVel, LimitsThirdPose, LimitsThirdVel,
};
use crate::segment::{Segment, SignBlock, Sweep, Touched};

/// Returns the trio of numerical tolerances used throughout this module.
#[inline]
fn tolerances<F: Float>() -> (F, F, F) {
    let eps12 = F::from(1e-12).unwrap();
    let eps10 = F::from(1e-10).unwrap();
    let eps8 = F::from(1e-8).unwrap();
    (eps12, eps10, eps8)
}

#[inline]
fn half<F: Float>() -> F {
    F::from(0.5).unwrap()
}

#[inline]
fn one_sixth<F: Float>() -> F {
    F::one() / F::from(6.0).unwrap()
}

pub mod first_order_pose {
    use super::*;

    /// Validate a first-order pose segment. Only the section-3 time matters;
    /// all jerks and most accelerations are clamped to zero. The segment is
    /// considered valid when the integrated terminal position equals `pf`
    /// to within tolerance.
    pub fn check_profile<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        cruise_velocity: F,
    ) -> bool {
        let (_, _, eps8) = tolerances::<F>();

        if profile.t[3] < F::zero() {
            return false;
        }
        profile.duration = profile.t[3];

        for i in 0..7 {
            profile.j[i] = F::zero();
        }
        for i in 0..7 {
            profile.a[i] = F::zero();
        }
        profile.a[7] = profile.af;

        profile.v[0] = F::zero();
        profile.v[1] = F::zero();
        profile.v[2] = F::zero();
        profile.v[3] = if profile.t[3] > F::zero() {
            cruise_velocity
        } else {
            F::zero()
        };
        profile.v[4] = F::zero();
        profile.v[5] = F::zero();
        profile.v[6] = F::zero();
        profile.v[7] = profile.vf;

        let h = half::<F>();
        for step in 0..7 {
            profile.p[step + 1] = profile.p[step]
                + profile.t[step] * (profile.v[step] + profile.t[step] * profile.a[step] * h);
        }

        profile.sign_block = signs;
        profile.touched = touched;
        profile.sweep = if cruise_velocity > F::zero() {
            Sweep::Up
        } else {
            Sweep::Down
        };

        (profile.p[7] - profile.pf).abs() < eps8
    }

    /// Variant that screens `cruise_velocity` against the supplied velocity
    /// envelope before validating the trajectory.
    pub fn check_profile2<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        _tf: F,
        cruise_velocity: F,
        limits: &LimitsFirstPose<F>,
    ) -> bool {
        let (eps12, _, _) = tolerances::<F>();
        if !(limits.min_vel - eps12 < cruise_velocity) {
            return false;
        }
        if !(cruise_velocity < limits.max_vel + eps12) {
            return false;
        }
        check_profile(profile, signs, touched, cruise_velocity)
    }
}

pub mod second_order_pose {
    use super::*;

    /// Validate a second-order pose segment with explicit
    /// `max_accel`/`min_accel`. Integrates the constant-acceleration sections
    /// (jerk is held at zero) and confirms the integrated terminal position
    /// and velocity match the prescribed final state while the intermediate
    /// velocities stay inside the velocity envelope.
    pub fn check_profile<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        max_accel: F,
        min_accel: F,
        limits: &LimitsSecondPose<F>,
    ) -> bool {
        let (eps12, _, eps8) = tolerances::<F>();
        let h = half::<F>();

        if profile.t[0] < F::zero() {
            return false;
        }
        profile.duration = profile.t[0];
        for step in 0..6 {
            if profile.t[step + 1] < F::zero() {
                return false;
            }
            profile.duration = profile.duration + profile.t[step + 1];
        }

        for i in 0..7 {
            profile.j[i] = F::zero();
        }

        profile.a[0] = if profile.t[0] > F::zero() {
            max_accel
        } else {
            F::zero()
        };
        profile.a[1] = F::zero();
        profile.a[2] = if profile.t[2] > F::zero() {
            min_accel
        } else {
            F::zero()
        };
        profile.a[3] = F::zero();
        match signs {
            SignBlock::Uddu => {
                profile.a[4] = if profile.t[4] > F::zero() {
                    min_accel
                } else {
                    F::zero()
                };
                profile.a[5] = F::zero();
                profile.a[6] = if profile.t[6] > F::zero() {
                    max_accel
                } else {
                    F::zero()
                };
            }
            SignBlock::Udud => {
                profile.a[4] = if profile.t[4] > F::zero() {
                    max_accel
                } else {
                    F::zero()
                };
                profile.a[5] = F::zero();
                profile.a[6] = if profile.t[6] > F::zero() {
                    min_accel
                } else {
                    F::zero()
                };
            }
        }
        profile.a[7] = profile.af;

        profile.sweep = if limits.max_vel > F::zero() {
            Sweep::Up
        } else {
            Sweep::Down
        };
        let vel_max = match profile.sweep {
            Sweep::Up => limits.max_vel,
            Sweep::Down => limits.min_vel,
        } + eps12;
        let vel_min = match profile.sweep {
            Sweep::Up => limits.min_vel,
            Sweep::Down => limits.max_vel,
        } - eps12;

        for step in 0..7 {
            profile.v[step + 1] = profile.v[step] + profile.t[step] * profile.a[step];
            profile.p[step + 1] = profile.p[step]
                + profile.t[step] * (profile.v[step] + profile.t[step] * profile.a[step] * h);
        }

        profile.sign_block = signs;
        profile.touched = touched;

        if (profile.p[7] - profile.pf).abs() >= eps8 {
            return false;
        }
        if (profile.v[7] - profile.vf).abs() >= eps8 {
            return false;
        }
        for k in 2..=6 {
            if profile.v[k] > vel_max {
                return false;
            }
            if profile.v[k] < vel_min {
                return false;
            }
        }
        true
    }

    /// `check_profile2` overload that pulls the per-section accelerations
    /// from `limits.max_accel` / `limits.min_accel`.
    #[inline]
    pub fn check_profile2<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        _tf: F,
        limits: &LimitsSecondPose<F>,
    ) -> bool {
        check_profile(
            profile,
            signs,
            touched,
            limits.max_accel,
            limits.min_accel,
            limits,
        )
    }

    /// `check_profile2` overload that screens the two explicit per-section
    /// accelerations against the supplied limits.
    pub fn check_profile2_accels<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        _tf: F,
        accel0: F,
        accel1: F,
        limits: &LimitsSecondPose<F>,
    ) -> bool {
        let (eps12, _, _) = tolerances::<F>();
        if !(limits.min_accel - eps12 < accel0) {
            return false;
        }
        if !(accel0 < limits.max_accel + eps12) {
            return false;
        }
        if !(limits.min_accel - eps12 < accel1) {
            return false;
        }
        if !(accel1 < limits.max_accel + eps12) {
            return false;
        }
        check_profile(profile, signs, touched, accel0, accel1, limits)
    }
}

pub mod third_order_pose {
    use super::*;

    /// Validate a third-order pose segment, drawing the jerk magnitude from
    /// `limits.jerk`.
    #[inline]
    pub fn check_profile<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        check_pf: bool,
        limits: &LimitsThirdPose<F>,
    ) -> bool {
        check_profile_jerk(profile, signs, touched, check_pf, limits.jerk, limits)
    }

    /// Validate a third-order pose segment with an explicit jerk magnitude.
    ///
    /// Sections that the touched-limit pattern says are clamped have their
    /// boundary accelerations overridden after integration. Sign reversals
    /// inside a section also force a check that the section-internal velocity
    /// peak stays inside the velocity envelope.
    ///
    /// `#[inline]` because every hot caller passes `signs`, `touched`, and
    /// `check_pf` as compile-time constants — inlining lets the compiler
    /// dead-strip the unused match arms.
    #[inline]
    pub fn check_profile_jerk<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        check_pf: bool,
        jerk_used: F,
        limits: &LimitsThirdPose<F>,
    ) -> bool {
        let (eps12, eps10, eps8) = tolerances::<F>();
        let h = half::<F>();
        let sixth = one_sixth::<F>();
        let dbl_eps = F::epsilon();
        let two = F::from(2.0).unwrap();

        if profile.t[0] < F::zero() || profile.t[0].is_nan() {
            return false;
        }
        profile.duration = profile.t[0];
        for step in 0..6 {
            let dt = profile.t[step + 1];
            if dt < F::zero() || dt.is_nan() {
                return false;
            }
            profile.duration = profile.duration + dt;
        }

        // Touched-limit guards that veto degenerate segment layouts.
        match touched {
            Touched::Acc0 | Touched::Acc0Acc1 if profile.t[1] == F::zero() => {
                return false;
            }
            _ => {}
        }
        match touched {
            Touched::Acc0Acc1Vel | Touched::Acc0Vel | Touched::Acc1Vel | Touched::Vel
                if profile.t[3] < dbl_eps =>
            {
                return false;
            }
            _ => {}
        }
        match touched {
            Touched::Acc1 | Touched::Acc0Acc1 if profile.t[5] == F::zero() => {
                return false;
            }
            _ => {}
        }

        // Section-wise jerk assignment.
        profile.j[0] = if profile.t[0] > F::zero() {
            jerk_used
        } else {
            F::zero()
        };
        profile.j[1] = F::zero();
        profile.j[2] = if profile.t[2] > F::zero() {
            -jerk_used
        } else {
            F::zero()
        };
        profile.j[3] = F::zero();
        match signs {
            SignBlock::Uddu => {
                profile.j[4] = if profile.t[4] > F::zero() {
                    -jerk_used
                } else {
                    F::zero()
                };
                profile.j[5] = F::zero();
                profile.j[6] = if profile.t[6] > F::zero() {
                    jerk_used
                } else {
                    F::zero()
                };
            }
            SignBlock::Udud => {
                profile.j[4] = if profile.t[4] > F::zero() {
                    jerk_used
                } else {
                    F::zero()
                };
                profile.j[5] = F::zero();
                profile.j[6] = if profile.t[6] > F::zero() {
                    -jerk_used
                } else {
                    F::zero()
                };
            }
        }

        profile.sweep = if limits.max_vel > F::zero() {
            Sweep::Up
        } else {
            Sweep::Down
        };
        let vel_max = match profile.sweep {
            Sweep::Up => limits.max_vel,
            Sweep::Down => limits.min_vel,
        } + eps12;
        let vel_min = match profile.sweep {
            Sweep::Up => limits.min_vel,
            Sweep::Down => limits.max_vel,
        } - eps12;

        // Touched-pattern override predicates evaluated once.
        let override_a3 = matches!(
            touched,
            Touched::Acc0Acc1Vel
                | Touched::Acc0Acc1
                | Touched::Acc0Vel
                | Touched::Acc1Vel
                | Touched::Vel
        );
        let override_a1 = matches!(
            touched,
            Touched::Acc0Acc1Vel | Touched::Acc0Acc1 | Touched::Acc0Vel | Touched::Acc0
        );
        let override_a5 = matches!(
            touched,
            Touched::Acc0Acc1Vel | Touched::Acc1Vel | Touched::Acc0Acc1 | Touched::Acc1
        );

        for step in 0..7 {
            profile.a[step + 1] = profile.a[step] + profile.t[step] * profile.j[step];
            profile.v[step + 1] = profile.v[step]
                + profile.t[step] * (profile.a[step] + profile.t[step] * profile.j[step] * h);
            profile.p[step + 1] = profile.p[step]
                + profile.t[step]
                    * (profile.v[step]
                        + profile.t[step]
                            * (profile.a[step] * h + profile.t[step] * profile.j[step] * sixth));

            if override_a3 && step == 2 {
                profile.a[3] = F::zero();
            }
            if override_a1 && step == 0 {
                profile.a[1] = limits.max_accel;
            }
            if override_a5 && step == 4 {
                match signs {
                    SignBlock::Uddu => {
                        profile.a[5] = limits.min_accel;
                    }
                    SignBlock::Udud => {
                        profile.a[5] = limits.max_accel;
                    }
                }
                if profile.t[3] == F::zero() && profile.t[4] == F::zero() {
                    profile.a[3] = profile.a[5];
                }
            }

            // Within-section velocity-peak guard on sign reversal of the
            // acceleration.
            if profile.a[step + 1] * profile.a[step] < -dbl_eps {
                let j_step = profile.j[step];
                let v_peak = profile.v[step] - (profile.a[step] * profile.a[step]) / (two * j_step);
                if v_peak > vel_max || v_peak < vel_min {
                    return false;
                }
            }
        }

        profile.sign_block = signs;
        profile.touched = touched;

        let accel_max = match profile.sweep {
            Sweep::Up => limits.max_accel,
            Sweep::Down => limits.min_accel,
        } + eps12;
        let accel_min = match profile.sweep {
            Sweep::Up => limits.min_accel,
            Sweep::Down => limits.max_accel,
        } - eps12;

        if (profile.p[7] - profile.pf).abs() >= eps8 {
            return false;
        }
        if (profile.v[7] - profile.vf).abs() >= eps8 {
            return false;
        }
        if check_pf && (profile.a[7] - profile.af).abs() >= eps10 {
            return false;
        }
        if profile.a[1] < accel_min || profile.a[1] > accel_max {
            return false;
        }
        if profile.a[3] < accel_min || profile.a[3] > accel_max {
            return false;
        }
        if profile.a[5] < accel_min || profile.a[5] > accel_max {
            return false;
        }
        for k in 3..=6 {
            if profile.v[k] > vel_max {
                return false;
            }
            if profile.v[k] < vel_min {
                return false;
            }
        }
        true
    }

    /// `check_profile2` overload that ignores its `tf` argument and uses
    /// `limits.jerk` for the jerk magnitude.
    #[inline]
    pub fn check_profile2<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        check_pf: bool,
        _tf: F,
        limits: &LimitsThirdPose<F>,
    ) -> bool {
        check_profile_jerk(profile, signs, touched, check_pf, limits.jerk, limits)
    }

    /// `check_profile2` overload that screens an explicit jerk against
    /// `limits.jerk` before validating.
    pub fn check_profile2_jerk<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        check_pf: bool,
        _tf: F,
        jerk_used: F,
        limits: &LimitsThirdPose<F>,
    ) -> bool {
        let (eps12, _, _) = tolerances::<F>();
        if !(jerk_used.abs() < limits.jerk.abs() + eps12) {
            return false;
        }
        check_profile_jerk(profile, signs, touched, check_pf, jerk_used, limits)
    }
}

pub mod second_order_vel {
    use super::*;

    /// Validate a second-order velocity segment. Only the section-1 time
    /// matters; the segment is valid when the integrated terminal velocity
    /// matches `vf` to within tolerance.
    pub fn check_profile<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        next_accel: F,
    ) -> bool {
        let (_, _, eps8) = tolerances::<F>();
        let h = half::<F>();

        if profile.t[1] < F::zero() {
            return false;
        }
        profile.duration = profile.t[1];

        for i in 0..7 {
            profile.j[i] = F::zero();
        }
        profile.a[0] = F::zero();
        profile.a[1] = if profile.t[1] > F::zero() {
            next_accel
        } else {
            F::zero()
        };
        for i in 2..7 {
            profile.a[i] = F::zero();
        }
        profile.a[7] = profile.af;

        for step in 0..7 {
            profile.v[step + 1] = profile.v[step] + profile.t[step] * profile.a[step];
            profile.p[step + 1] = profile.p[step]
                + profile.t[step] * (profile.v[step] + profile.t[step] * profile.a[step] * h);
        }

        profile.sign_block = signs;
        profile.touched = touched;
        profile.sweep = if next_accel > F::zero() {
            Sweep::Up
        } else {
            Sweep::Down
        };

        (profile.v[7] - profile.vf).abs() < eps8
    }

    /// Variant that screens the per-section acceleration `a_avg` against the
    /// acceleration envelope before validating.
    pub fn check_profile2<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        _tf: F,
        a_avg: F,
        limits: &LimitsSecondVel<F>,
    ) -> bool {
        let (eps12, _, _) = tolerances::<F>();
        if !(limits.min_accel - eps12 < a_avg) {
            return false;
        }
        if !(a_avg < limits.max_accel + eps12) {
            return false;
        }
        check_profile(profile, signs, touched, a_avg)
    }
}

pub mod third_order_vel {
    use super::*;

    /// Validate a third-order velocity segment with explicit jerk magnitude.
    /// The integrator runs the section-wise jerk pattern and the resulting
    /// trajectory must hit the prescribed final velocity and acceleration to
    /// within tolerance, with the per-section accelerations staying inside
    /// the acceleration envelope.
    pub fn check_profile<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        next_jerk: F,
        limits: &LimitsThirdVel<F>,
    ) -> bool {
        let (eps12, eps10, eps8) = tolerances::<F>();
        let h = half::<F>();
        let sixth = one_sixth::<F>();
        let dbl_eps = F::epsilon();

        if profile.t[0] < F::zero() {
            return false;
        }
        profile.duration = profile.t[0];
        for step in 0..6 {
            if profile.t[step + 1] < F::zero() {
                return false;
            }
            profile.duration = profile.duration + profile.t[step + 1];
        }

        if matches!(touched, Touched::Acc0) && profile.t[1] < dbl_eps {
            return false;
        }

        profile.j[0] = if profile.t[0] > F::zero() {
            next_jerk
        } else {
            F::zero()
        };
        profile.j[1] = F::zero();
        profile.j[2] = if profile.t[2] > F::zero() {
            -next_jerk
        } else {
            F::zero()
        };
        profile.j[3] = F::zero();
        match signs {
            SignBlock::Uddu => {
                profile.j[4] = if profile.t[4] > F::zero() {
                    -next_jerk
                } else {
                    F::zero()
                };
                profile.j[5] = F::zero();
                profile.j[6] = if profile.t[6] > F::zero() {
                    next_jerk
                } else {
                    F::zero()
                };
            }
            SignBlock::Udud => {
                profile.j[4] = if profile.t[4] > F::zero() {
                    next_jerk
                } else {
                    F::zero()
                };
                profile.j[5] = F::zero();
                profile.j[6] = if profile.t[6] > F::zero() {
                    -next_jerk
                } else {
                    F::zero()
                };
            }
        }

        for step in 0..7 {
            profile.a[step + 1] = profile.a[step] + profile.t[step] * profile.j[step];
            profile.v[step + 1] = profile.v[step]
                + profile.t[step] * (profile.a[step] + profile.t[step] * profile.j[step] * h);
            profile.p[step + 1] = profile.p[step]
                + profile.t[step]
                    * (profile.v[step]
                        + profile.t[step]
                            * (profile.a[step] * h + profile.t[step] * profile.j[step] * sixth));
        }

        profile.sign_block = signs;
        profile.touched = touched;
        profile.sweep = if limits.max_accel > F::zero() {
            Sweep::Up
        } else {
            Sweep::Down
        };

        let accel_max = match profile.sweep {
            Sweep::Up => limits.max_accel,
            Sweep::Down => limits.min_accel,
        } + eps12;
        let accel_min = match profile.sweep {
            Sweep::Up => limits.min_accel,
            Sweep::Down => limits.max_accel,
        } - eps12;

        if (profile.v[7] - profile.vf).abs() >= eps8 {
            return false;
        }
        if (profile.a[7] - profile.af).abs() >= eps10 {
            return false;
        }
        if profile.a[1] < accel_min || profile.a[1] > accel_max {
            return false;
        }
        if profile.a[3] < accel_min || profile.a[3] > accel_max {
            return false;
        }
        if profile.a[5] < accel_min || profile.a[5] > accel_max {
            return false;
        }
        true
    }

    /// Variant that screens an explicit jerk against `limits.jerk` before
    /// validating.
    pub fn check_profile2<F: Float>(
        profile: &mut Segment<F>,
        signs: SignBlock,
        touched: Touched,
        _tf: F,
        next_jerk: F,
        limits: &LimitsThirdVel<F>,
    ) -> bool {
        let (eps12, _, _) = tolerances::<F>();
        if !(next_jerk.abs() < limits.jerk.abs() + eps12) {
            return false;
        }
        check_profile(profile, signs, touched, next_jerk, limits)
    }
}
