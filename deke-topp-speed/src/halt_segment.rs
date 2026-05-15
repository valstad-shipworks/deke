//! Jerk-limited stopping ramps used when the start state already exceeds
//! kinematic limits.
//!
//! A halt segment is a short (one- or two-section) trajectory that drives an
//! axis from an out-of-bounds initial state back into the feasible region in
//! one of velocity/acceleration before the main shaping math takes over.

use num_traits::Float;

use crate::kin_state::{LimitsSecondPose, LimitsThirdPose, LimitsThirdVel};

/// Numerical fudge used to keep generated ramps strictly inside the feasible
/// region. Matches the magnitude used by the reference implementation.
const EPS: f64 = 2.2e-14;

/// Up to two-section stopping ramp expressed as parallel arrays over time,
/// jerk, acceleration, velocity and position.
#[derive(Debug, Clone, Copy)]
pub struct HaltSegment<F: Float> {
    /// Total duration of the ramp (sum of `t[0]` and `t[1]`).
    pub duration: F,
    pub t: [F; 2],
    pub j: [F; 2],
    pub a: [F; 2],
    pub v: [F; 2],
    pub p: [F; 2],
}

impl<F: Float> HaltSegment<F> {
    pub fn empty() -> Self {
        Self {
            duration: F::zero(),
            t: [F::zero(); 2],
            j: [F::zero(); 2],
            a: [F::zero(); 2],
            v: [F::zero(); 2],
            p: [F::zero(); 2],
        }
    }

    /// Roll the `[0]` entries forward using the integrator and store the
    /// resulting state into the `[1]` slot. Used to finalize a two-section
    /// ramp once the first section's `t[0]` and `j[0]` are known.
    /// Finalize a two-section halt ramp by integrating the supplied pre-halt
    /// state through both sections, populating the `[0]` / `[1]` slots, and
    /// returning the post-halt state as `(p_after, v_after, a_after)`.
    ///
    /// When both sections are zero-length the ramp collapses to no-op and the
    /// returned state equals the input.
    pub fn finalize_second_order(&mut self, p0: F, v0: F, a0: F) -> (F, F, F) {
        let two = F::from(2.0).unwrap();
        let six = F::from(6.0).unwrap();
        if self.t[0] <= F::zero() && self.t[1] <= F::zero() {
            self.duration = F::zero();
            return (p0, v0, a0);
        }
        let mut p = p0;
        let mut v = v0;
        let mut a = a0;
        // Section 0.
        self.duration = self.t[0];
        self.p[0] = p;
        self.v[0] = v;
        self.a[0] = a;
        let dt0 = self.t[0];
        let j0 = self.j[0];
        let new_p = p + dt0 * (v + dt0 * (a / two + dt0 * j0 / six));
        let new_v = v + dt0 * (a + dt0 * j0 / two);
        let new_a = a + dt0 * j0;
        p = new_p;
        v = new_v;
        a = new_a;
        // Optional section 1.
        if self.t[1] > F::zero() {
            self.duration = self.duration + self.t[1];
            self.p[1] = p;
            self.v[1] = v;
            self.a[1] = a;
            let dt1 = self.t[1];
            let j1 = self.j[1];
            let after_p = p + dt1 * (v + dt1 * (a / two + dt1 * j1 / six));
            let after_v = v + dt1 * (a + dt1 * j1 / two);
            let after_a = a + dt1 * j1;
            p = after_p;
            v = after_v;
            a = after_a;
        }
        (p, v, a)
    }

    /// Scale the time and value axes of the ramp.
    pub fn scale(&mut self, position_scale: F, time_scale: F) {
        let inv_t = F::one() / time_scale;
        let inv_t2 = inv_t * inv_t;
        let inv_t3 = inv_t2 * inv_t;
        self.duration = self.duration * time_scale;
        for i in 0..2 {
            self.t[i] = self.t[i] * time_scale;
            self.j[i] = self.j[i] * position_scale * inv_t3;
            self.a[i] = self.a[i] * position_scale * inv_t2;
            self.v[i] = self.v[i] * position_scale * inv_t;
            self.p[i] = self.p[i] * position_scale;
        }
    }
}

impl<F: Float> Default for HaltSegment<F> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Convert the numerical fudge constant into the generic float type.
fn eps<F: Float>() -> F {
    F::from(EPS).unwrap_or_else(F::epsilon)
}

// ---------------------------------------------------------------------------
// Per-order ramp generators.
// ---------------------------------------------------------------------------

/// Second-order position-domain halt: drive an axis back inside its velocity
/// envelope using a single constant-acceleration ramp.
pub mod second_order_pose {
    use super::*;

    /// Build a halt ramp that brings `v0` back into `[min_vel, max_vel]` using
    /// the available acceleration extremes. Returns an empty segment when the
    /// limits are degenerate or the start velocity is already feasible.
    pub fn get_profile<F: Float>(v0: F, limits: LimitsSecondPose<F>) -> HaltSegment<F> {
        let mut profile = HaltSegment::<F>::empty();
        profile.a[0] = F::zero();
        profile.a[1] = F::zero();
        if limits.max_accel == F::zero() || limits.min_accel == F::zero() {
            return profile;
        }
        let t_eps = eps::<F>();
        if v0 > limits.max_vel {
            profile.a[0] = limits.min_accel;
            profile.t[0] = (limits.max_vel - v0) / limits.min_accel + t_eps;
        } else if v0 < limits.min_vel {
            profile.a[0] = limits.max_accel;
            profile.t[0] = (limits.min_vel - v0) / limits.max_accel + t_eps;
        }
        profile
    }
}

/// Third-order position-domain halt: drive an axis back inside its
/// velocity/acceleration envelope using up to two constant-jerk sections.
pub mod third_order_pose {
    use super::*;

    /// Velocity reached after integrating `(v, a)` forward by `dt` under
    /// constant `jerk`.
    #[inline]
    fn v_at<F: Float>(v0: F, a0: F, jerk: F, dt: F) -> F {
        let two = F::from(2.0).unwrap();
        v0 + dt * (a0 + jerk * dt / two)
    }

    /// Velocity reached after a constant-jerk ramp that drives acceleration
    /// from `a` to zero.
    #[inline]
    fn v_after_ramp<F: Float>(v: F, a: F, jerk: F) -> F {
        let two = F::from(2.0).unwrap();
        v + (a * a) / (two * jerk)
    }

    /// Acceleration-leading branch: the start acceleration is already out of
    /// the `[min_accel, max_accel]` envelope and we must first ramp it back.
    fn get_profile_from_acc<F: Float>(v0: F, a0: F, limits: LimitsThirdPose<F>) -> HaltSegment<F> {
        let mut profile = HaltSegment::<F>::empty();
        profile.j[0] = -limits.jerk;
        let t_eps = eps::<F>();
        let two = F::from(2.0).unwrap();

        let t_to_max_accel = (a0 - limits.max_accel) / limits.jerk;
        let t_to_zero_accel = a0 / limits.jerk;
        let v_at_max_accel = v_at(v0, a0, -limits.jerk, t_to_max_accel);
        let v_at_zero_accel = v_at(v0, a0, -limits.jerk, t_to_zero_accel);

        if (v_at_zero_accel > limits.max_vel && limits.jerk > F::zero())
            || (v_at_zero_accel < limits.max_vel && limits.jerk < F::zero())
        {
            return get_profile_from_vel(v0, a0, limits);
        } else if (v_at_max_accel < limits.min_vel && limits.jerk > F::zero())
            || (v_at_max_accel > limits.min_vel && limits.jerk < F::zero())
        {
            let t_to_min_vel = -(v_at_max_accel - limits.min_vel) / limits.max_accel;
            let t_const_accel = -limits.max_accel / (two * limits.jerk)
                - (v_at_max_accel - limits.max_vel) / limits.max_accel;
            profile.t[0] = t_to_max_accel + t_eps;
            let second = if t_to_min_vel < t_const_accel - t_eps {
                t_to_min_vel
            } else {
                t_const_accel - t_eps
            };
            profile.t[1] = if second > F::zero() {
                second
            } else {
                F::zero()
            };
        } else {
            profile.t[0] = t_to_max_accel + t_eps;
        }
        profile
    }

    /// Velocity-leading branch: acceleration is in-bounds but velocity is
    /// outside the envelope (or heading the wrong way).
    fn get_profile_from_vel<F: Float>(v0: F, a0: F, limits: LimitsThirdPose<F>) -> HaltSegment<F> {
        let mut profile = HaltSegment::<F>::empty();
        profile.j[0] = -limits.jerk;
        let t_eps = eps::<F>();
        let two = F::from(2.0).unwrap();

        let t_to_min_accel = (a0 - limits.min_accel) / limits.jerk;
        let abs_jerk = limits.jerk.abs();
        let t_to_max_vel = a0 / limits.jerk
            + (a0 * a0 + two * limits.jerk * (v0 - limits.max_vel)).sqrt() / abs_jerk;
        let t_to_min_vel = a0 / limits.jerk
            + (a0 * a0 / two + limits.jerk * (v0 - limits.min_vel)).sqrt() / abs_jerk;
        let t_limit = if t_to_max_vel < t_to_min_vel {
            t_to_max_vel
        } else {
            t_to_min_vel
        };

        if t_to_min_accel < t_limit {
            let v_at_min_accel = v_at(v0, a0, -limits.jerk, t_to_min_accel);
            let t_a = -(v_at_min_accel - limits.max_vel) / limits.min_accel;
            let t_b = limits.min_accel / (two * limits.jerk)
                - (v_at_min_accel - limits.min_vel) / limits.min_accel;
            let first = t_to_min_accel - t_eps;
            profile.t[0] = if first > F::zero() { first } else { F::zero() };
            let pick = if t_a < t_b { t_a } else { t_b };
            profile.t[1] = if pick > F::zero() { pick } else { F::zero() };
        } else {
            let first = t_limit - t_eps;
            profile.t[0] = if first > F::zero() { first } else { F::zero() };
        }
        profile
    }

    /// Build a halt ramp that brings `(v0, a0)` back inside the feasible
    /// envelope. Dispatches to the acceleration- or velocity-leading branch
    /// (possibly with sign-flipped limits) based on which bound is violated.
    pub fn get_profile<F: Float>(v0: F, a0: F, limits: LimitsThirdPose<F>) -> HaltSegment<F> {
        if limits.jerk == F::zero()
            || limits.max_accel == F::zero()
            || limits.min_accel == F::zero()
        {
            return HaltSegment::empty();
        }
        if a0 > limits.max_accel {
            return get_profile_from_acc(v0, a0, limits);
        } else if a0 < limits.min_accel {
            return get_profile_from_acc(v0, a0, limits.inverse());
        } else if (v0 > limits.max_vel && v_after_ramp(v0, a0, -limits.jerk) > limits.min_vel)
            || (a0 > F::zero() && v_after_ramp(v0, a0, limits.jerk) > limits.max_vel)
        {
            return get_profile_from_vel(v0, a0, limits);
        } else if (v0 < limits.min_vel && v_after_ramp(v0, a0, limits.jerk) < limits.max_vel)
            || (a0 < F::zero() && v_after_ramp(v0, a0, -limits.jerk) < limits.min_vel)
        {
            return get_profile_from_vel(v0, a0, limits.inverse());
        }
        HaltSegment::empty()
    }
}

/// Third-order velocity-domain halt: drive acceleration back inside its
/// envelope using a single constant-jerk ramp.
pub mod third_order_vel {
    use super::*;

    /// Build a halt ramp that brings `a0` back into `[min_accel, max_accel]`
    /// using the configured jerk magnitude.
    pub fn get_profile<F: Float>(a0: F, limits: LimitsThirdVel<F>) -> HaltSegment<F> {
        let mut profile = HaltSegment::<F>::empty();
        if limits.jerk == F::zero() {
            return profile;
        }
        let t_eps = eps::<F>();
        if a0 > limits.max_accel {
            profile.j[0] = -limits.jerk;
            profile.t[0] = (a0 - limits.max_accel) / limits.jerk + t_eps;
        } else if a0 < limits.min_accel {
            profile.j[0] = limits.jerk;
            profile.t[0] = -(a0 - limits.min_accel) / limits.jerk + t_eps;
        }
        profile
    }
}
