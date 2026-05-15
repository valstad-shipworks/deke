//! Single-axis kinematic state at the start or end of a segment, plus the
//! per-axis kinematic-limit aggregates used by the shaping math.
//!
//! Three "orders" are represented:
//!
//! - First order: pose only (`KinFirstPose`).
//! - Second order: pose + velocity (`KinSecondPose`), or velocity only
//!   (`KinSecondVel`).
//! - Third order: pose + velocity + acceleration (`KinThirdPose`), or velocity
//!   + acceleration (`KinThirdVel`).

use num_traits::Float;

/// Pose-only state for first-order shaping.
#[derive(Debug, Clone, Copy)]
pub struct KinFirstPose<F: Float> {
    pub p: F,
}

impl<F: Float> KinFirstPose<F> {
    pub fn new(p: F) -> Self {
        Self { p }
    }
}

/// Pose + velocity state for second-order shaping.
#[derive(Debug, Clone, Copy)]
pub struct KinSecondPose<F: Float> {
    pub p: F,
    pub v: F,
}

impl<F: Float> KinSecondPose<F> {
    pub fn new(p: F, v: F) -> Self {
        Self { p, v }
    }
}

/// Pose + velocity + acceleration state for third-order shaping.
#[derive(Debug, Clone, Copy)]
pub struct KinThirdPose<F: Float> {
    pub p: F,
    pub v: F,
    pub a: F,
}

impl<F: Float> KinThirdPose<F> {
    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            p: F::zero(),
            v: F::zero(),
            a: F::zero(),
        }
    }

    #[inline(always)]
    pub fn new(p: F, v: F, a: F) -> Self {
        Self { p, v, a }
    }

    /// Integrate this state by `dt` under constant jerk `j`.
    #[inline(always)]
    pub fn next(&self, dt: F, j: F) -> Self {
        // 0.5 and 1/6 are exactly representable in IEEE 754, so the
        // `F::from` calls fold to constants when monomorphised for `f32`/`f64`.
        let half = F::from(0.5).unwrap();
        let six_inv = F::one() / F::from(6.0).unwrap();
        Self {
            p: self.p + dt * (self.v + dt * (self.a * half + dt * j * six_inv)),
            v: self.v + dt * (self.a + dt * j * half),
            a: self.a + dt * j,
        }
    }
}

/// Velocity-only state for second-order velocity shaping.
#[derive(Debug, Clone, Copy)]
pub struct KinSecondVel<F: Float> {
    pub v: F,
}

impl<F: Float> KinSecondVel<F> {
    pub fn new(v: F) -> Self {
        Self { v }
    }
}

/// Velocity + acceleration state for third-order velocity shaping.
#[derive(Debug, Clone, Copy)]
pub struct KinThirdVel<F: Float> {
    pub v: F,
    pub a: F,
}

impl<F: Float> KinThirdVel<F> {
    pub fn new(v: F, a: F) -> Self {
        Self { v, a }
    }
}

/// Kinematic-limit aggregate for first-order shaping.
#[derive(Debug, Clone, Copy)]
pub struct LimitsFirstPose<F: Float> {
    pub max_vel: F,
    pub min_vel: F,
}

impl<F: Float> LimitsFirstPose<F> {
    pub fn new(max_vel: F, min_vel: F) -> Self {
        Self { max_vel, min_vel }
    }
}

/// Kinematic-limit aggregate for second-order shaping.
#[derive(Debug, Clone, Copy)]
pub struct LimitsSecondPose<F: Float> {
    pub max_vel: F,
    pub min_vel: F,
    pub max_accel: F,
    pub min_accel: F,
}

impl<F: Float> LimitsSecondPose<F> {
    pub fn new(max_vel: F, min_vel: F, max_accel: F, min_accel: F) -> Self {
        Self {
            max_vel,
            min_vel,
            max_accel,
            min_accel,
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            max_vel: self.min_vel,
            min_vel: self.max_vel,
            max_accel: self.min_accel,
            min_accel: self.max_accel,
        }
    }
}

/// Kinematic-limit aggregate for third-order shaping.
#[derive(Debug, Clone, Copy)]
pub struct LimitsThirdPose<F: Float> {
    pub max_vel: F,
    pub min_vel: F,
    pub max_accel: F,
    pub min_accel: F,
    pub jerk: F,
}

impl<F: Float> LimitsThirdPose<F> {
    pub fn new(max_vel: F, min_vel: F, max_accel: F, min_accel: F, jerk: F) -> Self {
        Self {
            max_vel,
            min_vel,
            max_accel,
            min_accel,
            jerk,
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            max_vel: self.min_vel,
            min_vel: self.max_vel,
            max_accel: self.min_accel,
            min_accel: self.max_accel,
            jerk: -self.jerk,
        }
    }
}

/// Kinematic-limit aggregate for second-order velocity shaping.
#[derive(Debug, Clone, Copy)]
pub struct LimitsSecondVel<F: Float> {
    pub max_accel: F,
    pub min_accel: F,
}

impl<F: Float> LimitsSecondVel<F> {
    pub fn new(max_accel: F, min_accel: F) -> Self {
        Self {
            max_accel,
            min_accel,
        }
    }
}

/// Kinematic-limit aggregate for third-order velocity shaping.
#[derive(Debug, Clone, Copy)]
pub struct LimitsThirdVel<F: Float> {
    pub max_accel: F,
    pub min_accel: F,
    pub jerk: F,
}

impl<F: Float> LimitsThirdVel<F> {
    pub fn new(max_accel: F, min_accel: F, jerk: F) -> Self {
        Self {
            max_accel,
            min_accel,
            jerk,
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            max_accel: self.min_accel,
            min_accel: self.max_accel,
            jerk: -self.jerk,
        }
    }
}
