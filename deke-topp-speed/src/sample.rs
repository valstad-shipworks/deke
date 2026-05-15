//! Per-tick output of the live pursuer.

use deke_types::{FKScalar, SRobotQ};

/// A single discrete sample of the produced motion at a wall-clock instant.
#[derive(Debug, Clone, Copy)]
pub struct MotionSample<const N: usize, F: FKScalar = f32> {
    pub pose: SRobotQ<N, F>,
    pub vel: SRobotQ<N, F>,
    pub accel: SRobotQ<N, F>,
    pub jerk: SRobotQ<N, F>,
    /// Time stamp of this sample, measured from the start of the current solve.
    pub t: F,
    pub section_idx: usize,
    pub crossed_section: bool,
    pub fresh_solve: bool,
    pub solve_interrupted: bool,
    pub solve_micros: f64,
}

impl<const N: usize, F: FKScalar> MotionSample<N, F> {
    pub fn zero() -> Self {
        Self {
            pose: SRobotQ::zeros(),
            vel: SRobotQ::zeros(),
            accel: SRobotQ::zeros(),
            jerk: SRobotQ::zeros(),
            t: F::zero(),
            section_idx: 0,
            crossed_section: false,
            fresh_solve: false,
            solve_interrupted: false,
            solve_micros: 0.0,
        }
    }
}

impl<const N: usize, F: FKScalar> Default for MotionSample<N, F> {
    fn default() -> Self {
        Self::zero()
    }
}

/// The kinematic state of a moving goal that the live pursuer chases.
#[derive(Debug, Clone, Copy)]
pub struct PursuitTarget<const N: usize, F: FKScalar = f32> {
    pub pose: SRobotQ<N, F>,
    pub vel: SRobotQ<N, F>,
    pub accel: SRobotQ<N, F>,
}

impl<const N: usize, F: FKScalar> PursuitTarget<N, F> {
    pub fn zero() -> Self {
        Self {
            pose: SRobotQ::zeros(),
            vel: SRobotQ::zeros(),
            accel: SRobotQ::zeros(),
        }
    }

    pub fn new(pose: SRobotQ<N, F>, vel: SRobotQ<N, F>, accel: SRobotQ<N, F>) -> Self {
        Self { pose, vel, accel }
    }

    /// Linear interpolation toward `other` by fraction `alpha`.
    pub fn interpolate(&self, other: &Self, alpha: F) -> Self {
        Self {
            pose: self.pose.interpolate(&other.pose, alpha),
            vel: self.vel.interpolate(&other.vel, alpha),
            accel: self.accel.interpolate(&other.accel, alpha),
        }
    }
}

impl<const N: usize, F: FKScalar> Default for PursuitTarget<N, F> {
    fn default() -> Self {
        Self::zero()
    }
}
