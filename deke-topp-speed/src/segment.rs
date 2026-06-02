//! A single-axis seven-section jerk-limited motion segment.
//!
//! Each segment is represented as parallel arrays of time durations `t[0..7]`,
//! jerk values `j[0..7]`, and the resulting acceleration / velocity / position
//! at section boundaries (`a[0..8]`, `v[0..8]`, `p[0..8]`).

use num_traits::Float;

use crate::extent::Extent;
use crate::halt_segment::HaltSegment;
use crate::kin_state::KinThirdPose;
use crate::roots::solve_cubic;

/// Which kinematic limits the segment touches.
///
/// `None` is variant 0 (zero discriminant) so a zero-initialised `Segment`
/// has the same `touched` value as `Segment::empty()`. This lets buffer
/// constructors fold into a plain memset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Touched {
    None,
    Acc0Acc1Vel,
    Vel,
    Acc0,
    Acc1,
    Acc0Acc1,
    Acc0Vel,
    Acc1Vel,
}

/// Whether the segment ramps up or down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sweep {
    Up,
    Down,
}

/// Sign pattern of the segment's jerk sub-sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignBlock {
    Uddu,
    Udud,
}

/// Seven-section jerk-limited segment with optional pre-halt and pre-accel
/// halt ramps.
#[derive(Debug, Clone, Copy)]
pub struct Segment<F: Float> {
    pub t: [F; 7],
    pub j: [F; 7],
    pub a: [F; 8],
    pub v: [F; 8],
    pub p: [F; 8],
    pub halt: HaltSegment<F>,
    pub accel_halt: HaltSegment<F>,
    pub duration: F,
    pub pf: F,
    pub vf: F,
    pub af: F,
    pub touched: Touched,
    pub sweep: Sweep,
    pub sign_block: SignBlock,
    pub polynomial_root: F,
}

impl<F: Float> Segment<F> {
    /// Construct a zero-initialised segment.
    pub fn empty() -> Self {
        Self {
            t: [F::zero(); 7],
            j: [F::zero(); 7],
            a: [F::zero(); 8],
            v: [F::zero(); 8],
            p: [F::zero(); 8],
            halt: HaltSegment::empty(),
            accel_halt: HaltSegment::empty(),
            duration: F::zero(),
            pf: F::zero(),
            vf: F::zero(),
            af: F::zero(),
            touched: Touched::None,
            sweep: Sweep::Up,
            sign_block: SignBlock::Uddu,
            polynomial_root: F::zero(),
        }
    }

    /// If `delta_time` falls strictly inside `(0, duration)`, integrate the
    /// state and update the position extrema accordingly.
    fn extend_position_extrema_at(
        delta_time: F,
        time_offset: F,
        duration: F,
        state: &KinThirdPose<F>,
        jerk: F,
        bound: &mut Extent<F>,
    ) {
        if delta_time > F::zero() && delta_time < duration {
            let next = state.next(delta_time, jerk);
            if next.a > F::zero() && next.p < bound.min {
                bound.min = next.p;
                bound.t_min = time_offset + delta_time;
            } else if next.a < F::zero() && next.p > bound.max {
                bound.max = next.p;
                bound.t_max = time_offset + delta_time;
            }
        }
    }

    /// If `delta_time` falls strictly inside `(0, duration)`, integrate the
    /// state and update the velocity extrema accordingly.
    fn extend_velocity_extrema_at(
        delta_time: F,
        time_offset: F,
        duration: F,
        state: &KinThirdPose<F>,
        jerk: F,
        bound: &mut Extent<F>,
    ) {
        if delta_time > F::zero() && delta_time < duration {
            let next = state.next(delta_time, jerk);
            if next.v < bound.min {
                bound.min = next.v;
                bound.t_min = time_offset + delta_time;
            } else if next.v > bound.max {
                bound.max = next.v;
                bound.t_max = time_offset + delta_time;
            }
        }
    }

    /// Take the position at `state` plus any internal extrema reached during
    /// the section `[time_offset, time_offset + delta_time)` and fold them
    /// into `bound`.
    fn fold_position_extrema(
        time_offset: F,
        delta_time: F,
        state: &KinThirdPose<F>,
        jerk: F,
        bound: &mut Extent<F>,
    ) {
        if state.p < bound.min {
            bound.min = state.p;
            bound.t_min = time_offset;
        }
        if state.p > bound.max {
            bound.max = state.p;
            bound.t_max = time_offset;
        }
        let eps = F::epsilon();
        let two = F::from(2.0).unwrap();
        if jerk.abs() > eps {
            let discriminant = state.a * state.a - two * jerk * state.v;
            if discriminant.abs() < eps {
                Self::extend_position_extrema_at(
                    -state.a / jerk,
                    time_offset,
                    delta_time,
                    state,
                    jerk,
                    bound,
                );
            } else if discriminant > F::zero() {
                let sqrt_disc = discriminant.sqrt();
                Self::extend_position_extrema_at(
                    (-state.a - sqrt_disc) / jerk,
                    time_offset,
                    delta_time,
                    state,
                    jerk,
                    bound,
                );
                Self::extend_position_extrema_at(
                    (-state.a + sqrt_disc) / jerk,
                    time_offset,
                    delta_time,
                    state,
                    jerk,
                    bound,
                );
            }
        } else if state.a.abs() > eps {
            Self::extend_position_extrema_at(
                -state.v / state.a,
                time_offset,
                delta_time,
                state,
                jerk,
                bound,
            );
        }
    }

    /// Take the velocity at `state` plus any internal velocity extrema reached
    /// during the section `[time_offset, time_offset + duration)` and fold
    /// them into `bound`.
    fn fold_velocity_extrema(
        time_offset: F,
        duration: F,
        state: &KinThirdPose<F>,
        jerk: F,
        bound: &mut Extent<F>,
    ) {
        if state.v < bound.min {
            bound.min = state.v;
            bound.t_min = time_offset;
        }
        if state.v > bound.max {
            bound.max = state.v;
            bound.t_max = time_offset;
        }
        if jerk.abs() > F::epsilon() {
            Self::extend_velocity_extrema_at(
                -state.a / jerk,
                time_offset,
                duration,
                state,
                jerk,
                bound,
            );
        }
    }

    /// Bound on the position reached by this segment, including the optional
    /// pre-halt ramp and the terminal `pf` value.
    pub fn get_position_extrema(&self) -> Extent<F> {
        let mut bound = Extent::point(self.p[0]);
        if self.halt.duration > F::zero() && self.halt.t[0] > F::zero() {
            let s0 = KinThirdPose::new(self.halt.p[0], self.halt.v[0], self.halt.a[0]);
            Self::fold_position_extrema(F::zero(), self.halt.t[0], &s0, self.halt.j[0], &mut bound);
            if self.halt.t[1] > F::zero() {
                let s1 = KinThirdPose::new(self.halt.p[1], self.halt.v[1], self.halt.a[1]);
                Self::fold_position_extrema(
                    self.halt.t[0],
                    self.halt.t[1],
                    &s1,
                    self.halt.j[1],
                    &mut bound,
                );
            }
        }
        let mut time_in = F::zero();
        for step in 0..7 {
            let s = KinThirdPose::new(self.p[step], self.v[step], self.a[step]);
            Self::fold_position_extrema(
                time_in + self.halt.duration,
                self.t[step],
                &s,
                self.j[step],
                &mut bound,
            );
            time_in = time_in + self.t[step];
        }
        if self.pf < bound.min {
            bound.min = self.pf;
            bound.t_min = self.duration + self.halt.duration;
        }
        if self.pf > bound.max {
            bound.max = self.pf;
            bound.t_max = self.duration + self.halt.duration;
        }
        bound
    }

    /// Bound on the velocity reached by this segment, including the optional
    /// pre-halt ramp and the terminal `vf` value.
    pub fn get_velocity_extrema(&self) -> Extent<F> {
        let mut bound = Extent::point(self.v[0]);
        if self.halt.duration > F::zero() && self.halt.t[0] > F::zero() {
            let s0 = KinThirdPose::new(self.halt.p[0], self.halt.v[0], self.halt.a[0]);
            Self::fold_velocity_extrema(F::zero(), self.halt.t[0], &s0, self.halt.j[0], &mut bound);
            if self.halt.t[1] > F::zero() {
                let s1 = KinThirdPose::new(self.halt.p[1], self.halt.v[1], self.halt.a[1]);
                Self::fold_velocity_extrema(
                    self.halt.t[0],
                    self.halt.t[1],
                    &s1,
                    self.halt.j[1],
                    &mut bound,
                );
            }
        }
        let mut time_in = F::zero();
        for step in 0..7 {
            let s = KinThirdPose::new(self.p[step], self.v[step], self.a[step]);
            Self::fold_velocity_extrema(
                time_in + self.halt.duration,
                self.t[step],
                &s,
                self.j[step],
                &mut bound,
            );
            time_in = time_in + self.t[step];
        }
        if self.vf < bound.min {
            bound.min = self.vf;
            bound.t_min = self.duration + self.halt.duration;
        }
        if self.vf > bound.max {
            bound.max = self.vf;
            bound.t_max = self.duration + self.halt.duration;
        }
        bound
    }

    /// Initialise the section-0 state and the final velocity/acceleration.
    /// The final position `pf` is set elsewhere.
    pub fn set_initial(&mut self, p0: F, v0: F, a0: F, vf: F, af: F) {
        self.a[0] = a0;
        self.v[0] = v0;
        self.p[0] = p0;
        self.af = af;
        self.vf = vf;
    }

    /// Copy boundary conditions from another segment.
    pub fn set_boundary(&mut self, other: &Self) {
        self.a[0] = other.a[0];
        self.v[0] = other.v[0];
        self.p[0] = other.p[0];
        self.af = other.af;
        self.vf = other.vf;
        self.pf = other.pf;
        self.halt = other.halt;
        self.accel_halt = other.accel_halt;
    }

    /// Initialise from explicit `(p0, v0, a0, pf, vf, af)`.
    pub fn set_boundary_explicit(&mut self, p0: F, v0: F, a0: F, pf: F, vf: F, af: F) {
        self.a[0] = a0;
        self.v[0] = v0;
        self.p[0] = p0;
        self.af = af;
        self.vf = vf;
        self.pf = pf;
    }

    /// Find the earliest time `t >= t_min` at which the segment reaches the
    /// position `value`. On success writes the time into `*out_t` and returns
    /// `true`.
    pub fn find_first_time_at_pose(&self, value: F, out_t: &mut F, t_min: F) -> bool {
        let eps = F::epsilon();
        let two = F::from(2.0).unwrap();
        let six = F::from(6.0).unwrap();
        let mut time_in = F::zero();
        for step in 0..7 {
            if self.t[step] == F::zero() {
                continue;
            }
            if (self.p[step] - value).abs() < eps && time_in >= t_min {
                *out_t = time_in;
                return true;
            }
            let mut candidates = solve_cubic(
                self.j[step] / six,
                self.a[step] / two,
                self.v[step],
                self.p[step] - value,
            );
            for &root in candidates.as_sorted_slice() {
                if root > F::zero() && t_min - time_in <= root && root <= self.t[step] {
                    *out_t = root + time_in;
                    return true;
                }
            }
            time_in = time_in + self.t[step];
        }
        let close_tol = F::from(1e-9).unwrap();
        if (self.t[6] > F::zero() || self.duration == F::zero())
            && (self.pf - value).abs() < close_tol
            && self.duration >= t_min
        {
            *out_t = self.duration;
            return true;
        }
        false
    }

    /// Find the earliest time `t >= t_min` at which the segment reaches the
    /// velocity `value`. On success writes the time into `*out_t` and returns
    /// `true`.
    pub fn find_first_time_at_vel(&self, value: F, out_t: &mut F, t_min: F) -> bool {
        let eps = F::epsilon();
        let two = F::from(2.0).unwrap();
        let mut time_in = F::zero();
        for step in 0..7 {
            if self.t[step] == F::zero() {
                continue;
            }
            if (self.v[step] - value).abs() < eps && time_in >= t_min {
                *out_t = time_in;
                return true;
            }
            let discriminant =
                self.a[step] * self.a[step] - two * self.j[step] * (self.v[step] - value);
            if discriminant >= F::zero() && self.j[step] != F::zero() {
                let sqrt_disc = discriminant.sqrt();
                let mut candidates = [
                    (-self.a[step] + sqrt_disc) / self.j[step],
                    (-self.a[step] - sqrt_disc) / self.j[step],
                ];
                if candidates[0] > candidates[1] {
                    candidates.swap(0, 1);
                }
                for &root in candidates.iter() {
                    if root > F::zero() && t_min - time_in <= root && root <= self.t[step] {
                        *out_t = root + time_in;
                        return true;
                    }
                }
            }
            time_in = time_in + self.t[step];
        }
        let close_tol = F::from(1e-12).unwrap();
        if (self.t[6] > F::zero() || self.duration == F::zero())
            && (self.vf - value).abs() < close_tol
            && self.duration >= t_min
        {
            *out_t = self.duration;
            return true;
        }
        false
    }

    /// Scale the segment in time and position.
    pub fn scale(&mut self, position_scale: F, time_scale: F) {
        let inv_t = F::one() / time_scale;
        let inv_t2 = inv_t * inv_t;
        let inv_t3 = inv_t2 * inv_t;
        self.duration = self.duration * time_scale;
        for i in 0..7 {
            self.t[i] = self.t[i] * time_scale;
            self.j[i] = self.j[i] * position_scale * inv_t3;
        }
        for i in 0..8 {
            self.a[i] = self.a[i] * position_scale * inv_t2;
            self.v[i] = self.v[i] * position_scale * inv_t;
            self.p[i] = self.p[i] * position_scale;
        }
        self.halt.scale(position_scale, time_scale);
        self.accel_halt.scale(position_scale, time_scale);
        self.pf = self.pf * position_scale;
        self.vf = self.vf * position_scale * inv_t;
        self.af = self.af * position_scale * inv_t2;
    }

    /// Partial scale that only touches the section-0 state, the halts and the
    /// final state. Used when only the boundary conditions need rescaling
    /// before the interior sections are recomputed.
    pub fn scale_initial(&mut self, position_scale: F, time_scale: F) {
        let inv_t = F::one() / time_scale;
        let inv_t2 = inv_t * inv_t;
        self.a[0] = self.a[0] * position_scale * inv_t2;
        self.v[0] = self.v[0] * position_scale * inv_t;
        self.p[0] = self.p[0] * position_scale;
        self.halt.scale(position_scale, time_scale);
        self.accel_halt.scale(position_scale, time_scale);
        self.pf = self.pf * position_scale;
        self.vf = self.vf * position_scale * inv_t;
        self.af = self.af * position_scale * inv_t2;
    }
}

impl<F: Float> Default for Segment<F> {
    fn default() -> Self {
        Self::empty()
    }
}
