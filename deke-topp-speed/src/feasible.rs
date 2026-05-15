//! Per-axis collection of feasible time intervals.
//!
//! Each `Feasible` carries the minimum-duration profile plus optional
//! "forbidden" spans within which no consistent solution exists; queries pick
//! the appropriate profile for a given target time.

use num_traits::Float;

use crate::segment::Segment;

/// Span of time during which a profile is infeasible, bracketed by the two
/// neighbouring valid profiles. Only the right-hand profile is stored — the
/// optimizer never reads the left one, and dropping it halves the size of
/// `Span` (and thus `Feasible`).
#[derive(Debug, Clone, Copy)]
pub struct Span<F: Float> {
    pub left_time: F,
    pub right_time: F,
    pub profile_at_right: Segment<F>,
}

impl<F: Float> Span<F> {
    pub fn new(a: Segment<F>, b: Segment<F>) -> Self {
        let left_time = a.duration + a.halt.duration + a.accel_halt.duration;
        let right_time = b.duration + b.halt.duration + b.accel_halt.duration;
        Self {
            left_time,
            right_time,
            profile_at_right: b,
        }
    }

    /// Construct a span with explicit `left_time` and `right_time`. The
    /// right-hand profile defaults to a zero-initialised segment and may be
    /// overridden later.
    pub fn from_times(left_time: F, right_time: F) -> Self {
        Self {
            left_time,
            right_time,
            profile_at_right: Segment::empty(),
        }
    }
}

/// Per-axis feasible window.
#[derive(Debug, Clone, Copy)]
pub struct Feasible<F: Float> {
    /// Profile achieving the minimum total duration.
    pub p_min: Segment<F>,
    /// Minimum total duration achievable on this axis.
    pub t_min: F,
    pub blocked_interval_a: Option<Span<F>>,
    pub blocked_interval_b: Option<Span<F>>,
}

impl<F: Float> Feasible<F> {
    #[inline]
    pub fn empty() -> Self {
        Self {
            p_min: Segment::empty(),
            t_min: F::zero(),
            blocked_interval_a: None,
            blocked_interval_b: None,
        }
    }

    #[inline]
    pub fn set_min_profile(&mut self, profile: Segment<F>) {
        self.p_min = profile;
        self.t_min =
            self.p_min.duration + self.p_min.halt.duration + self.p_min.accel_halt.duration;
        self.blocked_interval_a = None;
        self.blocked_interval_b = None;
    }

    /// Choose the minimum-duration profile from a set of valid candidates and
    /// record any feasibility gaps as blocked intervals.
    pub fn pick_from_candidates(&mut self, profiles: &mut [Segment<F>], mut count: usize) -> bool {
        let eight = F::from(8.0).unwrap();
        let thirty_two = F::from(32.0).unwrap();
        let two_fifty_six = F::from(256.0).unwrap();
        let eps = F::epsilon();
        if count == 1 {
            self.set_min_profile(profiles[0]);
            return true;
        } else if count == 2 {
            if (profiles[0].duration - profiles[1].duration).abs() < eight * eps {
                self.set_min_profile(profiles[0]);
                return true;
            }
            let idx_min = if profiles[0].duration < profiles[1].duration {
                0
            } else {
                1
            };
            let idx_other = (idx_min + 1) % 2;
            self.set_min_profile(profiles[idx_min]);
            self.blocked_interval_a = Some(Span::new(profiles[idx_min], profiles[idx_other]));
            return true;
        } else if count == 4 {
            if (profiles[0].duration - profiles[1].duration).abs() < thirty_two * eps
                && profiles[0].sweep != profiles[1].sweep
            {
                Self::remove_at(profiles, &mut count, 1);
            } else if (profiles[2].duration - profiles[3].duration).abs() < two_fifty_six * eps
                && profiles[2].sweep != profiles[3].sweep
            {
                Self::remove_at(profiles, &mut count, 3);
            } else if (profiles[0].duration - profiles[3].duration).abs() < two_fifty_six * eps
                && profiles[0].sweep != profiles[3].sweep
            {
                Self::remove_at(profiles, &mut count, 3);
            } else {
                return false;
            }
        } else if count.is_multiple_of(2) {
            return false;
        }
        let mut idx_fastest = 0usize;
        let mut t_fastest = profiles[0].duration;
        for i in 1..count {
            let t_current = profiles[i].duration;
            if t_current < t_fastest {
                t_fastest = t_current;
                idx_fastest = i;
            }
        }
        self.set_min_profile(profiles[idx_fastest]);
        if count == 3 {
            let idx_a = (idx_fastest + 1) % 3;
            let idx_b = (idx_fastest + 2) % 3;
            self.blocked_interval_a = Some(Span::new(profiles[idx_a], profiles[idx_b]));
            return true;
        } else if count == 5 {
            let idx_0 = (idx_fastest + 1) % 5;
            let idx_1 = (idx_fastest + 2) % 5;
            let idx_2 = (idx_fastest + 3) % 5;
            let idx_3 = (idx_fastest + 4) % 5;
            if profiles[idx_0].sweep == profiles[idx_1].sweep {
                self.blocked_interval_a = Some(Span::new(profiles[idx_0], profiles[idx_1]));
                self.blocked_interval_b = Some(Span::new(profiles[idx_2], profiles[idx_3]));
            } else {
                self.blocked_interval_a = Some(Span::new(profiles[idx_0], profiles[idx_3]));
                self.blocked_interval_b = Some(Span::new(profiles[idx_1], profiles[idx_2]));
            }
            return true;
        }
        false
    }

    fn remove_at(profiles: &mut [Segment<F>], count: &mut usize, idx: usize) {
        for i in idx..(*count - 1) {
            profiles[i] = profiles[i + 1];
        }
        *count -= 1;
    }

    /// Whether the given target time falls inside a blocked span.
    #[inline]
    pub fn is_blocked(&self, time: F) -> bool {
        let in_a = self
            .blocked_interval_a
            .is_some_and(|s| s.left_time < time && time < s.right_time);
        let in_b = self
            .blocked_interval_b
            .is_some_and(|s| s.left_time < time && time < s.right_time);
        (time < self.t_min) || in_a || in_b
    }

    /// Return the appropriate profile achieving the given target time.
    #[inline]
    pub fn get_profile(&self, time: F) -> &Segment<F> {
        if let Some(b) = self.blocked_interval_b.as_ref()
            && time >= b.right_time
        {
            return &b.profile_at_right;
        }
        if let Some(a) = self.blocked_interval_a.as_ref()
            && time >= a.right_time
        {
            return &a.profile_at_right;
        }
        &self.p_min
    }
}

impl<F: Float> Default for Feasible<F> {
    fn default() -> Self {
        Self::empty()
    }
}
