use std::time::Duration;

use num_traits::Float;

use crate::{DekeError, DekeResult, RobotPath, SRobotPath, SRobotQ};

/// Dynamically-sized robot trajectory: a dense [`RobotPath`] sampled uniformly at `dt`.
#[derive(Debug, Clone)]
pub struct RobotTraj<T: Float = f32> {
    dt: Duration,
    path: RobotPath<T>,
}

impl<T: Float> RobotTraj<T> {
    pub fn new(dt: Duration, path: RobotPath<T>) -> Self {
        Self { dt, path }
    }

    pub fn dt(&self) -> Duration {
        self.dt
    }

    pub fn path(&self) -> &RobotPath<T> {
        &self.path
    }

    pub fn path_mut(&mut self) -> &mut RobotPath<T> {
        &mut self.path
    }

    pub fn into_path(self) -> RobotPath<T> {
        self.path
    }

    pub fn len(&self) -> usize {
        self.path.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.path.nrows() == 0
    }

    pub fn dof(&self) -> usize {
        self.path.ncols()
    }

    pub fn duration(&self) -> Duration {
        self.dt
            .saturating_mul(self.len().saturating_sub(1) as u32)
    }

    pub fn time_at(&self, index: usize) -> Duration {
        self.dt.saturating_mul(index as u32)
    }
}

/// Statically-sized robot trajectory: an [`SRobotPath`] sampled uniformly at `dt`.
#[derive(Debug, Clone)]
pub struct SRobotTraj<const N: usize, T: Float = f32> {
    dt: Duration,
    path: SRobotPath<N, T>,
}

impl<const N: usize, T: Float> SRobotTraj<N, T> {
    pub fn new(dt: Duration, path: SRobotPath<N, T>) -> Self {
        Self { dt, path }
    }

    pub fn try_from_waypoints(dt: Duration, waypoints: Vec<SRobotQ<N, T>>) -> DekeResult<Self> {
        Ok(Self {
            dt,
            path: SRobotPath::try_new(waypoints)?,
        })
    }

    pub fn dt(&self) -> Duration {
        self.dt
    }

    pub fn set_dt(&mut self, dt: Duration) {
        self.dt = dt;
    }

    pub fn path(&self) -> &SRobotPath<N, T> {
        &self.path
    }

    pub fn path_mut(&mut self) -> &mut SRobotPath<N, T> {
        &mut self.path
    }

    pub fn into_path(self) -> SRobotPath<N, T> {
        self.path
    }

    pub fn len(&self) -> usize {
        self.path.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dof(&self) -> usize {
        N
    }

    pub fn first(&self) -> &SRobotQ<N, T> {
        self.path.first()
    }

    pub fn last(&self) -> &SRobotQ<N, T> {
        self.path.last()
    }

    pub fn get(&self, index: usize) -> Option<&SRobotQ<N, T>> {
        self.path.get(index)
    }

    pub fn duration(&self) -> Duration {
        self.dt
            .saturating_mul(self.len().saturating_sub(1) as u32)
    }

    pub fn time_at(&self, index: usize) -> Duration {
        self.dt.saturating_mul(index as u32)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, SRobotQ<N, T>> {
        self.path.iter()
    }

    pub fn iter_timed(&self) -> impl Iterator<Item = (Duration, &SRobotQ<N, T>)> + '_ {
        let dt = self.dt;
        self.path
            .iter()
            .enumerate()
            .map(move |(i, q)| (dt.saturating_mul(i as u32), q))
    }

    /// Samples the trajectory at a given wall-clock time via linear interpolation between
    /// adjacent waypoints. Times outside `[0, duration()]` are clamped.
    pub fn sample_at_time(&self, t: Duration) -> Option<SRobotQ<N, T>> {
        let n = self.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some(*self.first());
        }

        let dt_secs = self.dt.as_secs_f64();
        if dt_secs <= 0.0 {
            return Some(*self.first());
        }

        let t_secs = t.as_secs_f64();
        let max_secs = dt_secs * (n - 1) as f64;
        let clamped = t_secs.clamp(0.0, max_secs);

        let f = clamped / dt_secs;
        let i = (f.floor() as usize).min(n - 2);
        let local = T::from(f - i as f64).unwrap_or_else(T::zero);

        let a = self.path.get(i)?;
        let b = self.path.get(i + 1)?;
        Some(a.interpolate(b, local))
    }

    /// Finite-difference velocity at sample index `i`, in joint-units per second.
    /// Uses forward difference at the start, backward at the end, central elsewhere.
    pub fn velocity_at(&self, i: usize) -> Option<SRobotQ<N, T>> {
        let n = self.len();
        if n < 2 || i >= n {
            return None;
        }
        let dt_secs = T::from(self.dt.as_secs_f64()).unwrap_or_else(T::zero);
        if dt_secs == T::zero() {
            return Some(SRobotQ::zeros());
        }

        let (a, b, span) = if i == 0 {
            (*self.path.get(0)?, *self.path.get(1)?, T::one())
        } else if i == n - 1 {
            (*self.path.get(n - 2)?, *self.path.get(n - 1)?, T::one())
        } else {
            (
                *self.path.get(i - 1)?,
                *self.path.get(i + 1)?,
                T::from(2.0).unwrap_or_else(T::one),
            )
        };

        Some((b - a) * (T::one() / (span * dt_secs)))
    }

    /// Finite-difference acceleration at sample index `i`, in joint-units per second^2.
    pub fn acceleration_at(&self, i: usize) -> Option<SRobotQ<N, T>> {
        let n = self.len();
        if n < 3 || i >= n {
            return None;
        }
        let dt_secs = T::from(self.dt.as_secs_f64()).unwrap_or_else(T::zero);
        if dt_secs == T::zero() {
            return Some(SRobotQ::zeros());
        }

        let idx = i.clamp(1, n - 2);
        let a = *self.path.get(idx - 1)?;
        let b = *self.path.get(idx)?;
        let c = *self.path.get(idx + 1)?;
        let inv_dt_sq = T::one() / (dt_secs * dt_secs);
        Some((a - b * T::from(2.0).unwrap_or_else(T::one) + c) * inv_dt_sq)
    }

    pub fn max_joint_velocity(&self) -> T {
        let mut peak = T::zero();
        for i in 0..self.len() {
            if let Some(v) = self.velocity_at(i) {
                peak = peak.max(v.linf_norm());
            }
        }
        peak
    }

    pub fn max_joint_acceleration(&self) -> T {
        let mut peak = T::zero();
        for i in 0..self.len() {
            if let Some(a) = self.acceleration_at(i) {
                peak = peak.max(a.linf_norm());
            }
        }
        peak
    }

    /// Reverses the trajectory in place (swaps start and end, preserves `dt`).
    pub fn reverse(&mut self) {
        self.path.reverse();
    }

    pub fn reversed(&self) -> Self {
        Self {
            dt: self.dt,
            path: self.path.reversed(),
        }
    }

    /// Rescales the trajectory by changing `dt`. `scale > 1.0` slows it down; `< 1.0` speeds it up.
    pub fn rescale_time(&mut self, scale: f64) {
        let new_secs = self.dt.as_secs_f64() * scale.max(0.0);
        self.dt = Duration::from_secs_f64(new_secs);
    }

    pub fn to_robot_traj(&self) -> RobotTraj<T> {
        RobotTraj {
            dt: self.dt,
            path: self.path.to_robot_path(),
        }
    }
}

impl<const N: usize, T: Float> std::ops::Index<usize> for SRobotTraj<N, T> {
    type Output = SRobotQ<N, T>;
    #[inline]
    fn index(&self, i: usize) -> &SRobotQ<N, T> {
        &self.path[i]
    }
}

impl<'a, const N: usize, T: Float> IntoIterator for &'a SRobotTraj<N, T> {
    type Item = &'a SRobotQ<N, T>;
    type IntoIter = std::slice::Iter<'a, SRobotQ<N, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.path.iter()
    }
}

impl<const N: usize, T: Float> AsRef<SRobotPath<N, T>> for SRobotTraj<N, T> {
    fn as_ref(&self) -> &SRobotPath<N, T> {
        &self.path
    }
}

impl<const N: usize, T: Float> From<SRobotTraj<N, T>> for SRobotPath<N, T> {
    fn from(traj: SRobotTraj<N, T>) -> Self {
        traj.path
    }
}

impl<const N: usize, T: Float> From<SRobotTraj<N, T>> for RobotTraj<T> {
    fn from(traj: SRobotTraj<N, T>) -> Self {
        traj.to_robot_traj()
    }
}

impl<const N: usize, T: Float> TryFrom<RobotTraj<T>> for SRobotTraj<N, T> {
    type Error = DekeError;

    fn try_from(traj: RobotTraj<T>) -> Result<Self, Self::Error> {
        Ok(Self {
            dt: traj.dt,
            path: SRobotPath::<N, T>::try_from(traj.path)?,
        })
    }
}
