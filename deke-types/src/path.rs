use ndarray::Array2;

use crate::{DekeResult, SRobotQ};

/// Dynamically-sized robot path as a 2D array (rows = waypoints, cols = joints).
pub type RobotPath = Array2<f32>;

/// Statically-sized robot path backed by `Vec<SRobotQ<N>>`.
/// 
/// SRobotPath is guranteed to have at least 2 waypoints, so it always has a defined start and end configuration.
#[derive(Debug, Clone)]
pub struct SRobotPath<const N: usize> {
    first: SRobotQ<N>,
    last: SRobotQ<N>,
    waypoints: Vec<SRobotQ<N>>,
}

impl<const N: usize> SRobotPath<N> {
    pub fn new(waypoints: Vec<SRobotQ<N>>) -> DekeResult<Self> {
        if waypoints.len() < 2 {
            return Err(crate::DekeError::PathTooShort(waypoints.len()));
        }
        Ok(Self {
            first: waypoints[0],
            last: waypoints[waypoints.len() - 1],
            waypoints,
        })
    }

    pub fn new_prechecked(first: SRobotQ<N>, last: SRobotQ<N>, middle: Vec<SRobotQ<N>>) -> Self {
        let mut waypoints = Vec::with_capacity(middle.len() + 2);
        waypoints.push(first);
        waypoints.extend(middle);
        waypoints.push(last);
        Self {
            first,
            last,
            waypoints,
        }
     }

    pub fn from_two(start: SRobotQ<N>, goal: SRobotQ<N>) -> Self {
        Self {
            first: start,
            last: goal,
            waypoints: vec![start, goal],
        }
    }

    pub fn len(&self) -> usize {
        self.waypoints.len()
    }

    pub fn get(&self, index: usize) -> Option<&SRobotQ<N>> {
        self.waypoints.get(index)
    }

    pub fn first(&self) -> &SRobotQ<N> {
        &self.first
    }

    pub fn last(&self) -> &SRobotQ<N> {
        &self.last
    }

    pub fn iter(&self) -> std::slice::Iter<'_, SRobotQ<N>> {
        self.waypoints.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, SRobotQ<N>> {
        self.waypoints.iter_mut()
    }

    pub fn segments(&self) -> impl Iterator<Item = (&SRobotQ<N>, &SRobotQ<N>)> {
        self.waypoints.windows(2).map(|w| (&w[0], &w[1]))
    }

    pub fn push(&mut self, q: SRobotQ<N>) {
        self.waypoints.push(q);
    }

    pub fn pop(&mut self) -> Option<SRobotQ<N>> {
        if self.waypoints.len() > 2 {
            let popped = self.waypoints.pop();
            if let Some(p) = popped {
                if let Some(last) = self.waypoints.last() {
                    self.last = *last;
                }
                Some(p)
            } else {
                None
            }

        } else {
            None
        }
    }

    pub fn truncate(&mut self, len: usize) {
        if len < 2 {
            return;
        }
        self.waypoints.truncate(len);
        if let Some(last) = self.waypoints.last() {
            self.last = *last;
        }
    }

    pub fn reverse(&mut self) {
        self.waypoints.reverse();
        if let Some(first) = self.waypoints.first() {
            self.first = *first;
        }
        if let Some(last) = self.waypoints.last() {
            self.last = *last;
        }
    }

    pub fn reversed(&self) -> Self {
        let mut wps = self.clone();
        wps.reverse();
        wps
    }

    pub fn arc_length(&self) -> f32 {
        self.segments().map(|(a, b)| a.distance(b)).sum()
    }

    pub fn segment_lengths(&self) -> Vec<f32> {
        self.segments().map(|(a, b)| a.distance(b)).collect()
    }

    pub fn cumulative_lengths(&self) -> Vec<f32> {
        let mut cum = Vec::with_capacity(self.len());
        let mut total = 0.0;
        cum.push(0.0);
        for (a, b) in self.segments() {
            total += a.distance(b);
            cum.push(total);
        }
        cum
    }

    pub fn max_segment_length(&self) -> f32 {
        self.segments()
            .map(|(a, b)| a.distance(b))
            .fold(0.0, f32::max)
    }

    pub fn max_joint_step(&self) -> f32 {
        self.segments()
            .map(|(a, b)| (*a - *b).linf_norm())
            .fold(0.0, f32::max)
    }

    pub fn sample(&self, t: f32) -> Option<SRobotQ<N>> {
        let n = self.len();
        if n < 2 {
            return if n == 1 {
                Some(self.waypoints[0])
            } else {
                None
            };
        }

        let t = t.clamp(0.0, 1.0);
        let total = self.arc_length();
        if total == 0.0 {
            return Some(self.waypoints[0]);
        }

        let target = t * total;
        let mut accumulated = 0.0;

        for (a, b) in self.segments() {
            let seg_len = a.distance(b);
            if accumulated + seg_len >= target {
                let local_t = if seg_len > 0.0 {
                    (target - accumulated) / seg_len
                } else {
                    0.0
                };
                return Some(a.interpolate(b, local_t));
            }
            accumulated += seg_len;
        }

        Some(*self.waypoints.last().unwrap())
    }

    pub fn densify(&self, max_dist: f32) -> Self {
        if self.len() < 2 {
            return self.clone();
        }

        let mut out = Vec::new();
        out.push(self.waypoints[0]);

        for (a, b) in self.segments() {
            let d = a.distance(b);
            let steps = (d / max_dist).ceil().max(1.0) as usize;
            for i in 1..=steps {
                let t = i as f32 / steps as f32;
                out.push(a.interpolate(b, t));
            }
        }

        Self::new(out).unwrap()
    }

    pub fn simplify(&self, tol: f32) -> Self {
        if self.len() <= 2 {
            return self.clone();
        }

        let mut keep = vec![true; self.len()];
        srdp_mark(&self.waypoints, 0, self.len() - 1, tol, &mut keep);

        let waypoints = self
            .waypoints
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, q)| *q)
            .collect();

        Self::new(waypoints).unwrap()
    }

    pub fn to_robot_path(&self) -> RobotPath {
        let n = self.waypoints.len();
        let mut arr = RobotPath::zeros((n, N));
        for (i, q) in self.waypoints.iter().enumerate() {
            arr.row_mut(i).assign(&ndarray::ArrayView1::from(&q.0));
        }
        arr
    }
}

impl<const N: usize> std::ops::Index<usize> for SRobotPath<N> {
    type Output = SRobotQ<N>;
    #[inline]
    fn index(&self, i: usize) -> &SRobotQ<N> {
        &self.waypoints[i]
    }
}

impl<const N: usize> std::ops::IndexMut<usize> for SRobotPath<N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut SRobotQ<N> {
        &mut self.waypoints[i]
    }
}

impl<const N: usize> IntoIterator for SRobotPath<N> {
    type Item = SRobotQ<N>;
    type IntoIter = std::vec::IntoIter<SRobotQ<N>>;

    fn into_iter(self) -> Self::IntoIter {
        self.waypoints.into_iter()
    }
}

impl<'a, const N: usize> IntoIterator for &'a SRobotPath<N> {
    type Item = &'a SRobotQ<N>;
    type IntoIter = std::slice::Iter<'a, SRobotQ<N>>;

    fn into_iter(self) -> Self::IntoIter {
        self.waypoints.iter()
    }
}

impl<const N: usize> AsRef<[SRobotQ<N>]> for SRobotPath<N> {
    fn as_ref(&self) -> &[SRobotQ<N>] {
        &self.waypoints
    }
}

impl<const N: usize> TryFrom<Vec<SRobotQ<N>>> for SRobotPath<N> {
    type Error = crate::DekeError;

    fn try_from(waypoints: Vec<SRobotQ<N>>) -> Result<Self, Self::Error> {
        Self::new(waypoints)
    }
}

impl<const N: usize> TryFrom<&[SRobotQ<N>]> for SRobotPath<N> {
    type Error = crate::DekeError;

    fn try_from(waypoints: &[SRobotQ<N>]) -> Result<Self, Self::Error> {
        Self::new(waypoints.to_vec())
    }
}

impl<const N: usize> TryFrom<&Vec<SRobotQ<N>>> for SRobotPath<N> {
    type Error = crate::DekeError;

    fn try_from(waypoints: &Vec<SRobotQ<N>>) -> Result<Self, Self::Error> {
        Self::new(waypoints.clone())
    }
}

impl<const N: usize> From<SRobotPath<N>> for RobotPath {
    fn from(sp: SRobotPath<N>) -> Self {
        sp.to_robot_path()
    }
}

impl<const N: usize> From<&SRobotPath<N>> for RobotPath {
    fn from(sp: &SRobotPath<N>) -> Self {
        sp.to_robot_path()
    }
}

impl<const N: usize> TryFrom<RobotPath> for SRobotPath<N> {
    type Error = crate::DekeError;

    fn try_from(arr: RobotPath) -> Result<Self, Self::Error> {
        if arr.ncols() != N {
            return Err(crate::DekeError::ShapeMismatch {
                expected: N,
                found: arr.ncols(),
            });
        }
        let mut waypoints = Vec::with_capacity(arr.nrows());
        for row in arr.rows() {
            let mut q = [0.0f32; N];
            q.copy_from_slice(row.as_slice().unwrap());
            waypoints.push(SRobotQ(q));
        }
        Self::new(waypoints)
    }
}

impl<const N: usize> TryFrom<&RobotPath> for SRobotPath<N> {
    type Error = crate::DekeError;

    fn try_from(arr: &RobotPath) -> Result<Self, Self::Error> {
        if arr.ncols() != N {
            return Err(crate::DekeError::ShapeMismatch {
                expected: N,
                found: arr.ncols(),
            });
        }
        let mut waypoints = Vec::with_capacity(arr.nrows());
        for row in arr.rows() {
            let mut q = [0.0f32; N];
            q.copy_from_slice(row.as_slice().unwrap());
            waypoints.push(SRobotQ(q));
        }
        Self::new(waypoints)
    }
}

fn srdp_mark<const N: usize>(
    pts: &[SRobotQ<N>],
    start: usize,
    end: usize,
    tol: f32,
    keep: &mut [bool],
) {
    if end <= start + 1 {
        return;
    }

    let seg = pts[end] - pts[start];
    let seg_len_sq = seg.norm_squared();

    let mut max_dist = 0.0f32;
    let mut max_idx = start;

    for i in (start + 1)..end {
        let v = pts[i] - pts[start];
        let dist = if seg_len_sq < 1e-30 {
            v.norm()
        } else {
            let t = (v.dot(&seg) / seg_len_sq).clamp(0.0, 1.0);
            (v - seg * t).norm()
        };
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    if max_dist > tol {
        keep[max_idx] = true;
        srdp_mark(pts, start, max_idx, tol, keep);
        srdp_mark(pts, max_idx, end, tol, keep);
    } else {
        for k in (start + 1)..end {
            keep[k] = false;
        }
    }
}
