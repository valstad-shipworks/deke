use crate::q::RobotQ;
use crate::{DekeError, DekeResult};

/// An ordered sequence of joint configurations representing a robot path.
#[derive(Debug, Clone)]
pub struct RobotPath {
    waypoints: Vec<RobotQ>,
}

impl RobotPath {
    pub fn new(waypoints: Vec<RobotQ>) -> DekeResult<Self> {
        if let Some(first) = waypoints.first() {
            let n = first.len();
            for q in waypoints.iter().skip(1) {
                if q.len() != n {
                    return Err(DekeError::ShapeMismatch {
                        expected: n,
                        found: q.len(),
                    });
                }
            }
        }
        Ok(Self { waypoints })
    }

    pub fn empty() -> Self {
        Self {
            waypoints: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            waypoints: Vec::with_capacity(cap),
        }
    }

    pub fn len(&self) -> usize {
        self.waypoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }

    /// Number of joints per waypoint, or 0 if the path is empty.
    pub fn ndof(&self) -> usize {
        self.waypoints.first().map_or(0, |q| q.len())
    }

    pub fn get(&self, index: usize) -> Option<&RobotQ> {
        self.waypoints.get(index)
    }

    pub fn first(&self) -> Option<&RobotQ> {
        self.waypoints.first()
    }

    pub fn last(&self) -> Option<&RobotQ> {
        self.waypoints.last()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, RobotQ> {
        self.waypoints.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, RobotQ> {
        self.waypoints.iter_mut()
    }

    /// Iterator over consecutive pairs of waypoints.
    pub fn segments(&self) -> impl Iterator<Item = (&RobotQ, &RobotQ)> {
        self.waypoints.windows(2).map(|w| (&w[0], &w[1]))
    }

    pub fn push(&mut self, q: RobotQ) -> DekeResult<()> {
        if let Some(first) = self.waypoints.first() {
            if q.len() != first.len() {
                return Err(DekeError::ShapeMismatch {
                    expected: first.len(),
                    found: q.len(),
                });
            }
        }
        self.waypoints.push(q);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<RobotQ> {
        self.waypoints.pop()
    }

    pub fn clear(&mut self) {
        self.waypoints.clear();
    }

    pub fn truncate(&mut self, len: usize) {
        self.waypoints.truncate(len);
    }

    pub fn reverse(&mut self) {
        self.waypoints.reverse();
    }

    pub fn reversed(&self) -> Self {
        let mut wps = self.waypoints.clone();
        wps.reverse();
        Self { waypoints: wps }
    }

    /// Total arc length (sum of Euclidean distances between consecutive waypoints).
    pub fn arc_length(&self) -> f32 {
        self.segments().map(|(a, b)| q_distance(a, b)).sum()
    }

    /// Euclidean distance of each segment.
    pub fn segment_lengths(&self) -> Vec<f32> {
        self.segments().map(|(a, b)| q_distance(a, b)).collect()
    }

    /// Cumulative arc length at each waypoint, starting at 0.0.
    pub fn cumulative_lengths(&self) -> Vec<f32> {
        let mut cum = Vec::with_capacity(self.len());
        let mut total = 0.0;
        cum.push(0.0);
        for (a, b) in self.segments() {
            total += q_distance(a, b);
            cum.push(total);
        }
        cum
    }

    /// Maximum Euclidean distance between any two consecutive waypoints.
    pub fn max_segment_length(&self) -> f32 {
        self.segments()
            .map(|(a, b)| q_distance(a, b))
            .fold(0.0, f32::max)
    }

    /// Maximum per-joint deviation (L-inf norm) across any segment.
    pub fn max_joint_step(&self) -> f32 {
        self.segments()
            .map(|(a, b)| q_linf_distance(a, b))
            .fold(0.0, f32::max)
    }

    /// Linearly interpolate along the path by normalized parameter `t` in `[0, 1]`.
    /// Returns `None` if the path has fewer than 2 waypoints.
    pub fn sample(&self, t: f32) -> Option<RobotQ> {
        let n = self.len();
        if n < 2 {
            return if n == 1 {
                Some(self.waypoints[0].clone())
            } else {
                None
            };
        }

        let t = t.clamp(0.0, 1.0);
        let total = self.arc_length();
        if total == 0.0 {
            return Some(self.waypoints[0].clone());
        }

        let target = t * total;
        let mut accumulated = 0.0;

        for (a, b) in self.segments() {
            let seg_len = q_distance(a, b);
            if accumulated + seg_len >= target {
                let local_t = if seg_len > 0.0 {
                    (target - accumulated) / seg_len
                } else {
                    0.0
                };
                return Some(a + &((b - a) * local_t));
            }
            accumulated += seg_len;
        }

        Some(self.waypoints.last().unwrap().clone())
    }

    /// Subdivide every segment so that no segment exceeds `max_dist`.
    pub fn densify(&self, max_dist: f32) -> Self {
        if self.len() < 2 {
            return self.clone();
        }

        let mut out = Vec::new();
        out.push(self.waypoints[0].clone());

        for (a, b) in self.segments() {
            let d = q_distance(a, b);
            let steps = (d / max_dist).ceil() as usize;
            let steps = steps.max(1);
            for i in 1..=steps {
                let t = i as f32 / steps as f32;
                out.push(a + &((b - a) * t));
            }
        }

        Self { waypoints: out }
    }

    /// Remove waypoints that are within `tol` of the line between their neighbors
    /// (Ramer-Douglas-Peucker in joint space).
    pub fn simplify(&self, tol: f32) -> Self {
        if self.len() <= 2 {
            return self.clone();
        }

        let mut keep = vec![true; self.len()];
        rdp_mark(&self.waypoints, 0, self.len() - 1, tol, &mut keep);

        let waypoints = self
            .waypoints
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, q)| q.clone())
            .collect();

        Self { waypoints }
    }

    /// Apply a function to every waypoint.
    pub fn map_waypoints(&self, f: impl Fn(&RobotQ) -> RobotQ) -> Self {
        Self {
            waypoints: self.waypoints.iter().map(f).collect(),
        }
    }
}

impl std::ops::Index<usize> for RobotPath {
    type Output = RobotQ;
    #[inline]
    fn index(&self, i: usize) -> &RobotQ {
        &self.waypoints[i]
    }
}

impl std::ops::IndexMut<usize> for RobotPath {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut RobotQ {
        &mut self.waypoints[i]
    }
}

impl FromIterator<RobotQ> for RobotPath {
    fn from_iter<I: IntoIterator<Item = RobotQ>>(iter: I) -> Self {
        Self {
            waypoints: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for RobotPath {
    type Item = RobotQ;
    type IntoIter = std::vec::IntoIter<RobotQ>;

    fn into_iter(self) -> Self::IntoIter {
        self.waypoints.into_iter()
    }
}

impl<'a> IntoIterator for &'a RobotPath {
    type Item = &'a RobotQ;
    type IntoIter = std::slice::Iter<'a, RobotQ>;

    fn into_iter(self) -> Self::IntoIter {
        self.waypoints.iter()
    }
}

fn q_distance(a: &RobotQ, b: &RobotQ) -> f32 {
    let diff = b - a;
    diff.mapv(|x| x * x).sum().sqrt()
}

fn q_linf_distance(a: &RobotQ, b: &RobotQ) -> f32 {
    let diff = b - a;
    diff.mapv(f32::abs).into_iter().fold(0.0, f32::max)
}

fn rdp_mark(pts: &[RobotQ], start: usize, end: usize, tol: f32, keep: &mut [bool]) {
    if end <= start + 1 {
        return;
    }

    let seg = &pts[end] - &pts[start];
    let seg_len_sq: f32 = seg.mapv(|x| x * x).sum();

    let mut max_dist = 0.0f32;
    let mut max_idx = start;

    for i in (start + 1)..end {
        let v = &pts[i] - &pts[start];
        let dist = if seg_len_sq < 1e-30 {
            v.mapv(|x| x * x).sum().sqrt()
        } else {
            let t = (v.dot(&seg) / seg_len_sq).clamp(0.0, 1.0);
            let proj = &v - &(&seg * t);
            proj.mapv(|x| x * x).sum().sqrt()
        };
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    if max_dist > tol {
        keep[max_idx] = true;
        rdp_mark(pts, start, max_idx, tol, keep);
        rdp_mark(pts, max_idx, end, tol, keep);
    } else {
        for k in (start + 1)..end {
            keep[k] = false;
        }
    }
}
