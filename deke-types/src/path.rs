use ndarray::Array2;
use num_traits::Float;

use crate::{DekeResult, SRobotQ};

pub type RobotPath<T = f32> = Array2<T>;

/// Statically-sized robot path backed by `Vec<SRobotQ<N, T>>`.
///
/// SRobotPath is guaranteed to have at least 2 waypoints, so it always has a defined start and end configuration.
#[derive(Debug, Clone)]
pub struct SRobotPath<const N: usize, F: Float = f32> {
    first: SRobotQ<N, F>,
    last: SRobotQ<N, F>,
    waypoints: Vec<SRobotQ<N, F>>,
}

impl<const N: usize, F: Float> SRobotPath<N, F> {
    pub fn try_new(waypoints: Vec<SRobotQ<N, F>>) -> DekeResult<Self> {
        if waypoints.len() < 2 {
            return Err(crate::DekeError::PathTooShort(waypoints.len()));
        }
        Ok(Self {
            first: waypoints[0],
            last: waypoints[waypoints.len() - 1],
            waypoints,
        })
    }

    pub fn new_prechecked(
        first: SRobotQ<N, F>,
        last: SRobotQ<N, F>,
        middle: Vec<SRobotQ<N, F>>,
    ) -> Self {
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

    pub fn from_two(start: SRobotQ<N, F>, goal: SRobotQ<N, F>) -> Self {
        Self {
            first: start,
            last: goal,
            waypoints: vec![start, goal],
        }
    }

    pub fn len(&self) -> usize {
        self.waypoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&SRobotQ<N, F>> {
        self.waypoints.get(index)
    }

    pub fn first(&self) -> &SRobotQ<N, F> {
        &self.first
    }

    pub fn last(&self) -> &SRobotQ<N, F> {
        &self.last
    }

    pub fn iter(&self) -> std::slice::Iter<'_, SRobotQ<N, F>> {
        self.waypoints.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, SRobotQ<N, F>> {
        self.waypoints.iter_mut()
    }

    pub fn segments(&self) -> impl Iterator<Item = (&SRobotQ<N, F>, &SRobotQ<N, F>)> {
        self.waypoints.windows(2).map(|w| (&w[0], &w[1]))
    }

    pub fn push(&mut self, q: SRobotQ<N, F>) {
        self.waypoints.push(q);
    }

    pub fn pop(&mut self) -> Option<SRobotQ<N, F>> {
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

    pub fn arc_length(&self) -> F {
        self.segments()
            .map(|(a, b)| a.distance(b))
            .fold(F::zero(), |acc, d| acc + d)
    }

    pub fn segment_lengths(&self) -> Vec<F> {
        self.segments().map(|(a, b)| a.distance(b)).collect()
    }

    pub fn cumulative_lengths(&self) -> Vec<F> {
        let mut cum = Vec::with_capacity(self.len());
        let mut total = F::zero();
        cum.push(F::zero());
        for (a, b) in self.segments() {
            total = total + a.distance(b);
            cum.push(total);
        }
        cum
    }

    pub fn max_segment_length(&self) -> F {
        self.segments()
            .map(|(a, b)| a.distance(b))
            .fold(F::zero(), |a, b| a.max(b))
    }

    pub fn max_joint_step(&self) -> F {
        self.segments()
            .map(|(a, b)| (*a - *b).linf_norm())
            .fold(F::zero(), |a, b| a.max(b))
    }

    pub fn sample(&self, t: F) -> Option<SRobotQ<N, F>> {
        let n = self.len();
        if n < 2 {
            return if n == 1 {
                Some(self.waypoints[0])
            } else {
                None
            };
        }

        let t = t.max(F::zero()).min(F::one());
        let total = self.arc_length();
        if total == F::zero() {
            return Some(self.waypoints[0]);
        }

        let target = t * total;
        let mut accumulated = F::zero();

        for (a, b) in self.segments() {
            let seg_len = a.distance(b);
            if accumulated + seg_len >= target {
                let local_t = if seg_len > F::zero() {
                    (target - accumulated) / seg_len
                } else {
                    F::zero()
                };
                return Some(a.interpolate(b, local_t));
            }
            accumulated = accumulated + seg_len;
        }

        Some(self.last)
    }

    pub fn densify(&self, max_dist: F) -> Self {
        if self.len() < 2 {
            return self.clone();
        }

        let mut out = Vec::new();
        out.push(self.waypoints[0]);

        for (a, b) in self.segments() {
            let d = a.distance(b);
            let steps = (d / max_dist).ceil().max(F::one()).to_usize().unwrap_or(1);
            for i in 1..=steps {
                let t = F::from(i).unwrap() / F::from(steps).unwrap();
                out.push(a.interpolate(b, t));
            }
        }

        Self {
            first: self.first,
            last: self.last,
            waypoints: out,
        }
    }

    /// Removes consecutive duplicate waypoints within `tol` distance of each other.
    pub fn deduplicate(&self, tol: F) -> Self {
        let mut out = Vec::with_capacity(self.len());
        out.push(self.waypoints[0]);
        for q in &self.waypoints[1..] {
            if out.last().unwrap().distance(q) > tol {
                out.push(*q);
            }
        }
        if out.len() < 2 {
            out.push(self.last);
        }
        Self {
            first: out[0],
            last: out[out.len() - 1],
            waypoints: out,
        }
    }

    pub fn simplify(&self, tol: F) -> Self {
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

        Self {
            first: self.first,
            last: self.last,
            waypoints,
        }
    }

    pub fn to_robot_path(&self) -> RobotPath<F> {
        let n = self.waypoints.len();
        let mut arr = RobotPath::zeros((n, N));
        for (i, q) in self.waypoints.iter().enumerate() {
            arr.row_mut(i).assign(&ndarray::ArrayView1::from(&q.0[..]));
        }
        arr
    }
}

impl<const N: usize, T: Float> std::ops::Index<usize> for SRobotPath<N, T> {
    type Output = SRobotQ<N, T>;
    #[inline]
    fn index(&self, i: usize) -> &SRobotQ<N, T> {
        &self.waypoints[i]
    }
}

impl<const N: usize, T: Float> std::ops::IndexMut<usize> for SRobotPath<N, T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut SRobotQ<N, T> {
        &mut self.waypoints[i]
    }
}

impl<const N: usize, T: Float> IntoIterator for SRobotPath<N, T> {
    type Item = SRobotQ<N, T>;
    type IntoIter = std::vec::IntoIter<SRobotQ<N, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.waypoints.into_iter()
    }
}

impl<'a, const N: usize, T: Float> IntoIterator for &'a SRobotPath<N, T> {
    type Item = &'a SRobotQ<N, T>;
    type IntoIter = std::slice::Iter<'a, SRobotQ<N, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.waypoints.iter()
    }
}

impl<const N: usize, T: Float> AsRef<[SRobotQ<N, T>]> for SRobotPath<N, T> {
    fn as_ref(&self) -> &[SRobotQ<N, T>] {
        &self.waypoints
    }
}

impl<const N: usize, T: Float> TryFrom<Vec<SRobotQ<N, T>>> for SRobotPath<N, T> {
    type Error = crate::DekeError;

    fn try_from(waypoints: Vec<SRobotQ<N, T>>) -> Result<Self, Self::Error> {
        Self::try_new(waypoints)
    }
}

impl<const N: usize, T: Float> TryFrom<&[SRobotQ<N, T>]> for SRobotPath<N, T> {
    type Error = crate::DekeError;

    fn try_from(waypoints: &[SRobotQ<N, T>]) -> Result<Self, Self::Error> {
        Self::try_new(waypoints.to_vec())
    }
}

impl<const N: usize, T: Float> TryFrom<&Vec<SRobotQ<N, T>>> for SRobotPath<N, T> {
    type Error = crate::DekeError;

    fn try_from(waypoints: &Vec<SRobotQ<N, T>>) -> Result<Self, Self::Error> {
        Self::try_new(waypoints.clone())
    }
}

impl<const N: usize, T: Float> From<SRobotPath<N, T>> for RobotPath<T> {
    fn from(sp: SRobotPath<N, T>) -> Self {
        sp.to_robot_path()
    }
}

impl<const N: usize, T: Float> From<&SRobotPath<N, T>> for RobotPath<T> {
    fn from(sp: &SRobotPath<N, T>) -> Self {
        sp.to_robot_path()
    }
}

impl<const N: usize, T: Float> TryFrom<RobotPath<T>> for SRobotPath<N, T> {
    type Error = crate::DekeError;

    fn try_from(arr: RobotPath<T>) -> Result<Self, Self::Error> {
        if arr.ncols() != N {
            return Err(crate::DekeError::ShapeMismatch {
                expected: N,
                found: arr.ncols(),
            });
        }
        let mut waypoints = Vec::with_capacity(arr.nrows());
        for row in arr.rows() {
            let mut q = [T::zero(); N];
            for (j, &v) in row.iter().enumerate() {
                q[j] = v;
            }
            waypoints.push(SRobotQ(q));
        }
        Self::try_new(waypoints)
    }
}

impl<const N: usize, F: Float> TryFrom<&RobotPath<F>> for SRobotPath<N, F> {
    type Error = crate::DekeError;

    fn try_from(arr: &RobotPath<F>) -> Result<Self, Self::Error> {
        if arr.ncols() != N {
            return Err(crate::DekeError::ShapeMismatch {
                expected: N,
                found: arr.ncols(),
            });
        }
        let mut waypoints = Vec::with_capacity(arr.nrows());
        for row in arr.rows() {
            let mut q = [F::zero(); N];
            for (j, &v) in row.iter().enumerate() {
                q[j] = v;
            }
            waypoints.push(SRobotQ(q));
        }
        Self::try_new(waypoints)
    }
}

impl<const N: usize> From<SRobotPath<N, f32>> for SRobotPath<N, f64> {
    fn from(path: SRobotPath<N, f32>) -> Self {
        Self {
            first: path.first.into(),
            last: path.last.into(),
            waypoints: path.waypoints.into_iter().map(Into::into).collect(),
        }
    }
}

impl<const N: usize> From<SRobotPath<N, f64>> for SRobotPath<N, f32> {
    fn from(path: SRobotPath<N, f64>) -> Self {
        Self {
            first: path.first.into(),
            last: path.last.into(),
            waypoints: path.waypoints.into_iter().map(Into::into).collect(),
        }
    }
}

impl<const N: usize> From<SRobotPath<N, f32>> for RobotPath<f64> {
    fn from(sp: SRobotPath<N, f32>) -> Self {
        let n = sp.waypoints.len();
        let mut arr = RobotPath::zeros((n, N));
        for (i, q) in sp.waypoints.iter().enumerate() {
            for (j, &v) in q.0.iter().enumerate() {
                arr[[i, j]] = v as f64;
            }
        }
        arr
    }
}

impl<const N: usize> From<SRobotPath<N, f64>> for RobotPath<f32> {
    fn from(sp: SRobotPath<N, f64>) -> Self {
        let n = sp.waypoints.len();
        let mut arr = RobotPath::zeros((n, N));
        for (i, q) in sp.waypoints.iter().enumerate() {
            for (j, &v) in q.0.iter().enumerate() {
                arr[[i, j]] = v as f32;
            }
        }
        arr
    }
}

impl<const N: usize> TryFrom<RobotPath<f64>> for SRobotPath<N, f32> {
    type Error = crate::DekeError;

    fn try_from(arr: RobotPath<f64>) -> Result<Self, Self::Error> {
        if arr.ncols() != N {
            return Err(crate::DekeError::ShapeMismatch {
                expected: N,
                found: arr.ncols(),
            });
        }
        let mut waypoints = Vec::with_capacity(arr.nrows());
        for row in arr.rows() {
            let mut q = [0.0f32; N];
            for (j, &v) in row.iter().enumerate() {
                q[j] = v as f32;
            }
            waypoints.push(SRobotQ(q));
        }
        Self::try_new(waypoints)
    }
}

impl<const N: usize> TryFrom<RobotPath<f32>> for SRobotPath<N, f64> {
    type Error = crate::DekeError;

    fn try_from(arr: RobotPath<f32>) -> Result<Self, Self::Error> {
        if arr.ncols() != N {
            return Err(crate::DekeError::ShapeMismatch {
                expected: N,
                found: arr.ncols(),
            });
        }
        let mut waypoints = Vec::with_capacity(arr.nrows());
        for row in arr.rows() {
            let mut q = [0.0f64; N];
            for (j, &v) in row.iter().enumerate() {
                q[j] = v as f64;
            }
            waypoints.push(SRobotQ(q));
        }
        Self::try_new(waypoints)
    }
}

#[cfg(feature = "valuable")]
mod valuable_impl {
    use super::SRobotPath;
    use ::valuable::{
        Fields, NamedField, NamedValues, StructDef, Structable, Valuable, Value, Visit,
    };

    const FIELDS: &[NamedField<'static>] = &[
        NamedField::new("first"),
        NamedField::new("last"),
        NamedField::new("waypoints"),
    ];

    macro_rules! impl_valuable {
        ($f:ty) => {
            impl<const N: usize> Valuable for SRobotPath<N, $f> {
                #[inline]
                fn as_value(&self) -> Value<'_> {
                    Value::Structable(self)
                }
                fn visit(&self, visit: &mut dyn Visit) {
                    let values = [
                        self.first.as_value(),
                        self.last.as_value(),
                        self.waypoints.as_value(),
                    ];
                    visit.visit_named_fields(&NamedValues::new(FIELDS, &values));
                }
            }

            impl<const N: usize> Structable for SRobotPath<N, $f> {
                fn definition(&self) -> StructDef<'_> {
                    StructDef::new_static("SRobotPath", Fields::Named(FIELDS))
                }
            }
        };
    }

    impl_valuable!(f32);
    impl_valuable!(f64);
}

fn srdp_mark<const N: usize, F: Float>(
    pts: &[SRobotQ<N, F>],
    start: usize,
    end: usize,
    tol: F,
    keep: &mut [bool],
) {
    if end <= start + 1 {
        return;
    }

    let seg = pts[end] - pts[start];
    let seg_len_sq = seg.norm_squared();

    let mut max_dist = F::zero();
    let mut max_idx = start;

    let near_zero: F = F::from(1e-30).unwrap_or_else(F::zero);

    for i in (start + 1)..end {
        let v = pts[i] - pts[start];
        let dist = if seg_len_sq < near_zero {
            v.norm()
        } else {
            let t = (v.dot(&seg) / seg_len_sq).max(F::zero()).min(F::one());
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
        keep[(start + 1)..end].fill(false);
    }
}
