//! Stage A — condition a raw Cartesian polyline into smooth, arc-length
//! parameterised runs.
//!
//! The polyline is split at *sharp* corners (turn angle above a threshold) into
//! runs that start and stop at rest. Inside a run, shallow corners are smoothed
//! by a Catmull–Rom [`squiggle::Spline`] through the vertices (it stays straight on
//! collinear stretches and bows through a shallow turn).
//! Orientation is carried as a unit quaternion slerped between the run's
//! vertices, keyed on arc length. Position geometry runs in `f32` (squiggle); poses
//! are returned in `f64` for the kinematics downstream.

use deke_types::glam::{DAffine3, DQuat, DVec3};
use glam::Vec3;
use squiggle::{Segmented, Spline};

use crate::constraints::PathConditioning;
use crate::error::LinearError;

/// One arc-length-parameterised Cartesian segment with no sharp corners.
#[derive(Clone, Debug)]
pub struct CartesianRun {
    spline: Spline,
    ts: Vec<f32>,
    ss: Vec<f32>,
    vtx_s: Vec<f64>,
    vtx_q: Vec<DQuat>,
    length: f64,
    weave: Option<crate::weave::WeaveOptions>,
}

impl CartesianRun {
    /// Total seam arc length of the run (metres). Unchanged by a weave overlay —
    /// the weave is transverse to this length, not added to it.
    pub fn length(&self) -> f64 {
        self.length
    }

    /// Overlay a spatial weave: `eval` then traces the seam pose offset transversely
    /// by the weave, so the planner follows the oscillation with no other change.
    pub fn with_weave(mut self, weave: crate::weave::WeaveOptions) -> Self {
        self.weave = Some(weave);
        self
    }

    /// Pose at seam arc length `s ∈ [0, length]`. With a weave overlay the
    /// translation is offset transversely (in the tool frame) by the weave; the
    /// orientation is unchanged (positional weave only).
    pub fn eval(&self, s: f64) -> DAffine3 {
        let s = s.clamp(0.0, self.length);
        let t = self.t_at_s(s as f32);
        let p = self.spline.point(t);
        let mut pos = DVec3::new(p.x as f64, p.y as f64, p.z as f64);
        let rot = self.orientation_at(s);
        if let Some(w) = &self.weave {
            pos += (rot * w.axis.vector()) * w.offset(s, self.length);
        }
        DAffine3::from_rotation_translation(rot, pos)
    }

    fn t_at_s(&self, s: f32) -> f32 {
        let s = s.clamp(0.0, *self.ss.last().unwrap_or(&0.0));
        match self.ss.binary_search_by(|v| v.partial_cmp(&s).unwrap()) {
            Ok(i) => self.ts[i],
            Err(i) => {
                if i == 0 {
                    return self.ts[0];
                }
                if i >= self.ss.len() {
                    return *self.ts.last().unwrap();
                }
                let (s0, s1) = (self.ss[i - 1], self.ss[i]);
                let (t0, t1) = (self.ts[i - 1], self.ts[i]);
                let f = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
                t0 + (t1 - t0) * f
            }
        }
    }

    fn orientation_at(&self, s: f64) -> DQuat {
        if self.vtx_q.len() == 1 {
            return self.vtx_q[0];
        }
        let mut k = 0usize;
        while k + 2 < self.vtx_s.len() && s > self.vtx_s[k + 1] {
            k += 1;
        }
        let (s0, s1) = (self.vtx_s[k], self.vtx_s[k + 1]);
        let f = if s1 > s0 {
            ((s - s0) / (s1 - s0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.vtx_q[k].slerp(self.vtx_q[k + 1], f)
    }
}

/// Split `poses` at sharp corners and condition each run.
pub fn condition(
    poses: &[DAffine3],
    cfg: &PathConditioning,
) -> Result<Vec<CartesianRun>, LinearError> {
    if poses.len() < 2 {
        return Err(LinearError::TooFewPoses(poses.len()));
    }
    // A non-finite pose survives the translation-distance dedup (`inf > 1e-9` is
    // true) and poisons the spline/arc tables → downstream usize-overflow abort or
    // a `partial_cmp().unwrap()` panic. Reject it here.
    if poses
        .iter()
        .any(|t| !t.translation.is_finite() || !t.matrix3.is_finite())
    {
        return Err(LinearError::NonFiniteInput);
    }

    let p: Vec<DVec3> = poses.iter().map(|t| t.translation).collect();

    let mut bounds = vec![0usize];
    for i in 1..poses.len() - 1 {
        if turn_angle(&p, i).is_some_and(|a| a > cfg.sharp_corner_angle) {
            bounds.push(i);
        }
    }
    bounds.push(poses.len() - 1);

    let mut runs = Vec::with_capacity(bounds.len() - 1);
    for (run_idx, w) in bounds.windows(2).enumerate() {
        runs.push(build_run(&poses[w[0]..=w[1]], run_idx)?);
    }
    Ok(runs)
}

fn turn_angle(p: &[DVec3], i: usize) -> Option<f64> {
    let a = (p[i] - p[i - 1]).normalize_or_zero();
    let b = (p[i + 1] - p[i]).normalize_or_zero();
    if a == DVec3::ZERO || b == DVec3::ZERO {
        return None;
    }
    Some(a.dot(b).clamp(-1.0, 1.0).acos())
}

fn build_run(poses: &[DAffine3], run_idx: usize) -> Result<CartesianRun, LinearError> {
    // Drop consecutive coincident vertices so the spline and slerp stay well-formed.
    let mut keep: Vec<usize> = vec![0];
    for i in 1..poses.len() {
        let prev = poses[*keep.last().unwrap()].translation;
        if poses[i].translation.distance(prev) > 1e-9 {
            keep.push(i);
        }
    }
    if keep.len() < 2 {
        return Err(LinearError::DegenerateRun { run: run_idx });
    }

    let knots: Vec<Vec3> = keep
        .iter()
        .map(|&i| {
            let t = poses[i].translation;
            Vec3::new(t.x as f32, t.y as f32, t.z as f32)
        })
        .collect();
    let spline = Spline::new(knots.clone());

    let mut vtx_q: Vec<DQuat> = keep
        .iter()
        .map(|&i| DQuat::from_mat3(&poses[i].matrix3).normalize())
        .collect();
    // Make consecutive quaternions take the short path so slerp never flips.
    for k in 1..vtx_q.len() {
        if vtx_q[k - 1].dot(vtx_q[k]) < 0.0 {
            vtx_q[k] = -vtx_q[k];
        }
    }

    let seg = spline.segment_count().max(1);
    let m = (seg * 24).max(64);
    let mut ts = Vec::with_capacity(m + 1);
    let mut ss = Vec::with_capacity(m + 1);
    let mut prev = spline.point(0.0);
    let mut acc = 0.0f32;
    ts.push(0.0);
    ss.push(0.0);
    for i in 1..=m {
        let t = i as f32 / m as f32;
        let pt = spline.point(t);
        acc += (pt - prev).length();
        prev = pt;
        ts.push(t);
        ss.push(acc);
    }
    let length = acc as f64;
    if length < 1e-9 {
        return Err(LinearError::DegenerateRun { run: run_idx });
    }

    // Arc length at each retained vertex: each is a knot at t = index / seg.
    let mut vtx_s: Vec<f64> = (0..knots.len())
        .map(|i| {
            let t = i as f32 / seg as f32;
            s_at_t(&ts, &ss, t) as f64
        })
        .collect();
    vtx_s[0] = 0.0;
    *vtx_s.last_mut().unwrap() = length;

    Ok(CartesianRun {
        spline,
        ts,
        ss,
        vtx_s,
        vtx_q,
        length,
        weave: None,
    })
}

fn s_at_t(ts: &[f32], ss: &[f32], t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match ts.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
        Ok(i) => ss[i],
        Err(i) => {
            if i == 0 {
                return ss[0];
            }
            if i >= ts.len() {
                return *ss.last().unwrap();
            }
            let (t0, t1) = (ts[i - 1], ts[i]);
            let (s0, s1) = (ss[i - 1], ss[i]);
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            s0 + (s1 - s0) * f
        }
    }
}
