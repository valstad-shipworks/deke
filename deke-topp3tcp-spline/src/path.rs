//! Spline-interpolated joint-space path with deviation-tube refinement.
//!
//! Builds a clamped B-spline through the waypoint joint configurations using
//! chord-length parameterization, refining support-point density until the
//! resulting curve lies within a configurable deviation tube around the
//! original polyline.

use crate::bspline::{BSpline, BcEntry};
use deke_types::{SRobotPath, SRobotQ};

/// Spline-interpolated robot path constrained to stay within a deviation
/// tube around the original piecewise-linear waypoint path.
#[derive(Clone)]
pub struct SplineInterpolatedRobotPath<const N: usize> {
    q_poly: Vec<[f64; N]>,
    s_poly: Vec<f64>,
    splines: Vec<BSpline>,
}

impl<const N: usize> SplineInterpolatedRobotPath<N> {
    /// Build the spline from the given joint-space waypoints.
    ///
    /// `start_direction` / `end_direction` are optional joint-space tangent
    /// directions that pin the spline's first / last derivative; their
    /// magnitude is renormalized to the magnitude of the polyline tangent
    /// at the corresponding endpoint.
    pub fn new(
        waypoints: &[SRobotQ<N, f64>],
        max_deviation: f64,
        max_refine_iters: usize,
        start_direction: Option<&SRobotQ<N, f64>>,
        end_direction: Option<&SRobotQ<N, f64>>,
    ) -> Self {
        assert!(
            waypoints.len() >= 2,
            "path must contain at least two joint states",
        );

        // De-duplicate coincident waypoints.
        let mut dedup: Vec<[f64; N]> = vec![waypoints[0].0];
        for row in &waypoints[1..] {
            let last = *dedup.last().expect("dedup must not be empty");
            let diff: f64 = (0..N)
                .map(|j| (row.0[j] - last[j]).powi(2))
                .sum::<f64>()
                .sqrt();
            if diff > 1e-9 {
                dedup.push(row.0);
            }
        }
        assert!(
            dedup.len() >= 2,
            "path collapsed to a single unique joint state",
        );

        let n_pts = dedup.len();

        // Chord-length parameterization.
        let mut chord = Vec::with_capacity(n_pts - 1);
        for i in 0..(n_pts - 1) {
            let d: f64 = (0..N)
                .map(|j| (dedup[i + 1][j] - dedup[i][j]).powi(2))
                .sum::<f64>()
                .sqrt();
            chord.push(d);
        }
        let mut s_poly = Vec::with_capacity(n_pts);
        s_poly.push(0.0);
        let mut cum = 0.0;
        for &c in &chord {
            cum += c;
            s_poly.push(cum);
        }
        assert!(cum > 1e-12, "path length is zero");
        for s in s_poly.iter_mut() {
            *s /= cum;
        }

        let bc_start = Self::prepare_bc(start_direction, &s_poly, &dedup, true);
        let bc_end = Self::prepare_bc(end_direction, &s_poly, &dedup, false);
        let has_bc = bc_start.is_some() || bc_end.is_some();

        let n_bc = 2 * ((bc_start.is_some() as usize) + (bc_end.is_some() as usize));
        let bc_degree = if has_bc { n_bc + 1 } else { 3 }.max(3);

        let mut support_count = n_pts;
        if has_bc {
            support_count = support_count.max(bc_degree + 1);
        }

        let mut splines: Vec<BSpline> = Vec::new();
        for _ in 0..=max_refine_iters {
            let support_s: Vec<f64> = (0..support_count)
                .map(|i| i as f64 / (support_count - 1) as f64)
                .collect();
            let support_q: Vec<[f64; N]> = support_s
                .iter()
                .map(|&ss| Self::polyline_eval(&dedup, &s_poly, ss))
                .collect();

            let degree = bc_degree.min(support_q.len() - 1);
            let (degree, support_s, support_q) = if degree == 1 {
                let s4: Vec<f64> = (0..4).map(|i| i as f64 / 3.0).collect();
                let q4: Vec<[f64; N]> = s4
                    .iter()
                    .map(|&ss| Self::polyline_eval(&dedup, &s_poly, ss))
                    .collect();
                (3usize, s4, q4)
            } else {
                (degree, support_s, support_q)
            };

            let use_bc = has_bc && degree >= bc_degree;

            splines = (0..N)
                .map(|j| {
                    let vals: Vec<f64> = support_q.iter().map(|row| row[j]).collect();
                    let (bcl, bcr) = if use_bc {
                        Self::bc_entries_for_dof(&bc_start, &bc_end, j)
                    } else {
                        (None, None)
                    };
                    BSpline::interpolate(
                        &support_s,
                        &vals,
                        degree,
                        bcl.as_deref(),
                        bcr.as_deref(),
                    )
                })
                .collect();

            if Self::estimate_max_deviation(&splines, &dedup, n_pts) <= max_deviation {
                break;
            }
            support_count = ((support_count * 2) - 1).min(4097);
        }
        if Self::estimate_max_deviation(&splines, &dedup, n_pts) > max_deviation {
            eprintln!("WARNING: failed to find a path that passed max deviation!");
        }

        Self {
            q_poly: dedup,
            s_poly,
            splines,
        }
    }

    /// Convenience: build directly from a [`SRobotPath`].
    pub fn from_path(
        path: &SRobotPath<N, f64>,
        max_deviation: f64,
        max_refine_iters: usize,
        start_direction: Option<&SRobotQ<N, f64>>,
        end_direction: Option<&SRobotQ<N, f64>>,
    ) -> Self {
        let waypoints: Vec<SRobotQ<N, f64>> = path.iter().copied().collect();
        Self::new(
            &waypoints,
            max_deviation,
            max_refine_iters,
            start_direction,
            end_direction,
        )
    }

    fn prepare_bc(
        direction: Option<&SRobotQ<N, f64>>,
        s_poly: &[f64],
        q_poly: &[[f64; N]],
        start: bool,
    ) -> Option<[f64; N]> {
        let dir = direction?.0;
        let d_norm = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        if d_norm < 1e-12 {
            return None;
        }
        let d: [f64; N] = std::array::from_fn(|j| dir[j] / d_norm);
        let (a, b, ds) = if start {
            (q_poly[0], q_poly[1], s_poly[1] - s_poly[0])
        } else {
            let last = q_poly.len() - 1;
            (q_poly[last - 1], q_poly[last], s_poly[last] - s_poly[last - 1])
        };
        let tangent: [f64; N] = if ds > 1e-12 {
            std::array::from_fn(|j| (b[j] - a[j]) / ds)
        } else {
            std::array::from_fn(|j| b[j] - a[j])
        };
        let t_norm = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();
        Some(std::array::from_fn(|j| d[j] * t_norm))
    }

    fn bc_entries_for_dof(
        bc_start: &Option<[f64; N]>,
        bc_end: &Option<[f64; N]>,
        j: usize,
    ) -> (Option<Vec<BcEntry>>, Option<Vec<BcEntry>>) {
        let left = bc_start
            .as_ref()
            .map(|v| vec![(1usize, v[j]), (2usize, 0.0)]);
        let right = bc_end
            .as_ref()
            .map(|v| vec![(1usize, v[j]), (2usize, 0.0)]);
        (left, right)
    }

    fn polyline_eval(q_poly: &[[f64; N]], s_poly: &[f64], s: f64) -> [f64; N] {
        let ss = s.clamp(0.0, 1.0);
        if ss <= 0.0 {
            return q_poly[0];
        }
        if ss >= 1.0 {
            return q_poly[q_poly.len() - 1];
        }
        let raw = s_poly.partition_point(|&v| v <= ss);
        let idx = raw.saturating_sub(1).min(s_poly.len() - 2);
        let (s0, s1) = (s_poly[idx], s_poly[idx + 1]);
        if s1 - s0 <= 1e-12 {
            return q_poly[idx];
        }
        let t = (ss - s0) / (s1 - s0);
        std::array::from_fn(|j| (1.0 - t) * q_poly[idx][j] + t * q_poly[idx + 1][j])
    }

    fn point_polyline_distance(point: &[f64; N], q_poly: &[[f64; N]]) -> f64 {
        let mut best = f64::INFINITY;
        for i in 0..(q_poly.len() - 1) {
            let a = &q_poly[i];
            let b = &q_poly[i + 1];
            let mut d_sq = 0.0;
            let mut ap_dot_ab = 0.0;
            for j in 0..N {
                let abj = b[j] - a[j];
                d_sq += abj * abj;
                ap_dot_ab += (point[j] - a[j]) * abj;
            }
            let t = if d_sq > 1e-18 {
                (ap_dot_ab / d_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut dist_sq = 0.0;
            for j in 0..N {
                let abj = b[j] - a[j];
                let d = point[j] - a[j] - t * abj;
                dist_sq += d * d;
            }
            let dist = dist_sq.sqrt();
            if dist < best {
                best = dist;
            }
        }
        best
    }

    fn eval_point(splines: &[BSpline], s: f64) -> [f64; N] {
        let ss = s.clamp(0.0, 1.0);
        std::array::from_fn(|j| splines[j].eval(ss))
    }

    fn estimate_max_deviation(splines: &[BSpline], q_poly: &[[f64; N]], n_pts: usize) -> f64 {
        let n_samples = 500usize.max(n_pts * 16);
        let mut max_dev = 0.0f64;
        for i in 0..=n_samples {
            let s = i as f64 / n_samples as f64;
            let pt = Self::eval_point(splines, s);
            let dev = Self::point_polyline_distance(&pt, q_poly);
            if dev > max_dev {
                max_dev = dev;
            }
        }
        max_dev
    }

    /// Polyline waypoints actually used by the spline (post de-duplication).
    pub fn polyline_waypoints(&self) -> &[[f64; N]] {
        &self.q_poly
    }

    /// Normalized arc-length parameter at each polyline waypoint.
    pub fn polyline_s(&self) -> &[f64] {
        &self.s_poly
    }

    /// Evaluate the spline path at parameter `s` ∈ [0, 1], returning
    /// `(q, q', q'', q''')` where primes denote derivatives w.r.t. `s`.
    pub fn eval(
        &self,
        s: f64,
    ) -> (
        SRobotQ<N, f64>,
        SRobotQ<N, f64>,
        SRobotQ<N, f64>,
        SRobotQ<N, f64>,
    ) {
        let ss = s.clamp(0.0, 1.0);
        let mut q = [0.0f64; N];
        let mut qp = [0.0f64; N];
        let mut qpp = [0.0f64; N];
        let mut qppp = [0.0f64; N];
        for j in 0..N {
            q[j] = self.splines[j].eval_deriv(ss, 0);
            qp[j] = self.splines[j].eval_deriv(ss, 1);
            qpp[j] = self.splines[j].eval_deriv(ss, 2);
            qppp[j] = self.splines[j].eval_deriv(ss, 3);
        }
        (
            SRobotQ::from_array(q),
            SRobotQ::from_array(qp),
            SRobotQ::from_array(qpp),
            SRobotQ::from_array(qppp),
        )
    }
}
