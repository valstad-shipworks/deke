use deke_types::{DekeError, DekeResult, FKChain, SRobotPath, SRobotQ};
use glam_traits_ext::{TAffine3, TVec3};

/// Constant geometric path data needed by the retimer NLP.
///
/// Holds the cumulative arc length `s[k]`, the segment lengths `ds[k]`, and analytical
/// derivatives of the joint-space path `q(s)` and the tool-center-point path `p(s)` evaluated
/// at each densified waypoint. Derivatives come from a chord-length-parameterised PCHIP
/// (Fritsch-Carlson monotone cubic Hermite) interpolant, not finite differences and not a
/// natural cubic spline — natural cubic spline overshoots near polyline corners (driving
/// `ppp` to large values right where the TCP a/j constraints are tightest), and FD spikes are
/// even worse. PCHIP is C¹ rather than C², but its monotonicity-preserving slopes keep
/// derivatives bounded at corners.
///
/// The fourth derivative is identically zero in the interior of every PCHIP segment (still
/// cubic), so `pppp` would normally be omitted; we keep it as a zero-filled vector matching
/// the length of `pp`/`ppp` so downstream code (constraints, diagnostics) can reference it
/// uniformly.
#[derive(Debug, Clone)]
pub struct PathDerivatives<const N: usize> {
    pub waypoints: Vec<SRobotQ<N, f64>>,
    pub s: Vec<f64>,
    pub ds: Vec<f64>,
    pub qp: Vec<[f64; N]>,
    pub qpp: Vec<[f64; N]>,
    pub qppp: Vec<[f64; N]>,
    pub pp: Vec<[f64; 3]>,
    pub ppp: Vec<[f64; 3]>,
    pub pppp: Vec<[f64; 3]>,
    pub tcp: Vec<[f64; 3]>,
}

impl<const N: usize> PathDerivatives<N> {
    /// Verifies that the first `locked_prefix` joints are constant across every waypoint in the
    /// input path. Returns the first violation as a [`DekeError::LockedPrefixViolation`].
    pub fn check_locked_prefix(
        path: &SRobotPath<N, f64>,
        locked_prefix: usize,
    ) -> DekeResult<()> {
        if locked_prefix == 0 {
            return Ok(());
        }
        let lock = locked_prefix.min(N);
        let first = path.first().0;
        for (i, wp) in path.iter().enumerate() {
            for j in 0..lock {
                if (wp.0[j] - first[j]).abs() > 1e-5 {
                    return Err(DekeError::LockedPrefixViolation {
                        waypoint: i as u32,
                        joint: j as u8,
                    });
                }
            }
        }
        Ok(())
    }

    pub fn new<FK: FKChain<N, f64>>(
        densified: &SRobotPath<N, f64>,
        fk: &FK,
    ) -> DekeResult<Self> {
        Self::build(densified, Some(fk))
    }

    /// Same as [`Self::new`] but skips all forward-kinematics evaluation. The resulting
    /// `PathDerivatives` has empty `tcp`, `pp`, `ppp`, and `pppp` vectors; callers must gate any
    /// TCP logic on `has_tcp()`.
    pub fn new_without_tcp(densified: &SRobotPath<N, f64>) -> DekeResult<Self> {
        Self::build::<crate::path_derivatives::NeverFK<N>>(densified, None)
    }

    fn build<FK: FKChain<N, f64>>(
        densified: &SRobotPath<N, f64>,
        fk: Option<&FK>,
    ) -> DekeResult<Self> {
        let m = densified.len();
        if m < 2 {
            return Err(DekeError::PathTooShort(m));
        }

        let mut ds = Vec::with_capacity(m - 1);
        let mut s = Vec::with_capacity(m);
        s.push(0.0_f64);
        let mut total = 0.0_f64;
        for k in 0..m - 1 {
            let a = densified.get(k).unwrap().0;
            let b = densified.get(k + 1).unwrap().0;
            let mut sq = 0.0_f64;
            for j in 0..N {
                let d = b[j] - a[j];
                sq += d * d;
            }
            let d = sq.sqrt();
            if d < 1e-9 {
                return Err(DekeError::DuplicateWaypoints);
            }
            ds.push(d);
            total += d;
            s.push(total);
        }

        let wps_arr: Vec<[f64; N]> = densified.iter().map(|wp| wp.0).collect();

        let (qp, qpp, qppp) = spline_derivatives::<N>(&wps_arr, &ds);

        let (tcp, pp, ppp, pppp) = if let Some(fk) = fk {
            let mut tcp = Vec::with_capacity(m);
            for wp in densified.iter() {
                let pose = fk.fk_end(wp).map_err(|e| e.into())?;
                let t = pose.translation();
                tcp.push([t.x(), t.y(), t.z()]);
            }
            let (pp, ppp, _pppp) = spline_derivatives::<3>(&tcp, &ds);
            let pppp = vec![[0.0_f64; 3]; m];
            (tcp, pp, ppp, pppp)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        Ok(Self {
            waypoints: densified.iter().copied().collect(),
            s,
            ds,
            qp,
            qpp,
            qppp,
            pp,
            ppp,
            pppp,
            tcp,
        })
    }

    pub fn num_waypoints(&self) -> usize {
        self.waypoints.len()
    }

    pub fn num_segments(&self) -> usize {
        self.ds.len()
    }

    pub fn total_length(&self) -> f64 {
        *self.s.last().unwrap_or(&0.0)
    }

    /// Returns true when TCP constants (`pp`, `ppp`, `pppp`, `tcp`) have been populated. False
    /// when the derivatives were built via [`Self::new_without_tcp`].
    pub fn has_tcp(&self) -> bool {
        !self.pp.is_empty()
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct NeverFK<const N: usize>;

impl<const N: usize> FKChain<N, f64> for NeverFK<N> {
    type Error = DekeError;
    fn fk(
        &self,
        _q: &deke_types::SRobotQ<N, f64>,
    ) -> Result<[deke_types::glam::DAffine3; N], Self::Error> {
        unreachable!("NeverFK is a placeholder — PathDerivatives::new_without_tcp never calls FK")
    }
    fn fk_end(
        &self,
        _q: &deke_types::SRobotQ<N, f64>,
    ) -> Result<deke_types::glam::DAffine3, Self::Error> {
        unreachable!()
    }
    fn joint_axes_positions(
        &self,
        _q: &deke_types::SRobotQ<N, f64>,
    ) -> Result<
        (
            [deke_types::glam::DVec3; N],
            [deke_types::glam::DVec3; N],
            deke_types::glam::DVec3,
        ),
        Self::Error,
    > {
        unreachable!()
    }
}

/// Analytical 1st, 2nd, and 3rd derivatives at every knot of a chord-length-parameterised
/// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) through `y[0..m]` with knot
/// spacing `ds[0..m-1]`.
///
/// We use Fritsch-Carlson monotone slopes — that's the whole point: a natural cubic spline
/// through a polyline (joint-space waypoint list) overshoots near corners because it
/// enforces C² continuity, and the overshoot lands exactly where the TCP a/j constraints
/// are tightest. PCHIP gives up C² (it's only C¹) but in exchange forbids overshoot per
/// segment. The 2nd derivative is discontinuous at knots; we average the two adjacent
/// segment values to land on a single value per knot. The 3rd derivative is piecewise
/// constant per segment; same averaging at interior knots.
///
/// Per-dimension; each output dimension fitted independently.
fn spline_derivatives<const D: usize>(
    y: &[[f64; D]],
    ds: &[f64],
) -> (Vec<[f64; D]>, Vec<[f64; D]>, Vec<[f64; D]>) {
    let m = y.len();
    let mut yp = vec![[0.0_f64; D]; m];
    let mut ypp = vec![[0.0_f64; D]; m];
    let mut yppp = vec![[0.0_f64; D]; m];

    if m < 2 {
        return (yp, ypp, yppp);
    }

    if m == 2 {
        let h = ds[0];
        for d in 0..D {
            let slope = (y[1][d] - y[0][d]) / h;
            yp[0][d] = slope;
            yp[1][d] = slope;
        }
        return (yp, ypp, yppp);
    }

    let mut delta = vec![[0.0_f64; D]; m - 1];
    for k in 0..m - 1 {
        let inv_h = 1.0 / ds[k];
        for d in 0..D {
            delta[k][d] = (y[k + 1][d] - y[k][d]) * inv_h;
        }
    }

    let slopes = pchip_slopes::<D>(&delta, ds);
    yp.copy_from_slice(&slopes);

    // ypp on segment k at t=0: (6·Δ_k - 4·d_k - 2·d_{k+1}) / h_k
    // ypp on segment k at t=1: (-6·Δ_k + 2·d_k + 4·d_{k+1}) / h_k
    for d in 0..D {
        ypp[0][d] = (6.0 * delta[0][d] - 4.0 * slopes[0][d] - 2.0 * slopes[1][d]) / ds[0];
        ypp[m - 1][d] = (-6.0 * delta[m - 2][d]
            + 2.0 * slopes[m - 2][d]
            + 4.0 * slopes[m - 1][d])
            / ds[m - 2];
    }
    for k in 1..m - 1 {
        let h_l = ds[k - 1];
        let h_r = ds[k];
        for d in 0..D {
            let left = (-6.0 * delta[k - 1][d]
                + 2.0 * slopes[k - 1][d]
                + 4.0 * slopes[k][d])
                / h_l;
            let right = (6.0 * delta[k][d]
                - 4.0 * slopes[k][d]
                - 2.0 * slopes[k + 1][d])
                / h_r;
            ypp[k][d] = 0.5 * (left + right);
        }
    }

    // yppp on segment k: (-12·Δ_k + 6·(d_k + d_{k+1})) / h_k²  (constant per segment)
    for d in 0..D {
        let h0 = ds[0];
        yppp[0][d] = (-12.0 * delta[0][d] + 6.0 * (slopes[0][d] + slopes[1][d])) / (h0 * h0);
        let hl = ds[m - 2];
        yppp[m - 1][d] = (-12.0 * delta[m - 2][d]
            + 6.0 * (slopes[m - 2][d] + slopes[m - 1][d]))
            / (hl * hl);
    }
    for k in 1..m - 1 {
        let h_l = ds[k - 1];
        let h_r = ds[k];
        for d in 0..D {
            let left = (-12.0 * delta[k - 1][d]
                + 6.0 * (slopes[k - 1][d] + slopes[k][d]))
                / (h_l * h_l);
            let right = (-12.0 * delta[k][d]
                + 6.0 * (slopes[k][d] + slopes[k + 1][d]))
                / (h_r * h_r);
            yppp[k][d] = 0.5 * (left + right);
        }
    }

    (yp, ypp, yppp)
}

/// Fritsch-Carlson PCHIP slopes at each knot, per dimension. Interior slopes use the
/// weighted harmonic mean of the secant slopes when those secants share a sign, and zero
/// at extrema (sign change or zero secant). Boundary slopes use the standard three-point
/// asymmetric formula with the two limiter checks recommended by Fritsch & Carlson —
/// (i) zero out slopes that point against the adjacent secant, (ii) cap to 3× the secant
/// to keep the cubic monotone on the boundary segment.
///
/// `delta[k]` must be `(y[k+1] - y[k]) / ds[k]`. `m = delta.len() + 1` is implicit.
fn pchip_slopes<const D: usize>(delta: &[[f64; D]], ds: &[f64]) -> Vec<[f64; D]> {
    let segs = delta.len();
    let m = segs + 1;
    let mut slopes = vec![[0.0_f64; D]; m];
    if segs == 0 {
        return slopes;
    }
    if segs == 1 {
        slopes[0] = delta[0];
        slopes[1] = delta[0];
        return slopes;
    }

    // Per-dimension max-secant magnitude. Used below to distinguish a "real" extremum
    // (sign change between large secants) from a "smooth small flip" (sign change between
    // small secants on a smoothly oscillating curve, e.g. one of many sin/cos extrema).
    // PCHIP's standard "slope = 0 on sign change" rule is right for the former and
    // overly-conservative for the latter: it creates a pseudo-corner at every smooth
    // extremum on a long sinusoidal path, which the IPM has to grind through.
    let mut max_abs = [0.0_f64; D];
    for k in 0..segs {
        for d in 0..D {
            let a = delta[k][d].abs();
            if a > max_abs[d] {
                max_abs[d] = a;
            }
        }
    }
    let small_flip_thresh: [f64; D] = {
        let mut out = [0.0_f64; D];
        for d in 0..D {
            out[d] = 0.05 * max_abs[d];
        }
        out
    };

    for k in 1..m - 1 {
        let h_l = ds[k - 1];
        let h_r = ds[k];
        let w1 = 2.0 * h_r + h_l;
        let w2 = h_r + 2.0 * h_l;
        for d in 0..D {
            let dl = delta[k - 1][d];
            let dr = delta[k][d];
            slopes[k][d] = if dl * dr <= 0.0 {
                let small_flip = dl.abs().max(dr.abs()) < small_flip_thresh[d];
                if small_flip {
                    // Smooth small-amplitude flip — use centered FD so the resulting
                    // cubic stays close to the actual smooth curve. We give up strict
                    // monotonicity on the two adjacent segments (the cubic may overshoot
                    // by O(small_flip_thresh × h)), which is a tiny relative overshoot
                    // and not where the constraint pressure actually lives.
                    let span = h_l + h_r;
                    (dl * h_l + dr * h_r) / span
                } else {
                    // Real extremum or corner — slope must be 0 to keep the cubic monotone
                    // on both adjacent segments.
                    0.0
                }
            } else {
                // Weighted harmonic mean. Both `dl` and `dr` strictly positive or strictly
                // negative here, so neither denominator term blows up.
                (w1 + w2) / (w1 / dl + w2 / dr)
            };
        }
    }

    // Boundary slopes: three-point asymmetric formula with two limiter checks.
    for d in 0..D {
        slopes[0][d] = boundary_slope(ds[0], ds[1], delta[0][d], delta[1][d]);
        slopes[m - 1][d] = boundary_slope(
            ds[segs - 1],
            ds[segs - 2],
            delta[segs - 1][d],
            delta[segs - 2][d],
        );
    }

    // Defensive fallback for sharp directional corners. PCHIP zeros the per-dimension
    // slope at every extremum; when *every* dimension simultaneously has a zero secant on
    // one side (a sharp corner where one set of joints stops and a different set starts),
    // the entire `qp` vector goes to zero. The chord-length parameter is still advancing,
    // so the resulting path-derivative inequality `qp_j·sd ≤ v_max` is vacuous in every
    // joint and the NLP's constraint Jacobian goes rank-deficient — Sleipnir's KKT
    // factorization then reads past the end of an Eigen vector and aborts the process.
    // Replace those samples with the centered-FD slope `(y[k+1] − y[k−1]) / (s[k+1] − s[k−1])`,
    // which gives a meaningful averaged direction at the cost of giving up local
    // monotonicity on the two adjacent segments — a much better trade than crashing.
    let max_norm_sq = slopes
        .iter()
        .map(|s| s.iter().map(|x| x * x).sum::<f64>())
        .fold(0.0_f64, f64::max);
    if max_norm_sq > 0.0 {
        let threshold_sq = 1e-12 * max_norm_sq;
        for k in 1..m - 1 {
            let norm_sq: f64 = slopes[k].iter().map(|x| x * x).sum();
            if norm_sq < threshold_sq {
                let span = ds[k - 1] + ds[k];
                for d in 0..D {
                    // delta[k-1] · ds[k-1] = y[k] - y[k-1];
                    // delta[k]   · ds[k]   = y[k+1] - y[k].
                    // Centered FD = ((y[k+1] - y[k-1])) / span.
                    let dy = delta[k - 1][d] * ds[k - 1] + delta[k][d] * ds[k];
                    slopes[k][d] = dy / span;
                }
            }
        }
    }

    slopes
}

/// PCHIP boundary slope: three-point asymmetric formula with the Fritsch-Carlson limiters.
/// `h_near, h_far` are the two segment widths next to the boundary; `d_near, d_far` are
/// the corresponding secant slopes.
fn boundary_slope(h_near: f64, h_far: f64, d_near: f64, d_far: f64) -> f64 {
    let raw = ((2.0 * h_near + h_far) * d_near - h_near * d_far) / (h_near + h_far);
    if raw * d_near <= 0.0 {
        0.0
    } else if d_near * d_far <= 0.0 && raw.abs() > 3.0 * d_near.abs() {
        3.0 * d_near
    } else {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_linear_data_recovers_slope() {
        let f: Vec<[f64; 1]> = (0..5).map(|i| [2.0 * i as f64 * 0.5]).collect();
        let ds = vec![0.5; 4];
        let (yp, ypp, yppp) = spline_derivatives::<1>(&f, &ds);
        for v in &yp {
            assert!((v[0] - 2.0).abs() < 1e-9, "yp = {}", v[0]);
        }
        for v in &ypp {
            assert!(v[0].abs() < 1e-9, "ypp = {}", v[0]);
        }
        for v in &yppp {
            assert!(v[0].abs() < 1e-9, "yppp = {}", v[0]);
        }
    }

    #[test]
    fn spline_quadratic_interior_second_derivative_bounded() {
        // f(s) = s^2 on a uniform grid. The natural cubic spline does NOT reproduce f
        // exactly (its second derivative is pinned to zero at the boundaries), but the
        // interior M_k must stay bounded near f''=2 — and that's all we need for a stable
        // NLP signal. Compare to the FD approach which gave exactly 2 everywhere.
        let ds = vec![0.25_f64; 16];
        let f: Vec<[f64; 1]> = (0..17).map(|i| [(i as f64 * 0.25).powi(2)]).collect();
        let (_yp, ypp, _yppp) = spline_derivatives::<1>(&f, &ds);
        // Interior samples (away from boundary) should be close to 2.
        for k in 4..13 {
            assert!(
                (ypp[k][0] - 2.0).abs() < 0.5,
                "ypp[{}] = {} not near 2",
                k,
                ypp[k][0]
            );
        }
    }

    #[test]
    fn spline_two_waypoints_handled() {
        let f: Vec<[f64; 1]> = vec![[0.0], [1.0]];
        let ds = vec![1.0];
        let (yp, ypp, yppp) = spline_derivatives::<1>(&f, &ds);
        assert_eq!(yp.len(), 2);
        assert!((yp[0][0] - 1.0).abs() < 1e-12);
        assert!((yp[1][0] - 1.0).abs() < 1e-12);
        assert!(ypp[0][0].abs() < 1e-12);
        assert!(ypp[1][0].abs() < 1e-12);
        assert!(yppp[0][0].abs() < 1e-12);
        assert!(yppp[1][0].abs() < 1e-12);
    }
}
