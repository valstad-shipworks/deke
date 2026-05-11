use deke_types::{DekeError, DekeResult, FKChain, SRobotPath, SRobotQ};
use glam_traits_ext::{TAffine3, TVec3};

/// Constant geometric path data needed by the retimer NLP.
///
/// Holds the cumulative arc length `s[k]`, the segment lengths `ds[k]`, and analytical
/// derivatives of the joint-space path `q(s)` and the tool-center-point path `p(s)` evaluated
/// at each densified waypoint. The derivatives are read off a chord-length-parameterised
/// natural cubic spline, not finite differences — that keeps `ppp`/`qppp` bounded at corners of
/// a densified polyline rather than spiking, which is what the NLP needs to stay feasible.
///
/// For a natural cubic spline the fourth derivative is identically zero in the interior of
/// every segment, so `pppp`/`qppp` would normally be omitted; we keep `pppp` as a zero-filled
/// vector matching the length of `pp`/`ppp` so downstream code (constraints, diagnostics) can
/// reference it uniformly.
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
/// natural cubic spline through `y[0..m]` with knot spacing `ds[0..m-1]`.
///
/// Each dimension is fitted independently. With natural boundary conditions M_0 = M_{m-1} = 0,
/// the 2nd derivative is exactly `M_k` at each knot, the 1st derivative is continuous (C^1) and
/// read from either adjacent segment, and the 3rd derivative is piecewise constant per segment;
/// at interior knots we average the two adjacent segment values, at the boundaries we use the
/// single available segment value.
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

    let big_m = solve_natural_cubic::<D>(y, ds);

    for k in 0..m {
        ypp[k] = big_m[k];
    }

    for k in 0..m - 1 {
        let h = ds[k];
        for d in 0..D {
            yp[k][d] = -big_m[k][d] * h / 2.0 + (y[k + 1][d] - y[k][d]) / h
                - (big_m[k + 1][d] - big_m[k][d]) * h / 6.0;
        }
    }
    {
        let k = m - 2;
        let h = ds[k];
        for d in 0..D {
            yp[m - 1][d] = big_m[k + 1][d] * h / 2.0
                + (y[k + 1][d] - y[k][d]) / h
                - (big_m[k + 1][d] - big_m[k][d]) * h / 6.0;
        }
    }

    for d in 0..D {
        yppp[0][d] = (big_m[1][d] - big_m[0][d]) / ds[0];
        yppp[m - 1][d] = (big_m[m - 1][d] - big_m[m - 2][d]) / ds[m - 2];
    }
    for k in 1..m - 1 {
        for d in 0..D {
            let left = (big_m[k][d] - big_m[k - 1][d]) / ds[k - 1];
            let right = (big_m[k + 1][d] - big_m[k][d]) / ds[k];
            yppp[k][d] = 0.5 * (left + right);
        }
    }

    (yp, ypp, yppp)
}

/// Solves the natural-cubic-spline tridiagonal system per dimension and returns the second
/// derivative M_k at each of the `m` knots. `m` must be >= 3; smaller cases are handled by the
/// caller.
fn solve_natural_cubic<const D: usize>(y: &[[f64; D]], ds: &[f64]) -> Vec<[f64; D]> {
    let m = y.len();
    let mut big_m = vec![[0.0_f64; D]; m];
    let n = m - 2;
    if n == 0 {
        return big_m;
    }

    let mut diag = vec![0.0_f64; n];
    let mut sub = vec![0.0_f64; n];
    let mut sup = vec![0.0_f64; n];
    for i in 0..n {
        let h_left = ds[i];
        let h_right = ds[i + 1];
        diag[i] = 2.0 * (h_left + h_right);
        sub[i] = h_left;
        sup[i] = h_right;
    }

    let mut c_prime = vec![0.0_f64; n];
    let mut rhs = vec![0.0_f64; n];
    let mut d_prime = vec![0.0_f64; n];

    for d in 0..D {
        for i in 0..n {
            let h_left = ds[i];
            let h_right = ds[i + 1];
            rhs[i] = 6.0
                * ((y[i + 2][d] - y[i + 1][d]) / h_right
                    - (y[i + 1][d] - y[i][d]) / h_left);
        }

        c_prime[0] = sup[0] / diag[0];
        d_prime[0] = rhs[0] / diag[0];
        for i in 1..n {
            let denom = diag[i] - sub[i] * c_prime[i - 1];
            if i < n - 1 {
                c_prime[i] = sup[i] / denom;
            }
            d_prime[i] = (rhs[i] - sub[i] * d_prime[i - 1]) / denom;
        }

        let mut x = vec![0.0_f64; n];
        x[n - 1] = d_prime[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }
        for i in 0..n {
            big_m[i + 1][d] = x[i];
        }
    }

    big_m
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
