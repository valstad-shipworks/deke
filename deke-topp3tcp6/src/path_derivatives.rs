use deke_types::{DekeError, DekeResult, FKChain, SRobotPath, SRobotQ};

/// Constant geometric path data needed by the retimer NLP.
///
/// Holds the cumulative arc length `s[k]`, the segment lengths `ds[k]`, and finite-difference
/// approximations to the first, second and third derivatives of the joint-space path `q(s)`
/// and the tool-center-point path `p(s)` at each densified waypoint.
#[derive(Debug, Clone)]
pub struct PathDerivatives<const N: usize> {
    pub waypoints: Vec<SRobotQ<N, f32>>,
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
        path: &SRobotPath<N, f32>,
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

    pub fn new<FK: FKChain<N>>(
        densified: &SRobotPath<N, f32>,
        fk: &FK,
    ) -> DekeResult<Self> {
        Self::build(densified, Some(fk))
    }

    /// Same as [`Self::new`] but skips all forward-kinematics evaluation. The resulting
    /// `PathDerivatives` has empty `tcp`, `pp`, `ppp`, and `pppp` vectors; callers must gate any
    /// TCP logic on `has_tcp()`.
    pub fn new_without_tcp(densified: &SRobotPath<N, f32>) -> DekeResult<Self> {
        Self::build::<crate::path_derivatives::NeverFK<N>>(densified, None)
    }

    fn build<FK: FKChain<N>>(
        densified: &SRobotPath<N, f32>,
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
                let d = (b[j] - a[j]) as f64;
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

        let wps_f64: Vec<[f64; N]> = densified
            .iter()
            .map(|wp| {
                let mut a = [0.0_f64; N];
                for j in 0..N {
                    a[j] = wp.0[j] as f64;
                }
                a
            })
            .collect();

        let qp = fd_first::<N>(&wps_f64, &ds);
        let qpp = fd_second::<N>(&wps_f64, &ds);
        let qppp = fd_third_of_second::<N>(&qpp, &ds);

        let (tcp, pp, ppp, pppp) = if let Some(fk) = fk {
            let mut tcp = Vec::with_capacity(m);
            for wp in densified.iter() {
                let pose = fk.fk_end(wp).map_err(|e| e.into())?;
                let t = pose.translation;
                tcp.push([t.x as f64, t.y as f64, t.z as f64]);
            }
            let pp = fd_first::<3>(&tcp, &ds);
            let ppp = fd_second::<3>(&tcp, &ds);
            let pppp = fd_third_of_second::<3>(&ppp, &ds);
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

impl<const N: usize> FKChain<N> for NeverFK<N> {
    type Error = DekeError;
    fn fk(&self, _q: &deke_types::SRobotQ<N>) -> Result<[deke_types::glam::Affine3A; N], Self::Error> {
        unreachable!("NeverFK is a placeholder — PathDerivatives::new_without_tcp never calls FK")
    }
    fn fk_end(&self, _q: &deke_types::SRobotQ<N>) -> Result<deke_types::glam::Affine3A, Self::Error> {
        unreachable!()
    }
    fn joint_axes_positions(
        &self,
        _q: &deke_types::SRobotQ<N>,
    ) -> Result<(
        [deke_types::glam::Vec3A; N],
        [deke_types::glam::Vec3A; N],
        deke_types::glam::Vec3A,
    ), Self::Error> {
        unreachable!()
    }
}

fn fd_first<const D: usize>(f: &[[f64; D]], ds: &[f64]) -> Vec<[f64; D]> {
    let m = f.len();
    let mut out = vec![[0.0_f64; D]; m];
    if m < 2 {
        return out;
    }
    for j in 0..D {
        out[0][j] = (f[1][j] - f[0][j]) / ds[0];
        out[m - 1][j] = (f[m - 1][j] - f[m - 2][j]) / ds[m - 2];
    }
    for k in 1..m - 1 {
        let span = ds[k - 1] + ds[k];
        for j in 0..D {
            out[k][j] = (f[k + 1][j] - f[k - 1][j]) / span;
        }
    }
    out
}

fn fd_second<const D: usize>(f: &[[f64; D]], ds: &[f64]) -> Vec<[f64; D]> {
    let m = f.len();
    let mut out = vec![[0.0_f64; D]; m];
    if m < 3 {
        return out;
    }
    for k in 1..m - 1 {
        let h0 = ds[k - 1];
        let h1 = ds[k];
        let scale = 2.0 / (h0 + h1);
        for j in 0..D {
            let right = (f[k + 1][j] - f[k][j]) / h1;
            let left = (f[k][j] - f[k - 1][j]) / h0;
            out[k][j] = scale * (right - left);
        }
    }
    out[0] = out[1];
    out[m - 1] = out[m - 2];
    out
}

fn fd_third_of_second<const D: usize>(qpp: &[[f64; D]], ds: &[f64]) -> Vec<[f64; D]> {
    let m = qpp.len();
    let mut out = vec![[0.0_f64; D]; m];
    if m < 3 {
        return out;
    }
    for k in 1..m - 1 {
        let span = ds[k - 1] + ds[k];
        for j in 0..D {
            out[k][j] = (qpp[k + 1][j] - qpp[k - 1][j]) / span;
        }
    }
    out[0] = out[1];
    out[m - 1] = out[m - 2];
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_first_linear_uniform() {
        // f(s) = 2s on uniform grid with ds=0.5 → f' = 2 everywhere
        let f: Vec<[f64; 1]> = (0..5).map(|i| [2.0 * i as f64 * 0.5]).collect();
        let ds = vec![0.5; 4];
        let fp = fd_first::<1>(&f, &ds);
        for v in fp {
            assert!((v[0] - 2.0).abs() < 1e-9, "got {}", v[0]);
        }
    }

    #[test]
    fn fd_second_quadratic_uniform() {
        // f(s) = s^2 on uniform grid; f'' = 2 everywhere (interior exact, endpoints copied)
        let ds = vec![0.25_f64; 8];
        let f: Vec<[f64; 1]> = (0..9).map(|i| [(i as f64 * 0.25).powi(2)]).collect();
        let fpp = fd_second::<1>(&f, &ds);
        for v in fpp {
            assert!((v[0] - 2.0).abs() < 1e-9, "got {}", v[0]);
        }
    }

    #[test]
    fn fd_third_is_zero_for_quadratic() {
        let ds = vec![0.25_f64; 8];
        let f: Vec<[f64; 1]> = (0..9).map(|i| [(i as f64 * 0.25).powi(2)]).collect();
        let fpp = fd_second::<1>(&f, &ds);
        let fppp = fd_third_of_second::<1>(&fpp, &ds);
        for v in fppp {
            assert!(v[0].abs() < 1e-9, "got {}", v[0]);
        }
    }
}
