use deke_types::SRobotQ;

/// Result of projecting a joint-space velocity and acceleration onto the fixed path tangent.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedBoundary {
    /// Scalar path-parameter speed (sd = ds/dt) that best matches `v`.
    pub sd: f64,
    /// Scalar path-parameter acceleration (sdd = d²s/dt²) that best matches `a` given `sd`.
    pub sdd: f64,
    /// L2 norm of the residual `v - sd·qp` in joint-space.
    pub velocity_residual: f64,
    /// L2 norm of the residual `a - (qpp·sd² + qp·sdd)` in joint-space.
    pub acceleration_residual: f64,
}

impl ProjectedBoundary {
    /// Largest residual across velocity and acceleration projections.
    pub fn max_residual(&self) -> f64 {
        self.velocity_residual.max(self.acceleration_residual)
    }
}

/// Projects a joint-space `(velocity, acceleration)` pair onto the path's scalar
/// `(sd, sdd)` at the waypoint whose first/second derivatives are `qp, qpp`.
///
/// The path fixes the direction of motion: `q̇(s) = qp · sd`, `q̈(s) = qpp · sd² + qp · sdd`.
/// If `v` or `a` have components perpendicular to `qp`, the projected scalars ignore them and
/// the perpendicular norm is returned as a residual for the caller to sanity-check.
pub fn project<const N: usize>(
    v: &SRobotQ<N>,
    a: &SRobotQ<N>,
    qp: &[f64; N],
    qpp: &[f64; N],
) -> ProjectedBoundary {
    let mut qp_norm_sq = 0.0_f64;
    for j in 0..N {
        qp_norm_sq += qp[j] * qp[j];
    }
    if qp_norm_sq < 1e-18 {
        return ProjectedBoundary {
            sd: 0.0,
            sdd: 0.0,
            velocity_residual: vector_norm(&v.0),
            acceleration_residual: vector_norm(&a.0),
        };
    }

    let mut v_dot_qp = 0.0_f64;
    for j in 0..N {
        v_dot_qp += v.0[j] as f64 * qp[j];
    }
    let sd = v_dot_qp / qp_norm_sq;

    let mut v_res_sq = 0.0_f64;
    for j in 0..N {
        let residual = v.0[j] as f64 - sd * qp[j];
        v_res_sq += residual * residual;
    }

    let mut a_minus_qpp_sd2_dot_qp = 0.0_f64;
    for j in 0..N {
        let adj = a.0[j] as f64 - qpp[j] * sd * sd;
        a_minus_qpp_sd2_dot_qp += adj * qp[j];
    }
    let sdd = a_minus_qpp_sd2_dot_qp / qp_norm_sq;

    let mut a_res_sq = 0.0_f64;
    for j in 0..N {
        let expected = qpp[j] * sd * sd + qp[j] * sdd;
        let residual = a.0[j] as f64 - expected;
        a_res_sq += residual * residual;
    }

    ProjectedBoundary {
        sd,
        sdd,
        velocity_residual: v_res_sq.sqrt(),
        acceleration_residual: a_res_sq.sqrt(),
    }
}

fn vector_norm<const N: usize>(v: &[f32; N]) -> f64 {
    let mut s = 0.0_f64;
    for x in v {
        s += (*x as f64) * (*x as f64);
    }
    s.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_projection_recovers_scalars() {
        // qp = [1, 0, 0], qpp = [0, 1, 0]
        // Take sd = 2, sdd = 3 ⇒ v = [2, 0, 0], a = qpp·4 + qp·3 = [3, 4, 0]
        let qp = [1.0_f64, 0.0, 0.0];
        let qpp = [0.0_f64, 1.0, 0.0];
        let v = SRobotQ::<3>::from_array([2.0, 0.0, 0.0]);
        let a = SRobotQ::<3>::from_array([3.0, 4.0, 0.0]);
        let out = project::<3>(&v, &a, &qp, &qpp);
        assert!((out.sd - 2.0).abs() < 1e-9);
        assert!((out.sdd - 3.0).abs() < 1e-9);
        assert!(out.velocity_residual < 1e-9);
        assert!(out.acceleration_residual < 1e-9);
    }

    #[test]
    fn perpendicular_velocity_reports_residual() {
        let qp = [1.0_f64, 0.0];
        let qpp = [0.0_f64, 0.0];
        let v = SRobotQ::<2>::from_array([0.0, 1.0]);
        let a = SRobotQ::<2>::from_array([0.0, 0.0]);
        let out = project::<2>(&v, &a, &qp, &qpp);
        assert!(out.sd.abs() < 1e-9);
        assert!((out.velocity_residual - 1.0).abs() < 1e-9);
    }
}
