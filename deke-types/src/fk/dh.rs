use glam::{Affine3A, Mat3A, Vec3A};

use crate::{DekeError, SRobotQ};

use super::{FKChain, check_finite, const_sin_cos};

#[derive(Debug, Clone, Copy)]
pub struct DHJoint {
    pub a: f32,
    pub alpha: f32,
    pub d: f32,
    pub theta_offset: f32,
}

/// Precomputed standard-DH chain with SoA layout.
///
/// Convention: `T_i = Rz(θ) · Tz(d) · Tx(a) · Rx(α)`
#[derive(Debug, Clone)]
pub struct DHChain<const N: usize> {
    a: [f32; N],
    d: [f32; N],
    sin_alpha: [f32; N],
    cos_alpha: [f32; N],
    theta_offset: [f32; N],
}

impl<const N: usize> DHChain<N> {
    pub const fn new(joints: [DHJoint; N]) -> Self {
        let mut a = [0.0; N];
        let mut d = [0.0; N];
        let mut sin_alpha = [0.0; N];
        let mut cos_alpha = [0.0; N];
        let mut theta_offset = [0.0; N];

        let mut i = 0;
        while i < N {
            a[i] = joints[i].a;
            d[i] = joints[i].d;
            let (sa, ca) = const_sin_cos(joints[i].alpha);
            sin_alpha[i] = sa;
            cos_alpha[i] = ca;
            theta_offset[i] = joints[i].theta_offset;
            i += 1;
        }

        Self {
            a,
            d,
            sin_alpha,
            cos_alpha,
            theta_offset,
        }
    }

    /// Construct from the row-major `DH_PARAMS` const array emitted by the
    /// workcell macro.
    ///
    /// `params`: `[[f64; N]; 4]` — rows are `(a, alpha, d, theta_offset)`
    /// across joints.
    pub const fn from_dh(params: &[[f64; N]; 4]) -> Self {
        let mut a = [0.0f32; N];
        let mut d = [0.0f32; N];
        let mut sin_alpha = [0.0f32; N];
        let mut cos_alpha = [0.0f32; N];
        let mut theta_offset = [0.0f32; N];

        let mut i = 0;
        while i < N {
            a[i] = params[0][i] as f32;
            let (sa, ca) = const_sin_cos(params[1][i] as f32);
            sin_alpha[i] = sa;
            cos_alpha[i] = ca;
            d[i] = params[2][i] as f32;
            theta_offset[i] = params[3][i] as f32;
            i += 1;
        }

        Self {
            a,
            d,
            sin_alpha,
            cos_alpha,
            theta_offset,
        }
    }
}

impl<const N: usize> FKChain<N> for DHChain<N> {
    #[cfg(debug_assertions)]
    type Error = DekeError;
    #[cfg(not(debug_assertions))]
    type Error = std::convert::Infallible;

    /// DH forward kinematics exploiting the structure of `Rz(θ)·Rx(α)`.
    ///
    /// The per-joint accumulation decomposes into two 2D column rotations:
    ///   1. Rotate `(c0, c1)` by θ  →  `(new_c0, perp)`
    ///   2. Rotate `(perp, c2)` by α  →  `(new_c1, new_c2)`
    /// Translation reuses `new_c0`:  `t += a·new_c0 + d·old_c2`
    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], Self::Error> {
        check_finite::<N>(q)?;
        let mut out = [Affine3A::IDENTITY; N];
        let mut c0 = Vec3A::X;
        let mut c1 = Vec3A::Y;
        let mut c2 = Vec3A::Z;
        let mut t = Vec3A::ZERO;

        let mut i = 0;
        while i < N {
            let (st, ct) = (q.0[i] + self.theta_offset[i]).sin_cos();
            let sa = self.sin_alpha[i];
            let ca = self.cos_alpha[i];

            let new_c0 = ct * c0 + st * c1;
            let perp = ct * c1 - st * c0;

            let new_c1 = ca * perp + sa * c2;
            let new_c2 = ca * c2 - sa * perp;

            t = self.a[i] * new_c0 + self.d[i] * c2 + t;

            c0 = new_c0;
            c1 = new_c1;
            c2 = new_c2;

            out[i] = Affine3A {
                matrix3: Mat3A::from_cols(c0, c1, c2),
                translation: t,
            };
            i += 1;
        }
        Ok(out)
    }

    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, Self::Error> {
        check_finite::<N>(q)?;
        let mut c0 = Vec3A::X;
        let mut c1 = Vec3A::Y;
        let mut c2 = Vec3A::Z;
        let mut t = Vec3A::ZERO;

        let mut i = 0;
        while i < N {
            let (st, ct) = (q.0[i] + self.theta_offset[i]).sin_cos();
            let sa = self.sin_alpha[i];
            let ca = self.cos_alpha[i];

            let new_c0 = ct * c0 + st * c1;
            let perp = ct * c1 - st * c0;

            let new_c1 = ca * perp + sa * c2;
            let new_c2 = ca * c2 - sa * perp;

            t = self.a[i] * new_c0 + self.d[i] * c2 + t;

            c0 = new_c0;
            c1 = new_c1;
            c2 = new_c2;
            i += 1;
        }

        Ok(Affine3A {
            matrix3: Mat3A::from_cols(c0, c1, c2),
            translation: t,
        })
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N>,
    ) -> Result<([Vec3A; N], [Vec3A; N], Vec3A), Self::Error> {
        let frames = self.fk(q)?;
        let mut axes = [Vec3A::Z; N];
        let mut positions = [Vec3A::ZERO; N];

        for i in 1..N {
            axes[i] = frames[i - 1].matrix3.z_axis;
            positions[i] = frames[i - 1].translation;
        }

        Ok((axes, positions, frames[N - 1].translation))
    }
}
