use const_soft_float::soft_f32::SoftF32;
use glam::{Affine3A, Mat3A, Vec3A};

use crate::{DekeError, SRobotQ};

use super::{FKChain, accumulate, check_finite, const_sin_cos};

#[derive(Debug, Clone, Copy)]
pub struct HPJoint {
    pub a: f32,
    pub alpha: f32,
    pub beta: f32,
    pub d: f32,
    pub theta_offset: f32,
}

/// Precomputed Hayati-Paul chain with SoA layout.
///
/// Convention: `T_i = Rz(θ) · Rx(α) · Ry(β) · Tx(a) · Tz(d)`
///
/// HP adds a `β` rotation about Y, which makes it numerically stable for
/// nearly-parallel consecutive joint axes where standard DH is singular.
#[derive(Debug, Clone)]
pub struct HPChain<const N: usize> {
    a: [f32; N],
    d: [f32; N],
    sin_alpha: [f32; N],
    cos_alpha: [f32; N],
    sin_beta: [f32; N],
    cos_beta: [f32; N],
    theta_offset: [f32; N],
}

impl<const N: usize> HPChain<N> {
    pub const fn new(joints: [HPJoint; N]) -> Self {
        let mut a = [0.0; N];
        let mut d = [0.0; N];
        let mut sin_alpha = [0.0; N];
        let mut cos_alpha = [0.0; N];
        let mut sin_beta = [0.0; N];
        let mut cos_beta = [0.0; N];
        let mut theta_offset = [0.0; N];

        let mut i = 0;
        while i < N {
            a[i] = joints[i].a;
            d[i] = joints[i].d;
            let (sa, ca) = const_sin_cos(joints[i].alpha);
            sin_alpha[i] = sa;
            cos_alpha[i] = ca;
            let (sb, cb) = const_sin_cos(joints[i].beta);
            sin_beta[i] = sb;
            cos_beta[i] = cb;
            theta_offset[i] = joints[i].theta_offset;
            i += 1;
        }

        Self {
            a,
            d,
            sin_alpha,
            cos_alpha,
            sin_beta,
            cos_beta,
            theta_offset,
        }
    }

    /// Construct from the row-major `HP_H` and `HP_P` const arrays emitted by
    /// the workcell macro.
    ///
    /// `h`: `[[f64; N]; 3]` — rows are (x, y, z) components across joints.
    /// `p`: `[[f64; N]; 3]` — rows are (x, y, z) components across points.
    ///
    /// Each `h[_][i]` is joint `i`'s axis in the base frame at zero config.
    /// `p[_][0]` is the base-to-joint-0 offset; `p[_][i]` for `1..N` is the
    /// offset from joint `i-1`'s origin to joint `i`'s origin. The tool
    /// offset from joint `N-1` to the flange is not part of this input
    /// because `HPChain` has no end-effector slot — wrap the result in a
    /// [`TransformedFK`](crate::TransformedFK) if a tool offset is required.
    ///
    /// `theta_offset` is set to zero for every joint: at zero config each
    /// local x-axis is pinned to `Rx(α) · Ry(β) · [1, 0, 0]` expressed in the
    /// parent frame.
    pub const fn from_hp(h: &[[f32; N]; 3], p: &[[f32; N]; 3]) -> Self {

        let mut a = [0.0f32; N];
        let mut d = [0.0f32; N];
        let mut sin_alpha = [0.0f32; N];
        let mut cos_alpha = [0.0f32; N];
        let mut sin_beta = [0.0f32; N];
        let mut cos_beta = [0.0f32; N];

        let mut c0 = [1.0f32, 0.0, 0.0];
        let mut c1 = [0.0f32, 1.0, 0.0];
        let mut c2 = [0.0f32, 0.0, 1.0];

        const EPS: f32 = 1e-12;

        let mut i = 0;
        while i < N {
            let hx = h[0][i];
            let hy = h[1][i];
            let hz = h[2][i];
            let px = p[0][i];
            let py = p[1][i];
            let pz = p[2][i];

            let vx = c0[0] * hx + c0[1] * hy + c0[2] * hz;
            let vy = c1[0] * hx + c1[1] * hy + c1[2] * hz;
            let vz = c2[0] * hx + c2[1] * hy + c2[2] * hz;
            let ux = c0[0] * px + c0[1] * py + c0[2] * pz;
            let uy = c1[0] * px + c1[1] * py + c1[2] * pz;
            let uz = c2[0] * px + c2[1] * py + c2[2] * pz;

            let sb = vx;
            let cb = SoftF32::from_f32(vy * vy + vz * vz).sqrt().to_f32();

            let (sa, ca) = if cb > EPS {
                (-vy / cb, vz / cb)
            } else {
                (0.0, 1.0)
            };

            let big_a = ux;
            let big_b = sa * uy - ca * uz;
            let ai = cb * big_a + sb * big_b;
            let di = sb * big_a - cb * big_b;

            a[i] = ai;
            d[i] = di;
            sin_alpha[i] = sa;
            cos_alpha[i] = ca;
            sin_beta[i] = sb;
            cos_beta[i] = cb;

            let sasb = sa * sb;
            let casb = ca * sb;
            let sacb = sa * cb;
            let cacb = ca * cb;
            let new_c0 = [
                cb * c0[0] + sasb * c1[0] - casb * c2[0],
                cb * c0[1] + sasb * c1[1] - casb * c2[1],
                cb * c0[2] + sasb * c1[2] - casb * c2[2],
            ];
            let new_c1 = [
                ca * c1[0] + sa * c2[0],
                ca * c1[1] + sa * c2[1],
                ca * c1[2] + sa * c2[2],
            ];
            let new_c2 = [
                sb * c0[0] - sacb * c1[0] + cacb * c2[0],
                sb * c0[1] - sacb * c1[1] + cacb * c2[1],
                sb * c0[2] - sacb * c1[2] + cacb * c2[2],
            ];
            c0 = new_c0;
            c1 = new_c1;
            c2 = new_c2;

            i += 1;
        }

        Self {
            a,
            d,
            sin_alpha,
            cos_alpha,
            sin_beta,
            cos_beta,
            theta_offset: [0.0f32; N],
        }
    }

    /// Build the local rotation columns and translation for joint `i`.
    ///
    /// R = Rz(θ) · Rx(α) · Ry(β), then t = R · [a, 0, d].
    ///
    /// Rx(α)·Ry(β) columns:
    ///   col0 = [ cβ,       sα·sβ,     -cα·sβ     ]
    ///   col1 = [ 0,        cα,          sα        ]
    ///   col2 = [ sβ,      -sα·cβ,      cα·cβ     ]
    ///
    /// Then Rz(θ) rotates each column: [cθ·x - sθ·y, sθ·x + cθ·y, z]
    ///
    /// Translation = a·col0 + d·col2  (since R·[a,0,d] = a·col0 + d·col2).
    #[inline(always)]
    fn local_frame(&self, i: usize, st: f32, ct: f32) -> (Vec3A, Vec3A, Vec3A, Vec3A) {
        let sa = self.sin_alpha[i];
        let ca = self.cos_alpha[i];
        let sb = self.sin_beta[i];
        let cb = self.cos_beta[i];
        let ai = self.a[i];
        let di = self.d[i];

        let sa_sb = sa * sb;
        let sa_cb = sa * cb;
        let ca_sb = ca * sb;
        let ca_cb = ca * cb;

        let c0 = Vec3A::new(ct * cb - st * sa_sb, st * cb + ct * sa_sb, -ca_sb);
        let c1 = Vec3A::new(-st * ca, ct * ca, sa);
        let c2 = Vec3A::new(ct * sb + st * sa_cb, st * sb - ct * sa_cb, ca_cb);
        let t = Vec3A::new(
            ai * c0.x + di * c2.x,
            ai * c0.y + di * c2.y,
            ai * c0.z + di * c2.z,
        );

        (c0, c1, c2, t)
    }
}

impl<const N: usize> FKChain<N> for HPChain<N> {
    #[cfg(debug_assertions)]
    type Error = DekeError;
    #[cfg(not(debug_assertions))]
    type Error = std::convert::Infallible;

    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], Self::Error> {
        check_finite::<N>(q)?;
        let mut out = [Affine3A::IDENTITY; N];
        let mut acc_m = Mat3A::IDENTITY;
        let mut acc_t = Vec3A::ZERO;

        let mut i = 0;
        while i < N {
            let (st, ct) = (q.0[i] + self.theta_offset[i]).sin_cos();
            let (c0, c1, c2, t) = self.local_frame(i, st, ct);
            accumulate(&mut acc_m, &mut acc_t, c0, c1, c2, t);

            out[i] = Affine3A {
                matrix3: acc_m,
                translation: acc_t,
            };
            i += 1;
        }
        Ok(out)
    }

    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, Self::Error> {
        check_finite(q)?;
        let mut acc_m = Mat3A::IDENTITY;
        let mut acc_t = Vec3A::ZERO;

        let mut i = 0;
        while i < N {
            let (st, ct) = (q.0[i] + self.theta_offset[i]).sin_cos();
            let (c0, c1, c2, t) = self.local_frame(i, st, ct);
            accumulate(&mut acc_m, &mut acc_t, c0, c1, c2, t);
            i += 1;
        }

        Ok(Affine3A {
            matrix3: acc_m,
            translation: acc_t,
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
