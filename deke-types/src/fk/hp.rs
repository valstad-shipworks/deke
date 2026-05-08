use const_soft_float::soft_f32::SoftF32;
use const_soft_float::soft_f64::SoftF64;
use glam_traits_ext::{FloatAffine, FloatMat, FloatVec, TAffine3, TMat3, TVec3};

#[cfg(debug_assertions)]
use crate::DekeError;
use crate::SRobotQ;

use super::{
    AAffine3, AMat3, AVec3, FKChain, FKScalar, check_finite, const_sin_cos, const_sin_cos_f64,
};

#[derive(Debug, Clone, Copy)]
pub struct HPJoint<F: FKScalar = f32> {
    pub a: F,
    pub alpha: F,
    pub beta: F,
    pub d: F,
    pub theta_offset: F,
}

/// Precomputed Hayati-Paul chain with SoA layout.
///
/// Convention: `T_i = Rz(θ) · Rx(α) · Ry(β) · Tx(a) · Tz(d)`
///
/// HP adds a `β` rotation about Y, which makes it numerically stable for
/// nearly-parallel consecutive joint axes where standard DH is singular.
#[derive(Debug, Clone)]
pub struct HPChain<const N: usize, F: FKScalar = f32> {
    a: [F; N],
    d: [F; N],
    sin_alpha: [F; N],
    cos_alpha: [F; N],
    sin_beta: [F; N],
    cos_beta: [F; N],
    theta_offset: [F; N],
}

impl<const N: usize> HPChain<N, f32> {
    pub const fn new(joints: [HPJoint<f32>; N]) -> Self {
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
    /// `h`: `[[f32; N]; 3]` — rows are (x, y, z) components across joints.
    /// `p`: `[[f32; N]; 3]` — rows are (x, y, z) components across points.
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
}

impl<const N: usize> HPChain<N, f64> {
    /// `const`-evaluable f64 constructor — analogue of [`HPChain::<N, f32>::new`].
    pub const fn new_f64(joints: [HPJoint<f64>; N]) -> Self {
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
            let (sa, ca) = const_sin_cos_f64(joints[i].alpha);
            sin_alpha[i] = sa;
            cos_alpha[i] = ca;
            let (sb, cb) = const_sin_cos_f64(joints[i].beta);
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

    /// `const`-evaluable f64 analogue of [`HPChain::from_hp`].
    pub const fn from_hp_f64(h: &[[f64; N]; 3], p: &[[f64; N]; 3]) -> Self {
        let mut a = [0.0f64; N];
        let mut d = [0.0f64; N];
        let mut sin_alpha = [0.0f64; N];
        let mut cos_alpha = [0.0f64; N];
        let mut sin_beta = [0.0f64; N];
        let mut cos_beta = [0.0f64; N];

        let mut c0 = [1.0f64, 0.0, 0.0];
        let mut c1 = [0.0f64, 1.0, 0.0];
        let mut c2 = [0.0f64, 0.0, 1.0];

        const EPS: f64 = 1e-15;

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
            let cb = SoftF64::from_f64(vy * vy + vz * vz).sqrt().to_f64();

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
            theta_offset: [0.0f64; N],
        }
    }
}

impl<const N: usize, F: FKScalar> HPChain<N, F> {
    /// Generic runtime constructor. For `f32` the const-evaluable
    /// [`HPChain::new`] is preferred; this exists so `HPChain<N, f64>` (and
    /// any other future scalar) is usable.
    pub fn from_joints(joints: [HPJoint<F>; N]) -> Self {
        let zero = F::zero();
        let mut a = [zero; N];
        let mut d = [zero; N];
        let mut sin_alpha = [zero; N];
        let mut cos_alpha = [zero; N];
        let mut sin_beta = [zero; N];
        let mut cos_beta = [zero; N];
        let mut theta_offset = [zero; N];

        for i in 0..N {
            a[i] = joints[i].a;
            d[i] = joints[i].d;
            let (sa, ca) = joints[i].alpha.sin_cos();
            sin_alpha[i] = sa;
            cos_alpha[i] = ca;
            let (sb, cb) = joints[i].beta.sin_cos();
            sin_beta[i] = sb;
            cos_beta[i] = cb;
            theta_offset[i] = joints[i].theta_offset;
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
    fn local_frame(&self, i: usize, st: F, ct: F) -> (AVec3<F>, AVec3<F>, AVec3<F>, AVec3<F>) {
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

        let c0 = AVec3::<F>::new(ct * cb - st * sa_sb, st * cb + ct * sa_sb, -ca_sb);
        let c1 = AVec3::<F>::new(-st * ca, ct * ca, sa);
        let c2 = AVec3::<F>::new(ct * sb + st * sa_cb, st * sb - ct * sa_cb, ca_cb);
        let t = c0 * ai + c2 * di;

        (c0, c1, c2, t)
    }
}

/// Accumulate a local rotation + translation into the running transform.
#[inline(always)]
fn accumulate<F: FKScalar>(
    acc_m: &mut AMat3<F>,
    acc_t: &mut AVec3<F>,
    local_c0: AVec3<F>,
    local_c1: AVec3<F>,
    local_c2: AVec3<F>,
    local_t: AVec3<F>,
) {
    let new_c0 = *acc_m * local_c0;
    let new_c1 = *acc_m * local_c1;
    let new_c2 = *acc_m * local_c2;
    *acc_t = *acc_m * local_t + *acc_t;
    *acc_m = AMat3::<F>::from_cols(new_c0, new_c1, new_c2);
}

impl<const N: usize, F: FKScalar> FKChain<N, F> for HPChain<N, F> {
    #[cfg(debug_assertions)]
    type Error = DekeError;
    #[cfg(not(debug_assertions))]
    type Error = std::convert::Infallible;

    fn fk(&self, q: &SRobotQ<N, F>) -> Result<[AAffine3<F>; N], Self::Error> {
        check_finite::<N, F>(q)?;
        let mut out = [AAffine3::<F>::IDENTITY; N];
        let mut acc_m = AMat3::<F>::IDENTITY;
        let mut acc_t = AVec3::<F>::ZERO;

        let mut i = 0;
        while i < N {
            let (st, ct) = (q.0[i] + self.theta_offset[i]).sin_cos();
            let (c0, c1, c2, t) = self.local_frame(i, st, ct);
            accumulate::<F>(&mut acc_m, &mut acc_t, c0, c1, c2, t);

            out[i] = AAffine3::<F>::from_mat3_translation(acc_m, acc_t);
            i += 1;
        }
        Ok(out)
    }

    fn fk_end(&self, q: &SRobotQ<N, F>) -> Result<AAffine3<F>, Self::Error> {
        check_finite::<N, F>(q)?;
        let mut acc_m = AMat3::<F>::IDENTITY;
        let mut acc_t = AVec3::<F>::ZERO;

        let mut i = 0;
        while i < N {
            let (st, ct) = (q.0[i] + self.theta_offset[i]).sin_cos();
            let (c0, c1, c2, t) = self.local_frame(i, st, ct);
            accumulate::<F>(&mut acc_m, &mut acc_t, c0, c1, c2, t);
            i += 1;
        }

        Ok(AAffine3::<F>::from_mat3_translation(acc_m, acc_t))
    }

    fn all_fk(
        &self,
        q: &SRobotQ<N, F>,
    ) -> Result<(AAffine3<F>, [AAffine3<F>; N], AAffine3<F>), Self::Error> {
        let frames = self.fk(q)?;
        // HP has no tool/suffix offset; the last accumulated frame is the
        // EE frame.
        let end = if N > 0 {
            frames[N - 1]
        } else {
            AAffine3::<F>::IDENTITY
        };
        Ok((self.base_tf(), frames, end))
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N, F>,
    ) -> Result<([AVec3<F>; N], [AVec3<F>; N], AVec3<F>), Self::Error> {
        let frames = self.fk(q)?;
        let mut axes = [AVec3::<F>::Z; N];
        let mut positions = [AVec3::<F>::ZERO; N];

        for i in 1..N {
            axes[i] = frames[i - 1].matrix3().z_axis();
            positions[i] = frames[i - 1].translation();
        }

        Ok((axes, positions, frames[N - 1].translation()))
    }
}

impl From<HPJoint<f32>> for HPJoint<f64> {
    #[inline]
    fn from(j: HPJoint<f32>) -> Self {
        HPJoint {
            a: j.a as f64,
            alpha: j.alpha as f64,
            beta: j.beta as f64,
            d: j.d as f64,
            theta_offset: j.theta_offset as f64,
        }
    }
}

impl From<HPJoint<f64>> for HPJoint<f32> {
    #[inline]
    fn from(j: HPJoint<f64>) -> Self {
        HPJoint {
            a: j.a as f32,
            alpha: j.alpha as f32,
            beta: j.beta as f32,
            d: j.d as f32,
            theta_offset: j.theta_offset as f32,
        }
    }
}

#[inline]
fn cast_arr<const N: usize, A: Copy, B: Copy>(src: [A; N], cast: impl Fn(A) -> B) -> [B; N] {
    std::array::from_fn(|i| cast(src[i]))
}

impl<const N: usize> From<HPChain<N, f32>> for HPChain<N, f64> {
    #[inline]
    fn from(c: HPChain<N, f32>) -> Self {
        HPChain::<N, f64> {
            a: cast_arr(c.a, |x| x as f64),
            d: cast_arr(c.d, |x| x as f64),
            sin_alpha: cast_arr(c.sin_alpha, |x| x as f64),
            cos_alpha: cast_arr(c.cos_alpha, |x| x as f64),
            sin_beta: cast_arr(c.sin_beta, |x| x as f64),
            cos_beta: cast_arr(c.cos_beta, |x| x as f64),
            theta_offset: cast_arr(c.theta_offset, |x| x as f64),
        }
    }
}

impl<const N: usize> From<HPChain<N, f64>> for HPChain<N, f32> {
    #[inline]
    fn from(c: HPChain<N, f64>) -> Self {
        HPChain::<N, f32> {
            a: cast_arr(c.a, |x| x as f32),
            d: cast_arr(c.d, |x| x as f32),
            sin_alpha: cast_arr(c.sin_alpha, |x| x as f32),
            cos_alpha: cast_arr(c.cos_alpha, |x| x as f32),
            sin_beta: cast_arr(c.sin_beta, |x| x as f32),
            cos_beta: cast_arr(c.cos_beta, |x| x as f32),
            theta_offset: cast_arr(c.theta_offset, |x| x as f32),
        }
    }
}
