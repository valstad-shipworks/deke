use glam_traits_ext::{FloatAffine, FloatVec, TAffine3, TMat3, TVec3};

#[cfg(debug_assertions)]
use crate::DekeError;
use crate::SRobotQ;

use super::{
    AAffine3, AMat3, AVec3, FKChain, FKScalar, check_finite, const_sin_cos, const_sin_cos_f64,
};

#[derive(Debug, Clone, Copy)]
pub struct DHJoint<F: FKScalar = f32> {
    pub a: F,
    pub alpha: F,
    pub d: F,
    pub theta_offset: F,
}

/// Precomputed standard-DH chain with SoA layout.
///
/// Convention: `T_i = Rz(θ) · Tz(d) · Tx(a) · Rx(α)`
#[derive(Debug, Clone)]
pub struct DHChain<const N: usize, F: FKScalar = f32> {
    a: [F; N],
    d: [F; N],
    sin_alpha: [F; N],
    cos_alpha: [F; N],
    theta_offset: [F; N],
}

impl<const N: usize> DHChain<N, f32> {
    pub const fn new(joints: [DHJoint<f32>; N]) -> Self {
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

impl<const N: usize> DHChain<N, f64> {
    /// `const`-evaluable f64 constructor — analogue of [`DHChain::<N, f32>::new`].
    pub const fn new_f64(joints: [DHJoint<f64>; N]) -> Self {
        let mut a = [0.0; N];
        let mut d = [0.0; N];
        let mut sin_alpha = [0.0; N];
        let mut cos_alpha = [0.0; N];
        let mut theta_offset = [0.0; N];

        let mut i = 0;
        while i < N {
            a[i] = joints[i].a;
            d[i] = joints[i].d;
            let (sa, ca) = const_sin_cos_f64(joints[i].alpha);
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

    /// Construct from the row-major `DH_PARAMS` const array, in `f64`.
    /// `params`: `[[f64; N]; 4]` — rows are `(a, alpha, d, theta_offset)`.
    pub const fn from_dh_f64(params: &[[f64; N]; 4]) -> Self {
        let mut a = [0.0f64; N];
        let mut d = [0.0f64; N];
        let mut sin_alpha = [0.0f64; N];
        let mut cos_alpha = [0.0f64; N];
        let mut theta_offset = [0.0f64; N];

        let mut i = 0;
        while i < N {
            a[i] = params[0][i];
            let (sa, ca) = const_sin_cos_f64(params[1][i]);
            sin_alpha[i] = sa;
            cos_alpha[i] = ca;
            d[i] = params[2][i];
            theta_offset[i] = params[3][i];
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

impl<const N: usize, F: FKScalar> DHChain<N, F> {
    /// Generic runtime constructor. For `f32` the const-evaluable
    /// [`DHChain::new`] is preferred; this exists so `DHChain<N, f64>` (and
    /// any other future scalar) is usable.
    pub fn from_joints(joints: [DHJoint<F>; N]) -> Self {
        let zero = F::zero();
        let mut a = [zero; N];
        let mut d = [zero; N];
        let mut sin_alpha = [zero; N];
        let mut cos_alpha = [zero; N];
        let mut theta_offset = [zero; N];

        for i in 0..N {
            a[i] = joints[i].a;
            d[i] = joints[i].d;
            let (sa, ca) = joints[i].alpha.sin_cos();
            sin_alpha[i] = sa;
            cos_alpha[i] = ca;
            theta_offset[i] = joints[i].theta_offset;
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

impl<const N: usize, F: FKScalar> FKChain<N, F> for DHChain<N, F> {
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
    fn fk(&self, q: &SRobotQ<N, F>) -> Result<[AAffine3<F>; N], Self::Error> {
        check_finite::<N, F>(q)?;
        let mut out = [AAffine3::<F>::IDENTITY; N];
        let mut c0 = AVec3::<F>::X;
        let mut c1 = AVec3::<F>::Y;
        let mut c2 = AVec3::<F>::Z;
        let mut t = AVec3::<F>::ZERO;

        let mut i = 0;
        while i < N {
            let (st, ct) = (q.0[i] + self.theta_offset[i]).sin_cos();
            let sa = self.sin_alpha[i];
            let ca = self.cos_alpha[i];

            let new_c0 = c0 * ct + c1 * st;
            let perp = c1 * ct - c0 * st;

            let new_c1 = perp * ca + c2 * sa;
            let new_c2 = c2 * ca - perp * sa;

            t = new_c0 * self.a[i] + c2 * self.d[i] + t;

            c0 = new_c0;
            c1 = new_c1;
            c2 = new_c2;

            out[i] = AAffine3::<F>::from_mat3_translation(
                AMat3::<F>::from_cols(c0, c1, c2),
                t,
            );
            i += 1;
        }
        Ok(out)
    }

    fn fk_end(&self, q: &SRobotQ<N, F>) -> Result<AAffine3<F>, Self::Error> {
        check_finite::<N, F>(q)?;
        let mut c0 = AVec3::<F>::X;
        let mut c1 = AVec3::<F>::Y;
        let mut c2 = AVec3::<F>::Z;
        let mut t = AVec3::<F>::ZERO;

        let mut i = 0;
        while i < N {
            let (st, ct) = (q.0[i] + self.theta_offset[i]).sin_cos();
            let sa = self.sin_alpha[i];
            let ca = self.cos_alpha[i];

            let new_c0 = c0 * ct + c1 * st;
            let perp = c1 * ct - c0 * st;

            let new_c1 = perp * ca + c2 * sa;
            let new_c2 = c2 * ca - perp * sa;

            t = new_c0 * self.a[i] + c2 * self.d[i] + t;

            c0 = new_c0;
            c1 = new_c1;
            c2 = new_c2;
            i += 1;
        }

        Ok(AAffine3::<F>::from_mat3_translation(
            AMat3::<F>::from_cols(c0, c1, c2),
            t,
        ))
    }

    fn all_fk(
        &self,
        q: &SRobotQ<N, F>,
    ) -> Result<(AAffine3<F>, [AAffine3<F>; N], AAffine3<F>), Self::Error> {
        let frames = self.fk(q)?;
        // DH has no tool/suffix offset, so the last accumulated frame *is*
        // the EE frame. For N == 0 there is nothing to accumulate; the EE
        // is identity.
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

impl From<DHJoint<f32>> for DHJoint<f64> {
    #[inline]
    fn from(j: DHJoint<f32>) -> Self {
        DHJoint {
            a: j.a as f64,
            alpha: j.alpha as f64,
            d: j.d as f64,
            theta_offset: j.theta_offset as f64,
        }
    }
}

impl From<DHJoint<f64>> for DHJoint<f32> {
    #[inline]
    fn from(j: DHJoint<f64>) -> Self {
        DHJoint {
            a: j.a as f32,
            alpha: j.alpha as f32,
            d: j.d as f32,
            theta_offset: j.theta_offset as f32,
        }
    }
}

#[inline]
fn cast_arr<const N: usize, A: Copy, B: Copy>(src: [A; N], cast: impl Fn(A) -> B) -> [B; N] {
    std::array::from_fn(|i| cast(src[i]))
}

impl<const N: usize> From<DHChain<N, f32>> for DHChain<N, f64> {
    #[inline]
    fn from(c: DHChain<N, f32>) -> Self {
        DHChain::<N, f64> {
            a: cast_arr(c.a, |x| x as f64),
            d: cast_arr(c.d, |x| x as f64),
            sin_alpha: cast_arr(c.sin_alpha, |x| x as f64),
            cos_alpha: cast_arr(c.cos_alpha, |x| x as f64),
            theta_offset: cast_arr(c.theta_offset, |x| x as f64),
        }
    }
}

impl<const N: usize> From<DHChain<N, f64>> for DHChain<N, f32> {
    #[inline]
    fn from(c: DHChain<N, f64>) -> Self {
        DHChain::<N, f32> {
            a: cast_arr(c.a, |x| x as f32),
            d: cast_arr(c.d, |x| x as f32),
            sin_alpha: cast_arr(c.sin_alpha, |x| x as f32),
            cos_alpha: cast_arr(c.cos_alpha, |x| x as f32),
            theta_offset: cast_arr(c.theta_offset, |x| x as f32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam_traits_ext::TAffine3;

    fn planar_2dof<F: FKScalar>() -> DHChain<2, F> {
        let zero = F::zero();
        let one = F::one();
        DHChain::<2, F>::from_joints([
            DHJoint { a: one, alpha: zero, d: zero, theta_offset: zero },
            DHJoint { a: one, alpha: zero, d: zero, theta_offset: zero },
        ])
    }

    #[test]
    fn f32_and_f64_agree_at_zero() {
        let f32_chain = planar_2dof::<f32>();
        let f64_chain = planar_2dof::<f64>();

        let q32 = SRobotQ::<2, f32>::zeros();
        let q64 = SRobotQ::<2, f64>::zeros();

        let end32 = f32_chain.fk_end(&q32).unwrap();
        let end64 = f64_chain.fk_end(&q64).unwrap();

        let t32 = end32.translation();
        let t64 = end64.translation();
        assert!((t32.x() as f64 - t64.x()).abs() < 1e-5);
        assert!((t32.y() as f64 - t64.y()).abs() < 1e-5);
        assert!((t32.z() as f64 - t64.z()).abs() < 1e-5);
    }

    #[test]
    fn const_f64_constructor_matches_runtime_f64() {
        const CHAIN_CONST: DHChain<2, f64> = DHChain::<2, f64>::new_f64([
            DHJoint { a: 1.0, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
            DHJoint { a: 1.0, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
        ]);
        let chain_runtime = planar_2dof::<f64>();
        let q = SRobotQ::<2, f64>::from_array([0.5, -0.3]);
        let end_const = CHAIN_CONST.fk_end(&q).unwrap().translation();
        let end_runtime = chain_runtime.fk_end(&q).unwrap().translation();
        assert!((end_const.x() - end_runtime.x()).abs() < 1e-12);
        assert!((end_const.y() - end_runtime.y()).abs() < 1e-12);
    }

    #[test]
    fn cast_f32_to_f64_and_back_preserves_fk() {
        let f32_chain = planar_2dof::<f32>();
        let f64_chain: DHChain<2, f64> = f32_chain.clone().into();
        let f32_again: DHChain<2, f32> = f64_chain.clone().into();

        let q32 = SRobotQ::<2, f32>::from_array([0.5, -0.3]);
        let q64 = SRobotQ::<2, f64>::from_array([0.5, -0.3]);

        let end32 = f32_chain.fk_end(&q32).unwrap().translation();
        let end64 = f64_chain.fk_end(&q64).unwrap().translation();
        let end32b = f32_again.fk_end(&q32).unwrap().translation();

        assert!((end32.x() as f64 - end64.x()).abs() < 1e-4);
        assert!((end32.y() as f64 - end64.y()).abs() < 1e-4);
        assert_eq!(end32, end32b);
    }
}
