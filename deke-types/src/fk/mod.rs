use std::ops::Mul;

use const_soft_float::soft_f32::SoftF32;
use const_soft_float::soft_f64::SoftF64;
use glam::{Affine3A, DAffine3, DMat3, DVec3, Mat3A, Vec3A};
use glam_traits_ext::{FloatAffine, FloatMat, FloatScalar, FloatVec, TAffine3, TMat3, TVec3};

use crate::{DekeError, SRobotQ};

mod dh;
mod dynamic;
mod fp_dispatch;
mod hp;
mod prismatic;
mod transformed;
mod urdf;

pub use dh::{DHChain, DHJoint};
pub use dynamic::{BoxFK, DynamicDHChain, DynamicHPChain, DynamicURDFChain};
pub use fp_dispatch::FPDispatch;
pub use hp::{HPChain, HPJoint};
pub use prismatic::PrismaticFK;
pub use transformed::TransformedFK;
pub use urdf::{
    URDFBuildError, URDFChain, URDFJoint, URDFJointType, compose_fixed_joints,
    compose_fixed_joints_f64,
};

/// Const-context sine/cosine via soft-float. Use only inside `const fn`
/// builders where the runtime intrinsic is unavailable; hot paths must call
/// `f32::sin_cos` directly.
#[inline(always)]
const fn const_sin_cos(x: f32) -> (f32, f32) {
    let sf = SoftF32::from_f32(x);
    (sf.sin().to_f32(), sf.cos().to_f32())
}

/// `f64` analogue of [`const_sin_cos`].
#[inline(always)]
const fn const_sin_cos_f64(x: f64) -> (f64, f64) {
    let sf = SoftF64::from_f64(x);
    (sf.sin().to_f64(), sf.cos().to_f64())
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Scalar bound for FK code: `FloatScalar` plus the SIMD-aligned vec/mat/affine
/// types the FK code operates on.
///
/// For `f32`, the aligned types are `Vec3A`/`Mat3A`/`Affine3A` (16-byte SIMD).
/// For `f64`, they are `DVec3`/`DMat3`/`DAffine3` (already efficient packing).
/// Both share a uniform interface via the `T*` traits in `glam-traits-ext`.
pub trait FKScalar: FloatScalar + Copy + std::fmt::Debug + Send + Sync + 'static + sealed::Sealed {
    type AVec3: TVec3<Self, MaybeAligned = Self::AVec3>;
    type AMat3: TMat3<Self, MaybeAligned = Self::AMat3>
        + FloatMat<Self, Col = Self::AVec3>
        + Mul<Self::AVec3, Output = Self::AVec3>;
    type AAffine3: TAffine3<Self, MaybeAligned = Self::AAffine3>
        + FloatAffine<Self, Vec = Self::AVec3, Mat = Self::AMat3>
        + Mul<Self::AAffine3, Output = Self::AAffine3>;
}

impl FKScalar for f32 {
    type AVec3 = Vec3A;
    type AMat3 = Mat3A;
    type AAffine3 = Affine3A;
}

impl FKScalar for f64 {
    type AVec3 = DVec3;
    type AMat3 = DMat3;
    type AAffine3 = DAffine3;
}

#[allow(type_alias_bounds)]
pub(crate) type AAffine3<F: FKScalar> = F::AAffine3;
#[allow(type_alias_bounds)]
pub(crate) type AMat3<F: FKScalar> = F::AMat3;
#[allow(type_alias_bounds)]
pub(crate) type AVec3<F: FKScalar> = F::AVec3;

pub trait FKChain<const N: usize, F: FKScalar = f32>: Clone + Send + Sync {
    type Error: Into<DekeError>;

    fn dof(&self) -> usize {
        N
    }
    /// Configuration-independent transform from the robot's base frame to the
    /// world frame. Defaults to identity; wrappers that install a static
    /// prefix (e.g. [`TransformedFK`] with a prefix set, or [`URDFChain`]
    /// with fixed leading joints baked in) override this so downstream
    /// consumers (collision validators, visualizers) can place the static
    /// base body at the correct pose.
    fn base_tf(&self) -> AAffine3<F> {
        AAffine3::<F>::IDENTITY
    }
    /// Theoretical maximum reach: sum of link lengths (upper bound, ignores joint limits).
    fn max_reach(&self) -> Result<F, Self::Error> {
        let (_, p, p_ee) = self.joint_axes_positions(&SRobotQ::zeros())?;
        let mut total = F::zero();
        let mut prev = p[0];
        for i in 1..N {
            total = total + (p[i] - prev).length();
            prev = p[i];
        }
        total = total + (p_ee - prev).length();
        Ok(total)
    }

    fn fk(&self, q: &SRobotQ<N, F>) -> Result<[AAffine3<F>; N], Self::Error>;
    fn fk_end(&self, q: &SRobotQ<N, F>) -> Result<AAffine3<F>, Self::Error>;
    /// Returns joint rotation axes and axis-origin positions in world frame at
    /// configuration `q`, plus the end-effector position.
    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N, F>,
    ) -> Result<([AVec3<F>; N], [AVec3<F>; N], AVec3<F>), Self::Error>;

    /// Geometric Jacobian (6×N) at configuration `q`.
    /// Rows 0–2: linear velocity, rows 3–5: angular velocity.
    fn jacobian(&self, q: &SRobotQ<N, F>) -> Result<[[F; N]; 6], Self::Error> {
        let (z, p, p_ee) = self.joint_axes_positions(q)?;
        let mut j = [[F::zero(); N]; 6];
        for i in 0..N {
            let dp = p_ee - p[i];
            let c = z[i].cross(dp);
            j[0][i] = c.x();
            j[1][i] = c.y();
            j[2][i] = c.z();
            j[3][i] = z[i].x();
            j[4][i] = z[i].y();
            j[5][i] = z[i].z();
        }
        Ok(j)
    }

    /// First time-derivative of the geometric Jacobian.
    fn jacobian_dot(
        &self,
        q: &SRobotQ<N, F>,
        qdot: &SRobotQ<N, F>,
    ) -> Result<[[F; N]; 6], Self::Error> {
        let (z, p, p_ee) = self.joint_axes_positions(q)?;

        let mut omega = AVec3::<F>::ZERO;
        let mut z_dot = [AVec3::<F>::ZERO; N];
        let mut p_dot = [AVec3::<F>::ZERO; N];
        let mut pdot_acc = AVec3::<F>::ZERO;

        for i in 0..N {
            p_dot[i] = pdot_acc;
            z_dot[i] = omega.cross(z[i]);
            omega += z[i] * qdot.0[i];
            let next_p = if i + 1 < N { p[i + 1] } else { p_ee };
            pdot_acc += omega.cross(next_p - p[i]);
        }
        let p_ee_dot = pdot_acc;

        let mut jd = [[F::zero(); N]; 6];
        for i in 0..N {
            let dp = p_ee - p[i];
            let dp_dot = p_ee_dot - p_dot[i];
            let c1 = z_dot[i].cross(dp);
            let c2 = z[i].cross(dp_dot);
            jd[0][i] = c1.x() + c2.x();
            jd[1][i] = c1.y() + c2.y();
            jd[2][i] = c1.z() + c2.z();
            jd[3][i] = z_dot[i].x();
            jd[4][i] = z_dot[i].y();
            jd[5][i] = z_dot[i].z();
        }
        Ok(jd)
    }

    /// Second time-derivative of the geometric Jacobian.
    fn jacobian_ddot(
        &self,
        q: &SRobotQ<N, F>,
        qdot: &SRobotQ<N, F>,
        qddot: &SRobotQ<N, F>,
    ) -> Result<[[F; N]; 6], Self::Error> {
        let (z, p, p_ee) = self.joint_axes_positions(q)?;

        let mut omega = AVec3::<F>::ZERO;
        let mut omega_dot = AVec3::<F>::ZERO;
        let mut z_dot = [AVec3::<F>::ZERO; N];
        let mut z_ddot = [AVec3::<F>::ZERO; N];
        let mut p_dot = [AVec3::<F>::ZERO; N];
        let mut p_ddot = [AVec3::<F>::ZERO; N];
        let mut pdot_acc = AVec3::<F>::ZERO;
        let mut pddot_acc = AVec3::<F>::ZERO;

        for i in 0..N {
            p_dot[i] = pdot_acc;
            p_ddot[i] = pddot_acc;
            let zd = omega.cross(z[i]);
            z_dot[i] = zd;
            z_ddot[i] = omega_dot.cross(z[i]) + omega.cross(zd);
            omega_dot += z[i] * qddot.0[i] + zd * qdot.0[i];
            omega += z[i] * qdot.0[i];
            let next_p = if i + 1 < N { p[i + 1] } else { p_ee };
            let delta = next_p - p[i];
            let delta_dot = omega.cross(delta);
            pdot_acc += delta_dot;
            pddot_acc += omega_dot.cross(delta) + omega.cross(delta_dot);
        }
        let p_ee_dot = pdot_acc;
        let p_ee_ddot = pddot_acc;

        let mut jdd = [[F::zero(); N]; 6];
        for i in 0..N {
            let dp = p_ee - p[i];
            let dp_dot = p_ee_dot - p_dot[i];
            let dp_ddot = p_ee_ddot - p_ddot[i];
            let c1 = z_ddot[i].cross(dp);
            let c2 = z_dot[i].cross(dp_dot);
            let c3 = z[i].cross(dp_ddot);
            let two = F::one() + F::one();
            jdd[0][i] = c1.x() + two * c2.x() + c3.x();
            jdd[1][i] = c1.y() + two * c2.y() + c3.y();
            jdd[2][i] = c1.z() + two * c2.z() + c3.z();
            jdd[3][i] = z_ddot[i].x();
            jdd[4][i] = z_ddot[i].y();
            jdd[5][i] = z_ddot[i].z();
        }
        Ok(jdd)
    }
}

#[inline(always)]
#[cfg(debug_assertions)]
fn check_finite<const N: usize, F: FloatScalar>(q: &SRobotQ<N, F>) -> Result<(), DekeError> {
    if q.any_non_finite() {
        return Err(DekeError::JointsNonFinite);
    }
    Ok(())
}

#[inline(always)]
#[cfg(not(debug_assertions))]
fn check_finite<const N: usize, F: FloatScalar>(_: &SRobotQ<N, F>) -> Result<(), std::convert::Infallible> {
    Ok(())
}

#[inline(always)]
const fn abs_f32(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// Const-friendly affine transform backed by plain f32 arrays. `glam`'s
/// `Vec3A`/`Mat3A` types use SIMD and expose components via a non-const
/// `Deref`, so `const fn` code that needs per-component arithmetic (compose,
/// identity check) routes through this type and only converts to
/// `Affine3A` at the boundary.
#[derive(Debug, Clone, Copy)]
struct AffineRaw {
    c0: [f32; 3],
    c1: [f32; 3],
    c2: [f32; 3],
    t: [f32; 3],
}

impl AffineRaw {
    const IDENTITY: Self = Self {
        c0: [1.0, 0.0, 0.0],
        c1: [0.0, 1.0, 0.0],
        c2: [0.0, 0.0, 1.0],
        t: [0.0, 0.0, 0.0],
    };

    /// `self * other` — applies `other` first, then `self`.
    #[inline(always)]
    const fn mul(self, other: Self) -> Self {
        let nc0 = [
            self.c0[0] * other.c0[0] + self.c1[0] * other.c0[1] + self.c2[0] * other.c0[2],
            self.c0[1] * other.c0[0] + self.c1[1] * other.c0[1] + self.c2[1] * other.c0[2],
            self.c0[2] * other.c0[0] + self.c1[2] * other.c0[1] + self.c2[2] * other.c0[2],
        ];
        let nc1 = [
            self.c0[0] * other.c1[0] + self.c1[0] * other.c1[1] + self.c2[0] * other.c1[2],
            self.c0[1] * other.c1[0] + self.c1[1] * other.c1[1] + self.c2[1] * other.c1[2],
            self.c0[2] * other.c1[0] + self.c1[2] * other.c1[1] + self.c2[2] * other.c1[2],
        ];
        let nc2 = [
            self.c0[0] * other.c2[0] + self.c1[0] * other.c2[1] + self.c2[0] * other.c2[2],
            self.c0[1] * other.c2[0] + self.c1[1] * other.c2[1] + self.c2[1] * other.c2[2],
            self.c0[2] * other.c2[0] + self.c1[2] * other.c2[1] + self.c2[2] * other.c2[2],
        ];
        let nt = [
            self.c0[0] * other.t[0]
                + self.c1[0] * other.t[1]
                + self.c2[0] * other.t[2]
                + self.t[0],
            self.c0[1] * other.t[0]
                + self.c1[1] * other.t[1]
                + self.c2[1] * other.t[2]
                + self.t[1],
            self.c0[2] * other.t[0]
                + self.c1[2] * other.t[1]
                + self.c2[2] * other.t[2]
                + self.t[2],
        ];
        Self {
            c0: nc0,
            c1: nc1,
            c2: nc2,
            t: nt,
        }
    }

    #[inline(always)]
    const fn is_identity(&self) -> bool {
        const EPS: f32 = 1e-6;
        abs_f32(self.c0[0] - 1.0) <= EPS
            && abs_f32(self.c0[1]) <= EPS
            && abs_f32(self.c0[2]) <= EPS
            && abs_f32(self.c1[0]) <= EPS
            && abs_f32(self.c1[1] - 1.0) <= EPS
            && abs_f32(self.c1[2]) <= EPS
            && abs_f32(self.c2[0]) <= EPS
            && abs_f32(self.c2[1]) <= EPS
            && abs_f32(self.c2[2] - 1.0) <= EPS
            && abs_f32(self.t[0]) <= EPS
            && abs_f32(self.t[1]) <= EPS
            && abs_f32(self.t[2]) <= EPS
    }

    /// Build the URDF RPY-convention rotation (`Rz(yaw)·Ry(pitch)·Rx(roll)`)
    /// and translate by `xyz`, using [`const_sin_cos`] for const evaluation.
    #[inline(always)]
    const fn from_xyz_rpy(xyz: (f64, f64, f64), rpy: (f64, f64, f64)) -> Self {
        let (ox, oy, oz) = xyz;
        let (roll, pitch, yaw) = rpy;
        let (sr, cr) = const_sin_cos(roll as f32);
        let (sp, cp) = const_sin_cos(pitch as f32);
        let (sy, cy) = const_sin_cos(yaw as f32);
        Self {
            c0: [cy * cp, sy * cp, -sp],
            c1: [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr],
            c2: [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr],
            t: [ox as f32, oy as f32, oz as f32],
        }
    }

    #[inline(always)]
    const fn to_affine3a(self) -> Affine3A {
        Affine3A {
            matrix3: Mat3A::from_cols(
                Vec3A::new(self.c0[0], self.c0[1], self.c0[2]),
                Vec3A::new(self.c1[0], self.c1[1], self.c1[2]),
                Vec3A::new(self.c2[0], self.c2[1], self.c2[2]),
            ),
            translation: Vec3A::new(self.t[0], self.t[1], self.t[2]),
        }
    }

    #[inline(always)]
    const fn c0_vec3a(&self) -> Vec3A {
        Vec3A::new(self.c0[0], self.c0[1], self.c0[2])
    }

    #[inline(always)]
    const fn c1_vec3a(&self) -> Vec3A {
        Vec3A::new(self.c1[0], self.c1[1], self.c1[2])
    }

    #[inline(always)]
    const fn c2_vec3a(&self) -> Vec3A {
        Vec3A::new(self.c2[0], self.c2[1], self.c2[2])
    }

    #[inline(always)]
    const fn t_vec3a(&self) -> Vec3A {
        Vec3A::new(self.t[0], self.t[1], self.t[2])
    }
}

#[inline(always)]
const fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

/// `f64` analogue of [`AffineRaw`] for const-context URDF construction in
/// `URDFChain<N, f64>`. Stores plain `[f64; 3]` columns + translation; converts
/// to `glam::DAffine3` only at the boundary.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AffineRaw64 {
    pub(crate) c0: [f64; 3],
    pub(crate) c1: [f64; 3],
    pub(crate) c2: [f64; 3],
    pub(crate) t: [f64; 3],
}

impl AffineRaw64 {
    pub(crate) const IDENTITY: Self = Self {
        c0: [1.0, 0.0, 0.0],
        c1: [0.0, 1.0, 0.0],
        c2: [0.0, 0.0, 1.0],
        t: [0.0, 0.0, 0.0],
    };

    /// `self * other` — applies `other` first, then `self`.
    #[inline(always)]
    pub(crate) const fn mul(self, other: Self) -> Self {
        let nc0 = [
            self.c0[0] * other.c0[0] + self.c1[0] * other.c0[1] + self.c2[0] * other.c0[2],
            self.c0[1] * other.c0[0] + self.c1[1] * other.c0[1] + self.c2[1] * other.c0[2],
            self.c0[2] * other.c0[0] + self.c1[2] * other.c0[1] + self.c2[2] * other.c0[2],
        ];
        let nc1 = [
            self.c0[0] * other.c1[0] + self.c1[0] * other.c1[1] + self.c2[0] * other.c1[2],
            self.c0[1] * other.c1[0] + self.c1[1] * other.c1[1] + self.c2[1] * other.c1[2],
            self.c0[2] * other.c1[0] + self.c1[2] * other.c1[1] + self.c2[2] * other.c1[2],
        ];
        let nc2 = [
            self.c0[0] * other.c2[0] + self.c1[0] * other.c2[1] + self.c2[0] * other.c2[2],
            self.c0[1] * other.c2[0] + self.c1[1] * other.c2[1] + self.c2[1] * other.c2[2],
            self.c0[2] * other.c2[0] + self.c1[2] * other.c2[1] + self.c2[2] * other.c2[2],
        ];
        let nt = [
            self.c0[0] * other.t[0]
                + self.c1[0] * other.t[1]
                + self.c2[0] * other.t[2]
                + self.t[0],
            self.c0[1] * other.t[0]
                + self.c1[1] * other.t[1]
                + self.c2[1] * other.t[2]
                + self.t[1],
            self.c0[2] * other.t[0]
                + self.c1[2] * other.t[1]
                + self.c2[2] * other.t[2]
                + self.t[2],
        ];
        Self { c0: nc0, c1: nc1, c2: nc2, t: nt }
    }

    #[inline(always)]
    pub(crate) const fn is_identity(&self) -> bool {
        const EPS: f64 = 1e-12;
        abs_f64(self.c0[0] - 1.0) <= EPS
            && abs_f64(self.c0[1]) <= EPS
            && abs_f64(self.c0[2]) <= EPS
            && abs_f64(self.c1[0]) <= EPS
            && abs_f64(self.c1[1] - 1.0) <= EPS
            && abs_f64(self.c1[2]) <= EPS
            && abs_f64(self.c2[0]) <= EPS
            && abs_f64(self.c2[1]) <= EPS
            && abs_f64(self.c2[2] - 1.0) <= EPS
            && abs_f64(self.t[0]) <= EPS
            && abs_f64(self.t[1]) <= EPS
            && abs_f64(self.t[2]) <= EPS
    }

    /// URDF RPY-convention rotation `Rz(yaw)·Ry(pitch)·Rx(roll)` plus `xyz`
    /// translation, evaluated in `const` via [`const_sin_cos_f64`].
    #[inline(always)]
    pub(crate) const fn from_xyz_rpy(xyz: (f64, f64, f64), rpy: (f64, f64, f64)) -> Self {
        let (ox, oy, oz) = xyz;
        let (roll, pitch, yaw) = rpy;
        let (sr, cr) = const_sin_cos_f64(roll);
        let (sp, cp) = const_sin_cos_f64(pitch);
        let (sy, cy) = const_sin_cos_f64(yaw);
        Self {
            c0: [cy * cp, sy * cp, -sp],
            c1: [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr],
            c2: [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr],
            t: [ox, oy, oz],
        }
    }

    #[inline(always)]
    pub(crate) const fn to_daffine3(self) -> DAffine3 {
        DAffine3 {
            matrix3: DMat3::from_cols(
                DVec3::new(self.c0[0], self.c0[1], self.c0[2]),
                DVec3::new(self.c1[0], self.c1[1], self.c1[2]),
                DVec3::new(self.c2[0], self.c2[1], self.c2[2]),
            ),
            translation: DVec3::new(self.t[0], self.t[1], self.t[2]),
        }
    }

    #[inline(always)]
    pub(crate) const fn c0_dvec3(&self) -> DVec3 {
        DVec3::new(self.c0[0], self.c0[1], self.c0[2])
    }

    #[inline(always)]
    pub(crate) const fn c1_dvec3(&self) -> DVec3 {
        DVec3::new(self.c1[0], self.c1[1], self.c1[2])
    }

    #[inline(always)]
    pub(crate) const fn c2_dvec3(&self) -> DVec3 {
        DVec3::new(self.c2[0], self.c2[1], self.c2[2])
    }

    #[inline(always)]
    pub(crate) const fn t_dvec3(&self) -> DVec3 {
        DVec3::new(self.t[0], self.t[1], self.t[2])
    }
}

