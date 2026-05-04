use const_soft_float::soft_f32::SoftF32;
use glam::{Affine3A, Mat3A, Vec3A};

use crate::{DekeError, SRobotQ};

mod dh;
mod dynamic;
mod hp;
mod prismatic;
mod transformed;
mod urdf;

pub use dh::{DHChain, DHJoint};
pub use dynamic::{BoxFK, DynamicDHChain, DynamicHPChain, DynamicURDFChain};
pub use hp::{HPChain, HPJoint};
pub use prismatic::PrismaticFK;
pub use transformed::TransformedFK;
pub use urdf::{URDFBuildError, URDFChain, URDFJoint, URDFJointType, compose_fixed_joints};

/// Const-context sine/cosine via soft-float. Use only inside `const fn`
/// builders where the runtime intrinsic is unavailable; hot paths must call
/// `f32::sin_cos` directly.
#[inline(always)]
const fn const_sin_cos(x: f32) -> (f32, f32) {
    let sf = SoftF32::from_f32(x);
    (sf.sin().to_f32(), sf.cos().to_f32())
}

pub trait FKChain<const N: usize>: Clone + Send + Sync {
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
    fn base_tf(&self) -> Affine3A {
        Affine3A::IDENTITY
    }
    /// Theoretical maximum reach: sum of link lengths (upper bound, ignores joint limits).
    fn max_reach(&self) -> Result<f32, Self::Error> {
        let (_, p, p_ee) = self.joint_axes_positions(&SRobotQ::zeros())?;
        let mut total = 0.0f32;
        let mut prev = p[0];
        for i in 1..N {
            total += (p[i] - prev).length();
            prev = p[i];
        }
        total += (p_ee - prev).length();
        Ok(total)
    }

    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], Self::Error>;
    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, Self::Error>;
    /// Returns joint rotation axes and axis-origin positions in world frame at
    /// configuration `q`, plus the end-effector position.
    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N>,
    ) -> Result<([Vec3A; N], [Vec3A; N], Vec3A), Self::Error>;

    /// Geometric Jacobian (6×N) at configuration `q`.
    /// Rows 0–2: linear velocity, rows 3–5: angular velocity.
    fn jacobian(&self, q: &SRobotQ<N>) -> Result<[[f32; N]; 6], Self::Error> {
        let (z, p, p_ee) = self.joint_axes_positions(q)?;
        let mut j = [[0.0f32; N]; 6];
        for i in 0..N {
            let dp = p_ee - p[i];
            let c = z[i].cross(dp);
            j[0][i] = c.x;
            j[1][i] = c.y;
            j[2][i] = c.z;
            j[3][i] = z[i].x;
            j[4][i] = z[i].y;
            j[5][i] = z[i].z;
        }
        Ok(j)
    }

    /// First time-derivative of the geometric Jacobian.
    fn jacobian_dot(
        &self,
        q: &SRobotQ<N>,
        qdot: &SRobotQ<N>,
    ) -> Result<[[f32; N]; 6], Self::Error> {
        let (z, p, p_ee) = self.joint_axes_positions(q)?;

        let mut omega = Vec3A::ZERO;
        let mut z_dot = [Vec3A::ZERO; N];
        let mut p_dot = [Vec3A::ZERO; N];
        let mut pdot_acc = Vec3A::ZERO;

        for i in 0..N {
            p_dot[i] = pdot_acc;
            z_dot[i] = omega.cross(z[i]);
            omega += qdot.0[i] * z[i];
            let next_p = if i + 1 < N { p[i + 1] } else { p_ee };
            pdot_acc += omega.cross(next_p - p[i]);
        }
        let p_ee_dot = pdot_acc;

        let mut jd = [[0.0f32; N]; 6];
        for i in 0..N {
            let dp = p_ee - p[i];
            let dp_dot = p_ee_dot - p_dot[i];
            let c1 = z_dot[i].cross(dp);
            let c2 = z[i].cross(dp_dot);
            jd[0][i] = c1.x + c2.x;
            jd[1][i] = c1.y + c2.y;
            jd[2][i] = c1.z + c2.z;
            jd[3][i] = z_dot[i].x;
            jd[4][i] = z_dot[i].y;
            jd[5][i] = z_dot[i].z;
        }
        Ok(jd)
    }

    /// Second time-derivative of the geometric Jacobian.
    fn jacobian_ddot(
        &self,
        q: &SRobotQ<N>,
        qdot: &SRobotQ<N>,
        qddot: &SRobotQ<N>,
    ) -> Result<[[f32; N]; 6], Self::Error> {
        let (z, p, p_ee) = self.joint_axes_positions(q)?;

        let mut omega = Vec3A::ZERO;
        let mut omega_dot = Vec3A::ZERO;
        let mut z_dot = [Vec3A::ZERO; N];
        let mut z_ddot = [Vec3A::ZERO; N];
        let mut p_dot = [Vec3A::ZERO; N];
        let mut p_ddot = [Vec3A::ZERO; N];
        let mut pdot_acc = Vec3A::ZERO;
        let mut pddot_acc = Vec3A::ZERO;

        for i in 0..N {
            p_dot[i] = pdot_acc;
            p_ddot[i] = pddot_acc;
            let zd = omega.cross(z[i]);
            z_dot[i] = zd;
            z_ddot[i] = omega_dot.cross(z[i]) + omega.cross(zd);
            omega_dot += qddot.0[i] * z[i] + qdot.0[i] * zd;
            omega += qdot.0[i] * z[i];
            let next_p = if i + 1 < N { p[i + 1] } else { p_ee };
            let delta = next_p - p[i];
            let delta_dot = omega.cross(delta);
            pdot_acc += delta_dot;
            pddot_acc += omega_dot.cross(delta) + omega.cross(delta_dot);
        }
        let p_ee_dot = pdot_acc;
        let p_ee_ddot = pddot_acc;

        let mut jdd = [[0.0f32; N]; 6];
        for i in 0..N {
            let dp = p_ee - p[i];
            let dp_dot = p_ee_dot - p_dot[i];
            let dp_ddot = p_ee_ddot - p_ddot[i];
            let c1 = z_ddot[i].cross(dp);
            let c2 = z_dot[i].cross(dp_dot);
            let c3 = z[i].cross(dp_ddot);
            jdd[0][i] = c1.x + 2.0 * c2.x + c3.x;
            jdd[1][i] = c1.y + 2.0 * c2.y + c3.y;
            jdd[2][i] = c1.z + 2.0 * c2.z + c3.z;
            jdd[3][i] = z_ddot[i].x;
            jdd[4][i] = z_ddot[i].y;
            jdd[5][i] = z_ddot[i].z;
        }
        Ok(jdd)
    }
}

#[inline(always)]
#[cfg(debug_assertions)]
fn check_finite<const N: usize>(q: &SRobotQ<N>) -> Result<(), DekeError> {
    if q.any_non_finite() {
        return Err(DekeError::JointsNonFinite);
    }
    Ok(())
}

#[inline(always)]
#[cfg(not(debug_assertions))]
fn check_finite<const N: usize>(_: &SRobotQ<N>) -> Result<(), std::convert::Infallible> {
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

/// Accumulate a local rotation + translation into the running transform.
/// Shared by both DH and HP — the only difference is how each convention
/// builds `local_c0..c2` and `local_t`.
#[inline(always)]
fn accumulate(
    acc_m: &mut Mat3A,
    acc_t: &mut Vec3A,
    local_c0: Vec3A,
    local_c1: Vec3A,
    local_c2: Vec3A,
    local_t: Vec3A,
) {
    let new_c0 = *acc_m * local_c0;
    let new_c1 = *acc_m * local_c1;
    let new_c2 = *acc_m * local_c2;
    *acc_t = *acc_m * local_t + *acc_t;
    *acc_m = Mat3A::from_cols(new_c0, new_c1, new_c2);
}
