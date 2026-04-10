use glam::{Affine3A, Mat3A, Vec3A};

use crate::{DekeError, SRobotQ};

#[inline(always)]
fn fast_sin_cos(x: f32) -> (f32, f32) {
    const FRAC_2_PI: f32 = std::f32::consts::FRAC_2_PI;
    const PI_2_HI: f32 = 1.570_796_4_f32;
    const PI_2_LO: f32 = -4.371_139e-8_f32;

    const S1: f32 = -0.166_666_67;
    const S2: f32 = 0.008_333_294;
    const S3: f32 = -0.000_198_074_14;

    const C1: f32 = -0.5;
    const C2: f32 = 0.041_666_52;
    const C3: f32 = -0.001_388_523_4;

    let q = (x * FRAC_2_PI).round();
    let qi = q as i32;
    let r = x - q * PI_2_HI - q * PI_2_LO;
    let r2 = r * r;

    let sin_r = r * (1.0 + r2 * (S1 + r2 * (S2 + r2 * S3)));
    let cos_r = 1.0 + r2 * (C1 + r2 * (C2 + r2 * C3));

    let (s, c) = match qi & 3 {
        0 => (sin_r, cos_r),
        1 => (cos_r, -sin_r),
        2 => (-sin_r, -cos_r),
        3 => (-cos_r, sin_r),
        _ => unsafe { std::hint::unreachable_unchecked() },
    };

    (s, c)
}

pub trait FKChain<const N: usize>: Clone + Send + Sync {
    type Error: Into<DekeError>;
    fn dof(&self) -> usize {
        N
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
    pub fn new(joints: [DHJoint; N]) -> Self {
        let mut a = [0.0; N];
        let mut d = [0.0; N];
        let mut sin_alpha = [0.0; N];
        let mut cos_alpha = [0.0; N];
        let mut theta_offset = [0.0; N];

        let mut i = 0;
        while i < N {
            a[i] = joints[i].a;
            d[i] = joints[i].d;
            let (sa, ca) = joints[i].alpha.sin_cos();
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
            let (st, ct) = fast_sin_cos(q.0[i] + self.theta_offset[i]);
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
            let (st, ct) = fast_sin_cos(q.0[i] + self.theta_offset[i]);
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
    pub fn new(joints: [HPJoint; N]) -> Self {
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
            let (sa, ca) = joints[i].alpha.sin_cos();
            sin_alpha[i] = sa;
            cos_alpha[i] = ca;
            let (sb, cb) = joints[i].beta.sin_cos();
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
            let (st, ct) = fast_sin_cos(q.0[i] + self.theta_offset[i]);
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
            let (st, ct) = fast_sin_cos(q.0[i] + self.theta_offset[i]);
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

#[derive(Debug, Clone, Copy)]
pub struct URDFJoint {
    pub origin_xyz: [f64; 3],
    pub origin_rpy: [f64; 3],
    pub axis: [f64; 3],
}

/// Precomputed per-joint axis type for column-rotation FK.
#[derive(Debug, Clone, Copy)]
enum JointAxis {
    Z,
    Y(f32),
    X(f32),
}

/// FK chain using exact URDF joint transforms.
///
/// Accumulation works directly on columns:
///   1. Translation: `t += fx·c0 + fy·c1 + fz·c2`
///   2. Fixed rotation: `(c0,c1,c2) = (c0,c1,c2) * fixed_rot`
///   3. Joint rotation: 2D rotation on the appropriate column pair
///
/// When `fixed_rot` is identity (RPY = 0, the common case), step 2 is
/// skipped entirely, making per-joint cost a single 2D column rotation
/// plus translation — cheaper than DH.
#[derive(Debug, Clone)]
pub struct URDFChain<const N: usize> {
    fr_c0: [Vec3A; N],
    fr_c1: [Vec3A; N],
    fr_c2: [Vec3A; N],
    fr_identity: [bool; N],
    fixed_trans: [Vec3A; N],
    axis: [JointAxis; N],
}

impl<const N: usize> URDFChain<N> {
    pub fn new(joints: [URDFJoint; N]) -> Self {
        let mut fr_c0 = [Vec3A::X; N];
        let mut fr_c1 = [Vec3A::Y; N];
        let mut fr_c2 = [Vec3A::Z; N];
        let mut fr_identity = [true; N];
        let mut fixed_trans = [Vec3A::ZERO; N];
        let mut axis = [JointAxis::Z; N];

        for i in 0..N {
            let [ox, oy, oz] = joints[i].origin_xyz;
            let [roll, pitch, yaw] = joints[i].origin_rpy;

            let is_identity = roll.abs() < 1e-10 && pitch.abs() < 1e-10 && yaw.abs() < 1e-10;
            fr_identity[i] = is_identity;

            if !is_identity {
                let (sr, cr) = roll.sin_cos();
                let (sp, cp) = pitch.sin_cos();
                let (sy, cy) = yaw.sin_cos();
                fr_c0[i] = Vec3A::new((cy * cp) as f32, (sy * cp) as f32, (-sp) as f32);
                fr_c1[i] = Vec3A::new(
                    (cy * sp * sr - sy * cr) as f32,
                    (sy * sp * sr + cy * cr) as f32,
                    (cp * sr) as f32,
                );
                fr_c2[i] = Vec3A::new(
                    (cy * sp * cr + sy * sr) as f32,
                    (sy * sp * cr - cy * sr) as f32,
                    (cp * cr) as f32,
                );
            }

            fixed_trans[i] = Vec3A::new(ox as f32, oy as f32, oz as f32);

            let [ax, ay, az] = joints[i].axis;
            if az.abs() > 0.5 {
                axis[i] = JointAxis::Z;
            } else if ay.abs() > 0.5 {
                axis[i] = JointAxis::Y(ay.signum() as f32);
            } else {
                axis[i] = JointAxis::X(ax.signum() as f32);
            }
        }

        Self {
            fr_c0,
            fr_c1,
            fr_c2,
            fr_identity,
            fixed_trans,
            axis,
        }
    }

    /// Apply fixed rotation + joint rotation to accumulator columns.
    #[inline(always)]
    fn accumulate_joint(
        &self,
        i: usize,
        st: f32,
        ct: f32,
        c0: &mut Vec3A,
        c1: &mut Vec3A,
        c2: &mut Vec3A,
        t: &mut Vec3A,
    ) {
        let ft = self.fixed_trans[i];
        *t = ft.x * *c0 + ft.y * *c1 + ft.z * *c2 + *t;

        let (f0, f1, f2) = if self.fr_identity[i] {
            (*c0, *c1, *c2)
        } else {
            let fc0 = self.fr_c0[i];
            let fc1 = self.fr_c1[i];
            let fc2 = self.fr_c2[i];
            (
                fc0.x * *c0 + fc0.y * *c1 + fc0.z * *c2,
                fc1.x * *c0 + fc1.y * *c1 + fc1.z * *c2,
                fc2.x * *c0 + fc2.y * *c1 + fc2.z * *c2,
            )
        };

        match self.axis[i] {
            JointAxis::Z => {
                let new_c0 = ct * f0 + st * f1;
                let new_c1 = ct * f1 - st * f0;
                *c0 = new_c0;
                *c1 = new_c1;
                *c2 = f2;
            }
            JointAxis::Y(s) => {
                let sst = s * st;
                let new_c0 = ct * f0 - sst * f2;
                let new_c2 = sst * f0 + ct * f2;
                *c0 = new_c0;
                *c1 = f1;
                *c2 = new_c2;
            }
            JointAxis::X(s) => {
                let sst = s * st;
                let new_c1 = ct * f1 + sst * f2;
                let new_c2 = ct * f2 - sst * f1;
                *c0 = f0;
                *c1 = new_c1;
                *c2 = new_c2;
            }
        }
    }
}

impl<const N: usize> FKChain<N> for URDFChain<N> {
    #[cfg(debug_assertions)]
    type Error = DekeError;
    #[cfg(not(debug_assertions))]
    type Error = std::convert::Infallible;

    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], Self::Error> {
        check_finite(q)?;
        let mut out = [Affine3A::IDENTITY; N];
        let mut c0 = Vec3A::X;
        let mut c1 = Vec3A::Y;
        let mut c2 = Vec3A::Z;
        let mut t = Vec3A::ZERO;

        let mut i = 0;
        while i < N {
            let (st, ct) = fast_sin_cos(q.0[i]);
            self.accumulate_joint(i, st, ct, &mut c0, &mut c1, &mut c2, &mut t);

            out[i] = Affine3A {
                matrix3: Mat3A::from_cols(c0, c1, c2),
                translation: t,
            };
            i += 1;
        }
        Ok(out)
    }

    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, Self::Error> {
        check_finite(q)?;
        let mut c0 = Vec3A::X;
        let mut c1 = Vec3A::Y;
        let mut c2 = Vec3A::Z;
        let mut t = Vec3A::ZERO;

        let mut i = 0;
        while i < N {
            let (st, ct) = fast_sin_cos(q.0[i]);
            self.accumulate_joint(i, st, ct, &mut c0, &mut c1, &mut c2, &mut t);
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
        let mut axes = [Vec3A::ZERO; N];
        let mut positions = [Vec3A::ZERO; N];

        for i in 0..N {
            axes[i] = match self.axis[i] {
                JointAxis::Z => frames[i].matrix3.z_axis,
                JointAxis::Y(s) => s * frames[i].matrix3.y_axis,
                JointAxis::X(s) => s * frames[i].matrix3.x_axis,
            };
            positions[i] = frames[i].translation;
        }

        Ok((axes, positions, frames[N - 1].translation))
    }
}

/// Wraps an `FKChain` with an optional prefix (base) and/or suffix (tool) transform.
///
/// - `fk` applies only the prefix — intermediate frames stay in world coordinates
///   without the tool offset.
/// - `fk_end` and `joint_axes_positions` apply both — the end-effector includes
///   the tool tip.
#[derive(Debug, Clone)]
pub struct TransformedFK<const N: usize, FK: FKChain<N>> {
    inner: FK,
    prefix: Option<Affine3A>,
    suffix: Option<Affine3A>,
}

impl<const N: usize, FK: FKChain<N>> TransformedFK<N, FK> {
    pub fn new(inner: FK) -> Self {
        Self {
            inner,
            prefix: None,
            suffix: None,
        }
    }

    pub fn with_prefix(mut self, prefix: Affine3A) -> Self {
        self.prefix = Some(prefix);
        self
    }

    pub fn with_suffix(mut self, suffix: Affine3A) -> Self {
        self.suffix = Some(suffix);
        self
    }

    pub fn set_prefix(&mut self, prefix: Option<Affine3A>) {
        self.prefix = prefix;
    }

    pub fn set_suffix(&mut self, suffix: Option<Affine3A>) {
        self.suffix = suffix;
    }

    pub fn prefix(&self) -> Option<&Affine3A> {
        self.prefix.as_ref()
    }

    pub fn suffix(&self) -> Option<&Affine3A> {
        self.suffix.as_ref()
    }

    pub fn inner(&self) -> &FK {
        &self.inner
    }
}

impl<const N: usize, FK: FKChain<N>> FKChain<N> for TransformedFK<N, FK> {
    type Error = FK::Error;

    fn max_reach(&self) -> Result<f32, Self::Error> {
        let mut reach = self.inner.max_reach()?;
        if let Some(suf) = &self.suffix {
            reach += Vec3A::from(suf.translation).length();
        }
        Ok(reach)
    }

    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], Self::Error> {
        let mut frames = self.inner.fk(q)?;
        if let Some(pre) = &self.prefix {
            for f in &mut frames {
                *f = *pre * *f;
            }
        }
        Ok(frames)
    }

    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, Self::Error> {
        let mut end = self.inner.fk_end(q)?;
        if let Some(pre) = &self.prefix {
            end = *pre * end;
        }
        if let Some(suf) = &self.suffix {
            end = end * *suf;
        }
        Ok(end)
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N>,
    ) -> Result<([Vec3A; N], [Vec3A; N], Vec3A), Self::Error> {
        let (mut axes, mut positions, inner_p_ee) = self.inner.joint_axes_positions(q)?;

        if let Some(pre) = &self.prefix {
            let rot = pre.matrix3;
            let t = Vec3A::from(pre.translation);
            for i in 0..N {
                axes[i] = rot * axes[i];
                positions[i] = rot * positions[i] + t;
            }
        }

        let p_ee = if self.prefix.is_some() || self.suffix.is_some() {
            self.fk_end(q)?.translation
        } else {
            inner_p_ee
        };

        Ok((axes, positions, p_ee))
    }
}

/// Wraps an `FKChain<N>` and prepends a prismatic (linear) joint, producing
/// an `FKChain<M>` where `M = N + 1`.
///
/// The prismatic joint always acts first in the kinematic chain — it
/// translates the entire arm along `axis` (world frame).  The
/// `q_index_first` flag only controls where the prismatic value is read
/// from in `SRobotQ<M>`: when `true` it is `q[0]`, when `false` it is
/// `q[M-1]`.
///
/// Jacobian columns for the prismatic joint are `[axis; 0]` (pure linear,
/// no angular contribution).  Because the prismatic uniformly shifts all
/// positions, the revolute Jacobian columns are identical to the inner
/// chain's.
#[derive(Debug, Clone)]
pub struct PrismaticFK<const M: usize, const N: usize, FK: FKChain<N>> {
    inner: FK,
    axis: Vec3A,
    q_index_first: bool,
}

impl<const M: usize, const N: usize, FK: FKChain<N>> PrismaticFK<M, N, FK> {
    pub fn new(inner: FK, axis: Vec3A, q_index_first: bool) -> Self {
        const { assert!(M == N + 1, "M must equal N + 1") };
        Self {
            inner,
            axis,
            q_index_first,
        }
    }

    pub fn inner(&self) -> &FK {
        &self.inner
    }

    pub fn axis(&self) -> Vec3A {
        self.axis
    }

    pub fn q_index_first(&self) -> bool {
        self.q_index_first
    }

    fn split_q(&self, q: &SRobotQ<M>) -> (f32, SRobotQ<N>) {
        let mut inner = [0.0f32; N];
        if self.q_index_first {
            inner.copy_from_slice(&q.0[1..M]);
            (q.0[0], SRobotQ(inner))
        } else {
            inner.copy_from_slice(&q.0[..N]);
            (q.0[M - 1], SRobotQ(inner))
        }
    }

    fn prismatic_col(&self) -> usize {
        if self.q_index_first { 0 } else { N }
    }

    fn revolute_offset(&self) -> usize {
        if self.q_index_first { 1 } else { 0 }
    }
}

impl<const M: usize, const N: usize, FK: FKChain<N>> FKChain<M> for PrismaticFK<M, N, FK> {
    type Error = FK::Error;

    fn fk(&self, q: &SRobotQ<M>) -> Result<[Affine3A; M], Self::Error> {
        let (q_p, inner_q) = self.split_q(q);
        let offset = q_p * self.axis;
        let inner_frames = self.inner.fk(&inner_q)?;
        let mut out = [Affine3A::IDENTITY; M];

        out[0] = Affine3A::from_translation(offset.into());
        for i in 0..N {
            let mut f = inner_frames[i];
            f.translation += offset;
            out[i + 1] = f;
        }

        Ok(out)
    }

    fn fk_end(&self, q: &SRobotQ<M>) -> Result<Affine3A, Self::Error> {
        let (q_p, inner_q) = self.split_q(q);
        let mut end = self.inner.fk_end(&inner_q)?;
        end.translation += q_p * self.axis;
        Ok(end)
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<M>,
    ) -> Result<([Vec3A; M], [Vec3A; M], Vec3A), Self::Error> {
        let (q_p, inner_q) = self.split_q(q);
        let offset = q_p * self.axis;
        let (inner_axes, inner_pos, inner_p_ee) = self.inner.joint_axes_positions(&inner_q)?;

        let mut axes = [Vec3A::ZERO; M];
        let mut positions = [Vec3A::ZERO; M];

        axes[0] = self.axis;
        for i in 0..N {
            axes[i + 1] = inner_axes[i];
            positions[i + 1] = inner_pos[i] + offset;
        }

        Ok((axes, positions, inner_p_ee + offset))
    }

    fn jacobian(&self, q: &SRobotQ<M>) -> Result<[[f32; M]; 6], Self::Error> {
        let (_q_p, inner_q) = self.split_q(q);
        let inner_j = self.inner.jacobian(&inner_q)?;
        let p_col = self.prismatic_col();
        let r_off = self.revolute_offset();

        let mut j = [[0.0f32; M]; 6];
        j[0][p_col] = self.axis.x;
        j[1][p_col] = self.axis.y;
        j[2][p_col] = self.axis.z;

        for row in 0..6 {
            for col in 0..N {
                j[row][col + r_off] = inner_j[row][col];
            }
        }

        Ok(j)
    }

    fn jacobian_dot(
        &self,
        q: &SRobotQ<M>,
        qdot: &SRobotQ<M>,
    ) -> Result<[[f32; M]; 6], Self::Error> {
        let (_q_p, inner_q) = self.split_q(q);
        let (_qdot_p, inner_qdot) = self.split_q(qdot);
        let inner_jd = self.inner.jacobian_dot(&inner_q, &inner_qdot)?;
        let r_off = self.revolute_offset();

        let mut jd = [[0.0f32; M]; 6];
        for row in 0..6 {
            for col in 0..N {
                jd[row][col + r_off] = inner_jd[row][col];
            }
        }

        Ok(jd)
    }

    fn jacobian_ddot(
        &self,
        q: &SRobotQ<M>,
        qdot: &SRobotQ<M>,
        qddot: &SRobotQ<M>,
    ) -> Result<[[f32; M]; 6], Self::Error> {
        let (_q_p, inner_q) = self.split_q(q);
        let (_qdot_p, inner_qdot) = self.split_q(qdot);
        let (_qddot_p, inner_qddot) = self.split_q(qddot);
        let inner_jdd = self
            .inner
            .jacobian_ddot(&inner_q, &inner_qdot, &inner_qddot)?;
        let r_off = self.revolute_offset();

        let mut jdd = [[0.0f32; M]; 6];
        for row in 0..6 {
            for col in 0..N {
                jdd[row][col + r_off] = inner_jdd[row][col];
            }
        }

        Ok(jdd)
    }
}
