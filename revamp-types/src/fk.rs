use glam::{Affine3A, Mat3A, Vec3A};

use crate::{RevampError, SRobotQ};

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
    type Error: Into<RevampError>;
    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], Self::Error>;
    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, Self::Error>;
}

#[inline(always)]
#[cfg(debug_assertions)]
fn check_finite<const N: usize>(q: &SRobotQ<N>) -> Result<(), RevampError> {
    if q.any_non_finite() {
        return Err(RevampError::JointsNonFinite);
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
    type Error = RevampError;
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
    type Error = RevampError;
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
    type Error = RevampError;
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
}
