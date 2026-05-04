use glam::{Affine3A, Mat3A, Vec3A};

use crate::{DekeError, SRobotQ};

use super::{AffineRaw, FKChain, check_finite, const_sin_cos};

/// Kind of URDF joint. Fixed joints have no motion; revolute and prismatic
/// joints move along `axis` (expressed in the joint's own frame, as per the
/// URDF spec).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum URDFJointType {
    Fixed,
    Revolute { axis: (f64, f64, f64) },
    Prismatic { axis: (f64, f64, f64) },
}

/// A URDF joint: its type plus the `<origin>` transform (xyz translation and
/// rpy Euler rotation) from the parent link's frame to the joint's own frame.
#[derive(Debug, Clone, Copy)]
pub struct URDFJoint {
    pub r#type: URDFJointType,
    pub xyz: (f64, f64, f64),
    pub rpy: (f64, f64, f64),
}

impl URDFJoint {
    pub const fn fixed(xyz: (f64, f64, f64), rpy: (f64, f64, f64)) -> Self {
        Self {
            r#type: URDFJointType::Fixed,
            xyz,
            rpy,
        }
    }

    pub const fn revolute(
        xyz: (f64, f64, f64),
        rpy: (f64, f64, f64),
        axis: (f64, f64, f64),
    ) -> Self {
        Self {
            r#type: URDFJointType::Revolute { axis },
            xyz,
            rpy,
        }
    }

    pub const fn prismatic(
        xyz: (f64, f64, f64),
        rpy: (f64, f64, f64),
        axis: (f64, f64, f64),
    ) -> Self {
        Self {
            r#type: URDFJointType::Prismatic { axis },
            xyz,
            rpy,
        }
    }

    /// Build the `Affine3A` corresponding to this joint's `<origin>`, using
    /// the URDF RPY convention `R = Rz(yaw) · Ry(pitch) · Rx(roll)`.
    pub const fn origin_affine(&self) -> Affine3A {
        AffineRaw::from_xyz_rpy(self.xyz, self.rpy).to_affine3a()
    }
}

/// Const-friendly error type for the `URDFChain` / `URDFJoint` const
/// constructors. Trivially `Copy`, so values can be matched/returned inside
/// `const fn`s (unlike [`DekeError`], whose `RetimerFailed(String)` variant
/// carries a non-const destructor). Converts into [`DekeError`] via `From`.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum URDFBuildError {
    #[error(
        "URDF joint at index {index} has an unexpected type: expected {expected}, found {found}"
    )]
    JointTypeMismatch {
        index: usize,
        expected: &'static str,
        found: &'static str,
    },
    #[error("URDFChain<{expected}> requires {expected} revolute joints, found {found}")]
    RevoluteCountMismatch { expected: usize, found: usize },
}

impl From<URDFBuildError> for DekeError {
    fn from(e: URDFBuildError) -> Self {
        match e {
            URDFBuildError::JointTypeMismatch {
                index,
                expected,
                found,
            } => DekeError::URDFJointTypeMismatch {
                index,
                expected,
                found,
            },
            URDFBuildError::RevoluteCountMismatch { expected, found } => {
                DekeError::URDFRevoluteCountMismatch { expected, found }
            }
        }
    }
}

const fn joint_kind_name(k: URDFJointType) -> &'static str {
    match k {
        URDFJointType::Fixed => "Fixed",
        URDFJointType::Revolute { .. } => "Revolute",
        URDFJointType::Prismatic { .. } => "Prismatic",
    }
}

const fn compose_fixed_joints_raw(joints: &[URDFJoint]) -> Result<AffineRaw, URDFBuildError> {
    let mut acc = AffineRaw::IDENTITY;
    let n = joints.len();
    let mut i = 0;
    while i < n {
        let j = &joints[i];
        if !matches!(j.r#type, URDFJointType::Fixed) {
            return Err(URDFBuildError::JointTypeMismatch {
                index: i,
                expected: "Fixed",
                found: joint_kind_name(j.r#type),
            });
        }
        acc = acc.mul(AffineRaw::from_xyz_rpy(j.xyz, j.rpy));
        i += 1;
    }
    Ok(acc)
}

/// Compose the `<origin>` transforms of a sequence of fixed joints
/// (parent→child order) into a single `Affine3A`. Returns an error if any
/// joint in `joints` is not `Fixed`.
pub const fn compose_fixed_joints(joints: &[URDFJoint]) -> Result<Affine3A, URDFBuildError> {
    match compose_fixed_joints_raw(joints) {
        Ok(a) => Ok(a.to_affine3a()),
        Err(e) => Err(e),
    }
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
    prefix_c0: Vec3A,
    prefix_c1: Vec3A,
    prefix_c2: Vec3A,
    prefix_t: Vec3A,
    prefix_identity: bool,
    suffix_c0: Vec3A,
    suffix_c1: Vec3A,
    suffix_c2: Vec3A,
    suffix_t: Vec3A,
    suffix_identity: bool,
}

impl<const N: usize> URDFChain<N> {
    /// Build a chain from exactly `N` actuated (revolute) joints. Returns
    /// [`URDFBuildError::JointTypeMismatch`] if any entry is `Fixed` or
    /// `Prismatic`. For a slice that mixes fixed joints in, use
    /// [`URDFChain::from_urdf`] instead.
    pub const fn new(joints: [URDFJoint; N]) -> Result<Self, URDFBuildError> {
        let mut fr_c0 = [Vec3A::X; N];
        let mut fr_c1 = [Vec3A::Y; N];
        let mut fr_c2 = [Vec3A::Z; N];
        let mut fr_identity = [true; N];
        let mut fixed_trans = [Vec3A::ZERO; N];
        let mut axis = [JointAxis::Z; N];

        let mut i = 0;
        while i < N {
            let (ox, oy, oz) = joints[i].xyz;
            let (roll, pitch, yaw) = joints[i].rpy;

            let is_identity = roll.abs() < 1e-10 && pitch.abs() < 1e-10 && yaw.abs() < 1e-10;
            fr_identity[i] = is_identity;

            if !is_identity {
                let (sr, cr) = const_sin_cos(roll as f32);
                let (sp, cp) = const_sin_cos(pitch as f32);
                let (sy, cy) = const_sin_cos(yaw as f32);
                fr_c0[i] = Vec3A::new(cy * cp, sy * cp, -sp);
                fr_c1[i] = Vec3A::new(cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr);
                fr_c2[i] = Vec3A::new(cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr);
            }

            fixed_trans[i] = Vec3A::new(ox as f32, oy as f32, oz as f32);

            let (ax, ay, az) = match joints[i].r#type {
                URDFJointType::Revolute { axis } => axis,
                _ => {
                    return Err(URDFBuildError::JointTypeMismatch {
                        index: i,
                        expected: "Revolute",
                        found: joint_kind_name(joints[i].r#type),
                    });
                }
            };
            if az.abs() > 0.5 {
                axis[i] = JointAxis::Z;
            } else if ay.abs() > 0.5 {
                axis[i] = JointAxis::Y(ay.signum() as f32);
            } else {
                axis[i] = JointAxis::X(ax.signum() as f32);
            }
            i += 1;
        }

        Ok(Self {
            fr_c0,
            fr_c1,
            fr_c2,
            fr_identity,
            fixed_trans,
            axis,
            prefix_c0: Vec3A::X,
            prefix_c1: Vec3A::Y,
            prefix_c2: Vec3A::Z,
            prefix_t: Vec3A::ZERO,
            prefix_identity: true,
            suffix_c0: Vec3A::X,
            suffix_c1: Vec3A::Y,
            suffix_c2: Vec3A::Z,
            suffix_t: Vec3A::ZERO,
            suffix_identity: true,
        })
    }

    /// Build a chain from a flat URDF joint list (any mix of `Fixed`,
    /// `Revolute`, and/or `Prismatic`). The list must describe a single
    /// branch in parent→child order.
    ///
    /// - Leading `Fixed` joints become the prefix (applied before joint 0).
    /// - Trailing `Fixed` joints become the suffix (applied after the last
    ///   actuated joint).
    /// - `Fixed` joints sandwiched between actuated joints are folded into
    ///   the origin of the next actuated joint so the kinematics are
    ///   preserved exactly.
    /// - The number of `Revolute` joints must equal `N`.
    ///
    /// Returns [`URDFBuildError::JointTypeMismatch`] if a `Prismatic` joint
    /// appears (not handled by `URDFChain` itself — wrap the result in
    /// [`PrismaticFK`](crate::PrismaticFK) for a prismatic joint at the start
    /// or end), or [`URDFBuildError::RevoluteCountMismatch`] if the revolute
    /// count doesn't match `N`.
    pub const fn from_urdf(joints: &[URDFJoint]) -> Result<Self, URDFBuildError> {
        let mut fr_c0 = [Vec3A::X; N];
        let mut fr_c1 = [Vec3A::Y; N];
        let mut fr_c2 = [Vec3A::Z; N];
        let mut fr_identity = [true; N];
        let mut fixed_trans = [Vec3A::ZERO; N];
        let mut axis_out = [JointAxis::Z; N];

        let mut pending = AffineRaw::IDENTITY;
        let mut prefix = AffineRaw::IDENTITY;
        let mut prefix_set = false;
        let mut r_count = 0usize;

        let n = joints.len();
        let mut i = 0;
        while i < n {
            let joint = &joints[i];
            match joint.r#type {
                URDFJointType::Fixed => {
                    pending = pending.mul(AffineRaw::from_xyz_rpy(joint.xyz, joint.rpy));
                }
                URDFJointType::Revolute { axis } => {
                    if r_count >= N {
                        return Err(URDFBuildError::RevoluteCountMismatch {
                            expected: N,
                            found: r_count + 1,
                        });
                    }
                    let local = AffineRaw::from_xyz_rpy(joint.xyz, joint.rpy);
                    let effective = if !prefix_set {
                        prefix = pending;
                        prefix_set = true;
                        local
                    } else {
                        pending.mul(local)
                    };

                    fr_identity[r_count] = effective.is_identity();
                    fr_c0[r_count] = effective.c0_vec3a();
                    fr_c1[r_count] = effective.c1_vec3a();
                    fr_c2[r_count] = effective.c2_vec3a();
                    fixed_trans[r_count] = effective.t_vec3a();

                    let (ax, ay, az) = axis;
                    axis_out[r_count] = if az.abs() > 0.5 {
                        JointAxis::Z
                    } else if ay.abs() > 0.5 {
                        JointAxis::Y(ay.signum() as f32)
                    } else {
                        JointAxis::X(ax.signum() as f32)
                    };

                    pending = AffineRaw::IDENTITY;
                    r_count += 1;
                }
                URDFJointType::Prismatic { .. } => {
                    return Err(URDFBuildError::JointTypeMismatch {
                        index: i,
                        expected: "Fixed or Revolute",
                        found: "Prismatic",
                    });
                }
            }
            i += 1;
        }
        if r_count != N {
            return Err(URDFBuildError::RevoluteCountMismatch {
                expected: N,
                found: r_count,
            });
        }

        let prefix_identity = !prefix_set || prefix.is_identity();
        let suffix_identity = pending.is_identity();

        Ok(Self {
            fr_c0,
            fr_c1,
            fr_c2,
            fr_identity,
            fixed_trans,
            axis: axis_out,
            prefix_c0: prefix.c0_vec3a(),
            prefix_c1: prefix.c1_vec3a(),
            prefix_c2: prefix.c2_vec3a(),
            prefix_t: prefix.t_vec3a(),
            prefix_identity,
            suffix_c0: pending.c0_vec3a(),
            suffix_c1: pending.c1_vec3a(),
            suffix_c2: pending.c2_vec3a(),
            suffix_t: pending.t_vec3a(),
            suffix_identity,
        })
    }

    /// Bake a sequence of URDF fixed-joint origins (parent→child order) into
    /// the base side of the chain. The composed transform is applied before
    /// joint 0, so every joint frame returned by [`FKChain::fk`] and every
    /// position returned by [`FKChain::joint_axes_positions`] reflects the
    /// fixed prefix.
    ///
    /// Each joint in `joints` must be `URDFJointType::Fixed`. An empty slice
    /// clears any previously set prefix. Returns
    /// [`DekeError::URDFJointTypeMismatch`] if any joint is non-Fixed.
    pub const fn with_fixed_prefix(
        mut self,
        joints: &[URDFJoint],
    ) -> Result<Self, URDFBuildError> {
        if joints.is_empty() {
            self.prefix_c0 = Vec3A::X;
            self.prefix_c1 = Vec3A::Y;
            self.prefix_c2 = Vec3A::Z;
            self.prefix_t = Vec3A::ZERO;
            self.prefix_identity = true;
        } else {
            let a = match compose_fixed_joints_raw(joints) {
                Ok(a) => a,
                Err(e) => return Err(e),
            };
            self.prefix_identity = a.is_identity();
            self.prefix_c0 = a.c0_vec3a();
            self.prefix_c1 = a.c1_vec3a();
            self.prefix_c2 = a.c2_vec3a();
            self.prefix_t = a.t_vec3a();
        }
        Ok(self)
    }

    /// Bake a sequence of URDF fixed-joint origins (parent→child order) into
    /// the tool side of the chain. The composed transform is applied after
    /// the last actuated joint, so the final frame of [`FKChain::fk`], the
    /// result of [`FKChain::fk_end`], and the `p_ee` returned by
    /// [`FKChain::joint_axes_positions`] all include the fixed suffix.
    ///
    /// Joint pivot positions and axes (`positions[0..N]`, `axes[0..N]`)
    /// remain at the actuated joint origins — they are not shifted by the
    /// suffix.
    ///
    /// Each joint in `joints` must be `URDFJointType::Fixed`. An empty slice
    /// clears any previously set suffix. Returns
    /// [`DekeError::URDFJointTypeMismatch`] if any joint is non-Fixed.
    pub const fn with_fixed_suffix(
        mut self,
        joints: &[URDFJoint],
    ) -> Result<Self, URDFBuildError> {
        if joints.is_empty() {
            self.suffix_c0 = Vec3A::X;
            self.suffix_c1 = Vec3A::Y;
            self.suffix_c2 = Vec3A::Z;
            self.suffix_t = Vec3A::ZERO;
            self.suffix_identity = true;
        } else {
            let a = match compose_fixed_joints_raw(joints) {
                Ok(a) => a,
                Err(e) => return Err(e),
            };
            self.suffix_identity = a.is_identity();
            self.suffix_c0 = a.c0_vec3a();
            self.suffix_c1 = a.c1_vec3a();
            self.suffix_c2 = a.c2_vec3a();
            self.suffix_t = a.t_vec3a();
        }
        Ok(self)
    }

    /// Convenience: set both a fixed-joint prefix and suffix in one call.
    pub const fn with_fixed_joints(
        self,
        prefix: &[URDFJoint],
        suffix: &[URDFJoint],
    ) -> Result<Self, URDFBuildError> {
        match self.with_fixed_prefix(prefix) {
            Ok(s) => s.with_fixed_suffix(suffix),
            Err(e) => Err(e),
        }
    }

    #[inline(always)]
    fn initial_frame(&self) -> (Vec3A, Vec3A, Vec3A, Vec3A) {
        if self.prefix_identity {
            (Vec3A::X, Vec3A::Y, Vec3A::Z, Vec3A::ZERO)
        } else {
            (self.prefix_c0, self.prefix_c1, self.prefix_c2, self.prefix_t)
        }
    }

    #[inline(always)]
    fn apply_suffix(
        &self,
        c0: &mut Vec3A,
        c1: &mut Vec3A,
        c2: &mut Vec3A,
        t: &mut Vec3A,
    ) {
        let st = self.suffix_t;
        *t = st.x * *c0 + st.y * *c1 + st.z * *c2 + *t;

        let fc0 = self.suffix_c0;
        let fc1 = self.suffix_c1;
        let fc2 = self.suffix_c2;
        let new_c0 = fc0.x * *c0 + fc0.y * *c1 + fc0.z * *c2;
        let new_c1 = fc1.x * *c0 + fc1.y * *c1 + fc1.z * *c2;
        let new_c2 = fc2.x * *c0 + fc2.y * *c1 + fc2.z * *c2;
        *c0 = new_c0;
        *c1 = new_c1;
        *c2 = new_c2;
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

    fn base_tf(&self) -> Affine3A {
        if self.prefix_identity {
            Affine3A::IDENTITY
        } else {
            Affine3A {
                matrix3: Mat3A::from_cols(self.prefix_c0, self.prefix_c1, self.prefix_c2),
                translation: self.prefix_t,
            }
        }
    }

    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], Self::Error> {
        check_finite(q)?;
        let mut out = [Affine3A::IDENTITY; N];
        let (mut c0, mut c1, mut c2, mut t) = self.initial_frame();

        let mut i = 0;
        while i < N {
            let (st, ct) = q.0[i].sin_cos();
            self.accumulate_joint(i, st, ct, &mut c0, &mut c1, &mut c2, &mut t);

            out[i] = Affine3A {
                matrix3: Mat3A::from_cols(c0, c1, c2),
                translation: t,
            };
            i += 1;
        }

        if N > 0 && !self.suffix_identity {
            self.apply_suffix(&mut c0, &mut c1, &mut c2, &mut t);
            out[N - 1] = Affine3A {
                matrix3: Mat3A::from_cols(c0, c1, c2),
                translation: t,
            };
        }
        Ok(out)
    }

    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, Self::Error> {
        check_finite(q)?;
        let (mut c0, mut c1, mut c2, mut t) = self.initial_frame();

        let mut i = 0;
        while i < N {
            let (st, ct) = q.0[i].sin_cos();
            self.accumulate_joint(i, st, ct, &mut c0, &mut c1, &mut c2, &mut t);
            i += 1;
        }

        if !self.suffix_identity {
            self.apply_suffix(&mut c0, &mut c1, &mut c2, &mut t);
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
        check_finite(q)?;
        let mut frames = [Affine3A::IDENTITY; N];
        let (mut c0, mut c1, mut c2, mut t) = self.initial_frame();

        let mut i = 0;
        while i < N {
            let (st, ct) = q.0[i].sin_cos();
            self.accumulate_joint(i, st, ct, &mut c0, &mut c1, &mut c2, &mut t);
            frames[i] = Affine3A {
                matrix3: Mat3A::from_cols(c0, c1, c2),
                translation: t,
            };
            i += 1;
        }

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

        let p_ee = if N == 0 {
            Vec3A::ZERO
        } else if !self.suffix_identity {
            self.apply_suffix(&mut c0, &mut c1, &mut c2, &mut t);
            t
        } else {
            frames[N - 1].translation
        };

        Ok((axes, positions, p_ee))
    }
}
