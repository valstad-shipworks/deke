//! [`Kinematics`] — the single forward-kinematics chain. Every constructor lowers
//! its input (DH / Hayati-Paul / URDF parameters, or a [`KinSpec`] directly)
//! into one common structure-of-arrays representation, and per-joint dispatch
//! picks the cheapest exact motion for each joint:
//!
//! - revolute about `+Z` / `±X` / `±Y` → a single 2D column rotation,
//! - revolute about an arbitrary axis → a full Rodrigues rotation,
//! - prismatic → a translation along the axis,
//!
//! and the fixed `parent_to_joint` rotation is skipped entirely when it is the
//! identity (the common URDF case). The exact [`KinSpec`] is recovered by
//! [`ContinuousFKChain::structure`], which powers the geometric Jacobians.

use deke_types::{DekeError, SRobotQ};
use glam_traits_ext::{FloatAffine, FloatMat, FloatVec, TAffine3, TMat3, TVec3};

use super::scalar_from_f64;
use super::{
    AAffine3, AMat3, AVec3, ContinuousFKChain, DHJoint, FKChain, HPJoint, JointSpec, KinScalar,
    KinSpec, URDFBuildError, URDFJoint, URDFJointType, check_finite,
};
use crate::IkRules;

/// Per-joint position limits required by every [`Kinematics`] constructor.
/// `lower[i] <= q[i] <= upper[i]` bounds joint `i`; IK output is filtered to
/// these ranges and [`IkRules::DiscreteAxis`] sweeps them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointLimits<const N: usize, F: KinScalar = f32> {
    pub lower: SRobotQ<N, F>,
    pub upper: SRobotQ<N, F>,
}

impl<const N: usize, F: KinScalar> JointLimits<N, F> {
    pub fn new(lower: SRobotQ<N, F>, upper: SRobotQ<N, F>) -> Self {
        Self { lower, upper }
    }

    /// Symmetric limits `[-bound, bound]` on every joint.
    pub fn symmetric(bound: F) -> Self {
        Self {
            lower: SRobotQ::from_array([-bound; N]),
            upper: SRobotQ::from_array([bound; N]),
        }
    }
}

#[inline]
fn scalar_to_f64<F: KinScalar>(x: F) -> f64 {
    num_traits::ToPrimitive::to_f64(&x).expect("scalar is representable as f64")
}

/// Per-joint motion, selected at construction from the joint's axis.
#[derive(Debug, Clone, Copy)]
enum JointKind<F: KinScalar> {
    RevoluteZ,
    RevoluteY(F),
    RevoluteX(F),
    RevoluteAxis(AVec3<F>),
    PrismaticAxis(AVec3<F>),
}

/// Forward-kinematics chain. Joints are stored as a structure-of-arrays: the
/// fixed `parent_to_joint` rotation columns + translation, an identity flag to
/// skip the rotation when possible, and the per-joint [`JointKind`].
///
/// `base_tf` / `ee_tf` are the world-base and tool transforms applied around
/// the joint chain (`base_tf · joints · intrinsic_ee · ee_tf`). `intrinsic_ee`
/// is a tool offset baked into the chain's own geometry (the trailing
/// `Tz·Tx·Rx` of a DH/HP chain) — it is preserved across
/// [`Kinematics::clone_with_ee_tf`], which only replaces the user `ee_tf`.
#[derive(Debug, Clone)]
pub struct Kinematics<const N: usize, F: KinScalar = f32> {
    fr_c0: [AVec3<F>; N],
    fr_c1: [AVec3<F>; N],
    fr_c2: [AVec3<F>; N],
    fr_identity: [bool; N],
    fixed_trans: [AVec3<F>; N],
    kind: [JointKind<F>; N],
    joints: [(AAffine3<F>, JointSpec<F>); N],
    intrinsic_ee: AAffine3<F>,
    base_tf: AAffine3<F>,
    ee_tf: AAffine3<F>,
    intrinsic_ee_id: bool,
    base_id: bool,
    ee_id: bool,
    /// IK strategy resolved eagerly at construction, shared across clones.
    ik: std::sync::Arc<crate::ik::IkResolved<N>>,
}

impl<const N: usize, F: KinScalar> Kinematics<N, F> {
    /// Build from standard Denavit-Hartenberg joints.
    ///
    /// `limits` are the per-joint `(lower, upper)` ranges (required), and `rules`
    /// constrain over-actuated chains so IK is solvable (see [`IkRules`]).
    pub fn from_dh(
        joints: [DHJoint<F>; N],
        limits: JointLimits<N, F>,
        rules: &[IkRules<f64>],
    ) -> Self {
        let (core, intrinsic_ee) =
            factor_offsets(&joints, |j| (dh_offset(j.a, j.alpha, j.d), j.theta_offset));
        Self::assemble(
            core,
            AAffine3::<F>::IDENTITY,
            AAffine3::<F>::IDENTITY,
            intrinsic_ee,
            limits,
            rules,
        )
    }

    /// Build from Hayati-Paul joints.
    pub fn from_hp(
        joints: [HPJoint<F>; N],
        limits: JointLimits<N, F>,
        rules: &[IkRules<f64>],
    ) -> Self {
        let (core, intrinsic_ee) = factor_offsets(&joints, |j| {
            (hp_offset(j.a, j.alpha, j.beta, j.d), j.theta_offset)
        });
        Self::assemble(
            core,
            AAffine3::<F>::IDENTITY,
            AAffine3::<F>::IDENTITY,
            intrinsic_ee,
            limits,
            rules,
        )
    }

    /// Build from a flat URDF joint list (parent→child order, any mix of
    /// `Fixed`/`Revolute`/`Prismatic`). Fixed joints between actuated joints
    /// are folded into the following actuated joint's origin (kinematics are
    /// preserved exactly); fixed joints before all actuated joints compose into
    /// the base transform, and those after all actuated joints into the
    /// end-effector transform.
    ///
    /// The number of actuated (revolute + prismatic) joints must equal `N`.
    pub fn from_urdf(
        joints: &[URDFJoint],
        limits: JointLimits<N, F>,
        rules: &[IkRules<f64>],
    ) -> Result<Self, URDFBuildError> {
        let mut pending = AAffine3::<F>::IDENTITY;
        let mut base = AAffine3::<F>::IDENTITY;
        let mut prefix_set = false;
        let mut acc: Vec<(AAffine3<F>, JointSpec<F>)> = Vec::with_capacity(N);

        for j in joints {
            let spec = match j.r#type {
                URDFJointType::Fixed => {
                    pending *= urdf_origin::<F>(j.xyz, j.rpy);
                    continue;
                }
                URDFJointType::Revolute { axis } => JointSpec::Revolute {
                    axis_local: axis_vec::<F>(axis),
                },
                URDFJointType::Prismatic { axis } => JointSpec::Prismatic {
                    axis_local: axis_vec::<F>(axis),
                },
            };
            let local = urdf_origin::<F>(j.xyz, j.rpy);
            let origin = if prefix_set {
                pending * local
            } else {
                base = pending;
                prefix_set = true;
                local
            };
            pending = AAffine3::<F>::IDENTITY;
            acc.push((origin, spec));
        }

        if acc.len() != N {
            return Err(URDFBuildError::RevoluteCountMismatch {
                expected: N,
                found: acc.len(),
            });
        }

        let joints: [(AAffine3<F>, JointSpec<F>); N] = match acc.try_into() {
            Ok(a) => a,
            Err(_) => unreachable!("length checked above"),
        };
        // trailing fixed joints (`pending`) become the tool transform.
        Ok(Self::assemble(
            joints,
            base,
            pending,
            AAffine3::<F>::IDENTITY,
            limits,
            rules,
        ))
    }

    /// Build directly from a [`KinSpec`].
    pub fn from_kinspec(
        spec: KinSpec<F, N>,
        limits: JointLimits<N, F>,
        rules: &[IkRules<f64>],
    ) -> Self {
        Self::assemble(
            spec.joints,
            spec.base_to_first,
            spec.end_to_ee,
            AAffine3::<F>::IDENTITY,
            limits,
            rules,
        )
    }

    /// Clone the chain with a replaced world-base transform.
    pub fn clone_with_base_tf(&self, base_tf: AAffine3<F>) -> Self {
        let mut c = self.clone();
        c.base_id = affine_is_identity::<F>(&base_tf);
        c.base_tf = base_tf;
        c
    }

    /// Clone the chain with a replaced end-effector (tool) transform. Any
    /// tool offset intrinsic to the chain geometry is preserved.
    pub fn clone_with_ee_tf(&self, ee_tf: AAffine3<F>) -> Self {
        let mut c = self.clone();
        c.ee_id = affine_is_identity::<F>(&ee_tf);
        c.ee_tf = ee_tf;
        c
    }

    fn assemble(
        joints: [(AAffine3<F>, JointSpec<F>); N],
        base_tf: AAffine3<F>,
        ee_tf: AAffine3<F>,
        intrinsic_ee: AAffine3<F>,
        limits: JointLimits<N, F>,
        rules: &[IkRules<f64>],
    ) -> Self {
        let mut fr_c0 = [AVec3::<F>::X; N];
        let mut fr_c1 = [AVec3::<F>::Y; N];
        let mut fr_c2 = [AVec3::<F>::Z; N];
        let mut fr_identity = [true; N];
        let mut fixed_trans = [AVec3::<F>::ZERO; N];
        let mut kind = [JointKind::RevoluteZ; N];

        for i in 0..N {
            let (origin, js) = joints[i];
            let m = origin.matrix3();
            let c0 = m.col(0);
            let c1 = m.col(1);
            let c2 = m.col(2);
            fr_c0[i] = c0;
            fr_c1[i] = c1;
            fr_c2[i] = c2;
            fixed_trans[i] = origin.translation();
            fr_identity[i] = cols_identity::<F>(c0, c1, c2);
            kind[i] = classify(&js);
        }

        let intrinsic_ee_id = affine_is_identity::<F>(&intrinsic_ee);
        let ee_id = affine_is_identity::<F>(&ee_tf);

        // Equivalent KinSpec (mirrors `structure()`), used to resolve the IK
        // strategy eagerly at construction.
        let end_to_ee = if ee_id {
            intrinsic_ee
        } else if intrinsic_ee_id {
            ee_tf
        } else {
            intrinsic_ee * ee_tf
        };
        let spec = KinSpec {
            base_to_first: base_tf,
            joints,
            end_to_ee,
        };
        let lim = crate::ik::Limits {
            lower: std::array::from_fn(|i| scalar_to_f64::<F>(limits.lower.0[i])),
            upper: std::array::from_fn(|i| scalar_to_f64::<F>(limits.upper.0[i])),
        };
        let ik = crate::ik::resolve_ik(&spec, lim, rules);

        Self {
            fr_c0,
            fr_c1,
            fr_c2,
            fr_identity,
            fixed_trans,
            kind,
            joints,
            intrinsic_ee,
            base_tf,
            ee_tf,
            intrinsic_ee_id,
            base_id: affine_is_identity::<F>(&base_tf),
            ee_id,
            ik,
        }
    }

    /// The IK strategy resolved for this chain at construction.
    pub fn ik_diagnostic(&self) -> &crate::ik::IkSolverDiagnostic {
        self.ik.diagnostic()
    }

    /// Internal accessor for the resolved IK engine.
    pub(crate) fn ik_resolved(&self) -> &crate::ik::IkResolved<N> {
        &self.ik
    }

    /// Rebuild this chain at `f64` precision, preserving geometry and joint
    /// limits. Useful when a chain is authored at `f32` but a consumer needs an
    /// `f64` [`FKChain`]. IK rules are *not* carried over (the result has none);
    /// re-supply them if the f64 chain needs IK.
    pub fn to_f64(&self) -> Kinematics<N, f64> {
        let spec64 = crate::ik::kinspec_to_f64(&self.structure());
        let (lower, upper) = self.ik.limits_f64();
        let limits = JointLimits::new(SRobotQ::from_array(lower), SRobotQ::from_array(upper));
        Kinematics::from_kinspec(spec64, limits, &[])
    }

    /// Accumulate joint `i`'s fixed origin and motion into the running column
    /// frame `(c0, c1, c2)` and translation `t`.
    #[inline(always)]
    fn step(
        &self,
        i: usize,
        q: F,
        c0: &mut AVec3<F>,
        c1: &mut AVec3<F>,
        c2: &mut AVec3<F>,
        t: &mut AVec3<F>,
    ) {
        let ft = self.fixed_trans[i];
        *t = *c0 * ft.x() + *c1 * ft.y() + *c2 * ft.z() + *t;

        let (f0, f1, f2) = if self.fr_identity[i] {
            (*c0, *c1, *c2)
        } else {
            let r0 = self.fr_c0[i];
            let r1 = self.fr_c1[i];
            let r2 = self.fr_c2[i];
            (
                *c0 * r0.x() + *c1 * r0.y() + *c2 * r0.z(),
                *c0 * r1.x() + *c1 * r1.y() + *c2 * r1.z(),
                *c0 * r2.x() + *c1 * r2.y() + *c2 * r2.z(),
            )
        };

        match self.kind[i] {
            JointKind::RevoluteZ => {
                let (st, ct) = q.sin_cos();
                *c0 = f0 * ct + f1 * st;
                *c1 = f1 * ct - f0 * st;
                *c2 = f2;
            }
            JointKind::RevoluteY(s) => {
                let (st, ct) = q.sin_cos();
                let sst = s * st;
                *c0 = f0 * ct - f2 * sst;
                *c1 = f1;
                *c2 = f0 * sst + f2 * ct;
            }
            JointKind::RevoluteX(s) => {
                let (st, ct) = q.sin_cos();
                let sst = s * st;
                *c0 = f0;
                *c1 = f1 * ct + f2 * sst;
                *c2 = f2 * ct - f1 * sst;
            }
            JointKind::RevoluteAxis(axis) => {
                let rot = AAffine3::<F>::from_axis_angle(axis, q).matrix3();
                let m = AMat3::<F>::from_cols(f0, f1, f2) * rot;
                *c0 = m.col(0);
                *c1 = m.col(1);
                *c2 = m.col(2);
            }
            JointKind::PrismaticAxis(axis) => {
                let world = f0 * axis.x() + f1 * axis.y() + f2 * axis.z();
                *t = world * q + *t;
                *c0 = f0;
                *c1 = f1;
                *c2 = f2;
            }
        }
    }

    fn core_frames(&self, q: &SRobotQ<N, F>) -> Result<[AAffine3<F>; N], DekeError> {
        check_finite::<N, F>(q).map_err(lift)?;
        let mut out = [AAffine3::<F>::IDENTITY; N];
        let mut c0 = AVec3::<F>::X;
        let mut c1 = AVec3::<F>::Y;
        let mut c2 = AVec3::<F>::Z;
        let mut t = AVec3::<F>::ZERO;
        for (i, slot) in out.iter_mut().enumerate() {
            self.step(i, q.0[i], &mut c0, &mut c1, &mut c2, &mut t);
            *slot = AAffine3::<F>::from_mat3_translation(AMat3::<F>::from_cols(c0, c1, c2), t);
        }
        Ok(out)
    }

    fn core_end(&self, q: &SRobotQ<N, F>) -> Result<AAffine3<F>, DekeError> {
        check_finite::<N, F>(q).map_err(lift)?;
        let mut c0 = AVec3::<F>::X;
        let mut c1 = AVec3::<F>::Y;
        let mut c2 = AVec3::<F>::Z;
        let mut t = AVec3::<F>::ZERO;
        for i in 0..N {
            self.step(i, q.0[i], &mut c0, &mut c1, &mut c2, &mut t);
        }
        let end = AAffine3::<F>::from_mat3_translation(AMat3::<F>::from_cols(c0, c1, c2), t);
        Ok(if self.intrinsic_ee_id {
            end
        } else {
            end * self.intrinsic_ee
        })
    }
}

impl<const N: usize, F: KinScalar> FKChain<N, F> for Kinematics<N, F> {
    type Error = DekeError;

    fn base_tf(&self) -> AAffine3<F> {
        self.base_tf
    }

    fn ee_tf(&self) -> AAffine3<F> {
        self.ee_tf
    }

    fn fk(&self, q: &SRobotQ<N, F>) -> Result<[AAffine3<F>; N], Self::Error> {
        let mut frames = self.core_frames(q)?;
        if !self.base_id {
            let base = self.base_tf;
            for f in &mut frames {
                *f = base * *f;
            }
        }
        Ok(frames)
    }

    fn fk_end(&self, q: &SRobotQ<N, F>) -> Result<AAffine3<F>, Self::Error> {
        let mut end = self.core_end(q)?;
        if !self.base_id {
            end = self.base_tf * end;
        }
        if !self.ee_id {
            end *= self.ee_tf;
        }
        Ok(end)
    }
}

impl<const N: usize, F: KinScalar> ContinuousFKChain<N, F> for Kinematics<N, F> {
    fn structure(&self) -> KinSpec<F, N> {
        let end_to_ee = if self.ee_id {
            self.intrinsic_ee
        } else if self.intrinsic_ee_id {
            self.ee_tf
        } else {
            self.intrinsic_ee * self.ee_tf
        };
        KinSpec {
            base_to_first: self.base_tf,
            joints: self.joints,
            end_to_ee,
        }
    }
}

/// Lift a backend FK error into [`DekeError`]. `check_finite` returns
/// [`DekeError`] in debug builds and is infallible in release.
#[cfg(debug_assertions)]
#[inline(always)]
fn lift(e: DekeError) -> DekeError {
    e
}
#[cfg(not(debug_assertions))]
#[inline(always)]
fn lift(e: std::convert::Infallible) -> DekeError {
    match e {}
}

#[inline]
fn classify<F: KinScalar>(js: &JointSpec<F>) -> JointKind<F> {
    let eps = s::<F>(1e-6);
    let one = F::one();
    let neg = s::<F>(-1.0);
    let near = |v: F, t: F| (v - t).abs() <= eps;
    let zero = |v: F| v.abs() <= eps;
    match js {
        JointSpec::Revolute { axis_local } => {
            let a = axis_local.normalize();
            let (x, y, z) = (a.x(), a.y(), a.z());
            if near(z, one) && zero(x) && zero(y) {
                JointKind::RevoluteZ
            } else if near(x, one) && zero(y) && zero(z) {
                JointKind::RevoluteX(one)
            } else if near(x, neg) && zero(y) && zero(z) {
                JointKind::RevoluteX(neg)
            } else if near(y, one) && zero(x) && zero(z) {
                JointKind::RevoluteY(one)
            } else if near(y, neg) && zero(x) && zero(z) {
                JointKind::RevoluteY(neg)
            } else {
                JointKind::RevoluteAxis(a)
            }
        }
        JointSpec::Prismatic { axis_local } => JointKind::PrismaticAxis(axis_local.normalize()),
    }
}

/// Factor a rotate-first joint list (DH/HP) into the `(origin, axis)` joints of
/// an equivalent origin-then-rotate [`KinSpec`] plus the trailing intrinsic
/// offset.
///
/// A chain `∏ Rz(θ_i + off_i)·M_i` is rewritten as `origin_0 = Rz(off_0)`,
/// `origin_i = M_{i-1}·Rz(off_i)`, with the final `M_{N-1}` becoming the
/// intrinsic end-effector offset. `decode` returns each joint's `(M_i, off_i)`.
type CoreChain<const N: usize, F> = ([(AAffine3<F>, JointSpec<F>); N], AAffine3<F>);

fn factor_offsets<const N: usize, F: KinScalar, J: Copy>(
    joints: &[J; N],
    decode: impl Fn(J) -> (AAffine3<F>, F),
) -> CoreChain<N, F> {
    let z = AVec3::<F>::Z;
    let mut core = [(
        AAffine3::<F>::IDENTITY,
        JointSpec::Revolute { axis_local: z },
    ); N];
    let mut prev_m = AAffine3::<F>::IDENTITY;
    let mut intrinsic_ee = AAffine3::<F>::IDENTITY;

    for i in 0..N {
        let (m, off) = decode(joints[i]);
        let rz_off = AAffine3::<F>::from_axis_angle(z, off);
        core[i] = (prev_m * rz_off, JointSpec::Revolute { axis_local: z });
        prev_m = m;
        if i + 1 == N {
            intrinsic_ee = m;
        }
    }
    (core, intrinsic_ee)
}

/// `Tz(d)·Tx(a)·Rx(α)` — the fixed part of a standard DH joint.
fn dh_offset<F: KinScalar>(a: F, alpha: F, d: F) -> AAffine3<F> {
    let rx = AAffine3::<F>::from_axis_angle(AVec3::<F>::X, alpha);
    let txz = AAffine3::<F>::from_translation(AVec3::<F>::X * a + AVec3::<F>::Z * d);
    txz * rx
}

/// `Rx(α)·Ry(β)·Tx(a)·Tz(d)` — the fixed part of a Hayati-Paul joint.
fn hp_offset<F: KinScalar>(a: F, alpha: F, beta: F, d: F) -> AAffine3<F> {
    let rx = AAffine3::<F>::from_axis_angle(AVec3::<F>::X, alpha);
    let ry = AAffine3::<F>::from_axis_angle(AVec3::<F>::Y, beta);
    let txz = AAffine3::<F>::from_translation(AVec3::<F>::X * a + AVec3::<F>::Z * d);
    rx * ry * txz
}

#[inline]
fn urdf_origin<F: KinScalar>(xyz: (f64, f64, f64), rpy: (f64, f64, f64)) -> AAffine3<F> {
    let (ox, oy, oz) = xyz;
    let (roll, pitch, yaw) = rpy;
    let (sr, cr) = (roll.sin(), roll.cos());
    let (sp, cp) = (pitch.sin(), pitch.cos());
    let (sy, cy) = (yaw.sin(), yaw.cos());
    let c0 = AVec3::<F>::new(s(cy * cp), s(sy * cp), s(-sp));
    let c1 = AVec3::<F>::new(
        s(cy * sp * sr - sy * cr),
        s(sy * sp * sr + cy * cr),
        s(cp * sr),
    );
    let c2 = AVec3::<F>::new(
        s(cy * sp * cr + sy * sr),
        s(sy * sp * cr - cy * sr),
        s(cp * cr),
    );
    let t = AVec3::<F>::new(s(ox), s(oy), s(oz));
    AAffine3::<F>::from_mat3_translation(AMat3::<F>::from_cols(c0, c1, c2), t)
}

#[inline]
fn axis_vec<F: KinScalar>(axis: (f64, f64, f64)) -> AVec3<F> {
    AVec3::<F>::new(s(axis.0), s(axis.1), s(axis.2))
}

fn cols_identity<F: KinScalar>(c0: AVec3<F>, c1: AVec3<F>, c2: AVec3<F>) -> bool {
    let eps = s::<F>(1e-9);
    let one = F::one();
    (c0.x() - one).abs() <= eps
        && c0.y().abs() <= eps
        && c0.z().abs() <= eps
        && c1.x().abs() <= eps
        && (c1.y() - one).abs() <= eps
        && c1.z().abs() <= eps
        && c2.x().abs() <= eps
        && c2.y().abs() <= eps
        && (c2.z() - one).abs() <= eps
}

fn affine_is_identity<F: KinScalar>(a: &AAffine3<F>) -> bool {
    let m = a.to_cols_array();
    let eps = s::<F>(1e-9);
    const ID: [f64; 12] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    for k in 0..12 {
        if (m[k] - s::<F>(ID[k])).abs() > eps {
            return false;
        }
    }
    true
}

#[inline(always)]
fn s<F: KinScalar>(x: f64) -> F {
    scalar_from_f64(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DAffine3, DVec3};

    /// Wide symmetric limits (±100 rad) for tests that don't care about limits.
    fn wide<const M: usize>() -> JointLimits<M, f64> {
        JointLimits::symmetric(100.0)
    }

    fn puma() -> Kinematics<6, f64> {
        let pi = std::f64::consts::PI;
        let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
        let a = [0.0, 0.4318, -0.0203, 0.0, 0.0, 0.0];
        let d = [0.6718, 0.1397, 0.0, 0.4318, 0.0, 0.0565];
        let joints = std::array::from_fn(|i| DHJoint {
            a: a[i],
            alpha: alpha[i],
            d: d[i],
            theta_offset: 0.0,
        });
        Kinematics::from_dh(joints, wide(), &[])
    }

    fn q6(v: [f64; 6]) -> SRobotQ<6, f64> {
        SRobotQ::from_array(v)
    }

    /// Reference DH FK by direct affine composition, independent of `Kinematics`.
    fn dh_reference(q: [f64; 6]) -> DAffine3 {
        let pi = std::f64::consts::PI;
        let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
        let a = [0.0, 0.4318, -0.0203, 0.0, 0.0, 0.0];
        let d = [0.6718, 0.1397, 0.0, 0.4318, 0.0, 0.0565];
        let mut t = DAffine3::IDENTITY;
        for i in 0..6 {
            let rz = DAffine3::from_axis_angle(DVec3::Z, q[i]);
            let tz = DAffine3::from_translation(DVec3::Z * d[i]);
            let tx = DAffine3::from_translation(DVec3::X * a[i]);
            let rx = DAffine3::from_axis_angle(DVec3::X, alpha[i]);
            t = t * rz * tz * tx * rx;
        }
        t
    }

    #[test]
    fn from_dh_matches_reference() {
        let chain = puma();
        for cfg in [
            [0.0; 6],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [-1.0, 0.7, -0.3, 1.2, -0.9, 0.4],
        ] {
            let got = chain.fk_end(&q6(cfg)).unwrap();
            let want = dh_reference(cfg);
            let g = got.to_cols_array();
            let w = want.to_cols_array();
            for k in 0..12 {
                assert!(
                    (g[k] - w[k]).abs() < 1e-9,
                    "cfg {cfg:?} elem {k}: {} vs {}",
                    g[k],
                    w[k]
                );
            }
        }
    }

    /// The geometric Jacobian from `structure()` must match a finite-difference
    /// of `fk_end` — this is the core `ContinuousFKChain` correctness guarantee.
    #[test]
    fn jacobian_matches_finite_difference() {
        let chain = puma();
        let q0 = [0.2, -0.4, 0.7, 0.3, -0.6, 0.5];
        let j = chain.jacobian(&q6(q0)).unwrap();

        let h = 1e-6;
        let p0 = chain.fk_end(&q6(q0)).unwrap().translation;
        for i in 0..6 {
            let mut qp = q0;
            qp[i] += h;
            let pp = chain.fk_end(&q6(qp)).unwrap().translation;
            let dx = (pp.x - p0.x) / h;
            let dy = (pp.y - p0.y) / h;
            let dz = (pp.z - p0.z) / h;
            assert!((j[0][i] - dx).abs() < 1e-4, "Jx[{i}] {} vs {}", j[0][i], dx);
            assert!((j[1][i] - dy).abs() < 1e-4, "Jy[{i}] {} vs {}", j[1][i], dy);
            assert!((j[2][i] - dz).abs() < 1e-4, "Jz[{i}] {} vs {}", j[2][i], dz);
        }
    }

    /// Yoshikawa manipulability must equal `|det(J)|` for a square (6-DOF)
    /// Jacobian, since `det(J Jᵀ) = det(J)²`. The implementation forms `J Jᵀ`
    /// and takes a root, so checking it against the determinant of the raw
    /// Jacobian exercises a genuinely independent path.
    #[test]
    fn manipulability_matches_jacobian_determinant() {
        let chain = puma();
        for cfg in [
            [0.2, -0.4, 0.7, 0.3, -0.6, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ] {
            let w = chain.manipulability(&q6(cfg)).unwrap();
            let det = det6(chain.jacobian(&q6(cfg)).unwrap());
            assert!(
                w > 0.0,
                "cfg {cfg:?}: expected positive manipulability, got {w}"
            );
            assert!(
                (w - det.abs()).abs() < 1e-6,
                "cfg {cfg:?}: manip {w} vs |det J| {}",
                det.abs()
            );
        }
    }

    fn det6(j: [[f64; 6]; 6]) -> f64 {
        let mut m = j;
        let mut det = 1.0;
        for col in 0..6 {
            let mut piv = col;
            for (r, row) in m.iter().enumerate().skip(col + 1) {
                if row[col].abs() > m[piv][col].abs() {
                    piv = r;
                }
            }
            if m[piv][col].abs() < 1e-12 {
                return 0.0;
            }
            if piv != col {
                m.swap(piv, col);
                det = -det;
            }
            let pivot_row = m[col];
            det *= pivot_row[col];
            for row in m.iter_mut().skip(col + 1) {
                let f = row[col] / pivot_row[col];
                for (c, &pv) in pivot_row.iter().enumerate().skip(col) {
                    row[c] -= f * pv;
                }
            }
        }
        det
    }

    /// Routing the same robot through `from_kinspec(structure())` must preserve
    /// the end-effector pose exactly.
    #[test]
    fn kinspec_roundtrip() {
        let chain = puma();
        let rebuilt = Kinematics::from_kinspec(chain.structure(), wide(), &[]);
        for cfg in [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.4, -0.5, 0.6, 0.2, -0.3, 0.4],
        ] {
            let a = chain.fk_end(&q6(cfg)).unwrap().to_cols_array();
            let b = rebuilt.fk_end(&q6(cfg)).unwrap().to_cols_array();
            for k in 0..12 {
                assert!((a[k] - b[k]).abs() < 1e-9, "elem {k}: {} vs {}", a[k], b[k]);
            }
        }
    }

    #[test]
    fn urdf_arbitrary_and_canonical_agree() {
        // A 2R arm: revolute about +Z then about a 45° axis in the XZ plane.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let joints = [
            URDFJoint::revolute((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            URDFJoint::revolute((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), (s, 0.0, s)),
        ];
        let chain: Kinematics<2, f64> = Kinematics::from_urdf(&joints, wide(), &[]).unwrap();
        // Build the same thing as a KinSpec and confirm agreement.
        let spec = chain.structure();
        let viaspec = Kinematics::from_kinspec(spec, wide(), &[]);
        let q = SRobotQ::<2, f64>::from_array([0.6, -0.8]);
        let a = chain.fk_end(&q).unwrap().to_cols_array();
        let b = viaspec.fk_end(&q).unwrap().to_cols_array();
        for k in 0..12 {
            assert!((a[k] - b[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn urdf_fixed_joints_become_base_and_ee() {
        let joints = [
            URDFJoint::fixed((0.0, 0.0, 0.1), (0.0, 0.0, 0.0)), // leading -> base
            URDFJoint::revolute((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            URDFJoint::fixed((0.2, 0.0, 0.0), (0.0, 0.0, 0.0)), // trailing -> ee
        ];
        let chain: Kinematics<1, f64> = Kinematics::from_urdf(&joints, wide(), &[]).unwrap();
        let spec = chain.structure();
        assert!((spec.base_to_first.translation.z - 0.1).abs() < 1e-12);
        assert!((spec.end_to_ee.translation.x - 0.2).abs() < 1e-12);

        // At q=0 the tip sits at base(0,0,0.1) + ee(0.2,0,0) = (0.2, 0, 0.1).
        let end = chain.fk_end(&SRobotQ::<1, f64>::from_array([0.0])).unwrap();
        let t = end.translation;
        assert!((t.x - 0.2).abs() < 1e-12 && t.y.abs() < 1e-12 && (t.z - 0.1).abs() < 1e-12);
    }

    #[test]
    fn clone_with_base_and_ee() {
        let chain = puma();
        let base = DAffine3::from_translation(DVec3::new(1.0, 2.0, 3.0));
        let ee = DAffine3::from_translation(DVec3::new(0.0, 0.0, 0.1));
        let moved = chain.clone_with_base_tf(base).clone_with_ee_tf(ee);

        let q = q6([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let want = base * chain.fk_end(&q).unwrap() * ee;
        let got = moved.fk_end(&q).unwrap();
        let g = got.to_cols_array();
        let w = want.to_cols_array();
        for k in 0..12 {
            assert!((g[k] - w[k]).abs() < 1e-9, "elem {k}");
        }
    }

    /// A spherical-wrist chain resolves to the analytic strategy, and
    /// `Kinematics::ik` round-trips through the unified API.
    #[test]
    fn ik_analytic_strategy_and_roundtrip() {
        use crate::IkStrategy;
        use deke_types::IkSolver;

        let chain = puma();
        let diag = chain.ik_diagnostic();
        assert!(diag.viable);
        assert!(
            matches!(diag.strategy, IkStrategy::Analytic { .. }),
            "got {:?}",
            diag.strategy
        );
        assert_eq!(diag.effective_dof, 6);
        assert!(diag.family().unwrap().contains("6R"));

        let q = q6([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let target = chain.fk_end(&q).unwrap();
        let sols = chain.ik(target).unwrap().unwrap();
        assert!(!sols.is_empty());
        let want = target.to_cols_array();
        let matched = sols.iter().any(|s| {
            let got = chain.fk_end(s).unwrap().to_cols_array();
            want.iter()
                .zip(got.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6)
        });
        assert!(matched, "no analytic IK solution reproduced the pose");
    }

    /// A generic 6R chain (no closed form) resolves to the eigenvalue fallback
    /// and still inverts through `Kinematics::ik`.
    #[test]
    fn ik_generic_fallback_strategy_and_roundtrip() {
        use crate::IkStrategy;
        use deke_types::IkSolver;

        // Arbitrary non-DH, non-Z axes → no recognised analytic class.
        let axes = [
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(0.0, 1.0, 0.3).normalize(),
            DVec3::new(0.2, 1.0, 0.0).normalize(),
            DVec3::new(1.0, 0.2, 0.4).normalize(),
            DVec3::new(0.0, 1.0, 0.5).normalize(),
            DVec3::new(0.3, 0.2, 1.0).normalize(),
        ];
        let offs = [
            DVec3::new(0.0, 0.0, 0.30),
            DVec3::new(0.10, 0.02, 0.05),
            DVec3::new(0.30, 0.0, 0.04),
            DVec3::new(0.0, 0.05, 0.28),
            DVec3::new(0.06, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.07),
        ];
        let joints: [URDFJoint; 6] = std::array::from_fn(|i| {
            URDFJoint::revolute(
                (offs[i].x, offs[i].y, offs[i].z),
                (0.0, 0.0, 0.0),
                (axes[i].x, axes[i].y, axes[i].z),
            )
        });
        let chain: Kinematics<6, f64> = Kinematics::from_urdf(&joints, wide(), &[]).unwrap();
        let diag = chain.ik_diagnostic();
        assert_eq!(
            diag.strategy,
            IkStrategy::Generic6R,
            "got {:?}",
            diag.strategy
        );
        assert!(diag.viable);

        let q = q6([0.4, -0.8, 1.0, -0.5, 0.9, -0.3]);
        let target = chain.fk_end(&q).unwrap();
        let sols = chain.ik(target).unwrap().unwrap();
        assert!(!sols.is_empty(), "generic fallback found no solutions");
        let want = target.to_cols_array();
        let matched = sols.iter().any(|s| {
            let got = chain.fk_end(s).unwrap().to_cols_array();
            want.iter()
                .zip(got.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6)
        });
        assert!(matched, "no generic IK solution reproduced the pose");
    }

    /// A chain with a prismatic joint is fundamentally not IK-viable: the
    /// diagnostic flags it and every `ik` call returns `Err(IkNotViable)`.
    #[test]
    fn ik_not_viable_for_prismatic_chain() {
        use crate::IkStrategy;
        use deke_types::DekeError;
        use deke_types::IkSolver;

        let joints = [
            URDFJoint::revolute((0.0, 0.0, 0.1), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            URDFJoint::prismatic((0.0, 0.0, 0.2), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ];
        let chain: Kinematics<2, f64> = Kinematics::from_urdf(&joints, wide(), &[]).unwrap();
        let diag = chain.ik_diagnostic();
        assert!(!diag.viable);
        assert_eq!(diag.strategy, IkStrategy::None);

        // FK still works on a non-IK-viable chain.
        let _ = chain
            .fk_end(&SRobotQ::<2, f64>::from_array([0.1, 0.2]))
            .unwrap();

        let target = AAffine3::<f64>::IDENTITY;
        match chain.ik(target) {
            Err(DekeError::IkNotViable(_)) => {}
            Err(e) => panic!("expected IkNotViable, got {e:?}"),
            Ok(_) => panic!("expected IkNotViable error, got Ok"),
        }
    }

    fn panda7() -> [DHJoint<f64>; 7] {
        let pi = std::f64::consts::PI;
        let alpha = [
            pi / 2.0,
            -pi / 2.0,
            -pi / 2.0,
            pi / 2.0,
            -pi / 2.0,
            pi / 2.0,
            0.0,
        ];
        let a = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088];
        let d = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107];
        std::array::from_fn(|i| DHJoint {
            a: a[i],
            alpha: alpha[i],
            d: d[i],
            theta_offset: 0.0,
        })
    }

    /// A 7-DOF chain is not IK-viable on its own (7 free revolute joints > 6),
    /// but a single FixedAxis rule on joint 6 reduces it to a solvable 6R.
    #[test]
    fn fixed_axis_makes_7dof_viable() {
        use crate::IkRules;
        use deke_types::IkSolver;

        // No rules: 7 free DOF → not viable, but still constructs (FK works).
        let bare: Kinematics<7, f64> = Kinematics::from_dh(panda7(), wide(), &[]);
        assert!(!bare.ik_diagnostic().viable);
        let _ = bare
            .fk_end(&SRobotQ::<7, f64>::from_array([0.0; 7]))
            .unwrap();

        // FixedAxis on joint 6 → 6 free DOF, viable.
        let rules = [IkRules::FixedAxis { idx: 6, pos: 0.0 }];
        let chain: Kinematics<7, f64> = Kinematics::from_dh(panda7(), wide(), &rules);
        let diag = chain.ik_diagnostic();
        assert!(diag.viable, "reason: {}", diag.reason);
        assert_eq!(diag.effective_dof, 6);

        let q = SRobotQ::<7, f64>::from_array([0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.0]);
        let target = chain.fk_end(&q).unwrap();
        let sols = chain.ik(target).unwrap().unwrap();
        assert!(
            !sols.is_empty(),
            "FixedAxis-reduced chain found no solutions"
        );
        for s in &sols {
            // The fixed joint holds its value, and FK reproduces the pose.
            assert!((s.0[6] - 0.0).abs() < 1e-9, "joint 6 not held: {}", s.0[6]);
            let got = chain.fk_end(s).unwrap().to_cols_array();
            let want = target.to_cols_array();
            assert!(
                want.iter()
                    .zip(got.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-6),
                "solution does not reproduce pose"
            );
        }
    }

    /// A prismatic rail prefixed to a 6R arm (7 DOF, one linear) becomes solvable
    /// with a DiscreteAxis rule sweeping the rail; the swept axis is folded out
    /// at each sample, leaving a 6R solve.
    #[test]
    fn discrete_linear_axis_rail_plus_arm() {
        use crate::{IkRules, IkStrategy};
        use deke_types::IkSolver;

        // Build a correct Puma 6R as a KinSpec, then prepend a prismatic rail
        // (joint 0, along +X) to form a 7-DOF rail+arm KinSpec.
        let pi = std::f64::consts::PI;
        let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
        let a = [0.0, 0.4318, -0.0203, 0.0, 0.0, 0.0];
        let d = [0.6718, 0.1397, 0.0, 0.4318, 0.0, 0.0565];
        let puma: Kinematics<6, f64> = Kinematics::from_dh(
            std::array::from_fn(|i| DHJoint {
                a: a[i],
                alpha: alpha[i],
                d: d[i],
                theta_offset: 0.0,
            }),
            wide(),
            &[],
        );
        let pspec = puma.structure();
        let joints: [(DAffine3, JointSpec<f64>); 7] = std::array::from_fn(|i| {
            if i == 0 {
                (
                    DAffine3::IDENTITY,
                    JointSpec::Prismatic {
                        axis_local: DVec3::X,
                    },
                )
            } else {
                pspec.joints[i - 1]
            }
        });
        let spec = KinSpec::new(pspec.base_to_first, joints, pspec.end_to_ee);

        let lower = SRobotQ::<7, f64>::from_array([0.0, -pi, -pi, -pi, -pi, -pi, -pi]);
        let upper = SRobotQ::<7, f64>::from_array([0.5, pi, pi, pi, pi, pi, pi]);
        let limits = JointLimits::new(lower, upper);
        let rules = [IkRules::DiscreteAxis {
            idx: 0,
            step_size: 0.1,
        }];
        let chain: Kinematics<7, f64> = Kinematics::from_kinspec(spec, limits, &rules);

        let diag = chain.ik_diagnostic();
        assert!(diag.viable, "reason: {}", diag.reason);
        assert!(
            matches!(diag.strategy, IkStrategy::Ruled { discrete: 1, .. }),
            "got {:?}",
            diag.strategy
        );

        // Plant a config with the rail at a grid value (0.2) so a sample hits it.
        let q = SRobotQ::<7, f64>::from_array([0.2, 0.1, -0.5, 0.6, 0.2, -0.3, 0.4]);
        let target = chain.fk_end(&q).unwrap();
        let sols = chain.ik(target).unwrap().unwrap();
        assert!(!sols.is_empty(), "rail+arm found no solutions");
        // Every solution must be within limits and reproduce the pose.
        for s in &sols {
            assert!(
                s.0[0] >= -1e-9 && s.0[0] <= 0.5 + 1e-9,
                "rail out of limits: {}",
                s.0[0]
            );
            let got = chain.fk_end(s).unwrap().to_cols_array();
            let want = target.to_cols_array();
            assert!(
                want.iter()
                    .zip(got.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-6)
            );
        }
        // The planted rail value should be recovered by some solution.
        assert!(
            sols.iter().any(|s| (s.0[0] - 0.2).abs() < 1e-6),
            "planted rail value not found"
        );
    }

    /// IncludeWrapped emits an extra solution with the joint wrapped ±2π when the
    /// wrapped value stays within limits.
    #[test]
    fn include_wrapped_emits_extra_solution() {
        use crate::IkRules;
        use deke_types::IkSolver;

        // Joint 6 (FixedAxis-reduced Panda) with wide limits so ±2π stays inside.
        let rules = [
            IkRules::FixedAxis { idx: 6, pos: 0.0 },
            IkRules::IncludeWrapped { idx: 0 },
        ];
        let chain: Kinematics<7, f64> = Kinematics::from_dh(panda7(), wide(), &rules);
        let q = SRobotQ::<7, f64>::from_array([0.3, 0.2, 0.3, -0.4, 0.5, 0.6, 0.0]);
        let target = chain.fk_end(&q).unwrap();
        let sols = chain.ik(target).unwrap().unwrap();
        assert!(!sols.is_empty());

        // For some base solution there must also be one with joint 0 wrapped by 2π.
        let has_wrapped_pair = sols.iter().any(|s| {
            sols.iter().any(|t| {
                (t.0[0] - (s.0[0] + std::f64::consts::TAU)).abs() < 1e-6
                    && (1..7).all(|k| (t.0[k] - s.0[k]).abs() < 1e-6)
            })
        });
        assert!(has_wrapped_pair, "IncludeWrapped produced no ±2π variant");
        // Every wrapped solution still reproduces the pose (2π is a no-op for FK).
        for s in &sols {
            let got = chain.fk_end(s).unwrap().to_cols_array();
            let want = target.to_cols_array();
            assert!(
                want.iter()
                    .zip(got.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-6)
            );
        }
    }

    /// Solutions outside the joint limits are filtered from the output.
    #[test]
    fn limits_filter_ik_output() {
        use deke_types::IkSolver;

        // Tight limits around the planted config drop the alternate branches.
        let pi = std::f64::consts::PI;
        let lower = SRobotQ::<6, f64>::from_array([-0.5, -1.5, -0.5, -1.0, -1.0, -1.0]);
        let upper = SRobotQ::<6, f64>::from_array([0.5, 0.5, 1.5, 1.0, 0.0, 1.0]);
        let chain = {
            let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
            let a = [0.0, 0.4318, -0.0203, 0.0, 0.0, 0.0];
            let d = [0.6718, 0.1397, 0.0, 0.4318, 0.0, 0.0565];
            Kinematics::<6, f64>::from_dh(
                std::array::from_fn(|i| DHJoint {
                    a: a[i],
                    alpha: alpha[i],
                    d: d[i],
                    theta_offset: 0.0,
                }),
                JointLimits::new(lower, upper),
                &[],
            )
        };
        let q = SRobotQ::<6, f64>::from_array([0.1, 0.2, 0.3, 0.4, -0.5, 0.6]);
        let target = chain.fk_end(&q).unwrap();
        let sols = chain.ik(target).unwrap().unwrap();
        assert!(!sols.is_empty());
        // Every returned solution lies within the limits.
        for s in &sols {
            for k in 0..6 {
                assert!(
                    s.0[k] >= lower.0[k] - 1e-9 && s.0[k] <= upper.0[k] + 1e-9,
                    "joint {k} = {} out of [{}, {}]",
                    s.0[k],
                    lower.0[k],
                    upper.0[k]
                );
            }
        }
    }
}
