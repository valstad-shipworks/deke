use std::ops::Mul;

use glam::{Affine3A, DAffine3, DMat3, DVec3, Mat3A, Vec3A};
use glam_traits_ext::{FloatAffine, FloatMat, FloatScalar, FloatVec, TAffine3, TMat3, TVec3};

use crate::{DekeError, SRobotQ};

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
pub trait KinScalar: FloatScalar + Copy + std::fmt::Debug + Send + Sync + 'static + sealed::Sealed {
    type AVec3: TVec3<Self, MaybeAligned = Self::AVec3>;
    type AMat3: TMat3<Self, MaybeAligned = Self::AMat3>
        + FloatMat<Self, Col = Self::AVec3>
        + Mul<Self::AVec3, Output = Self::AVec3>;
    type AAffine3: TAffine3<Self, MaybeAligned = Self::AAffine3>
        + FloatAffine<Self, Vec = Self::AVec3, Mat = Self::AMat3>
        + Mul<Self::AAffine3, Output = Self::AAffine3>;
}

impl KinScalar for f32 {
    type AVec3 = Vec3A;
    type AMat3 = Mat3A;
    type AAffine3 = Affine3A;
}

impl KinScalar for f64 {
    type AVec3 = DVec3;
    type AMat3 = DMat3;
    type AAffine3 = DAffine3;
}

#[allow(type_alias_bounds)]
pub type AAffine3<F: KinScalar> = F::AAffine3;
#[allow(type_alias_bounds)]
pub type AVec3<F: KinScalar> = F::AVec3;

#[inline(always)]
#[cfg(debug_assertions)]
pub fn check_finite<const N: usize, F: FloatScalar>(q: &SRobotQ<N, F>) -> Result<(), DekeError> {
    if q.any_non_finite() {
        return Err(DekeError::JointsNonFinite);
    }
    Ok(())
}

#[inline(always)]
#[cfg(not(debug_assertions))]
pub fn check_finite<const N: usize, F: FloatScalar>(_: &SRobotQ<N, F>) -> Result<(), std::convert::Infallible> {
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointSpec<F: KinScalar> {
    Revolute  { axis_local: AVec3<F> },
    Prismatic { axis_local: AVec3<F> },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KinSpec<F: KinScalar, const N: usize> {
    pub base_to_first: AAffine3<F>,
    /// each tuple represents a `parent_to_joint` transform and the joint type.
    pub joints: [(AAffine3<F>, JointSpec<F>); N],
    pub end_to_ee: AAffine3<F>,
}

impl<F: KinScalar, const N: usize> KinSpec<F, N> {
    pub fn new(
        base_to_first: AAffine3<F>,
        joints: [(AAffine3<F>, JointSpec<F>); N],
        end_to_ee: AAffine3<F>,
    ) -> Self {
        Self {
            base_to_first,
            joints,
            end_to_ee,
        }
    }
}

pub trait FKChain<const N: usize, F: KinScalar = f32>: Clone + Send + Sync {
    type Error: Into<DekeError>;

    fn dof(&self) -> usize {
        N
    }
    /// Configuration-independent transform from the robot's base frame to the
    /// world frame.
    fn base_tf(&self) -> AAffine3<F> {
        AAffine3::<F>::IDENTITY
    }

    fn ee_tf(&self) -> AAffine3<F> {
        AAffine3::<F>::IDENTITY
    }

    fn fk(&self, q: &SRobotQ<N, F>) -> Result<[AAffine3<F>; N], Self::Error>;

    /// End-effector frame at configuration `q`.
    fn fk_end(&self, q: &SRobotQ<N, F>) -> Result<AAffine3<F>, Self::Error> {
        let frames = self.fk(q)?;
        Ok(if N > 0 {
            frames[N - 1] * self.ee_tf()
        } else {
            AAffine3::<F>::IDENTITY
        })
    }

    /// Compute base transform, per-link frames, and the end-effector frame
    /// in one call.
    fn all_fk(
        &self,
        q: &SRobotQ<N, F>,
    ) -> Result<(AAffine3<F>, [AAffine3<F>; N], AAffine3<F>), Self::Error> {
        let base = self.base_tf();
        let frames = self.fk(q)?;
        let end = self.fk_end(q)?;
        Ok((base, frames, end))
    }
}

/// Extension trait over [`FKChain`] for chains that can describe their
/// kinematic structure as a [`KinSpec`]. From the structure plus a joint
/// configuration the trait derives geometric Jacobian computations
/// (`jacobian`, `jacobian_dot`, `jacobian_ddot`) and the link-length-sum
/// `max_reach` estimate, all provided as defaults that respect each joint's
/// [`JointSpec`] (so prismatic and revolute columns are formed correctly).
pub trait ContinuousFKChain<const N: usize, F: KinScalar = f32>: FKChain<N, F, Error = DekeError> {
    fn structure(&self) -> KinSpec<F, N>;

    /// Theoretical maximum reach: sum of link lengths at `q = 0` (upper bound,
    /// ignores joint limits).
    fn max_reach(&self) -> Result<F, Self::Error> {
        let spec = self.structure();
        let (_, p, p_ee) = forward_pass(&spec, &SRobotQ::zeros());
        let mut total = F::zero();
        let mut prev = p[0];
        for i in 1..N {
            total = total + (p[i] - prev).length();
            prev = p[i];
        }
        total = total + (p_ee - prev).length();
        Ok(total)
    }

    /// Geometric Jacobian (6×N) at configuration `q`.
    /// Rows 0–2: linear velocity, rows 3–5: angular velocity.
    fn jacobian(&self, q: &SRobotQ<N, F>) -> Result<[[F; N]; 6], Self::Error> {
        #[cfg(debug_assertions)]
        check_finite::<N, F>(q).map_err(Self::Error::from)?;
        let spec = self.structure();
        let (z, p, p_ee) = forward_pass(&spec, q);
        let mut j = [[F::zero(); N]; 6];
        for i in 0..N {
            match spec.joints[i].1 {
                JointSpec::Revolute { .. } => {
                    let c = z[i].cross(p_ee - p[i]);
                    j[0][i] = c.x();
                    j[1][i] = c.y();
                    j[2][i] = c.z();
                    j[3][i] = z[i].x();
                    j[4][i] = z[i].y();
                    j[5][i] = z[i].z();
                }
                JointSpec::Prismatic { .. } => {
                    j[0][i] = z[i].x();
                    j[1][i] = z[i].y();
                    j[2][i] = z[i].z();
                }
            }
        }
        Ok(j)
    }

    /// First time-derivative of the geometric Jacobian.
    fn jacobian_dot(
        &self,
        q: &SRobotQ<N, F>,
        qdot: &SRobotQ<N, F>,
    ) -> Result<[[F; N]; 6], Self::Error> {
        #[cfg(debug_assertions)]
        {
            check_finite::<N, F>(q).map_err(Self::Error::from)?;
            check_finite::<N, F>(qdot).map_err(Self::Error::from)?;
        }
        let spec = self.structure();
        let (z, p, p_ee) = forward_pass(&spec, q);

        let mut omega = AVec3::<F>::ZERO;
        let mut z_dot = [AVec3::<F>::ZERO; N];
        let mut p_dot = [AVec3::<F>::ZERO; N];
        let mut pdot_acc = AVec3::<F>::ZERO;

        for i in 0..N {
            p_dot[i] = pdot_acc;
            z_dot[i] = omega.cross(z[i]);
            match spec.joints[i].1 {
                JointSpec::Revolute { .. } => {
                    omega += z[i] * qdot.0[i];
                }
                JointSpec::Prismatic { .. } => {
                    pdot_acc += z[i] * qdot.0[i];
                }
            }
            let next_p = if i + 1 < N { p[i + 1] } else { p_ee };
            pdot_acc += omega.cross(next_p - p[i]);
        }
        let p_ee_dot = pdot_acc;

        let mut jd = [[F::zero(); N]; 6];
        for i in 0..N {
            match spec.joints[i].1 {
                JointSpec::Revolute { .. } => {
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
                JointSpec::Prismatic { .. } => {
                    jd[0][i] = z_dot[i].x();
                    jd[1][i] = z_dot[i].y();
                    jd[2][i] = z_dot[i].z();
                }
            }
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
        #[cfg(debug_assertions)]
        {
            check_finite::<N, F>(q).map_err(Self::Error::from)?;
            check_finite::<N, F>(qdot).map_err(Self::Error::from)?;
            check_finite::<N, F>(qddot).map_err(Self::Error::from)?;
        }
        let spec = self.structure();
        let (z, p, p_ee) = forward_pass(&spec, q);

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
            match spec.joints[i].1 {
                JointSpec::Revolute { .. } => {
                    omega_dot += z[i] * qddot.0[i] + zd * qdot.0[i];
                    omega += z[i] * qdot.0[i];
                }
                JointSpec::Prismatic { .. } => {
                    pddot_acc += z[i] * qddot.0[i] + zd * qdot.0[i];
                    pdot_acc += z[i] * qdot.0[i];
                }
            }
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
            match spec.joints[i].1 {
                JointSpec::Revolute { .. } => {
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
                JointSpec::Prismatic { .. } => {
                    jdd[0][i] = z_ddot[i].x();
                    jdd[1][i] = z_ddot[i].y();
                    jdd[2][i] = z_ddot[i].z();
                }
            }
        }
        Ok(jdd)
    }
}

/// Walk a [`KinSpec`] at configuration `q` and return per-joint world-frame
/// axes, per-joint world-frame origins (before each joint's motion is
/// applied), and the end-effector world position. Joint axes from the spec
/// are normalised here, so callers may pass unnormalised axes in
/// [`JointSpec`].
fn forward_pass<F: KinScalar, const N: usize>(
    spec: &KinSpec<F, N>,
    q: &SRobotQ<N, F>,
) -> ([AVec3<F>; N], [AVec3<F>; N], AVec3<F>) {
    let mut z_out = [AVec3::<F>::ZERO; N];
    let mut p_out = [AVec3::<F>::ZERO; N];
    let mut current = spec.base_to_first;

    for i in 0..N {
        current = current * spec.joints[i].0;
        p_out[i] = current.translation();
        match spec.joints[i].1 {
            JointSpec::Revolute { axis_local } => {
                let axis = axis_local.normalize();
                z_out[i] = current.matrix3() * axis;
                current = current * AAffine3::<F>::from_axis_angle(axis, q.0[i]);
            }
            JointSpec::Prismatic { axis_local } => {
                let axis = axis_local.normalize();
                z_out[i] = current.matrix3() * axis;
                current = current * AAffine3::<F>::from_translation(axis * q.0[i]);
            }
        }
    }

    current = current * spec.end_to_ee;
    let p_ee = current.translation();
    (z_out, p_out, p_ee)
}


/// Inverse-kinematics solution set. Stays inline on the stack for the common
/// case of ≤8 solutions (analytic branches) and spills to the heap when a
/// discrete/enumerated solve produces more.
pub type IkSolutions<const N: usize, F> = smallvec::SmallVec<[SRobotQ<N, F>; 8]>;

pub enum IkOutcome<const N: usize, F: KinScalar> {
    Solved(IkSolutions<N, F>),
    Failed { partial: Option<IkSolutions<N, F>>, residual: F }
}

impl<const N: usize, F: KinScalar> IkOutcome<N, F> {
    pub fn unwrap(self) -> IkSolutions<N, F> {
        match self {
            IkOutcome::Solved(solutions) => solutions,
            _ => IkSolutions::new(),
        }
    }

    pub fn as_result(self) -> Result<IkSolutions<N, F>, DekeError> {
        match self {
            IkOutcome::Solved(solutions) => Ok(solutions),
            IkOutcome::Failed { residual, .. } => Err(DekeError::IkSolverFailed(
                residual.to_f64().unwrap_or(f64::MAX)
            )),
        }
    }

    pub fn residual(&self) -> Option<F> {
        match self {
            IkOutcome::Solved(_) => Some(F::zero()),
            IkOutcome::Failed { residual, .. } => Some(*residual),
        }
    }

    pub fn is_solved(&self) -> bool {
        matches!(self, IkOutcome::Solved(_))
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, IkOutcome::Failed { .. })
    }
}

pub trait IkSolver<const N: usize, F: KinScalar = f32>: FKChain<N, F> {
    type IkConfig: Default + Clone + Send + Sync + 'static;

    fn ik_with_config(&self, target: AAffine3<F>, config: &Self::IkConfig) -> Result<IkOutcome<N, F>, Self::Error>;
    fn ik(&self, target: AAffine3<F>) -> Result<IkOutcome<N, F>, Self::Error> {
        self.ik_with_config(target, &Self::IkConfig::default())
    }
}

trait ErasedFK<const N: usize, F: KinScalar>: Send + Sync {
    fn base_tf(&self) -> AAffine3<F>;
    fn fk(&self, q: &SRobotQ<N, F>) -> Result<[AAffine3<F>; N], DekeError>;
    fn fk_end(&self, q: &SRobotQ<N, F>) -> Result<AAffine3<F>, DekeError>;
    fn all_fk(
        &self,
        q: &SRobotQ<N, F>,
    ) -> Result<(AAffine3<F>, [AAffine3<F>; N], AAffine3<F>), DekeError>;
    fn clone_box(&self) -> Box<dyn ErasedFK<N, F>>;
}

impl<const N: usize, F: KinScalar, FK: FKChain<N, F> + 'static> ErasedFK<N, F> for FK {
    fn base_tf(&self) -> AAffine3<F> {
        FKChain::base_tf(self)
    }

    fn fk(&self, q: &SRobotQ<N, F>) -> Result<[AAffine3<F>; N], DekeError> {
        FKChain::fk(self, q).map_err(Into::into)
    }

    fn fk_end(&self, q: &SRobotQ<N, F>) -> Result<AAffine3<F>, DekeError> {
        FKChain::fk_end(self, q).map_err(Into::into)
    }

    fn all_fk(
        &self,
        q: &SRobotQ<N, F>,
    ) -> Result<(AAffine3<F>, [AAffine3<F>; N], AAffine3<F>), DekeError> {
        FKChain::all_fk(self, q).map_err(Into::into)
    }

    fn clone_box(&self) -> Box<dyn ErasedFK<N, F>> {
        Box::new(self.clone())
    }
}

pub struct BoxFK<const N: usize, F: KinScalar = f32>(Box<dyn ErasedFK<N, F>>);

impl<const N: usize, F: KinScalar> BoxFK<N, F> {
    pub fn new(fk: impl FKChain<N, F> + 'static) -> Self {
        Self(Box::new(fk))
    }
}

impl<const N: usize, F: KinScalar> Clone for BoxFK<N, F> {
    fn clone(&self) -> Self {
        Self(self.0.clone_box())
    }
}

impl<const N: usize, F: KinScalar> FKChain<N, F> for BoxFK<N, F> {
    type Error = DekeError;

    fn base_tf(&self) -> AAffine3<F> {
        self.0.base_tf()
    }

    fn fk(&self, q: &SRobotQ<N, F>) -> Result<[AAffine3<F>; N], DekeError> {
        self.0.fk(q)
    }

    fn fk_end(&self, q: &SRobotQ<N, F>) -> Result<AAffine3<F>, DekeError> {
        self.0.fk_end(q)
    }

    fn all_fk(
        &self,
        q: &SRobotQ<N, F>,
    ) -> Result<(AAffine3<F>, [AAffine3<F>; N], AAffine3<F>), DekeError> {
        self.0.all_fk(q)
    }
}
