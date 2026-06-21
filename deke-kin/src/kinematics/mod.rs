//! Kinematics: the unified forward-kinematics chain [`Kinematics`] plus the
//! construction-time helpers (DH→HP, chain reversal, fixed-axis locking) used
//! by the IK solvers.
//!
//! The FK trait machinery (`FKChain`, `ContinuousFKChain`, `KinScalar`, `KinSpec`)
//! lives in [`deke_types`]; the aligned vector/matrix aliases are defined in this
//! crate's root. Both are re-exported here so the chain implementation can refer
//! to them via `super::`.

mod chain;
mod fk_chain;
mod params;

pub use chain::{
    apply_r6t, create_normal_vector, dh_to_hp, fwdkin, inverse_homogeneous,
    partial_joint_parametrization, reverse_chain,
};
pub use fk_chain::{JointLimits, Kinematics};
pub use params::{DHJoint, HPJoint, URDFBuildError, URDFJoint, URDFJointType};

pub(crate) use crate::{AAffine3, AMat3, AVec3};
pub(crate) use deke_types::{
    ContinuousFKChain, FKChain, JointSpec, KinScalar, KinSpec, check_finite,
};

/// Convert an `f64` value into the chain's scalar type `F` (`f32` or `f64`).
/// Used by the generic builders to lower URDF parameters and tolerance
/// constants into `F`.
#[inline]
pub(crate) fn scalar_from_f64<F: KinScalar>(x: f64) -> F {
    num_traits::NumCast::from(x).expect("f64 value is representable in F")
}
