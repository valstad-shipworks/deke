//! Input parameter types for [`Kinematics`](super::Kinematics) constructors. These
//! describe a chain; they are lowered into a [`KinSpec`](deke_types::KinSpec)
//! at construction time.

use deke_types::DekeError;

use super::KinScalar;

/// A standard Denavit-Hartenberg joint: `T_i = Rz(θ + theta_offset)·Tz(d)·Tx(a)·Rx(α)`.
#[derive(Debug, Clone, Copy)]
pub struct DHJoint<F: KinScalar = f32> {
    pub a: F,
    pub alpha: F,
    pub d: F,
    pub theta_offset: F,
}

/// A Hayati-Paul joint: `T_i = Rz(θ + theta_offset)·Rx(α)·Ry(β)·Tx(a)·Tz(d)`.
///
/// The extra `β` rotation about Y keeps the parameterization stable for
/// nearly-parallel consecutive joint axes, where standard DH is singular.
#[derive(Debug, Clone, Copy)]
pub struct HPJoint<F: KinScalar = f32> {
    pub a: F,
    pub alpha: F,
    pub beta: F,
    pub d: F,
    pub theta_offset: F,
}

/// Kind of URDF joint. Fixed joints have no motion; revolute and prismatic
/// joints move along `axis` (expressed in the joint's own frame, per the URDF
/// spec).
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
}

/// Error returned by [`Kinematics::from_urdf`](super::Kinematics::from_urdf).
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
    #[error("Kinematics<{expected}> requires {expected} actuated joints, found {found}")]
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
