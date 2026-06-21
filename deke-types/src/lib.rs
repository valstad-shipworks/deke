use std::{
    convert::Infallible,
    fmt::{Debug, Display},
    time::Duration,
};

pub use glam;

pub mod rexports {
    pub use glam;
    pub use glam_traits_ext;
    pub use ndarray;
    pub use num_traits;
    pub use smallvec;
}

mod fk;
mod path;
mod q;
mod traj;
mod validator;
mod validator_dynamic;

pub use fk::{
    BoxFK, ContinuousFKChain, FKChain, IkOutcome, IkSolutions, IkSolver, JointSpec, KinScalar,
    KinSpec, check_finite,
};
pub use path::{RobotPath, SRobotPath};
pub use q::{RobotQ, SRobotQ, SRobotQLike, robotq};
pub use traj::{RobotTraj, SRobotTraj};
#[doc(hidden)]
pub use validator::BatchLimits;
pub use validator::{
    FromFlattened, JointValidator, Leaf, MaybeValidator, Validator, ValidatorAnd, ValidatorContext,
    ValidatorNot, ValidatorOr,
};
pub use validator_dynamic::DynamicJointValidator;

use crate::validator::ValidatorRet;

#[derive(Debug, Clone, thiserror::Error)]
pub enum DekeError {
    #[error("Expected {expected} joints, but found {found}")]
    ShapeMismatch { expected: usize, found: usize },
    #[error("Path has {0} waypoints, needs at least 2")]
    PathTooShort(usize),
    #[error("Joints contain non-finite values")]
    JointsNonFinite,
    #[error("Self-collision detected between joints {0} and {1}")]
    SelfCollision(i16, i16),
    #[error("Environment collision detected between joint {0} and object {1}")]
    EnvironmentCollision(i16, i16),
    #[error("Joints exceed their limits")]
    ExceedJointLimits,
    #[error("Out of iterations")]
    OutOfIterations,
    #[error("Locked-prefix constraint violated at waypoint {waypoint} joint {joint}")]
    LockedPrefixViolation { waypoint: u32, joint: u8 },
    #[error("Boundary conditions not parallel to path tangent (residual {0})")]
    BoundaryInfeasible(f32),
    #[error("Path has consecutive zero-length segments")]
    DuplicateWaypoints,
    #[error("Retimer failed: {0}")]
    RetimerFailed(String),
    #[error(
        "Retimer output exceeds {limit_type} limit on dof {dof}: observed {observed_value}, limit {limit_value} (dt_in={dt_in:?})"
    )]
    ExceedsDynamicsLimits {
        dt_in: Duration,
        limit_type: &'static str,
        dof: u8,
        limit_value: f64,
        observed_value: f64,
    },
    #[error(
        "URDF joint at index {index} has an unexpected type: expected {expected}, found {found}"
    )]
    URDFJointTypeMismatch {
        index: usize,
        expected: &'static str,
        found: &'static str,
    },
    #[error("URDFChain<{expected}> requires {expected} revolute joints, found {found}")]
    URDFRevoluteCountMismatch { expected: usize, found: usize },
    #[error("IkSolver failed to converge: {0}")]
    IkSolverFailed(f64),
    #[error("inverse kinematics is not viable for this chain: {0}")]
    IkNotViable(String),
    #[error("Super error")]
    SuperError,
}

impl From<Infallible> for DekeError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

pub type DekeResult<T> = Result<T, DekeError>;

pub trait Planner<const N: usize, F: KinScalar = f32, R: ValidatorRet = ()>: Sized {
    type Diagnostic: Display + Debug;
    type Config;
    type Waypoints;

    fn plan<E: Into<DekeError>, V: Validator<N, R, F>>(
        &self,
        config: &Self::Config,
        waypoints: &Self::Waypoints,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotPath<N, F>>, Self::Diagnostic);
}

pub trait Retimer<const N: usize, F: KinScalar = f32, R: ValidatorRet = ()>: Sized {
    type Diagnostic: Display + Debug;
    type Constraints;

    fn retime<V: Validator<N, R, F>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, F>,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, F>>, Self::Diagnostic);
}
