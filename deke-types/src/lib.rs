use std::{
    convert::Infallible,
    fmt::{Debug, Display},
};

pub use glam;
pub use wide;

mod fk;
mod fk_dynamic;
mod path;
mod q;
mod validator;
mod validator_dynamic;

pub use fk::{
    DHChain, DHJoint, FKChain, HPChain, HPJoint, PrismaticFK, TransformedFK, URDFChain, URDFJoint,
};
pub use fk_dynamic::{BoxFK, DynamicDHChain, DynamicHPChain, DynamicURDFChain};
pub use path::{RobotPath, SRobotPath};
pub use q::{RobotQ, SRobotQ, robotq, SRobotQLike};
pub use validator::{JointValidator, Validator, ValidatorAnd, ValidatorNot, ValidatorOr};
pub use validator_dynamic::DynamicJointValidator;

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
    #[error("Super error")]
    SuperError,
}

impl From<Infallible> for DekeError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

pub type DekeResult<T> = Result<T, DekeError>;

pub trait Planner<const N: usize>: Sized + Clone + Debug + Send + Sync + 'static {
    type Diagnostic: Display + Send + Sync;

    fn plan<
        E: Into<DekeError>,
        A: SRobotQLike<N, E>,
        B: SRobotQLike<N, E>,
    >(
        &self,
        start: A,
        goal: B,
        validators: &mut impl Validator<N>,
    ) -> (DekeResult<SRobotPath<N>>, Self::Diagnostic);
}
