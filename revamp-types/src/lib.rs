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

pub use fk::{DHChain, DHJoint, FKChain, HPChain, HPJoint, URDFChain, URDFJoint};
pub use fk_dynamic::{BoxFK, DynamicDHChain, DynamicHPChain, DynamicURDFChain};
pub use path::RobotPath;
pub use q::{RobotQ, SRobotQ};
pub use validator::{JointValidator, Validator, ValidatorAnd, ValidatorNot, ValidatorOr};
pub use validator_dynamic::DynamicJointValidator;

#[derive(Debug, Clone)]
pub enum RevampError {
    ShapeMismatch { expected: usize, found: usize },
    JointsNonFinite,
    SelfCollison(i16, i16),
    EnvironmentCollision(i16, i16),
    ExceedJointLimits,
    OutOfIterations,
    SuperError,
}

impl From<Infallible> for RevampError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

pub type RevampResult<T> = Result<T, RevampError>;

pub trait Planner<const N: usize>: Sized + Clone + Debug + Send + Sync + 'static {
    type Diagnostic: Display + Send + Sync;

    fn plan<
        E: Into<RevampError>,
        A: TryInto<SRobotQ<N>, Error = E>,
        B: TryInto<SRobotQ<N>, Error = E>,
    >(
        &self,
        start: A,
        goal: B,
        validators: &mut impl Validator<N>,
    ) -> (RevampResult<RobotPath>, Self::Diagnostic);
}
