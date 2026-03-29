use std::{
    convert::Infallible, fmt::{Debug, Display}, hash::Hash
};

pub use glam;
pub use wide;

mod path;
mod q;
mod validator;
mod fk;

pub use fk::{DHChain, DHJoint, FKChain, HPChain, HPJoint, URDFChain, URDFJoint};
pub use path::RobotPath;
pub use q::{RobotQ, SRobotQ};
pub use validator::{Validator, ValidatorAnd, ValidatorOr, ValidatorNot, JointValidator};

#[derive(Debug, Clone)]
pub enum RevampError<TKN: Token = NoToken> {
    ShapeMismatch { expected: usize, found: usize },
    SelfCollison(TKN, TKN),
    EnvironmentCollision(TKN, TKN),
    ExceedJointLimits,
    OutOfIterations,
    SuperError
}

impl<TKN: Token> From<Infallible> for RevampError<TKN> {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

pub type RevampResult<T, TKN = NoToken> = Result<T, RevampError<TKN>>;

pub trait Token: Debug + Display + Hash + Clone + Send + Sync + PartialEq + Eq {}
impl Token for String {}
impl<'a> Token for &'a str {}
impl Token for usize {}
impl Token for isize {}
impl Token for u64 {}
impl Token for u32 {}
impl Token for u16 {}
impl Token for u8 {}
impl Token for i64 {}
impl Token for i32 {}
impl Token for i16 {}
impl Token for i8 {}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct NoToken;
impl std::fmt::Display for NoToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NoToken")
    }
}
impl Token for NoToken {}


pub trait Planner<const N: usize, TKN: Token>:
    Sized + Clone + Debug + Send + Sync + 'static
{
    type Diagnostic: Display + Send + Sync;

    fn plan<E: Into<RevampError<TKN>>, A: TryInto<SRobotQ<N>, Error = E>, B: TryInto<SRobotQ<N>, Error = E>>(
        &self,
        start: A,
        goal: B,
        validators: &mut impl Validator<N, TKN>,
    ) -> (RevampResult<RobotPath, TKN>, Self::Diagnostic);
}
