use std::{fmt::Debug, sync::Arc};

use crate::{RevampError, RevampResult, SRobotQ, Token};


pub trait Validator<const N: usize, TKN: Token>:
    Sized + Clone + Debug + Send + Sync + 'static
{
    fn validate<E: Into<RevampError<TKN>>, A: TryInto<SRobotQ<N>, Error = E>>(&mut self, q: A) -> RevampResult<(), TKN>;
    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> RevampResult<(), TKN>;
}

#[derive(Debug, Clone)]
pub struct ValidatorAnd<A, B>(pub A, pub B);

#[derive(Debug, Clone)]
pub struct ValidatorOr<A, B>(pub A, pub B);

#[derive(Debug, Clone)]
pub struct ValidatorNot<A>(pub A);

impl<const N: usize, TKN: Token, A, B> Validator<N, TKN> for ValidatorAnd<A, B>
where
    A: Validator<N, TKN>,
    B: Validator<N, TKN>,
{
    #[inline]
    fn validate<E: Into<RevampError<TKN>>, Q: TryInto<SRobotQ<N>, Error = E>>(
        &mut self,
        q: Q,
    ) -> RevampResult<(), TKN> {
        let q = q.try_into().map_err(|e| e.into())?;
        self.0.validate(q)?;
        self.1.validate(q)
    }

    #[inline]
    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> RevampResult<(), TKN> {
        self.0.validate_motion(qs)?;
        self.1.validate_motion(qs)
    }
}

impl<const N: usize, TKN: Token, A, B> Validator<N, TKN> for ValidatorOr<A, B>
where
    A: Validator<N, TKN>,
    B: Validator<N, TKN>,
{
    #[inline]
    fn validate<E: Into<RevampError<TKN>>, Q: TryInto<SRobotQ<N>, Error = E>>(
        &mut self,
        q: Q,
    ) -> RevampResult<(), TKN> {
        let q = q.try_into().map_err(|e| e.into())?;
        match self.0.validate(q) {
            Ok(()) => Ok(()),
            Err(_) => self.1.validate(q),
        }
    }

    #[inline]
    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> RevampResult<(), TKN> {
        match self.0.validate_motion(qs) {
            Ok(()) => Ok(()),
            Err(_) => self.1.validate_motion(qs),
        }
    }
}

impl<const N: usize, TKN: Token, A> Validator<N, TKN> for ValidatorNot<A>
where
    A: Validator<N, TKN>,
{
    #[inline]
    fn validate<E: Into<RevampError<TKN>>, Q: TryInto<SRobotQ<N>, Error = E>>(
        &mut self,
        q: Q,
    ) -> RevampResult<(), TKN> {
        let q = q.try_into().map_err(|e| e.into())?;
        match self.0.validate(q) {
            Ok(()) => Err(RevampError::SuperError),
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> RevampResult<(), TKN> {
        match self.0.validate_motion(qs) {
            Ok(()) => Err(RevampError::SuperError),
            Err(_) => Ok(()),
        }
    }
}

#[derive(Clone)]
pub struct JointValidator<const N: usize, TKN: Token> {
    lower: SRobotQ<N>,
    upper: SRobotQ<N>,
    extras: Option<Arc<[Box<dyn Fn(&SRobotQ<N>) -> bool + Send + Sync>]>>,
    phantom: std::marker::PhantomData<TKN>
}

impl<const N: usize, TKN: Token> Debug for JointValidator<N, TKN> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JointValidator")
            .field("lower", &self.lower)
            .field("upper", &self.upper)
            .field("extras", &format!("[{} extra checks]", self.extras.as_ref().map(|e| e.len()).unwrap_or(0)))
            .finish()
    }
}

impl<const N: usize, TKN: Token> JointValidator<N, TKN> {
    pub fn new(lower: SRobotQ<N>, upper: SRobotQ<N>) -> Self {
        Self {
            lower,
            upper,
            extras: None,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn new_with_extras(
        lower: SRobotQ<N>,
        upper: SRobotQ<N>,
        extras: Vec<Box<dyn Fn(&SRobotQ<N>) -> bool + Send + Sync>>,
    ) -> Self {
        Self {
            lower,
            upper,
            extras: Some(extras.into()),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<const N: usize, TKN: Token + 'static> Validator<N, TKN> for JointValidator<N, TKN> {
    #[inline]
    fn validate<E: Into<RevampError<TKN>>, Q: TryInto<SRobotQ<N>, Error = E>>(
        &mut self,
        q: Q,
    ) -> RevampResult<(), TKN> {
        let q = q.try_into().map_err(|e| e.into())?;
        if q.any_lt(&self.lower) || q.any_gt(&self.upper) {
            return Err(RevampError::ExceedJointLimits);
        }
        if let Some(extras) = &self.extras {
            for check in extras.iter() {
                if !check(&q) {
                    return Err(RevampError::ExceedJointLimits);
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> RevampResult<(), TKN> {
        for q in qs {
            self.validate(*q)?;
        }
        Ok(())
    }
}