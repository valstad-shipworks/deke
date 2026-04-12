use std::{fmt::Debug, sync::Arc};

use crate::{DekeError, DekeResult, SRobotQ, SRobotQLike};

pub trait Validator<const N: usize>: Sized + Clone + Debug + Send + Sync + 'static {
    fn validate<E: Into<DekeError>, A: SRobotQLike<N, E>>(
        &mut self,
        q: A,
    ) -> DekeResult<()>;
    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> DekeResult<()>;
}

#[derive(Debug, Clone)]
pub struct ValidatorAnd<A, B>(pub A, pub B);

#[derive(Debug, Clone)]
pub struct ValidatorOr<A, B>(pub A, pub B);

#[derive(Debug, Clone)]
pub struct ValidatorNot<A>(pub A);

impl<const N: usize, A, B> Validator<N> for ValidatorAnd<A, B>
where
    A: Validator<N>,
    B: Validator<N>,
{
    #[inline]
    fn validate<E: Into<DekeError>, Q: SRobotQLike<N, E>>(
        &mut self,
        q: Q,
    ) -> DekeResult<()> {
        let q = q.to_srobotq().map_err(Into::into)?;
        self.0.validate(q)?;
        self.1.validate(q)
    }

    #[inline]
    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> DekeResult<()> {
        self.0.validate_motion(qs)?;
        self.1.validate_motion(qs)
    }
}

impl<const N: usize, A, B> Validator<N> for ValidatorOr<A, B>
where
    A: Validator<N>,
    B: Validator<N>,
{
    #[inline]
    fn validate<E: Into<DekeError>, Q: SRobotQLike<N, E>>(
        &mut self,
        q: Q,
    ) -> DekeResult<()> {
        let q = q.to_srobotq().map_err(Into::into)?;
        match self.0.validate(q) {
            Ok(()) => Ok(()),
            Err(_) => self.1.validate(q),
        }
    }

    #[inline]
    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> DekeResult<()> {
        match self.0.validate_motion(qs) {
            Ok(()) => Ok(()),
            Err(_) => self.1.validate_motion(qs),
        }
    }
}

impl<const N: usize, A> Validator<N> for ValidatorNot<A>
where
    A: Validator<N>,
{
    #[inline]
    fn validate<E: Into<DekeError>, Q: SRobotQLike<N, E>>(
        &mut self,
        q: Q,
    ) -> DekeResult<()> {
        let q = q.to_srobotq().map_err(Into::into)?;
        match self.0.validate(q) {
            Ok(()) => Err(DekeError::SuperError),
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> DekeResult<()> {
        match self.0.validate_motion(qs) {
            Ok(()) => Err(DekeError::SuperError),
            Err(_) => Ok(()),
        }
    }
}

#[derive(Clone)]
pub struct JointValidator<const N: usize> {
    lower: SRobotQ<N>,
    upper: SRobotQ<N>,
    extras: Option<Arc<[Box<dyn Fn(&SRobotQ<N>) -> bool + Send + Sync>]>>,
}

impl<const N: usize> Debug for JointValidator<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JointValidator")
            .field("lower", &self.lower)
            .field("upper", &self.upper)
            .field(
                "extras",
                &format!(
                    "[{} extra checks]",
                    self.extras.as_ref().map(|e| e.len()).unwrap_or(0)
                ),
            )
            .finish()
    }
}

impl<const N: usize> JointValidator<N> {
    pub fn new(lower: SRobotQ<N>, upper: SRobotQ<N>) -> Self {
        Self {
            lower,
            upper,
            extras: None,
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
        }
    }
}

impl<const N: usize> Validator<N> for JointValidator<N> {
    #[inline]
    fn validate<E: Into<DekeError>, Q: SRobotQLike<N, E>>(
        &mut self,
        q: Q,
    ) -> DekeResult<()> {
        let q = q.to_srobotq().map_err(Into::into)?;
        if q.any_lt(&self.lower) || q.any_gt(&self.upper) {
            return Err(DekeError::ExceedJointLimits);
        }
        if let Some(extras) = &self.extras {
            for check in extras.iter() {
                if !check(&q) {
                    return Err(DekeError::ExceedJointLimits);
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> DekeResult<()> {
        for q in qs {
            self.validate(*q)?;
        }
        Ok(())
    }
}
