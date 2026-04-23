use std::{fmt::Debug, sync::Arc};

use crate::{DekeError, DekeResult, SRobotQ, SRobotQLike};


pub trait ValidatorContext: Sized {}

#[doc(hidden)]
pub trait Leaf: ValidatorContext {}

impl ValidatorContext for () {}

#[macro_export]
macro_rules! validator_context_type_impl {
    ($($ident:ident),*) => {
        $(
            impl $crate::ValidatorContext for $ident {}
            impl $crate::Leaf for $ident {}
        )*
    };
}

impl<A: ValidatorContext, B: ValidatorContext> ValidatorContext for (A, B) {}

pub trait FromFlattened<Flattened>: ValidatorContext {
    fn nest(flattened: Flattened) -> Self;
}

macro_rules! validator_context_tuple_impl {
    ($tup:tt) => {
        validator_context_tuple_impl!(@rewrite $tup [emit_impl]);
    };

    (@rewrite ($l:tt, $r:tt) [$($cb:tt)*]) => {
        validator_context_tuple_impl!(@rewrite $l [pair_right $r [$($cb)*]]);
    };
    (@rewrite $ident:ident [$($cb:tt)*]) => {
        validator_context_tuple_impl!(@invoke [$($cb)*] [$ident] $ident);
        validator_context_tuple_impl!(@invoke [$($cb)*] [] ());
    };

    (@invoke [pair_right $r:tt [$($cb:tt)*]] [$($kept_l:ident)*] $rew_l:tt) => {
        validator_context_tuple_impl!(@rewrite $r [pair_combine [$($kept_l)*] $rew_l [$($cb)*]]);
    };
    (@invoke [pair_combine [$($kept_l:ident)*] $rew_l:tt [$($cb:tt)*]] [$($kept_r:ident)*] $rew_r:tt) => {
        validator_context_tuple_impl!(@invoke [$($cb)*] [$($kept_l)* $($kept_r)*] ($rew_l, $rew_r));
    };
    (@invoke [emit_impl] [$($kept:ident)*] $shape:tt) => {
        #[allow(non_snake_case)]
        impl<$($kept: Leaf),*> FromFlattened<($($kept,)*)> for $shape {
            #[inline]
            fn nest(flattened: ($($kept,)*)) -> Self {
                let ($($kept,)*) = flattened;
                $shape
            }
        }
    };
}

validator_context_tuple_impl! { (A, (B, C)) }
validator_context_tuple_impl! { ((A, B), C) }

validator_context_tuple_impl! { (A, (B, (C, D))) }
validator_context_tuple_impl! { (A, ((B, C), D)) }
validator_context_tuple_impl! { ((A, B), (C, D)) }
validator_context_tuple_impl! { ((A, (B, C)), D) }
validator_context_tuple_impl! { (((A, B), C), D) }

validator_context_tuple_impl! { (A, (B, (C, (D, E)))) }
validator_context_tuple_impl! { (A, (B, ((C, D), E))) }
validator_context_tuple_impl! { (A, ((B, C), (D, E))) }
validator_context_tuple_impl! { (A, ((B, (C, D)), E)) }
validator_context_tuple_impl! { (A, (((B, C), D), E)) }
validator_context_tuple_impl! { ((A, B), (C, (D, E))) }
validator_context_tuple_impl! { ((A, B), ((C, D), E)) }
validator_context_tuple_impl! { ((A, (B, C)), (D, E)) }
validator_context_tuple_impl! { (((A, B), C), (D, E)) }
validator_context_tuple_impl! { ((A, (B, (C, D))), E) }
validator_context_tuple_impl! { ((A, ((B, C), D)), E) }
validator_context_tuple_impl! { (((A, B), (C, D)), E) }
validator_context_tuple_impl! { (((A, (B, C)), D), E) }
validator_context_tuple_impl! { ((((A, B), C), D), E) }

validator_context_tuple_impl! { ((A, B), ((C, D), (E, F))) }
validator_context_tuple_impl! { (((A, B), (C, D)), (E, F)) }
validator_context_tuple_impl! { ((A, (B, C)), ((D, E), F)) }
validator_context_tuple_impl! { (((A, B), C), ((D, E), F)) }

#[doc(hidden)]
mod sealed {
    pub trait Sealed {}
}

pub trait ValidatorRet: Sized + sealed::Sealed + Copy {
    fn as_f64(&self) -> f64;
}

impl sealed::Sealed for () {}
impl ValidatorRet for () {
    #[inline]
    fn as_f64(&self) -> f64 {
        f64::INFINITY
    }
}

impl sealed::Sealed for f32 {}
impl ValidatorRet for f32 {
    #[inline]
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}

impl sealed::Sealed for f64 {}
impl ValidatorRet for f64 {
    #[inline]
    fn as_f64(&self) -> f64 {
        *self
    }
}

pub trait Validator<const N: usize, R: ValidatorRet = ()>: Sized + Clone + Debug + Send + Sync + 'static {
    type Context<'ctx>: ValidatorContext;

    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<N, E>>(
        &self,
        q: A,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<R>;
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<R>;
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
    type Context<'ctx> = (A::Context<'ctx>, B::Context<'ctx>);

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E>>(
        &self,
        q: Q,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        let q = q.to_srobotq().map_err(Into::into)?;
        self.0.validate(q, &ctx.0)?;
        self.1.validate(q, &ctx.1)
    }

    #[inline]
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        self.0.validate_motion(qs, &ctx.0)?;
        self.1.validate_motion(qs, &ctx.1)
    }
}

impl<const N: usize, A, B> Validator<N> for ValidatorOr<A, B>
where
    A: Validator<N>,
    B: Validator<N>,
{
    type Context<'ctx> = (A::Context<'ctx>, B::Context<'ctx>);

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E>>(
        &self,
        q: Q,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        let q = q.to_srobotq().map_err(Into::into)?;
        match self.0.validate(q, &ctx.0) {
            Ok(()) => Ok(()),
            Err(_) => self.1.validate(q, &ctx.1),
        }
    }

    #[inline]
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        match self.0.validate_motion(qs, &ctx.0) {
            Ok(()) => Ok(()),
            Err(_) => self.1.validate_motion(qs, &ctx.1),
        }
    }
}

impl<const N: usize, A> Validator<N> for ValidatorNot<A>
where
    A: Validator<N>,
{
    type Context<'ctx> = A::Context<'ctx>;

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E>>(
        &self,
        q: Q,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        let q = q.to_srobotq().map_err(Into::into)?;
        match self.0.validate(q, ctx) {
            Ok(()) => Err(DekeError::SuperError),
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        match self.0.validate_motion(qs, ctx) {
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
    type Context<'ctx> = ();

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E>>(
        &self,
        q: Q,
        _ctx: &Self::Context<'ctx>,
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
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N>],
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        for q in qs {
            self.validate(*q, _ctx)?;
        }
        Ok(())
    }
}
