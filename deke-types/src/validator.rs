use std::{fmt::Debug, sync::Arc};

use bitvec::vec::BitVec;
use wide::{f32x8, f64x4, CmpGt, CmpLt};

use crate::{DekeError, DekeResult, KinScalar, SRobotQ, SRobotQLike};


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
    /// Maximal-margin value used when a validator is disabled (no
    /// constraint applied). Mirrors the `as_f64() == INFINITY` convention
    /// for the unit return type.
    fn passing() -> Self;
}

impl sealed::Sealed for () {}
impl ValidatorRet for () {
    #[inline]
    fn as_f64(&self) -> f64 {
        f64::INFINITY
    }
    #[inline]
    fn passing() -> Self {}
}

impl sealed::Sealed for f32 {}
impl ValidatorRet for f32 {
    #[inline]
    fn as_f64(&self) -> f64 {
        *self as f64
    }
    #[inline]
    fn passing() -> Self {
        f32::INFINITY
    }
}

impl sealed::Sealed for f64 {}
impl ValidatorRet for f64 {
    #[inline]
    fn as_f64(&self) -> f64 {
        *self
    }
    #[inline]
    fn passing() -> Self {
        f64::INFINITY
    }
}

/// SIMD joint-limit batch check, sealed to the scalar types [`KinScalar`]
/// permits (`f32` → `f32x8`, `f64` → `f64x4`). Bit `i` of `out` is set when
/// `qs[i]` lies outside `[lower, upper]` on any axis. It is a supertrait of
/// [`KinScalar`] so the generic [`JointValidator`] batch path vectorises
/// without narrowing the impl with an extra bound.
#[doc(hidden)]
pub trait BatchLimits: num_traits::Float {
    fn fill_oob<const N: usize>(
        qs: &[SRobotQ<N, Self>],
        lower: &[Self; N],
        upper: &[Self; N],
        out: &mut BitVec,
    );
}

macro_rules! impl_batch_limits {
    ($scalar:ty, $simd:ty, $lanes:literal) => {
        impl BatchLimits for $scalar {
            fn fill_oob<const N: usize>(
                qs: &[SRobotQ<N, $scalar>],
                lower: &[$scalar; N],
                upper: &[$scalar; N],
                out: &mut BitVec,
            ) {
                let n = qs.len();
                out.clear();
                out.resize(n, false);
                let mut i = 0usize;
                while i + $lanes <= n {
                    let mut fail = <$simd>::new([0.0; $lanes]);
                    let mut j = 0usize;
                    while j < N {
                        let mut col = [0.0; $lanes];
                        let mut l = 0usize;
                        while l < $lanes {
                            col[l] = qs[i + l].0[j];
                            l += 1;
                        }
                        let cv = <$simd>::new(col);
                        fail = fail | cv.simd_lt(lower[j]) | cv.simd_gt(upper[j]);
                        j += 1;
                    }
                    let arr = fail.to_array();
                    let mut l = 0usize;
                    while l < $lanes {
                        if arr[l].to_bits() != 0 {
                            out.set(i + l, true);
                        }
                        l += 1;
                    }
                    i += $lanes;
                }
                while i < n {
                    let q = &qs[i].0;
                    let mut j = 0usize;
                    while j < N {
                        if q[j] < lower[j] || q[j] > upper[j] {
                            out.set(i, true);
                            break;
                        }
                        j += 1;
                    }
                    i += 1;
                }
            }
        }
    };
}
impl_batch_limits!(f32, f32x8, 8);
impl_batch_limits!(f64, f64x4, 4);

pub trait Validator<const N: usize, R: ValidatorRet = (), F: KinScalar = f32>: Sized + Clone + Debug + Send + Sync + 'static {
    type Context<'ctx>: ValidatorContext;
    const VALIDATE_MOTION_IS_CONTINUOUS: bool = false;

    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<N, E, F>>(
        &self,
        q: A,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<R>;
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N, F>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<R>;

    /// Validate a batch of configurations at once, returning a bitvec whose
    /// `i`-th bit is set iff `qs[i]` is **invalid** (rejected). The default
    /// runs [`Validator::validate`] per config; implementors with a batched
    /// fast path (SIMD, GPU) override it.
    fn validate_batched<'ctx>(
        &self,
        qs: &[SRobotQ<N, F>],
        ctx: &Self::Context<'ctx>,
    ) -> BitVec {
        qs.iter().map(|q| self.validate(*q, ctx).is_err()).collect()
    }
}

#[derive(Debug, Clone)]
pub struct ValidatorAnd<A, B>(pub A, pub B);

#[derive(Debug, Clone)]
pub struct ValidatorOr<A, B>(pub A, pub B);

#[derive(Debug, Clone)]
pub struct ValidatorNot<A>(pub A);

impl<A, B> ValidatorAnd<A, B> {
    /// Construct an AND combinator after a compile-time assertion that
    /// `A` and `B` share the `Validator<N, R, F>` signature passed via
    /// turbofish or inferred at the call site.
    ///
    /// Direct tuple-struct construction (`ValidatorAnd(a, b)`) skips this
    /// check and is what the [`combine_validators!`] macro emits — both
    /// forms produce the same value, and the trait impl below only fires
    /// for `(N, R, F)` triples both members support, so callers that
    /// dispatch through the trait are safe either way.
    ///
    /// [`combine_validators!`]: deke-cricket
    pub fn new<const N: usize, R: ValidatorRet, F: KinScalar>(a: A, b: B) -> Self
    where
        A: Validator<N, R, F>,
        B: Validator<N, R, F>,
    {
        Self(a, b)
    }
}

impl<A, B> ValidatorOr<A, B> {
    /// See [`ValidatorAnd::new`].
    pub fn new<const N: usize, R: ValidatorRet, F: KinScalar>(a: A, b: B) -> Self
    where
        A: Validator<N, R, F>,
        B: Validator<N, R, F>,
    {
        Self(a, b)
    }
}

impl<A> ValidatorNot<A> {
    /// Construct a NOT combinator after a compile-time assertion that `A`
    /// implements `Validator<N, (), F>`. `Not` is restricted to the unit
    /// return type because inverting a scalar score isn't well-defined.
    pub fn new<const N: usize, F: KinScalar>(a: A) -> Self
    where
        A: Validator<N, (), F>,
    {
        Self(a)
    }
}

/// Blanket impls below cover every `(N, R, F)` triple that **both** member
/// validators implement: a single generic `impl` over `R` and `F` (and `N`
/// since validators are const-generic over DOF) means monomorphization
/// fires the impl for every shared signature without manual enumeration.
impl<const N: usize, F: KinScalar, R: ValidatorRet, A, B> Validator<N, R, F>
    for ValidatorAnd<A, B>
where
    A: Validator<N, R, F>,
    B: Validator<N, R, F>,
{
    type Context<'ctx> = (A::Context<'ctx>, B::Context<'ctx>);

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E, F>>(
        &self,
        q: Q,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<R> {
        let q = q.to_srobotq().map_err(Into::into)?;
        self.0.validate(q, &ctx.0)?;
        self.1.validate(q, &ctx.1)
    }

    #[inline]
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N, F>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<R> {
        self.0.validate_motion(qs, &ctx.0)?;
        self.1.validate_motion(qs, &ctx.1)
    }

    #[inline]
    fn validate_batched<'ctx>(
        &self,
        qs: &[SRobotQ<N, F>],
        ctx: &Self::Context<'ctx>,
    ) -> BitVec {
        let a = self.0.validate_batched(qs, &ctx.0);
        let b = self.1.validate_batched(qs, &ctx.1);
        a.iter().zip(b.iter()).map(|(x, y)| *x | *y).collect()
    }
}

impl<const N: usize, F: KinScalar, R: ValidatorRet, A, B> Validator<N, R, F> for ValidatorOr<A, B>
where
    A: Validator<N, R, F>,
    B: Validator<N, R, F>,
{
    type Context<'ctx> = (A::Context<'ctx>, B::Context<'ctx>);

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E, F>>(
        &self,
        q: Q,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<R> {
        let q = q.to_srobotq().map_err(Into::into)?;
        match self.0.validate(q, &ctx.0) {
            Ok(r) => Ok(r),
            Err(_) => self.1.validate(q, &ctx.1),
        }
    }

    #[inline]
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N, F>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<R> {
        match self.0.validate_motion(qs, &ctx.0) {
            Ok(r) => Ok(r),
            Err(_) => self.1.validate_motion(qs, &ctx.1),
        }
    }

    #[inline]
    fn validate_batched<'ctx>(
        &self,
        qs: &[SRobotQ<N, F>],
        ctx: &Self::Context<'ctx>,
    ) -> BitVec {
        let a = self.0.validate_batched(qs, &ctx.0);
        let b = self.1.validate_batched(qs, &ctx.1);
        a.iter().zip(b.iter()).map(|(x, y)| *x & *y).collect()
    }
}

/// `Not` is only meaningful for `R = ()` (a pass/fail validator). For
/// scalar-returning validators the inversion of the return value isn't
/// well-defined, so the impl is restricted to the unit case.
impl<const N: usize, F: KinScalar, A> Validator<N, (), F> for ValidatorNot<A>
where
    A: Validator<N, (), F>,
{
    type Context<'ctx> = A::Context<'ctx>;

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E, F>>(
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
        qs: &[SRobotQ<N, F>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        match self.0.validate_motion(qs, ctx) {
            Ok(()) => Err(DekeError::SuperError),
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn validate_batched<'ctx>(
        &self,
        qs: &[SRobotQ<N, F>],
        ctx: &Self::Context<'ctx>,
    ) -> BitVec {
        self.0.validate_batched(qs, ctx).iter().map(|x| !*x).collect()
    }
}

#[derive(Debug, Clone)]
pub enum MaybeValidator<V> {
    Active(V),
    Disabled,
}

impl<const N: usize, F: KinScalar, R: ValidatorRet, V> Validator<N, R, F> for MaybeValidator<V>
where
    V: Validator<N, R, F>,
{
    type Context<'ctx> = V::Context<'ctx>;

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E, F>>(
        &self,
        q: Q,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<R> {
        match self {
            MaybeValidator::Active(v) => v.validate(q, ctx),
            MaybeValidator::Disabled => Ok(R::passing()),
        }
    }

    #[inline]
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N, F>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<R> {
        match self {
            MaybeValidator::Active(v) => v.validate_motion(qs, ctx),
            MaybeValidator::Disabled => Ok(R::passing()),
        }
    }

    #[inline]
    fn validate_batched<'ctx>(
        &self,
        qs: &[SRobotQ<N, F>],
        ctx: &Self::Context<'ctx>,
    ) -> BitVec {
        match self {
            MaybeValidator::Active(v) => v.validate_batched(qs, ctx),
            MaybeValidator::Disabled => core::iter::repeat(false).take(qs.len()).collect(),
        }
    }
}

#[derive(Clone)]
pub struct JointValidator<const N: usize, F: KinScalar = f32> {
    lower: SRobotQ<N, F>,
    upper: SRobotQ<N, F>,
    extras: Option<Arc<[Box<dyn Fn(&SRobotQ<N, F>) -> bool + Send + Sync>]>>,
}

impl<const N: usize, F: KinScalar> Debug for JointValidator<N, F> {
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

impl<const N: usize, F: KinScalar> JointValidator<N, F> {
    pub fn new(lower: SRobotQ<N, F>, upper: SRobotQ<N, F>) -> Self {
        Self {
            lower,
            upper,
            extras: None,
        }
    }

    pub fn new_with_extras(
        lower: SRobotQ<N, F>,
        upper: SRobotQ<N, F>,
        extras: Vec<Box<dyn Fn(&SRobotQ<N, F>) -> bool + Send + Sync>>,
    ) -> Self {
        Self {
            lower,
            upper,
            extras: Some(extras.into()),
        }
    }
}

impl<const N: usize, F: KinScalar> Validator<N, (), F> for JointValidator<N, F> {
    type Context<'ctx> = ();

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E, F>>(
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
        qs: &[SRobotQ<N, F>],
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        for q in qs {
            self.validate(*q, _ctx)?;
        }
        Ok(())
    }

    #[inline]
    fn validate_batched<'ctx>(
        &self,
        qs: &[SRobotQ<N, F>],
        _ctx: &Self::Context<'ctx>,
    ) -> BitVec {
        let mut out = BitVec::with_capacity(qs.len());
        F::fill_oob::<N>(qs, &self.lower.0, &self.upper.0, &mut out);
        if let Some(extras) = &self.extras {
            for (i, q) in qs.iter().enumerate() {
                if out[i] {
                    continue;
                }
                if extras.iter().any(|check| !check(q)) {
                    out.set(i, true);
                }
            }
        }
        out
    }
}

/// Cross-precision entry point: f32-storage `JointValidator` accepting f64
/// inputs. The input is narrowed to f32 at the boundary so the same limits
/// govern both precisions; comparison is done in storage precision.
impl<const N: usize> Validator<N, (), f64> for JointValidator<N, f32> {
    type Context<'ctx> = ();

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E, f64>>(
        &self,
        q: Q,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        let q64 = q.to_srobotq().map_err(Into::into)?;
        let q32: SRobotQ<N, f32> = q64.into();
        <Self as Validator<N, (), f32>>::validate(self, q32, ctx)
    }

    #[inline]
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N, f64>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        for q in qs {
            let q32: SRobotQ<N, f32> = (*q).into();
            <Self as Validator<N, (), f32>>::validate(self, q32, ctx)?;
        }
        Ok(())
    }

    #[inline]
    fn validate_batched<'ctx>(
        &self,
        qs: &[SRobotQ<N, f64>],
        ctx: &Self::Context<'ctx>,
    ) -> BitVec {
        let q32: Vec<SRobotQ<N, f32>> = qs.iter().map(|q| (*q).into()).collect();
        <Self as Validator<N, (), f32>>::validate_batched(self, &q32, ctx)
    }
}

/// Cross-precision entry point: f64-storage `JointValidator` accepting f32
/// inputs. The f32 input is widened to f64 (lossless) before comparison.
impl<const N: usize> Validator<N, (), f32> for JointValidator<N, f64> {
    type Context<'ctx> = ();

    #[inline]
    fn validate<'ctx, E: Into<DekeError>, Q: SRobotQLike<N, E, f32>>(
        &self,
        q: Q,
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        let q32 = q.to_srobotq().map_err(Into::into)?;
        let q64: SRobotQ<N, f64> = q32.into();
        <Self as Validator<N, (), f64>>::validate(self, q64, ctx)
    }

    #[inline]
    fn validate_motion<'ctx>(
        &self,
        qs: &[SRobotQ<N, f32>],
        ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        for q in qs {
            let q64: SRobotQ<N, f64> = (*q).into();
            <Self as Validator<N, (), f64>>::validate(self, q64, ctx)?;
        }
        Ok(())
    }

    #[inline]
    fn validate_batched<'ctx>(
        &self,
        qs: &[SRobotQ<N, f32>],
        ctx: &Self::Context<'ctx>,
    ) -> BitVec {
        let q64: Vec<SRobotQ<N, f64>> = qs.iter().map(|q| (*q).into()).collect();
        <Self as Validator<N, (), f64>>::validate_batched(self, &q64, ctx)
    }
}
