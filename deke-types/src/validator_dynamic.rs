use crate::{DekeError, DekeResult, JointValidator, SRobotQ, SRobotQLike, Validator};

macro_rules! dynamic_joint_new {
    ($lower:ident, $upper:ident, $($variant:ident $n:literal),+) => {
        match $lower.len() {
            $($n => {
                let lo: [f32; $n] = $lower.as_slice().try_into().map_err(|_| DekeError::ShapeMismatch { expected: $n, found: $lower.len() })?;
                let hi: [f32; $n] = $upper.as_slice().try_into().map_err(|_| DekeError::ShapeMismatch { expected: $n, found: $upper.len() })?;
                Ok(DynamicJointValidator::$variant(JointValidator::new(SRobotQ(lo), SRobotQ(hi))))
            }),+,
            _ => Err(DekeError::ShapeMismatch { expected: 8, found: $lower.len() }),
        }
    };
}

#[derive(Debug, Clone)]
pub enum DynamicJointValidator {
    J1(JointValidator<1>),
    J2(JointValidator<2>),
    J3(JointValidator<3>),
    J4(JointValidator<4>),
    J5(JointValidator<5>),
    J6(JointValidator<6>),
    J7(JointValidator<7>),
    J8(JointValidator<8>),
}

impl DynamicJointValidator {
    pub fn try_new(lower: Vec<f32>, upper: Vec<f32>) -> DekeResult<Self> {
        if lower.len() != upper.len() {
            return Err(DekeError::ShapeMismatch {
                expected: lower.len(),
                found: upper.len(),
            });
        }
        dynamic_joint_new!(lower, upper, J1 1, J2 2, J3 3, J4 4, J5 5, J6 6, J7 7, J8 8)
    }

    pub fn dof(&self) -> usize {
        match self {
            Self::J1(_) => 1,
            Self::J2(_) => 2,
            Self::J3(_) => 3,
            Self::J4(_) => 4,
            Self::J5(_) => 5,
            Self::J6(_) => 6,
            Self::J7(_) => 7,
            Self::J8(_) => 8,
        }
    }

    pub fn validate_dyn(&self, q: &[f32]) -> DekeResult<()> {
        match self {
            Self::J1(v) => {
                let arr: &[f32; 1] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 1,
                    found: q.len(),
                })?;
                <JointValidator<_, f32> as Validator<_, (), f32>>::validate(v, SRobotQ(*arr), &())
            }
            Self::J2(v) => {
                let arr: &[f32; 2] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 2,
                    found: q.len(),
                })?;
                <JointValidator<_, f32> as Validator<_, (), f32>>::validate(v, SRobotQ(*arr), &())
            }
            Self::J3(v) => {
                let arr: &[f32; 3] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 3,
                    found: q.len(),
                })?;
                <JointValidator<_, f32> as Validator<_, (), f32>>::validate(v, SRobotQ(*arr), &())
            }
            Self::J4(v) => {
                let arr: &[f32; 4] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 4,
                    found: q.len(),
                })?;
                <JointValidator<_, f32> as Validator<_, (), f32>>::validate(v, SRobotQ(*arr), &())
            }
            Self::J5(v) => {
                let arr: &[f32; 5] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 5,
                    found: q.len(),
                })?;
                <JointValidator<_, f32> as Validator<_, (), f32>>::validate(v, SRobotQ(*arr), &())
            }
            Self::J6(v) => {
                let arr: &[f32; 6] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 6,
                    found: q.len(),
                })?;
                <JointValidator<_, f32> as Validator<_, (), f32>>::validate(v, SRobotQ(*arr), &())
            }
            Self::J7(v) => {
                let arr: &[f32; 7] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 7,
                    found: q.len(),
                })?;
                <JointValidator<_, f32> as Validator<_, (), f32>>::validate(v, SRobotQ(*arr), &())
            }
            Self::J8(v) => {
                let arr: &[f32; 8] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 8,
                    found: q.len(),
                })?;
                <JointValidator<_, f32> as Validator<_, (), f32>>::validate(v, SRobotQ(*arr), &())
            }
        }
    }

    pub fn validate_motion_dyn(&self, qs: &[&[f32]]) -> DekeResult<()> {
        for q in qs {
            self.validate_dyn(q)?;
        }
        Ok(())
    }
}

macro_rules! impl_dynamic_joint {
    ($($n:literal $variant:ident),+) => {
        $(
            impl Validator<$n> for DynamicJointValidator {
                type Context<'ctx> = ();

                fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<$n, E>>(
                    &self,
                    q: A,
                    ctx: &Self::Context<'ctx>,
                ) -> DekeResult<()> {
                    match self {
                        Self::$variant(v) => v.validate(q, ctx),
                        _ => Err(DekeError::ShapeMismatch {
                            expected: self.dof(),
                            found: $n,
                        }),
                    }
                }

                fn validate_motion<'ctx>(
                    &self,
                    qs: &[SRobotQ<$n>],
                    ctx: &Self::Context<'ctx>,
                ) -> DekeResult<()> {
                    match self {
                        Self::$variant(v) => v.validate_motion(qs, ctx),
                        _ => Err(DekeError::ShapeMismatch {
                            expected: self.dof(),
                            found: $n,
                        }),
                    }
                }
            }

            /// f64 entry point — downcasts to f32 and dispatches to the f32 impl.
            /// `DynamicJointValidator` stores `JointValidator<N, f32>`, so f64 inputs
            /// are narrowed at the boundary; precision is governed by the f32 limits
            /// the validator was configured with.
            impl Validator<$n, (), f64> for DynamicJointValidator {
                type Context<'ctx> = ();

                fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<$n, E, f64>>(
                    &self,
                    q: A,
                    ctx: &Self::Context<'ctx>,
                ) -> DekeResult<()> {
                    let q64 = q.to_srobotq().map_err(Into::into)?;
                    let q32: SRobotQ<$n, f32> = q64.into();
                    <Self as Validator<$n, (), f32>>::validate(self, q32, ctx)
                }

                fn validate_motion<'ctx>(
                    &self,
                    qs: &[SRobotQ<$n, f64>],
                    ctx: &Self::Context<'ctx>,
                ) -> DekeResult<()> {
                    for q in qs {
                        let q32: SRobotQ<$n, f32> = (*q).into();
                        <Self as Validator<$n, (), f32>>::validate(self, q32, ctx)?;
                    }
                    Ok(())
                }
            }

            impl From<JointValidator<$n>> for DynamicJointValidator {
                fn from(v: JointValidator<$n>) -> Self {
                    Self::$variant(v)
                }
            }
        )+
    };
}

impl_dynamic_joint!(1 J1, 2 J2, 3 J3, 4 J4, 5 J5, 6 J6, 7 J7, 8 J8);

impl DynamicJointValidator {
    pub fn from_validator(v: impl Into<Self>) -> Self {
        v.into()
    }
}
