use crate::{JointValidator, RevampError, RevampResult, SRobotQ, Validator};

macro_rules! dynamic_joint_new {
    ($lower:ident, $upper:ident, $($variant:ident $n:literal),+) => {
        match $lower.len() {
            $($n => Some(DynamicJointValidator::$variant(JointValidator::new(
                SRobotQ(<[f32; $n]>::try_from($lower.as_slice()).unwrap()),
                SRobotQ(<[f32; $n]>::try_from($upper.as_slice()).unwrap()),
            )))),+,
            _ => None,
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
    pub fn try_new(lower: Vec<f32>, upper: Vec<f32>) -> RevampResult<Self> {
        if lower.len() != upper.len() {
            return Err(RevampError::ShapeMismatch { expected: lower.len(), found: upper.len() });
        }
        dynamic_joint_new!(lower, upper, J1 1, J2 2, J3 3, J4 4, J5 5, J6 6, J7 7, J8 8)
            .ok_or(RevampError::ShapeMismatch { expected: 8, found: lower.len() })
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

    pub fn validate_dyn(&mut self, q: &[f32]) -> RevampResult<()> {
        match self {
            Self::J1(v) => {
                let arr: &[f32; 1] = q.try_into().map_err(|_| RevampError::ShapeMismatch {
                    expected: 1,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J2(v) => {
                let arr: &[f32; 2] = q.try_into().map_err(|_| RevampError::ShapeMismatch {
                    expected: 2,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J3(v) => {
                let arr: &[f32; 3] = q.try_into().map_err(|_| RevampError::ShapeMismatch {
                    expected: 3,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J4(v) => {
                let arr: &[f32; 4] = q.try_into().map_err(|_| RevampError::ShapeMismatch {
                    expected: 4,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J5(v) => {
                let arr: &[f32; 5] = q.try_into().map_err(|_| RevampError::ShapeMismatch {
                    expected: 5,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J6(v) => {
                let arr: &[f32; 6] = q.try_into().map_err(|_| RevampError::ShapeMismatch {
                    expected: 6,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J7(v) => {
                let arr: &[f32; 7] = q.try_into().map_err(|_| RevampError::ShapeMismatch {
                    expected: 7,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J8(v) => {
                let arr: &[f32; 8] = q.try_into().map_err(|_| RevampError::ShapeMismatch {
                    expected: 8,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
        }
    }

    pub fn validate_motion_dyn(&mut self, qs: &[&[f32]]) -> RevampResult<()> {
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
                fn validate<E: Into<RevampError>, A: TryInto<SRobotQ<$n>, Error = E>>(
                    &mut self,
                    q: A,
                ) -> RevampResult<()> {
                    match self {
                        Self::$variant(v) => v.validate(q),
                        _ => Err(RevampError::ShapeMismatch {
                            expected: self.dof(),
                            found: $n,
                        }),
                    }
                }

                fn validate_motion(&mut self, qs: &[SRobotQ<$n>]) -> RevampResult<()> {
                    match self {
                        Self::$variant(v) => v.validate_motion(qs),
                        _ => Err(RevampError::ShapeMismatch {
                            expected: self.dof(),
                            found: $n,
                        }),
                    }
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
    pub fn from_validator(v: impl Into<Self>) -> Self { v.into() }
}
