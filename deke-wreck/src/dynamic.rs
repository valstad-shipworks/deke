use std::fmt::Debug;

use deke_types::{BoxFK, FKChain, DekeError, DekeResult, SRobotQ, Validator};

use crate::WreckValidator;

pub enum DynamicWreckValidator {
    J1(Box<WreckValidator<1, BoxFK<1>>>),
    J2(Box<WreckValidator<2, BoxFK<2>>>),
    J3(Box<WreckValidator<3, BoxFK<3>>>),
    J4(Box<WreckValidator<4, BoxFK<4>>>),
    J5(Box<WreckValidator<5, BoxFK<5>>>),
    J6(Box<WreckValidator<6, BoxFK<6>>>),
    J7(Box<WreckValidator<7, BoxFK<7>>>),
    J8(Box<WreckValidator<8, BoxFK<8>>>),
}

impl Clone for DynamicWreckValidator {
    fn clone(&self) -> Self {
        match self {
            Self::J1(v) => Self::J1(v.clone()),
            Self::J2(v) => Self::J2(v.clone()),
            Self::J3(v) => Self::J3(v.clone()),
            Self::J4(v) => Self::J4(v.clone()),
            Self::J5(v) => Self::J5(v.clone()),
            Self::J6(v) => Self::J6(v.clone()),
            Self::J7(v) => Self::J7(v.clone()),
            Self::J8(v) => Self::J8(v.clone()),
        }
    }
}

impl Debug for DynamicWreckValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::J1(v) => f.debug_tuple("J1").field(v).finish(),
            Self::J2(v) => f.debug_tuple("J2").field(v).finish(),
            Self::J3(v) => f.debug_tuple("J3").field(v).finish(),
            Self::J4(v) => f.debug_tuple("J4").field(v).finish(),
            Self::J5(v) => f.debug_tuple("J5").field(v).finish(),
            Self::J6(v) => f.debug_tuple("J6").field(v).finish(),
            Self::J7(v) => f.debug_tuple("J7").field(v).finish(),
            Self::J8(v) => f.debug_tuple("J8").field(v).finish(),
        }
    }
}

impl DynamicWreckValidator {
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

    pub fn validate_dyn(&mut self, q: &[f32]) -> DekeResult<()> {
        match self {
            Self::J1(v) => {
                let arr: &[f32; 1] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 1,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J2(v) => {
                let arr: &[f32; 2] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 2,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J3(v) => {
                let arr: &[f32; 3] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 3,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J4(v) => {
                let arr: &[f32; 4] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 4,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J5(v) => {
                let arr: &[f32; 5] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 5,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J6(v) => {
                let arr: &[f32; 6] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 6,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J7(v) => {
                let arr: &[f32; 7] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 7,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
            Self::J8(v) => {
                let arr: &[f32; 8] = q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: 8,
                    found: q.len(),
                })?;
                v.validate(SRobotQ(*arr))
            }
        }
    }

    pub fn validate_motion_dyn(&mut self, qs: &[&[f32]]) -> DekeResult<()> {
        for q in qs {
            self.validate_dyn(q)?;
        }
        Ok(())
    }
}

macro_rules! impl_dynamic_wreck {
    ($($n:literal $variant:ident),+) => {
        $(
            impl Validator<$n> for DynamicWreckValidator {
                fn validate<E: Into<DekeError>, A: TryInto<SRobotQ<$n>, Error = E>>(
                    &mut self,
                    q: A,
                ) -> DekeResult<()> {
                    match self {
                        Self::$variant(v) => v.validate(q),
                        _ => Err(DekeError::ShapeMismatch {
                            expected: self.dof(),
                            found: $n,
                        }),
                    }
                }

                fn validate_motion(&mut self, qs: &[SRobotQ<$n>]) -> DekeResult<()> {
                    match self {
                        Self::$variant(v) => v.validate_motion(qs),
                        _ => Err(DekeError::ShapeMismatch {
                            expected: self.dof(),
                            found: $n,
                        }),
                    }
                }
            }

            impl<FK: FKChain<$n> + 'static> From<WreckValidator<$n, FK>> for DynamicWreckValidator {
                fn from(v: WreckValidator<$n, FK>) -> Self {
                    let (links, ee, base, env, fk) = v.into_parts();
                    Self::$variant(Box::new(WreckValidator::new(links, ee, base, env, BoxFK::new(fk))))
                }
            }
        )+
    };
}

impl_dynamic_wreck!(1 J1, 2 J2, 3 J3, 4 J4, 5 J5, 6 J6, 7 J7, 8 J8);

impl DynamicWreckValidator {
    pub fn from_validator(v: impl Into<Self>) -> Self { v.into() }
}
