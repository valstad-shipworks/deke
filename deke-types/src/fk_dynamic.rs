use glam::{Affine3A, Vec3A};

use crate::{DHChain, DHJoint, HPChain, HPJoint, URDFChain, URDFJoint};
use crate::{DekeError, DekeResult, FKChain, SRobotQ};

macro_rules! dynamic_fk {
    ($name:ident, $chain:ident, $joint:ident) => {
        #[derive(Debug, Clone)]
        pub enum $name {
            J1($chain<1>),
            J2($chain<2>),
            J3($chain<3>),
            J4($chain<4>),
            J5($chain<5>),
            J6($chain<6>),
            J7($chain<7>),
            J8($chain<8>),
        }

        impl $name {
            pub fn try_new(joints: Vec<$joint>) -> DekeResult<Self> {
                let n = joints.len();
                let err = || DekeError::ShapeMismatch { expected: n, found: n };
                Ok(match n {
                    1 => Self::J1($chain::new(joints.try_into().map_err(|_| err())?)),
                    2 => Self::J2($chain::new(joints.try_into().map_err(|_| err())?)),
                    3 => Self::J3($chain::new(joints.try_into().map_err(|_| err())?)),
                    4 => Self::J4($chain::new(joints.try_into().map_err(|_| err())?)),
                    5 => Self::J5($chain::new(joints.try_into().map_err(|_| err())?)),
                    6 => Self::J6($chain::new(joints.try_into().map_err(|_| err())?)),
                    7 => Self::J7($chain::new(joints.try_into().map_err(|_| err())?)),
                    8 => Self::J8($chain::new(joints.try_into().map_err(|_| err())?)),
                    _ => return Err(DekeError::ShapeMismatch { expected: 8, found: n }),
                })
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

            pub fn fk_dyn(&self, q: &[f32]) -> DekeResult<Vec<Affine3A>> {
                dynamic_fk!(@dispatch_fk self, q,
                    J1 1, J2 2, J3 3, J4 4, J5 5, J6 6, J7 7, J8 8
                )
            }

            pub fn fk_end_dyn(&self, q: &[f32]) -> DekeResult<Affine3A> {
                dynamic_fk!(@dispatch_fk_end self, q,
                    J1 1, J2 2, J3 3, J4 4, J5 5, J6 6, J7 7, J8 8
                )
            }
        }

        dynamic_fk!(@impl_fkchain $name, $chain, 1 J1, 2 J2, 3 J3, 4 J4, 5 J5, 6 J6, 7 J7, 8 J8);
    };

    (@dispatch_fk $self:ident, $q:ident, $($variant:ident $n:literal),+) => {
        match $self {
            $(Self::$variant(chain) => {
                let arr: &[f32; $n] = $q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: $n,
                    found: $q.len(),
                })?;
                Ok(FKChain::<$n>::fk(chain, &SRobotQ(*arr)).map_err(|e| -> DekeError { e.into() })?.to_vec())
            }),+
        }
    };

    (@dispatch_fk_end $self:ident, $q:ident, $($variant:ident $n:literal),+) => {
        match $self {
            $(Self::$variant(chain) => {
                let arr: &[f32; $n] = $q.try_into().map_err(|_| DekeError::ShapeMismatch {
                    expected: $n,
                    found: $q.len(),
                })?;
                FKChain::<$n>::fk_end(chain, &SRobotQ(*arr)).map_err(|e| -> DekeError { e.into() })
            }),+
        }
    };

    (@impl_fkchain $name:ident, $chain:ident, $($n:literal $variant:ident),+) => {
        $(
            impl FKChain<$n> for $name {
                type Error = DekeError;

                fn fk(&self, q: &SRobotQ<$n>) -> Result<[Affine3A; $n], Self::Error> {
                    match self {
                        Self::$variant(chain) => FKChain::<$n>::fk(chain, q).map_err(Into::into),
                        _ => Err(DekeError::ShapeMismatch {
                            expected: self.dof(),
                            found: $n,
                        }),
                    }
                }

                fn fk_end(&self, q: &SRobotQ<$n>) -> Result<Affine3A, Self::Error> {
                    match self {
                        Self::$variant(chain) => FKChain::<$n>::fk_end(chain, q).map_err(Into::into),
                        _ => Err(DekeError::ShapeMismatch {
                            expected: self.dof(),
                            found: $n,
                        }),
                    }
                }

                fn joint_axes_positions(&self, q: &SRobotQ<$n>) -> Result<([Vec3A; $n], [Vec3A; $n], Vec3A), Self::Error> {
                    match self {
                        Self::$variant(chain) => FKChain::<$n>::joint_axes_positions(chain, q).map_err(Into::into),
                        _ => Err(DekeError::ShapeMismatch {
                            expected: self.dof(),
                            found: $n,
                        }),
                    }
                }
            }

            impl From<$chain<$n>> for $name {
                fn from(chain: $chain<$n>) -> Self {
                    Self::$variant(chain)
                }
            }
        )+
    };
}

dynamic_fk!(DynamicDHChain, DHChain, DHJoint);
dynamic_fk!(DynamicHPChain, HPChain, HPJoint);
dynamic_fk!(DynamicURDFChain, URDFChain, URDFJoint);

impl DynamicDHChain {
    pub fn from_chain(chain: impl Into<Self>) -> Self {
        chain.into()
    }
}

impl DynamicHPChain {
    pub fn from_chain(chain: impl Into<Self>) -> Self {
        chain.into()
    }
}

impl DynamicURDFChain {
    pub fn from_chain(chain: impl Into<Self>) -> Self {
        chain.into()
    }
}

trait ErasedFK<const N: usize>: Send + Sync {
    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], DekeError>;
    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, DekeError>;
    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N>,
    ) -> Result<([Vec3A; N], [Vec3A; N], Vec3A), DekeError>;
    fn clone_box(&self) -> Box<dyn ErasedFK<N>>;
}

impl<const N: usize, FK: FKChain<N> + 'static> ErasedFK<N> for FK {
    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], DekeError> {
        FKChain::fk(self, q).map_err(Into::into)
    }

    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, DekeError> {
        FKChain::fk_end(self, q).map_err(Into::into)
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N>,
    ) -> Result<([Vec3A; N], [Vec3A; N], Vec3A), DekeError> {
        FKChain::joint_axes_positions(self, q).map_err(Into::into)
    }

    fn clone_box(&self) -> Box<dyn ErasedFK<N>> {
        Box::new(self.clone())
    }
}

pub struct BoxFK<const N: usize>(Box<dyn ErasedFK<N>>);

impl<const N: usize> BoxFK<N> {
    pub fn new(fk: impl FKChain<N> + 'static) -> Self {
        Self(Box::new(fk))
    }
}

impl<const N: usize> Clone for BoxFK<N> {
    fn clone(&self) -> Self {
        Self(self.0.clone_box())
    }
}

impl<const N: usize> FKChain<N> for BoxFK<N> {
    type Error = DekeError;

    fn fk(&self, q: &SRobotQ<N>) -> Result<[Affine3A; N], DekeError> {
        self.0.fk(q)
    }

    fn fk_end(&self, q: &SRobotQ<N>) -> Result<Affine3A, DekeError> {
        self.0.fk_end(q)
    }

    fn joint_axes_positions(
        &self,
        q: &SRobotQ<N>,
    ) -> Result<([Vec3A; N], [Vec3A; N], Vec3A), DekeError> {
        self.0.joint_axes_positions(q)
    }
}
