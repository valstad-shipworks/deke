#![allow(dead_code)]

use std::f64::consts::FRAC_PI_2;

use deke_types::{DHChain, DHJoint, JointValidator, SRobotQ};

/// UR5-ish 6-DOF DH chain — used in every multi-DOF test so that TCP metrics are realistic.
pub fn dh_6dof() -> DHChain<6, f64> {
    DHChain::<6, f64>::from_joints([
        DHJoint { a: 0.0, alpha: FRAC_PI_2, d: 0.089, theta_offset: 0.0 },
        DHJoint { a: -0.425, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
        DHJoint { a: -0.392, alpha: 0.0, d: 0.0, theta_offset: 0.0 },
        DHJoint { a: 0.0, alpha: FRAC_PI_2, d: 0.109, theta_offset: 0.0 },
        DHJoint { a: 0.0, alpha: -FRAC_PI_2, d: 0.094, theta_offset: 0.0 },
        DHJoint { a: 0.0, alpha: 0.0, d: 0.082, theta_offset: 0.0 },
    ])
}

pub fn dh_1dof() -> DHChain<1, f64> {
    DHChain::<1, f64>::from_joints([DHJoint {
        a: 0.3,
        alpha: 0.0,
        d: 0.0,
        theta_offset: 0.0,
    }])
}

pub fn wide_validator<const N: usize>() -> JointValidator<N, f64> {
    JointValidator::<N, f64>::new(
        SRobotQ::from_array([-10.0; N]),
        SRobotQ::from_array([10.0; N]),
    )
}

/// f32-precision sibling of [`wide_validator`]. Useful when the test wants to
/// exercise the f32 dispatch arm of an `FPDispatch`-style validator.
pub fn wide_validator_f32<const N: usize>() -> JointValidator<N, f32> {
    JointValidator::<N, f32>::new(
        SRobotQ::from_array([-10.0; N]),
        SRobotQ::from_array([10.0; N]),
    )
}
