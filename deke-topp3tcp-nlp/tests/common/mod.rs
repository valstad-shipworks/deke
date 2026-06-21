#![allow(dead_code)]

use std::f64::consts::FRAC_PI_2;

use deke_kin::deke_types::{ContinuousFKChain, JointSpec, KinSpec};
use deke_kin::glam::{DAffine3, DVec3};
use deke_kin::{DHJoint, JointLimits, Kinematics};
use deke_types::{JointValidator, SRobotQ};

/// UR5-ish 6-DOF DH chain — used in every multi-DOF test so that TCP metrics are realistic.
pub fn dh_6dof() -> Kinematics<6, f64> {
    Kinematics::from_dh(
        [
            DHJoint {
                a: 0.0,
                alpha: FRAC_PI_2,
                d: 0.089,
                theta_offset: 0.0,
            },
            DHJoint {
                a: -0.425,
                alpha: 0.0,
                d: 0.0,
                theta_offset: 0.0,
            },
            DHJoint {
                a: -0.392,
                alpha: 0.0,
                d: 0.0,
                theta_offset: 0.0,
            },
            DHJoint {
                a: 0.0,
                alpha: FRAC_PI_2,
                d: 0.109,
                theta_offset: 0.0,
            },
            DHJoint {
                a: 0.0,
                alpha: -FRAC_PI_2,
                d: 0.094,
                theta_offset: 0.0,
            },
            DHJoint {
                a: 0.0,
                alpha: 0.0,
                d: 0.082,
                theta_offset: 0.0,
            },
        ],
        JointLimits::symmetric(10.0),
        &[],
    )
}

pub fn dh_1dof() -> Kinematics<1, f64> {
    Kinematics::from_dh(
        [DHJoint {
            a: 0.3,
            alpha: 0.0,
            d: 0.0,
            theta_offset: 0.0,
        }],
        JointLimits::symmetric(10.0),
        &[],
    )
}

/// 7-DOF chain: a prismatic rail (q[0]) along world +X carrying a UR5-ish 6-DOF arm.
/// Models the common "mobile base + arm" or "vertical lift + arm" setup. The redundancy
/// is deliberate — TCP velocity has a null-space (the rail can shift while the arm
/// counter-rotates).
pub fn dh_7dof_prismatic() -> Kinematics<7, f64> {
    // Prepend a prismatic rail (along +X) to the 6-DOF arm's KinSpec.
    let arm = dh_6dof().structure();
    let joints: [(DAffine3, JointSpec<f64>); 7] = std::array::from_fn(|i| {
        if i == 0 {
            (
                DAffine3::IDENTITY,
                JointSpec::Prismatic {
                    axis_local: DVec3::X,
                },
            )
        } else {
            arm.joints[i - 1]
        }
    });
    let spec = KinSpec::new(arm.base_to_first, joints, arm.end_to_ee);
    Kinematics::from_kinspec(spec, JointLimits::symmetric(10.0), &[])
}

pub fn wide_validator<const N: usize>() -> JointValidator<N, f64> {
    JointValidator::<N, f64>::new(
        SRobotQ::from_array([-10.0; N]),
        SRobotQ::from_array([10.0; N]),
    )
}

/// f32-precision sibling of [`wide_validator`].
pub fn wide_validator_f32<const N: usize>() -> JointValidator<N, f32> {
    JointValidator::<N, f32>::new(
        SRobotQ::from_array([-10.0; N]),
        SRobotQ::from_array([10.0; N]),
    )
}
