#![allow(dead_code)]

use std::f64::consts::FRAC_PI_2;

use deke_kin::{DHJoint, JointLimits, Kinematics};
use deke_types::glam::{DAffine3, DVec3};
use deke_types::{ContinuousFKChain, JointSpec, JointValidator, KinSpec, SRobotQ, SRobotTraj};

/// UR5-ish 6-DOF DH chain — mirrors the deke-topp3tcp-nlp parity harness.
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

/// 7-DOF chain: a prismatic rail (q[0]) along world +X carrying the 6-DOF arm.
pub fn dh_7dof_prismatic() -> Kinematics<7, f64> {
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

/// A real production 7-DOF RTU chain (prismatic rail at q[0], metres, carrying a
/// 6-DOF arm), KinSpec extracted from orchestra's `nanopanel-material` handler.
/// Used by the long-chord TCP-cap regression / perf tests because its arm has a
/// large TCP Jacobian gain on the shoulder joints, which is what surfaces the
/// per-segment kappa under-estimate (see `docs/FUTURE.md`). The synthetic
/// `dh_7dof_prismatic` UR5 arm does not reproduce it.
pub fn material_7dof() -> Kinematics<7, f64> {
    let base = DAffine3::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -2.3285, -0.2049, -0.5005,
    ]);
    let rev = |c: &[f64; 12]| {
        (
            DAffine3::from_cols_array(c),
            JointSpec::Revolute { axis_local: DVec3::Z },
        )
    };
    let joints: [(DAffine3, JointSpec<f64>); 7] = [
        (
            DAffine3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            JointSpec::Prismatic { axis_local: DVec3::X },
        ),
        rev(&[-0.000003673, 1.0, 0.0, -1.0, -0.000003673, 0.0, 0.0, 0.0, 1.0, 0.315, -0.412, 0.8335]),
        rev(&[1.0, 0.0, 0.0, 0.0, -0.000003673, -1.0, 0.0, 1.0, -0.000003673, 0.312, 0.108, 0.4095]),
        rev(&[1.0, 0.0, 0.0, 0.0, -1.0, 0.000002654, 0.0, -0.000002654, -1.0, 0.0, -1.075, 0.0565]),
        rev(&[-0.000003673, 0.0, -1.0, 0.000002654, -1.0, 0.0, -1.0, -0.000002654, 0.000003673, 1.0146, 0.225, 0.1645]),
        rev(&[-0.000003673, 0.0, -1.0, -0.000002654, -1.0, 0.0, -1.0, 0.000002654, 0.000003673, -0.0664, 0.0, -0.2654]),
        rev(&[-0.000003673, 0.0, -1.0, 0.000002654, -1.0, 0.0, -1.0, -0.000002654, 0.000003673, 0.1814, 0.0, -0.0664]),
    ];
    let end = DAffine3::from_cols_array(&[
        0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -0.0586,
    ]);
    Kinematics::from_kinspec(KinSpec::new(base, joints, end), JointLimits::symmetric(100.0), &[])
}

/// Two real `nanopanel-material` transitions that exercise the long-chord TCP cap:
/// each is a single 2-waypoint chord that slides the rail (metres) and reconfigures
/// the arm a lot at once, so the TCP sweeps several metres.
pub const MATERIAL_PATH_A: [[f64; 7]; 2] = [
    [6.098, 0.518589, 0.522595, -0.086285, -0.001229, -0.965298, -0.511949],
    [7.875, 1.170435, 1.205216, 0.407282, 0.538306, 0.871359, -0.362264],
];
pub const MATERIAL_PATH_B: [[f64; 7]; 2] = [
    [7.575, 1.293054, 1.173025, 0.343524, -1.35973, 1.380238, 0.724128],
    [8.5, 0.002574, 0.150109, 0.000204, 0.000738, 0.147501, 0.003309],
];

pub fn wide_validator<const N: usize>() -> JointValidator<N, f64> {
    JointValidator::<N, f64>::new(
        SRobotQ::from_array([-10.0; N]),
        SRobotQ::from_array([10.0; N]),
    )
}

/// Per-step max normalised finite difference (v/a/j over their symmetric caps),
/// averaged over every step — "is the trajectory pressing against SOME limit?".
pub fn avg_utilization<const N: usize>(traj: &SRobotTraj<N, f64>, v: f64, a: f64, j: f64) -> f64 {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let n = p.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 1..n {
        let mut u = 0.0f64;
        for jj in 0..N {
            u = u.max((p[i].0[jj] - p[i - 1].0[jj]).abs() / dt / v);
        }
        if i >= 2 {
            for jj in 0..N {
                let acc = (p[i].0[jj] - 2.0 * p[i - 1].0[jj] + p[i - 2].0[jj]).abs() / (dt * dt);
                u = u.max(acc / a);
            }
        }
        if i >= 3 {
            for jj in 0..N {
                let jk = (p[i].0[jj] - 3.0 * p[i - 1].0[jj] + 3.0 * p[i - 2].0[jj]
                    - p[i - 3].0[jj])
                    .abs()
                    / (dt * dt * dt);
                u = u.max(jk / j);
            }
        }
        sum += u;
    }
    sum / (n - 1) as f64
}
