// Shared across several test binaries; each uses a subset.
#![allow(dead_code)]

use std::time::Duration;

use deke_kin::{DHJoint, JointLimits as KinJointLimits, Kinematics};
use deke_linear::{FollowConfig, JointLimits, LinearConstraints, PathConditioning, PlannerOptions};
use deke_types::glam::{DAffine3, DMat3, DVec3};
use deke_types::{FKChain, SRobotQ, SRobotTraj};

/// UR10-ish 6R chain (spherical wrist → analytic IK), generous joint limits.
pub fn ur() -> Kinematics<6, f64> {
    use std::f64::consts::PI;
    let alpha = [PI / 2.0, 0.0, 0.0, PI / 2.0, -PI / 2.0, 0.0];
    let a = [0.0, -0.612, -0.573, 0.0, 0.0, 0.0];
    let d = [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922];
    Kinematics::from_dh(
        std::array::from_fn(|i| DHJoint {
            a: a[i],
            alpha: alpha[i],
            d: d[i],
            theta_offset: 0.0,
        }),
        KinJointLimits::symmetric(2.0 * PI),
        &[],
    )
}

/// A well-conditioned mid-workspace configuration to anchor test paths.
pub fn anchor() -> SRobotQ<6, f64> {
    SRobotQ::from_array([0.2, -1.0, 1.2, -1.3, -std::f64::consts::FRAC_PI_2, 0.3])
}

/// A validator that accepts everything, for tests that don't exercise obstacles.
pub fn noop() -> deke_linear::NoopValidator<6> {
    deke_linear::NoopValidator
}

pub fn config(tcp_speed: f64) -> FollowConfig<6> {
    config_flag(tcp_speed, false)
}

pub fn config_flag(tcp_speed: f64, forbid_interior_dips: bool) -> FollowConfig<6> {
    FollowConfig {
        conditioning: PathConditioning::default(),
        planner: PlannerOptions::default(),
        redundant: None,
        constraints: LinearConstraints {
            joint: JointLimits::symmetric(2.0, 8.0, 80.0),
            tcp_speed,
            output_dt: Duration::from_millis(8),
            forbid_interior_dips,
        },
    }
}

/// A straight Cartesian pose line of `n` vertices through the anchor's pose,
/// translating along `dir` for `len` metres with fixed orientation.
pub fn straight(robot: &Kinematics<6, f64>, dir: DVec3, len: f64, n: usize) -> Vec<DAffine3> {
    let base = robot.fk_end(&anchor()).unwrap();
    let rot = base.matrix3;
    let dir = dir.normalize();
    (0..n)
        .map(|i| {
            let f = i as f64 / (n - 1) as f64;
            DAffine3::from_mat3_translation(rot, base.translation + dir * (f * len))
        })
        .collect()
}

/// Two straight legs meeting at a corner, fixed orientation. `turn` is applied
/// in the XY plane of the world between the legs.
pub fn corner(
    robot: &Kinematics<6, f64>,
    leg: f64,
    turn_rad: f64,
    per_leg: usize,
) -> Vec<DAffine3> {
    let base = robot.fk_end(&anchor()).unwrap();
    let rot = base.matrix3;
    let d0 = DVec3::X;
    let d1 = DVec3::new(turn_rad.cos(), turn_rad.sin(), 0.0);
    let mut out = Vec::new();
    let p0 = base.translation;
    for i in 0..per_leg {
        let f = i as f64 / (per_leg - 1) as f64;
        out.push(DAffine3::from_mat3_translation(rot, p0 + d0 * (f * leg)));
    }
    let corner = p0 + d0 * leg;
    for i in 1..per_leg {
        let f = i as f64 / (per_leg - 1) as f64;
        out.push(DAffine3::from_mat3_translation(
            rot,
            corner + d1 * (f * leg),
        ));
    }
    out
}

/// TCP linear speed (m/s) between consecutive output samples via FK.
pub fn tcp_speeds(robot: &Kinematics<6, f64>, traj: &SRobotTraj<6, f64>) -> Vec<f64> {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    (0..p.len().saturating_sub(1))
        .map(|i| {
            let a = robot.fk_end(&p[i]).unwrap().translation;
            let b = robot.fk_end(&p[i + 1]).unwrap().translation;
            a.distance(b) / dt
        })
        .collect()
}

/// Peak per-axis joint velocity (rad/s) from the output trajectory.
pub fn joint_vel_peak(traj: &SRobotTraj<6, f64>) -> f64 {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let mut peak = 0.0f64;
    for i in 0..p.len().saturating_sub(1) {
        for j in 0..6 {
            peak = peak.max(((p[i + 1].0[j] - p[i].0[j]) / dt).abs());
        }
    }
    peak
}

/// Peak per-axis joint acceleration (rad/s²) via second difference.
pub fn joint_acc_peak(traj: &SRobotTraj<6, f64>) -> f64 {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let mut peak = 0.0f64;
    for i in 1..p.len().saturating_sub(1) {
        for j in 0..6 {
            let acc = (p[i + 1].0[j] - 2.0 * p[i].0[j] + p[i - 1].0[j]) / (dt * dt);
            peak = peak.max(acc.abs());
        }
    }
    peak
}

#[allow(dead_code)]
pub fn identity_rot() -> DMat3 {
    DMat3::IDENTITY
}
