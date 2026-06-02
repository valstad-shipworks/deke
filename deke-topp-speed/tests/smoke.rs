//! End-to-end smoke tests for the public API.
//!
//! These verify that types compose correctly and the `Retimer` impl can be
//! invoked. They do not assert numerical fidelity — the per-axis profile
//! validators are currently permissive stubs, so the produced trajectory may
//! be degenerate.

use std::time::Duration;

use deke_topp_speed::{
    ControlMode, Coordination, DurationGrid, FollowMode, GoalOutOfBounds, MotionSpec, Pursuer,
    PursuitTarget, Retimer, SRobotPath, SRobotQ, ToppSolver,
};
use deke_kin::{DHJoint, JointLimits, Kinematics};
use deke_types::JointValidator;

fn make_fk() -> Kinematics<3, f64> {
    let joints = [
        DHJoint::<f64> {
            a: 0.0,
            alpha: 0.0,
            d: 0.0,
            theta_offset: 0.0,
        },
        DHJoint::<f64> {
            a: 0.0,
            alpha: 0.0,
            d: 0.0,
            theta_offset: 0.0,
        },
        DHJoint::<f64> {
            a: 0.0,
            alpha: 0.0,
            d: 0.0,
            theta_offset: 0.0,
        },
    ];
    Kinematics::from_dh(joints, JointLimits::symmetric(1e6), &[])
}

#[test]
fn construct_motion_spec() {
    let mut spec: MotionSpec<3, f64> = MotionSpec::new();
    spec.max_vel = SRobotQ::from_array([1.0, 2.0, 1.0]);
    spec.max_accel = SRobotQ::from_array([3.0, 2.0, 2.0]);
    spec.max_jerk = SRobotQ::from_array([6.0, 10.0, 20.0]);
    spec.current_pose = SRobotQ::from_array([0.2, 0.0, -0.3]);
    spec.goal_pose = SRobotQ::from_array([0.5, 1.0, 0.0]);

    assert_eq!(spec.control_mode, ControlMode::Position);
    assert_eq!(spec.coordination, Coordination::TimeLocked);
    assert_eq!(spec.duration_grid, DurationGrid::Smooth);
    assert_eq!(spec.goal_overflow, GoalOutOfBounds::Reject);
}

#[test]
fn topp_solver_compiles_with_retimer_trait() {
    let spec: MotionSpec<3, f64> = {
        let mut s = MotionSpec::new();
        s.max_vel = SRobotQ::from_array([1.0, 2.0, 1.0]);
        s.max_accel = SRobotQ::from_array([3.0, 2.0, 2.0]);
        s.max_jerk = SRobotQ::from_array([6.0, 10.0, 20.0]);
        s
    };

    let start = SRobotQ::from_array([0.2, 0.0, -0.3]);
    let goal = SRobotQ::from_array([0.5, 1.0, 0.0]);
    let path = SRobotPath::from_two(start, goal);

    let fk = make_fk();
    let validator = JointValidator::<3, f64>::new(
        SRobotQ::from_array([-10.0, -10.0, -10.0]),
        SRobotQ::from_array([10.0, 10.0, 10.0]),
    );

    // Just verify retime() can be called; do not assert numerical content,
    // since the per-axis validators are currently permissive stubs.
    let (_result, diagnostic) =
        ToppSolver::new(Duration::from_millis(10), &fk).retime(&spec, &path, &validator, &());
    assert!(diagnostic.solve_micros >= 0.0);
}

#[test]
fn pursuer_construction() {
    let pursuer: Pursuer<3, f64> = Pursuer::new(Duration::from_millis(10));
    assert_eq!(pursuer.mode(), FollowMode::Tuned);
    assert_eq!(pursuer.reactiveness(), 1.0);
    assert_eq!(pursuer.look_ahead_cycles(), 1);
    assert_eq!(pursuer.max_iterations(), 64);
    assert_eq!(pursuer.last_iteration_count(), 0);
}

#[test]
fn pursuer_tick_returns_status() {
    let mut pursuer: Pursuer<3, f64> = Pursuer::new(Duration::from_millis(10));
    pursuer.set_mode(FollowMode::Quick);

    let mut spec: MotionSpec<3, f64> = MotionSpec::new();
    spec.max_vel = SRobotQ::splat(1.0);
    spec.max_accel = SRobotQ::splat(2.0);
    spec.max_jerk = SRobotQ::splat(20.0);

    let target = PursuitTarget::new(
        SRobotQ::from_array([1.0, 0.0, 0.0]),
        SRobotQ::zeros(),
        SRobotQ::zeros(),
    );

    let (_status, sample) = pursuer.tick(&target, &mut spec);
    assert!(sample.t.is_finite());
}

#[test]
fn pursuit_target_interpolation() {
    let a = PursuitTarget::<3, f64>::new(
        SRobotQ::from_array([0.0_f64, 0.0, 0.0]),
        SRobotQ::from_array([0.0_f64, 0.0, 0.0]),
        SRobotQ::zeros(),
    );
    let b = PursuitTarget::<3, f64>::new(
        SRobotQ::from_array([1.0_f64, 2.0, 3.0]),
        SRobotQ::from_array([0.0_f64, 0.0, 0.0]),
        SRobotQ::zeros(),
    );
    let mid = a.interpolate(&b, 0.5_f64);
    assert!((mid.pose[0] - 0.5_f64).abs() < 1e-12);
    assert!((mid.pose[1] - 1.0_f64).abs() < 1e-12);
    assert!((mid.pose[2] - 1.5_f64).abs() < 1e-12);
}
