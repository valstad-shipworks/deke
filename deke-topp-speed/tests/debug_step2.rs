//! Debug test that runs the full target solve and reports exactly where it
//! fails for the canonical 3-DoF long-motion scenario.

use std::time::Duration;

use deke_topp_speed::{
    ControlMode, Coordination, DurationGrid, GoalOutOfBounds, MotionSpec, Retimer, SRobotPath,
    SRobotQ, ToppSolver,
};
use deke_kin::{DHJoint, JointLimits, Kinematics};
use deke_types::JointValidator;

#[test]
fn debug_long_motion_failure() {
    let mut spec: MotionSpec<3, f64> = MotionSpec::new();
    spec.current_pose = SRobotQ::from_array([0.0, 0.0, 0.5]);
    spec.current_vel = SRobotQ::from_array([0.0, -2.2, -0.5]);
    spec.current_accel = SRobotQ::from_array([0.0, 2.5, -0.5]);
    spec.goal_pose = SRobotQ::from_array([5.0, -2.0, -3.5]);
    spec.goal_vel = SRobotQ::from_array([0.0, -0.5, -2.0]);
    spec.goal_accel = SRobotQ::from_array([0.0, 0.0, 0.5]);
    spec.max_vel = SRobotQ::from_array([3.0, 1.0, 3.0]);
    spec.max_accel = SRobotQ::from_array([3.0, 2.0, 1.0]);
    spec.max_jerk = SRobotQ::from_array([4.0, 3.0, 2.0]);
    spec.control_mode = ControlMode::Position;
    spec.coordination = Coordination::TimeLocked;
    spec.duration_grid = DurationGrid::Smooth;
    spec.goal_overflow = GoalOutOfBounds::Reject;

    let path = SRobotPath::from_two(spec.current_pose, spec.goal_pose);

    let joint = DHJoint::<f64> {
        a: 0.0,
        alpha: 0.0,
        d: 0.0,
        theta_offset: 0.0,
    };
    let fk: Kinematics<3, f64> =
        Kinematics::from_dh([joint, joint, joint], JointLimits::symmetric(1e6), &[]);
    let validator: JointValidator<3, f64> =
        JointValidator::new(SRobotQ::splat(-1e6), SRobotQ::splat(1e6));

    let (result, diag) =
        ToppSolver::new(Duration::from_millis(10), &fk).retime(&spec, &path, &validator, &());
    eprintln!("diagnostic: {diag}");
    let traj = result.expect("retime ok");
    eprintln!("samples: {}, dt: {:?}", traj.len(), traj.dt());

    // Print DoF 1 at a few times so we can compare against the C++ binding.
    for i in [0, 50, 58, 100, 200, 300, traj.len() - 1] {
        if let Some(q) = traj.get(i) {
            eprintln!(
                "  t={:.3}  pose = [{:>10.5}, {:>10.5}, {:>10.5}]",
                i as f64 * 0.01,
                q[0],
                q[1],
                q[2]
            );
        }
    }
}
