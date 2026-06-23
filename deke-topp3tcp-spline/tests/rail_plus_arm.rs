//! End-to-end: plan a 7-DOF motion with the RRT planner, then retime it with
//! the spline retimer. The mechanism is a 6R arm carried on a 10 m linear rail;
//! the rail is joint 0, a prismatic axis along world +X. This exercises the
//! mixed prismatic/revolute chain through both crates and checks that the
//! retimed trajectory honors the rail's distinct linear limits rather than
//! treating every joint as a revolute radian/second axis.

use std::f64::consts::{FRAC_PI_2, PI};

use deke_kin::glam::{DAffine3, DVec3};
use deke_kin::{DHJoint, JointLimits as KinLimits, Kinematics};
use deke_rrt::{RrtcPlanner, RrtcSettings, StartEnd};
use deke_topp3tcp_spline::{
    JointLimits, SearchOptions, SplinePathOptions, TcpLimits, Topp3TcpSpline,
    Topp3TcpSplineConstraints,
};
use deke_types::{
    ContinuousFKChain, DekeError, JointSpec, JointValidator, KinSpec, Planner, Retimer, SRobotQ,
};

const N: usize = 7;
const RAIL_TRAVEL: f64 = 10.0;

/// 6R arm (UR-ish DH) mounted on a prismatic rail along world +X, expressed as
/// a single `ContinuousFKChain<7>`. Joint 0 is the rail (meters); joints 1..=6
/// are the arm (radians).
fn rail_plus_arm() -> Kinematics<N, f64> {
    let arm: Kinematics<6, f64> = Kinematics::from_dh(
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
        KinLimits::symmetric(10.0),
        &[],
    );

    let arm_spec = arm.structure();
    let joints: [(DAffine3, JointSpec<f64>); N] = std::array::from_fn(|i| {
        if i == 0 {
            (
                DAffine3::IDENTITY,
                JointSpec::Prismatic {
                    axis_local: DVec3::X,
                },
            )
        } else {
            arm_spec.joints[i - 1]
        }
    });
    let spec = KinSpec::new(arm_spec.base_to_first, joints, arm_spec.end_to_ee);

    let lower = SRobotQ::<N, f64>::from_array([0.0, -PI, -PI, -PI, -PI, -PI, -PI]);
    let upper = SRobotQ::<N, f64>::from_array([RAIL_TRAVEL, PI, PI, PI, PI, PI, PI]);
    Kinematics::from_kinspec(spec, KinLimits::new(lower, upper), &[])
}

#[test]
fn rrt_path_retimes_on_six_axis_arm_with_linear_rail() {
    let fk = rail_plus_arm();

    let lower = SRobotQ::<N, f64>::from_array([0.0, -PI, -PI, -PI, -PI, -PI, -PI]);
    let upper = SRobotQ::<N, f64>::from_array([RAIL_TRAVEL, PI, PI, PI, PI, PI, PI]);

    // A short coordinated move: the rail traverses 0.5 m while the arm reposes.
    // The spline retimer's jerk DFS is worst-case `branch^(time/dt)`, so a
    // multi-second traverse (e.g. the full 10 m rail) is intractable for it;
    // long hauls must be retimed segment-by-segment. This keeps the search
    // shallow while still exercising the rail-as-joint-0 plumbing.
    let start = SRobotQ::<N, f64>::from_array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let goal = SRobotQ::<N, f64>::from_array([0.8, 0.3, -0.25, 0.2, -0.15, 0.1, 0.2]);

    let validator = JointValidator::<N, f64>::new(lower, upper);
    let ctx = ();

    let mut settings = RrtcSettings::<N>::new(lower, upper);
    // The rail spans 10 m while the arm joints span a few radians. Down-weight
    // the rail in the planner's distance metric so a meter of rail travel is
    // commensurate with ~a radian of joint motion; left at 1.0 the rail would
    // dominate nearest-neighbor and steering.
    settings.dof_cost_weights = SRobotQ::<N, f64>::from_array([0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    settings.seed = 7;

    let waypoints = StartEnd { start, end: goal };
    let planner = RrtcPlanner::<N>::new();
    let (path_res, diag) = planner.plan::<DekeError, _>(&settings, &waypoints, &validator, &ctx);
    let path = path_res.unwrap_or_else(|e| panic!("RRT failed for 7-DOF rail+arm: {e} ({diag})"));

    assert!(path.len() >= 2, "degenerate path: {} waypoints", path.len());
    assert!(
        path.first().distance(&start) < 1e-9,
        "path does not start at start"
    );
    assert!(
        path.last().distance(&goal) < 1e-9,
        "path does not end at goal"
    );

    let (rail_lo, rail_hi) = path
        .iter()
        .map(|q| q.0[0])
        .fold((f64::MAX, f64::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
    assert!(
        rail_hi - rail_lo > 0.4,
        "rail barely moved across the plan: span [{rail_lo}, {rail_hi}]"
    );
    assert!(
        rail_lo >= -1e-9 && rail_hi <= RAIL_TRAVEL + 1e-9,
        "rail left its 10 m travel: [{rail_lo}, {rail_hi}]"
    );

    let rail_v = 3.0;
    let arm_v = 1.5;
    let joint = JointLimits::<N> {
        v_max: SRobotQ::from_array([rail_v, arm_v, arm_v, arm_v, arm_v, arm_v, arm_v]),
        a_max: SRobotQ::from_array([10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
        j_max: SRobotQ::from_array([200.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]),
    };
    let cfg = Topp3TcpSplineConstraints::<N> {
        joint,
        tcp: TcpLimits::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        path: SplinePathOptions {
            max_deviation: 1e-2,
            max_refine_iters: 8,
            start_direction: None,
            end_direction: None,
        },
        search: SearchOptions {
            dt: 0.05,
            verify_dt: 0.05,
            output_dt: None,
            jerk_smoothing_passes: 0,
            fd_safety_slack: 0.05,
            max_jerk_jump: None,
            start_sdot: 0.0,
            end_sdot: 0.0,
            max_sdot: 10.0,
        },
    };

    let (retime_res, rdiag) = Topp3TcpSpline::new(&fk).retime(&cfg, &path, &validator, &ctx);
    let traj =
        retime_res.unwrap_or_else(|e| panic!("retimer failed on rail+arm path: {e} ({rdiag})"));

    assert!(
        traj.len() >= 4,
        "trajectory too short: {} samples",
        traj.len()
    );
    assert!(
        traj.first().distance(&start) < 1e-3,
        "trajectory drifts from start"
    );
    assert!(
        traj.last().distance(&goal) < 1e-3,
        "trajectory drifts from goal"
    );

    // The retimer must respect each joint's own velocity bound, the rail's 2 m/s
    // included. A finite-difference check on the emitted samples; the internal
    // search already runs with `fd_safety_slack`, so the small tolerance here is
    // for sampling curvature, not constraint violation.
    for i in 0..traj.len() {
        let Some(v) = traj.velocity_at(i) else {
            continue;
        };
        for j in 0..N {
            let lim = cfg.joint.v_max.0[j];
            assert!(
                v.0[j].abs() <= lim * 1.05 + 1e-6,
                "joint {j} velocity {} exceeds limit {lim} at sample {i}",
                v.0[j]
            );
        }
    }
}
