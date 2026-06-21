//! Regression test: short, kinematically-loose paths must succeed.
//!
//! The DFS's `state_could_possibly_reach_target` cutoff was hard-coded to
//! a magic constant (`175.0`) that only made sense at the reference
//! implementation's default jerk limit of ~100. With realistic robot jerk
//! limits (hundreds to thousands of rad/s³), the constant caused the DFS
//! to prune feasible root candidates and exit with `SearchExhausted` even
//! on trivially-feasible inputs. The fix derives the cutoff from the
//! configured joint jerk limit instead.

use std::f64::consts::FRAC_PI_2;

use deke_kin::{DHJoint, JointLimits as KinLimits, Kinematics};
use deke_topp3tcp_spline::{
    JointLimits, SearchOptions, SplinePathOptions, TcpLimits, Topp3TcpSpline,
    Topp3TcpSplineConstraints,
};
use deke_types::{JointValidator, Retimer, SRobotPath, SRobotQ};

fn dh_6dof() -> Kinematics<6, f64> {
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
        KinLimits::symmetric(10.0),
        &[],
    )
}

#[test]
fn straight_short_path_with_large_jerk_limits_solves() {
    let fk = dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.2, -0.1, 0.1, -0.1, 0.0, 0.1]),
    ])
    .unwrap();
    // j_max here is ~4× the reference implementation's hardcoded `175.0`
    // pruning threshold. Pre-fix, the DFS exhausted on this input.
    let cfg = Topp3TcpSplineConstraints::<6> {
        joint: JointLimits::symmetric(1.5, 8.0, 400.0),
        tcp: TcpLimits::new(1.2, f64::INFINITY, f64::INFINITY),
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
    let validator = JointValidator::<6, f64>::new(
        SRobotQ::from_array([-1e9; 6]),
        SRobotQ::from_array([1e9; 6]),
    );
    let (result, diag) = Topp3TcpSpline::new(&fk).retime(&cfg, &path, &validator, &());
    assert!(
        result.is_ok(),
        "spline retimer failed on a feasible short path: {}",
        diag
    );
    let traj = result.unwrap();
    assert!(
        traj.len() >= 4,
        "trajectory too short: {} samples",
        traj.len()
    );
}
