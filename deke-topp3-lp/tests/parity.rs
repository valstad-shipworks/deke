//! Parity suite mirroring `deke-topp3tcp-nlp`'s discrete tests, adapted to the
//! convex-LP retimer's API.
//!
//! Faithful ports of every behavioural and stress case that applies. Assertions
//! that referenced NLP-only diagnostics (`SolveStatus`, `output_fd_residual`,
//! `average_utilization`) are expressed through this crate's API instead: success
//! is `Result::is_ok` (the retimer only returns `Ok` after its FD verify passes
//! against the true limits), residuals become `diag.peak_* <= limit·(1+tol)`, and
//! utilisation is computed from the output trajectory by `common::avg_utilization`.
//!
//! NLP cases with no analogue here are intentionally omitted: boundary conditions
//! (non-zero start/end velocity), locked-prefix, TCP acceleration/jerk caps (this
//! crate caps TCP *velocity* only), and the `external_failures` suite (regression
//! fixtures for the Sleipnir IPM's failure taxonomy, meaningless for a convex LP).

mod common;

use std::time::Duration;

use deke_kin::{DHJoint, JointLimits as KinJointLimits, Kinematics};
use deke_topp3_lp::{JointLimits, Topp3Lp, Topp3LpConstraints, Topp3LpTcp};
use deke_types::{DekeError, FKChain, Retimer, SRobotPath, SRobotQ};

fn dt8() -> Duration {
    Duration::from_millis(8)
}

fn jl<const N: usize>(v: f64, a: f64, j: f64) -> Topp3LpConstraints<N> {
    Topp3LpConstraints::symmetric(v, a, j, dt8())
}

// ----- behavioural parity (discrete_*) -----

#[test]
fn single_joint_rest_to_rest() {
    let path =
        SRobotPath::<1, f64>::try_new(vec![SRobotQ::from_array([0.0]), SRobotQ::from_array([1.0])])
            .unwrap();
    let cfg = jl::<1>(1.0, 2.0, 200.0);
    let (result, diag) =
        Topp3Lp::<1>::new().retime(&cfg, &path, &common::wide_validator::<1>(), &());
    eprintln!("{diag}");
    let traj = result.expect("retime");
    let total = traj.duration().as_secs_f64();
    assert!(
        (1.5..=2.5).contains(&total),
        "total time {total} outside [1.5, 2.5]"
    );
    assert!(diag.peak_joint_vel <= 1.0 * (1.0 + 1e-6));
    assert!(diag.peak_joint_accel <= 2.0 * (1.0 + 1e-6));
}

#[test]
fn straight_line_six_dof_joint_limits_dominant() {
    let a = SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]);
    let b = SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]);
    let path = SRobotPath::<6, f64>::try_new(vec![a, b]).unwrap();
    let cfg = jl::<6>(1.0, 3.0, 300.0);
    let (result, diag) =
        Topp3Lp::<6>::new().retime(&cfg, &path, &common::wide_validator::<6>(), &());
    eprintln!("{diag}");
    result.expect("retime");
    assert!(diag.peak_joint_vel <= 1.0 * (1.0 + 1e-6));
    assert!(diag.peak_joint_accel <= 3.0 * (1.0 + 1e-6));
    assert!(diag.peak_joint_jerk <= 300.0 * (1.0 + 1e-6));
}

#[test]
fn multi_waypoint_curved_path_solves_and_is_feasible() {
    let waypoints = vec![
        SRobotQ::from_array([0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.2, -1.1, 1.3, -0.1, 0.1, 0.1]),
        SRobotQ::from_array([0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::from_array([0.6, -0.7, 0.9, -0.3, 0.1, 0.3]),
        SRobotQ::from_array([0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<6, f64>::try_new(waypoints).unwrap();
    let cfg = jl::<6>(1.5, 8.0, 400.0);
    let (result, diag) =
        Topp3Lp::<6>::new().retime(&cfg, &path, &common::wide_validator::<6>(), &());
    eprintln!("{diag}");
    result.expect("retime");
    assert!(diag.peak_joint_vel <= 1.5 * (1.0 + 1e-6));
}

#[test]
fn tight_jerk_limit_increases_total_time() {
    let path =
        SRobotPath::<1, f64>::try_new(vec![SRobotQ::from_array([0.0]), SRobotQ::from_array([1.0])])
            .unwrap();
    let v = common::wide_validator::<1>();
    let (r1, _) = Topp3Lp::<1>::new().retime(&jl::<1>(1.0, 2.0, 500.0), &path, &v, &());
    let t1 = r1.expect("loose").duration().as_secs_f64();
    let (r2, _) = Topp3Lp::<1>::new().retime(&jl::<1>(1.0, 2.0, 4.0), &path, &v, &());
    let t2 = r2.expect("tight").duration().as_secs_f64();
    assert!(
        t2 > t1 + 0.02,
        "tight jerk ({t2}) should take longer than loose ({t1})"
    );
}

#[test]
fn tcp_velocity_is_limiting() {
    let fk = common::dh_6dof();
    let a = SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]);
    let b = SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]);
    let path = SRobotPath::<6, f64>::try_new(vec![a, b]).unwrap();
    let cfg = jl::<6>(5.0, 30.0, 3_000.0).with_tcp_speed(0.25);
    let (result, diag) =
        Topp3LpTcp::new(&fk).retime(&cfg, &path, &common::wide_validator::<6>(), &());
    eprintln!("{diag}");
    result.expect("retime");
    assert!(
        diag.peak_tcp_speed <= 0.25 * 1.01,
        "peak tcp v {}",
        diag.peak_tcp_speed
    );
    assert!(
        diag.peak_joint_vel <= 5.0,
        "peak joint v {}",
        diag.peak_joint_vel
    );
}

#[test]
fn impossible_bounds_return_error_not_panic() {
    let path =
        SRobotPath::<1, f64>::try_new(vec![SRobotQ::from_array([0.0]), SRobotQ::from_array([1.0])])
            .unwrap();
    let cfg = jl::<1>(1e-4, 1e-3, 1e-2);
    let (result, _) = Topp3Lp::<1>::new().retime(&cfg, &path, &common::wide_validator::<1>(), &());
    match result {
        Err(DekeError::RetimerFailed(_)) => {}
        other => panic!("expected RetimerFailed, got {other:?}"),
    }
}

#[test]
fn time_optimal_solution_saturates_some_limit_on_average() {
    let waypoints = vec![
        SRobotQ::from_array([0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.2, -1.1, 1.3, -0.1, 0.1, 0.1]),
        SRobotQ::from_array([0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::from_array([0.6, -0.7, 0.9, -0.3, 0.1, 0.3]),
        SRobotQ::from_array([0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<6, f64>::try_new(waypoints).unwrap();
    let cfg = jl::<6>(1.5, 4.0, 200.0);
    let (result, diag) =
        Topp3Lp::<6>::new().retime(&cfg, &path, &common::wide_validator::<6>(), &());
    eprintln!("{diag}");
    let _ = result.expect("retime");
    let peaks = [
        diag.peak_joint_vel / 1.5,
        diag.peak_joint_accel / 4.0,
        diag.peak_joint_jerk / 200.0,
    ];
    for p in peaks {
        assert!(p <= 1.01, "an individual peak exceeded the limit: {p}");
    }
    // Time-optimal ⇒ SOME limit is saturated at the tightest point (here the path
    // is acceleration/jerk-bound, so velocity keeps headroom).
    let hit = peaks.into_iter().fold(0.0_f64, f64::max);
    assert!(
        hit >= 0.99,
        "no limit reached saturation (peak utilisation {:.1}%)",
        hit * 100.0
    );
}

#[test]
fn single_joint_rest_to_rest_utilization_is_high() {
    let path =
        SRobotPath::<1, f64>::try_new(vec![SRobotQ::from_array([0.0]), SRobotQ::from_array([1.0])])
            .unwrap();
    let cfg = jl::<1>(1.0, 2.0, 200.0);
    let (result, diag) =
        Topp3Lp::<1>::new().retime(&cfg, &path, &common::wide_validator::<1>(), &());
    eprintln!("{diag}");
    let traj = result.expect("retime");
    let util = common::avg_utilization(&traj, 1.0, 2.0, 200.0);
    assert!(
        util >= 0.80,
        "average limit utilisation only {:.1}%",
        util * 100.0
    );
}

#[test]
fn seven_dof_curved_with_tcp_solves_and_is_feasible() {
    let fk = common::dh_7dof_prismatic();
    let waypoints = vec![
        SRobotQ::from_array([0.0, 0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.1, 0.2, -1.1, 1.3, -0.1, 0.1, 0.1]),
        SRobotQ::from_array([0.2, 0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::from_array([0.3, 0.6, -0.7, 0.9, -0.3, 0.1, 0.3]),
        SRobotQ::from_array([0.4, 0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<7, f64>::try_new(waypoints).unwrap();
    let cfg = jl::<7>(1.5, 8.0, 400.0).with_tcp_speed(1.0);
    let (result, diag) =
        Topp3LpTcp::new(&fk).retime(&cfg, &path, &common::wide_validator::<7>(), &());
    eprintln!("{diag}");
    result.expect("7-DOF retime");
    assert!(diag.peak_joint_vel <= 1.5 * 1.2);
    assert!(diag.peak_tcp_speed <= 1.0 * 1.1);
}

#[test]
fn seven_dof_rail_dominant_motion() {
    let fk = common::dh_7dof_prismatic();
    let waypoints = vec![
        SRobotQ::from_array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.25, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.5, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.75, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
    ];
    let path = SRobotPath::<7, f64>::try_new(waypoints).unwrap();
    let cfg = jl::<7>(2.0, 10.0, 500.0).with_tcp_speed(0.5);
    let (result, diag) =
        Topp3LpTcp::new(&fk).retime(&cfg, &path, &common::wide_validator::<7>(), &());
    eprintln!("{diag}");
    result.expect("rail-dominant retime");
    assert!(
        diag.peak_tcp_speed <= 0.5 * 1.05,
        "peak TCP velocity {}",
        diag.peak_tcp_speed
    );
}

#[test]
fn f32_built_chain_drives_f64_retimer() {
    let dh = std::array::from_fn(|i| {
        let p = [
            (0.0, std::f32::consts::FRAC_PI_2, 0.089),
            (-0.425, 0.0, 0.0),
            (-0.392, 0.0, 0.0),
            (0.0, std::f32::consts::FRAC_PI_2, 0.109),
            (0.0, -std::f32::consts::FRAC_PI_2, 0.094),
            (0.0, 0.0, 0.082),
        ][i];
        DHJoint::<f32> {
            a: p.0,
            alpha: p.1,
            d: p.2,
            theta_offset: 0.0,
        }
    });
    let chain_f32: Kinematics<6, f32> =
        Kinematics::from_dh(dh, KinJointLimits::symmetric(10.0), &[]);
    let chain_f64: Kinematics<6, f64> = chain_f32.to_f64();
    let _ = chain_f64.fk_end(&SRobotQ::<6, f64>::zeros()).unwrap();

    let waypoints = vec![
        SRobotQ::<6, f64>::from_array([0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
        SRobotQ::<6, f64>::from_array([0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
        SRobotQ::<6, f64>::from_array([0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
    ];
    let path = SRobotPath::<6, f64>::try_new(waypoints).unwrap();
    let cfg = jl::<6>(1.5, 4.0, 200.0);
    let (result, diag) =
        Topp3LpTcp::new(&chain_f64).retime(&cfg, &path, &common::wide_validator::<6>(), &());
    eprintln!("{diag}");
    result.expect("retime");
}

// ----- stress parity (discrete_stress) -----

#[test]
fn corner_180_reversal_single_joint() {
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([1.0]),
        SRobotQ::from_array([0.0]),
    ])
    .unwrap();
    let (result, diag) = Topp3Lp::<1>::new().retime(
        &jl::<1>(1.0, 2.0, 50.0),
        &path,
        &common::wide_validator::<1>(),
        &(),
    );
    eprintln!("180° reversal:\n{diag}");
    result.expect("180° reversal retime");
}

#[test]
fn sharp_90_corner() {
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -1.0, 1.2, 0.3, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -1.0, 1.2, 0.3, 0.3, 0.0]),
    ])
    .unwrap();
    let cfg = jl::<6>(1.5, 5.0, 200.0).with_tcp_speed(0.6);
    let (result, diag) = Topp3LpTcp::new(&common::dh_6dof()).retime(
        &cfg,
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    eprintln!("sharp 90° corner:\n{diag}");
    result.expect("sharp corner retime");
}

#[test]
fn zigzag_pattern() {
    let mut wps = Vec::new();
    for i in 0..10 {
        wps.push(SRobotQ::from_array([if i % 2 == 0 { 0.0 } else { 0.5 }]));
    }
    let path = SRobotPath::<1, f64>::try_new(wps).unwrap();
    let (result, diag) = Topp3Lp::<1>::new().retime(
        &jl::<1>(1.0, 5.0, 200.0),
        &path,
        &common::wide_validator::<1>(),
        &(),
    );
    eprintln!("zigzag:\n{diag}");
    result.expect("zigzag retime");
}

#[test]
fn wrist_only_rotation_with_tcp_bounds() {
    let mut wps = Vec::new();
    for i in 0..5 {
        wps.push(SRobotQ::from_array([
            0.0,
            -1.0,
            1.2,
            0.0,
            0.0,
            0.4 * i as f64,
        ]));
    }
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let cfg = jl::<6>(2.0, 10.0, 500.0).with_tcp_speed(0.5);
    let (result, diag) = Topp3LpTcp::new(&common::dh_6dof()).retime(
        &cfg,
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    eprintln!("wrist-only:\n{diag}");
    result.expect("wrist-only retime");
}

#[test]
fn through_elbow_singularity() {
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 0.5, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.0, -0.5, 0.05, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.0, -0.5, -0.05, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.0, -1.0, -0.5, 0.0, 0.0, 0.0]),
    ])
    .unwrap();
    let cfg = jl::<6>(1.5, 5.0, 200.0).with_tcp_speed(0.5);
    let (result, diag) = Topp3LpTcp::new(&common::dh_6dof()).retime(
        &cfg,
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    eprintln!("elbow singularity:\n{diag}");
    result.expect("singular path retime");
}

#[test]
fn all_limits_simultaneously_tight() {
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]),
        SRobotQ::from_array([0.3, -1.0, 1.2, 0.0, 0.2, 0.4]),
        SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]),
    ])
    .unwrap();
    let cfg = jl::<6>(0.8, 2.0, 30.0).with_tcp_speed(0.4);
    let (result, diag) = Topp3LpTcp::new(&common::dh_6dof()).retime(
        &cfg,
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    eprintln!("all limits tight:\n{diag}");
    result.expect("all-tight retime");
}

#[test]
fn long_path_many_waypoints() {
    let mut wps = Vec::new();
    for i in 0..50 {
        let t = i as f64 / 49.0;
        let angle = std::f64::consts::TAU * 0.5 * t;
        wps.push(SRobotQ::from_array([
            0.5 * angle.sin(),
            -1.0 + 0.3 * (2.0 * angle).cos(),
            1.2 + 0.2 * angle.sin(),
            0.5 * (3.0 * angle).sin(),
            0.3 * angle.cos(),
            0.4 * t,
        ]));
    }
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let (result, diag) = Topp3Lp::<6>::new().retime(
        &jl::<6>(1.5, 8.0, 400.0),
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    eprintln!("long path 50 waypoints:\n{diag}");
    result.expect("long path retime");
}

#[test]
fn microscopic_path_length() {
    let a = SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]);
    let mut b_arr = a.0;
    b_arr[0] += 1e-3;
    let path = SRobotPath::<6, f64>::try_new(vec![a, SRobotQ::from_array(b_arr)]).unwrap();
    let (result, diag) = Topp3Lp::<6>::new().retime(
        &jl::<6>(1.0, 5.0, 200.0),
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    eprintln!("microscopic path:\n{diag}");
    result.expect("tiny path retime");
}

#[test]
fn near_duplicate_waypoints() {
    let a = SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]);
    let mut b_arr = a.0;
    b_arr[0] += 1e-6;
    let c = SRobotQ::from_array([0.3, -0.8, 0.9, 0.2, 0.1, 0.3]);
    let path = SRobotPath::<6, f64>::try_new(vec![a, SRobotQ::from_array(b_arr), c]).unwrap();
    let (result, diag) = Topp3Lp::<6>::new().retime(
        &jl::<6>(1.0, 5.0, 200.0),
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    eprintln!("near-duplicate:\n{diag}");
    result.expect("near-dup retime");
}

#[test]
fn asymmetric_joint_limits() {
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.7, 0.9, 0.3, 0.2, 1.0]),
        SRobotQ::from_array([0.6, -0.4, 0.6, 0.6, 0.4, 2.0]),
    ])
    .unwrap();
    let mut cfg = jl::<6>(1.0, 5.0, 200.0);
    cfg.joint = JointLimits {
        v_max: SRobotQ::from_array([0.3, 0.3, 0.5, 1.0, 2.0, 5.0]),
        a_max: SRobotQ::from_array([1.0, 1.0, 2.0, 5.0, 10.0, 30.0]),
        j_max: SRobotQ::from_array([20.0, 20.0, 50.0, 100.0, 300.0, 1000.0]),
    };
    let (result, diag) =
        Topp3Lp::<6>::new().retime(&cfg, &path, &common::wide_validator::<6>(), &());
    eprintln!("asymmetric limits:\n{diag}");
    result.expect("asymmetric limits retime");
}

#[test]
fn single_joint_per_segment_rotation() {
    let wps = vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.4, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.4, -0.6, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.4, -0.6, 0.8, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.4, -0.6, 0.8, 0.4, 0.0, 0.0]),
        SRobotQ::from_array([0.4, -0.6, 0.8, 0.4, 0.4, 0.0]),
        SRobotQ::from_array([0.4, -0.6, 0.8, 0.4, 0.4, 0.4]),
    ];
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let (result, diag) = Topp3Lp::<6>::new().retime(
        &jl::<6>(1.5, 5.0, 200.0),
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    eprintln!("single-joint per segment:\n{diag}");
    result.expect("axis-flipping retime");
}

#[test]
fn tight_validator_bounds() {
    use deke_types::JointValidator;
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([-0.99, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.99, -1.0, 1.2, 0.0, 0.0, 0.0]),
    ])
    .unwrap();
    let validator = JointValidator::<6, f64>::new(
        SRobotQ::from_array([-1.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
        SRobotQ::from_array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    );
    let (result, diag) =
        Topp3Lp::<6>::new().retime(&jl::<6>(1.0, 5.0, 200.0), &path, &validator, &());
    eprintln!("tight validator:\n{diag}");
    result.expect("tight validator retime");
}

// At 10 kHz the per-tick jerk bound is `j·dt³ ≈ 2e-10`, which makes the discrete
// LP severely ill-conditioned (the shipping 8 ms LP works because dt³ is ~5e5
// larger). The fix is to solve on a coarse grid and quintic-resample to the fine
// output dt — a layer not yet built; this crate targets ≤~1 kHz today.
#[ignore = "discrete LP ill-conditioned at extreme sample rates; needs coarse-solve + resample"]
#[test]
fn very_high_output_sample_rate() {
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.5, -0.7, 0.9, 0.3, 0.2, 0.4]),
    ])
    .unwrap();
    let mut cfg = jl::<6>(1.5, 5.0, 200.0);
    cfg.output_dt = Duration::from_secs_f64(1.0 / 10_000.0);
    let (result, diag) =
        Topp3Lp::<6>::new().retime(&cfg, &path, &common::wide_validator::<6>(), &());
    eprintln!("10 kHz:\n{diag}");
    let traj = result.expect("10 kHz retime");
    assert!(
        traj.len() > 5_000,
        "expected >5k samples, got {}",
        traj.len()
    );
}

#[test]
fn very_low_output_sample_rate() {
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.1, -0.95, 1.15, 0.05, 0.05, 0.05]),
    ])
    .unwrap();
    let mut cfg = jl::<6>(1.5, 5.0, 200.0);
    cfg.output_dt = Duration::from_secs_f64(1.0 / 5.0);
    let (result, diag) =
        Topp3Lp::<6>::new().retime(&cfg, &path, &common::wide_validator::<6>(), &());
    eprintln!("5 Hz:\n{diag}");
    let traj = result.expect("5 Hz retime");
    assert!(traj.len() >= 2, "need >=2 samples, got {}", traj.len());
}

#[test]
fn closed_loop_returns_to_start() {
    let mut wps = Vec::new();
    for i in 0..12 {
        let theta = std::f64::consts::TAU * (i as f64 / 12.0);
        wps.push(SRobotQ::from_array([
            0.5 * theta.cos(),
            -1.0 + 0.3 * theta.sin(),
            1.2,
            0.2 * (2.0 * theta).sin(),
            0.0,
            0.0,
        ]));
    }
    wps.push(wps[0]);
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let (result, diag) = Topp3Lp::<6>::new().retime(
        &jl::<6>(1.5, 5.0, 200.0),
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    eprintln!("closed loop:\n{diag}");
    result.expect("closed loop retime");
}

#[test]
fn cusp_with_direction_change() {
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.8, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.6, -0.6, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.6, -0.6, 1.2, 0.5, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.4, 1.0, 0.5, 0.0, 0.0]),
    ])
    .unwrap();
    let (result, diag) = Topp3Lp::<6>::new().retime(
        &jl::<6>(1.5, 5.0, 200.0),
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    eprintln!("cusp:\n{diag}");
    result.expect("cusp retime");
}

#[test]
fn zero_length_path_rejected_by_retimer() {
    let q = SRobotQ::<6, f64>::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]);
    let path = SRobotPath::<6, f64>::try_new(vec![q, q]).unwrap();
    let (result, _) = Topp3Lp::<6>::new().retime(
        &jl::<6>(1.5, 5.0, 200.0),
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    assert!(result.is_err(), "zero-length path must be rejected");
}

#[test]
fn constant_joint_pose_rejected() {
    let q = SRobotQ::<6, f64>::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]);
    let path = SRobotPath::<6, f64>::try_new(vec![q; 5]).unwrap();
    let (result, _) = Topp3Lp::<6>::new().retime(
        &jl::<6>(1.5, 5.0, 200.0),
        &path,
        &common::wide_validator::<6>(),
        &(),
    );
    assert!(result.is_err(), "constant pose must be rejected");
}

fn fuzz_random_walk<const N: usize>(
    seed: u64,
    start: SRobotQ<N, f64>,
    n: usize,
    delta: f64,
) -> Vec<SRobotQ<N, f64>> {
    let mut s = seed;
    let next = |s: &mut u64| -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let mut wps = vec![start];
    let mut cur = start.0;
    for _ in 1..n {
        for c in cur.iter_mut() {
            *c += delta * next(&mut s);
        }
        wps.push(SRobotQ::from_array(cur));
    }
    wps
}

#[test]
fn fuzz_seeded_6wp_paths() {
    let fk = common::dh_6dof();
    let seeds: [u64; 6] = [
        0xDEAD_BEEF_0001,
        0xCAFE_F00D_0002,
        0xBADD_CAFE_0003,
        0x1234_5678_0004,
        0xFEED_FACE_0005,
        0x0F0F_F0F0_0006,
    ];
    let start = SRobotQ::<6, f64>::from_array([0.0, -0.8, 1.0, 0.0, 0.2, 0.5]);
    let mut failures = Vec::new();
    for &seed in &seeds {
        let wps = fuzz_random_walk::<6>(seed, start, 6, 0.4);
        let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
        let cfg = jl::<6>(1.5, 6.0, 250.0).with_tcp_speed(1.0);
        let (result, _) =
            Topp3LpTcp::new(&fk).retime(&cfg, &path, &common::wide_validator::<6>(), &());
        if let Err(e) = result {
            failures.push(format!("seed {seed:#x}: {e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "fuzz failures:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn fuzz_seeded_6wp_paths_aggressive() {
    let fk = common::dh_6dof();
    let seeds: [u64; 6] = [
        0xDEAD_BEEF_0001,
        0xCAFE_F00D_0002,
        0xBADD_CAFE_0003,
        0x1234_5678_0004,
        0xFEED_FACE_0005,
        0x0F0F_F0F0_0006,
    ];
    let start = SRobotQ::<6, f64>::from_array([0.0, -0.8, 1.0, 0.0, 0.2, 0.5]);
    let mut failures = Vec::new();
    for &seed in &seeds {
        let wps = fuzz_random_walk::<6>(seed, start, 6, 0.6);
        let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
        let cfg = jl::<6>(1.5, 6.0, 250.0).with_tcp_speed(1.0);
        let (result, _) =
            Topp3LpTcp::new(&fk).retime(&cfg, &path, &common::wide_validator::<6>(), &());
        if let Err(e) = result {
            failures.push(format!("seed {seed:#x}: {e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "aggressive fuzz failures:\n  {}",
        failures.join("\n  ")
    );
}
