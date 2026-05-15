//! Stress tests aimed at finding failure modes in `Topp3Tcp6`.
//!
//! Each test exercises a deliberately extreme condition: sharp corners, near-singular
//! geometry, all-limits-tight optimization, large sample counts, tiny paths, high-velocity
//! boundaries, mixed length scales, or near-duplicate waypoints. Some of these are
//! expected to expose weaknesses in the current implementation — *that's the point*.
//! When one fails, the diagnostic tells you which group went infeasible and where, and
//! the test becomes a fixture for hardening the retimer.
//!
//! Run with `cargo test --test stress -- --nocapture` to see diagnostics.

mod common;

use deke_topp3tcp6::{
    BoundaryConditions, SolveStatus, TcpLimits, Topp3Tcp6, Topp3Tcp6Constraints,
};
use deke_types::{JointValidator, Retimer, SRobotPath, SRobotQ};

// ----------------------------------------------------------------------------
// Sharp corners and direction reversals
// ----------------------------------------------------------------------------

/// Path bends 180° back on itself. Spline through this is degenerate at the reversal
/// (|pp| → 0 over a short stretch, ppp spikes). Likely failure: TCP a/j infeasible at
/// the reversal sample, or boundary projection succeeds but the NLP can't navigate the
/// near-singular geometry.
#[test]
fn corner_180_reversal_single_joint() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([1.0]),
        SRobotQ::from_array([0.0]),
    ])
    .unwrap();

    let cfg = Topp3Tcp6Constraints::<1>::symmetric(1.0, 2.0, 50.0);
    let mut validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("180° reversal:\n{}", diag);
    assert!(result.is_ok(), "180° reversal retime failed: {}", diag);
}

/// Sharp ~90° corner in joint space with TCP acceleration tight. The natural cubic
/// spline overshoots near corners (M_k inflates) which is exactly where the TCP a
/// constraint LHS lives.
#[test]
fn sharp_90_corner_with_tight_tcp_a() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -1.0, 1.2, 0.3, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -1.0, 1.2, 0.3, 0.3, 0.0]),
    ])
    .unwrap();

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 0.6,
        a_max: 1.5,
        j_max: 50.0,
    });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("sharp 90° corner with tight TCP a:\n{}", diag);
    assert!(result.is_ok(), "sharp corner retime failed: {}", diag);
}

/// Zig-zag with many small reversals. Each reversal is a chance for the spline
/// overshoot to dominate the constraint values.
#[test]
fn zigzag_pattern() {
    let fk = common::dh_1dof();
    let mut wps = Vec::new();
    for i in 0..10 {
        let v = if i % 2 == 0 { 0.0 } else { 0.5 };
        wps.push(SRobotQ::from_array([v]));
    }
    let path = SRobotPath::<1, f64>::try_new(wps).unwrap();

    let cfg = Topp3Tcp6Constraints::<1>::symmetric(1.0, 5.0, 200.0);
    let mut validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("zigzag pattern:\n{}", diag);
    assert!(result.is_ok(), "zigzag retime failed: {}", diag);
}

// ----------------------------------------------------------------------------
// Singular / near-singular geometry
// ----------------------------------------------------------------------------

/// Wrist-only motion: only the last joint changes, so the TCP barely moves and
/// |pp| ≈ 0 across most of the path. Exercises the relative-|pp| cutoff (fix E).
#[test]
fn wrist_only_rotation_with_tcp_bounds() {
    let fk = common::dh_6dof();
    let mut wps = Vec::new();
    for i in 0..5 {
        let theta = 0.4 * i as f64;
        wps.push(SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, theta]));
    }
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(2.0, 10.0, 500.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 0.5,
        a_max: 5.0,
        j_max: 200.0,
    });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("wrist-only with TCP bounds:\n{}", diag);
    assert!(result.is_ok(), "wrist-only retime failed: {}", diag);
}

/// UR5-style "elbow straight" singularity: joint 2 near zero straightens the arm,
/// making the Jacobian rank-deficient — TCP velocity in the radial direction is zero
/// for any joint motion.
#[test]
fn through_elbow_singularity() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 0.5, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.0, -0.5, 0.05, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.0, -0.5, -0.05, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.0, -1.0, -0.5, 0.0, 0.0, 0.0]),
    ])
    .unwrap();

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 0.5,
        a_max: 5.0,
        j_max: 200.0,
    });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("through elbow singularity:\n{}", diag);
    assert!(result.is_ok(), "singular path retime failed: {}", diag);
}

// ----------------------------------------------------------------------------
// All-limits-tight: forces optimization to a true corner
// ----------------------------------------------------------------------------

/// Every per-joint and per-TCP bound is tight relative to the path length. The
/// time-optimal solution lives at the intersection of many active constraints, which
/// is the worst case for IPM line search.
#[test]
fn all_limits_simultaneously_tight() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]),
        SRobotQ::from_array([0.3, -1.0, 1.2, 0.0, 0.2, 0.4]),
        SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]),
    ])
    .unwrap();

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(0.8, 2.0, 30.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 0.4,
        a_max: 2.0,
        j_max: 30.0,
    });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("all limits tight:\n{}", diag);
    assert!(result.is_ok(), "all-tight retime failed: {}", diag);
}

// ----------------------------------------------------------------------------
// Size & scale extremes
// ----------------------------------------------------------------------------

/// Long path: 50 waypoints, will densify to the 200-sample cap. Stress test on
/// problem size and on the iteration budget.
#[test]
fn long_path_many_waypoints() {
    let fk = common::dh_6dof();
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
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 8.0, 400.0);
    // PCHIP zeros the slope at every detected extremum; sin/cos-rich long paths have many
    // pseudo-corners and need a higher IPM budget than the default 1500 to converge.
    cfg.solver.max_iterations = 3000;
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("long path 50 waypoints:\n{}", diag);
    assert!(result.is_ok(), "long path retime failed: {}", diag);
}

/// Tiny path: 1 mm total joint-space distance. After densification, ds is microscopic
/// and the integrator equality dt = ds/sd_avg gives dt near 1e-6, which is the
/// problem's lower bound on dt itself. Tests numerical floor handling.
#[test]
fn microscopic_path_length() {
    let fk = common::dh_6dof();
    let a = SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]);
    let mut b_arr = a.0;
    b_arr[0] += 1e-3;
    let b = SRobotQ::from_array(b_arr);
    let path = SRobotPath::<6, f64>::try_new(vec![a, b]).unwrap();

    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.0, 5.0, 200.0);
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("microscopic path:\n{}", diag);
    assert!(result.is_ok(), "tiny path retime failed: {}", diag);
}

/// Near-duplicate waypoints with one normal-size segment between them. The duplicate
/// check (`DekeError::DuplicateWaypoints`) uses an absolute 1e-9 threshold, so a
/// 1e-6 gap should pass the check but stress chord-length parameterization heavily.
#[test]
fn near_duplicate_waypoints() {
    let fk = common::dh_6dof();
    let a = SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]);
    let mut b_arr = a.0;
    b_arr[0] += 1e-6;
    let b = SRobotQ::from_array(b_arr);
    let c = SRobotQ::from_array([0.3, -0.8, 0.9, 0.2, 0.1, 0.3]);
    let path = SRobotPath::<6, f64>::try_new(vec![a, b, c]).unwrap();

    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.0, 5.0, 200.0);
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("near-duplicate waypoints:\n{}", diag);
    assert!(result.is_ok(), "near-dup retime failed: {}", diag);
}

// ----------------------------------------------------------------------------
// Boundary conditions
// ----------------------------------------------------------------------------

/// Starts and ends at high velocity (90% of v_max). The soft-boundary slack absorbs
/// the projection residual but tests how well the integrator-consistent initial guess
/// handles non-zero start.sdd.
#[test]
fn high_velocity_boundary() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([0.5]),
        SRobotQ::from_array([1.0]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<1>::symmetric(1.0, 5.0, 200.0);
    cfg.boundary = BoundaryConditions {
        v_start: SRobotQ::from_array([0.9]),
        a_start: SRobotQ::from_array([0.0]),
        v_end: SRobotQ::from_array([0.9]),
        a_end: SRobotQ::from_array([0.0]),
        projection_tolerance: 1e-3,
    };
    let mut validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("high-velocity boundary:\n{}", diag);
    assert!(result.is_ok(), "high-v boundary retime failed: {}", diag);
}

/// Non-zero start *acceleration*, requiring the boundary projection to absorb a real
/// `sdd` rather than zero. Pairs well with the high-velocity-boundary test.
#[test]
fn accelerated_start_boundary() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([1.0]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<1>::symmetric(1.0, 5.0, 200.0);
    cfg.boundary = BoundaryConditions {
        v_start: SRobotQ::from_array([0.3]),
        a_start: SRobotQ::from_array([2.0]),
        v_end: SRobotQ::zeros(),
        a_end: SRobotQ::zeros(),
        projection_tolerance: 1e-3,
    };
    let mut validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("accelerated start boundary:\n{}", diag);
    assert!(result.is_ok(), "a_start retime failed: {}", diag);
}

// ----------------------------------------------------------------------------
// Mixed scales
// ----------------------------------------------------------------------------

/// 7-DOF: rail (q[0]) moves 2m while the arm joints rotate by only 0.02 rad. The
/// chord-length parameterization is dominated by the rail, so qp in joint dims is
/// nearly zero in arm dofs — exercises the qp.abs() < 1e-12 cutoff and any
/// per-joint constraint scaling assumptions.
#[test]
fn rail_dominant_with_tiny_arm_motion() {
    let fk = common::dh_7dof_prismatic();
    let path = SRobotPath::<7, f64>::try_new(vec![
        SRobotQ::from_array([0.0, 0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([1.0, 0.005, -1.005, 1.205, 0.005, 0.005, 0.005]),
        SRobotQ::from_array([2.0, 0.01, -1.01, 1.21, 0.01, 0.01, 0.01]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<7>::symmetric(1.0, 5.0, 200.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 1.5,
        a_max: 10.0,
        j_max: 500.0,
    });
    let mut validator = common::wide_validator::<7>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("rail-dominant + tiny arm:\n{}", diag);
    assert!(result.is_ok(), "rail-dominant retime failed: {}", diag);
}

/// Asymmetric joint limits: base joints slow, wrist joints fast. The peak utilization
/// limit varies wildly per joint and the constraint Hessian is more anisotropic than
/// the symmetric-limits case.
#[test]
fn asymmetric_joint_limits() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.7, 0.9, 0.3, 0.2, 1.0]),
        SRobotQ::from_array([0.6, -0.4, 0.6, 0.6, 0.4, 2.0]),
    ])
    .unwrap();

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.0, 5.0, 200.0);
    cfg.joint.v_max = SRobotQ::from_array([0.3, 0.3, 0.5, 1.0, 2.0, 5.0]);
    cfg.joint.a_max = SRobotQ::from_array([1.0, 1.0, 2.0, 5.0, 10.0, 30.0]);
    cfg.joint.j_max = SRobotQ::from_array([20.0, 20.0, 50.0, 100.0, 300.0, 1000.0]);
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("asymmetric joint limits:\n{}", diag);
    assert!(result.is_ok(), "asymmetric limits retime failed: {}", diag);
}

// ----------------------------------------------------------------------------
// TCP / joint coupling extremes
// ----------------------------------------------------------------------------

/// TCP jerk *extremely* tight relative to path. The TCP jerk constraint involves
/// `pppp·sd³ + 3·ppp·sd·sdd + pp·sddd` — with cubic spline pppp = 0 and a tight bound,
/// this becomes a tight constraint on a quadratic-in-vars LHS, which can be hard for
/// the IPM to handle near the boundary.
#[test]
fn extremely_tight_tcp_jerk() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.7, 0.9, 0.3, 0.2, 0.3]),
        SRobotQ::from_array([0.6, -0.4, 0.6, 0.6, 0.4, 0.6]),
    ])
    .unwrap();

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(2.0, 10.0, 1000.0);
    cfg.tcp = Some(TcpLimits {
        v_max: f64::INFINITY,
        a_max: f64::INFINITY,
        j_max: 2.0, // very tight
    });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("extremely tight TCP jerk:\n{}", diag);
    assert!(result.is_ok(), "tight TCP jerk retime failed: {}", diag);
}

/// Path where only one joint moves between every pair of waypoints, but a different
/// joint each time. Each segment has a single-axis tangent — chord lengths are equal
/// but `qp` jumps across the unit sphere from segment to segment, putting all the
/// curvature into the spline at the waypoints.
#[test]
fn single_joint_per_segment_rotation() {
    let fk = common::dh_6dof();
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
    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("single-joint per segment:\n{}", diag);
    assert!(result.is_ok(), "axis-flipping retime failed: {}", diag);
}

/// Tight validator post-validation will reject if any sample is out of bounds. Force a
/// near-bound trajectory and see whether the retimer's output stays inside.
#[test]
fn tight_validator_bounds() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([-0.99, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.99, -1.0, 1.2, 0.0, 0.0, 0.0]),
    ])
    .unwrap();

    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.0, 5.0, 200.0);
    let mut validator = JointValidator::<6, f64>::new(
        SRobotQ::from_array([-1.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
        SRobotQ::from_array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    );
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("tight validator bounds:\n{}", diag);
    assert!(result.is_ok(), "tight validator retime failed: {}", diag);
}

// ----------------------------------------------------------------------------
// Sample at success and report iteration cost
// ----------------------------------------------------------------------------

/// Sanity-test for the test file itself: every successful test above should report
/// `SolveStatus::Success`. This test catches the case where the retimer returns Ok but
/// not Success (e.g. NotAttempted with degenerate input) — which would have slipped
/// past the `result.is_ok()` checks in the asserts.
#[test]
fn smoke_easy_baseline() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.7, 0.9, 0.0, 0.0, 0.0]),
    ])
    .unwrap();
    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("baseline:\n{}", diag);
    assert!(result.is_ok());
    assert_eq!(diag.status, SolveStatus::Success);
}

// ----------------------------------------------------------------------------
// Round 2: edge-case probes
// ----------------------------------------------------------------------------

/// Helical TCP path: joint 0 sweeps a circle at constant radius while joint 2 lifts in z.
/// Smooth, no corners — should converge fast; if it doesn't, PCHIP slope accuracy on
/// curved smooth data is worse than I think.
#[test]
fn helical_tcp_path_with_tight_a() {
    let fk = common::dh_6dof();
    let mut wps = Vec::new();
    for i in 0..16 {
        let t = i as f64 / 15.0;
        let theta = std::f64::consts::TAU * 0.75 * t;
        wps.push(SRobotQ::from_array([
            theta,
            -1.0,
            1.2 + 0.3 * t,
            0.0,
            0.0,
            0.0,
        ]));
    }
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(2.0, 8.0, 400.0);
    cfg.tcp = Some(TcpLimits {
        v_max: 0.4,
        a_max: 1.0,
        j_max: 50.0,
    });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("helical TCP path:\n{}", diag);
    assert!(result.is_ok(), "helical retime failed: {}", diag);
}

/// Very high output sample rate. The retimer integrates a closed-form S-curve over each
/// segment and then re-samples; at 10 kHz output on a 1-second trajectory it produces
/// ~10 000 samples, stressing the resampler's segment-walk and any per-sample work.
#[test]
fn very_high_output_sample_rate() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.5, -0.7, 0.9, 0.3, 0.2, 0.4]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.sample_rate_hz = 10_000.0;
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("10 kHz output:\n{}", diag);
    let traj = result.expect("10 kHz retime failed");
    assert!(traj.len() > 5_000, "expected >5k output samples, got {}", traj.len());
}

/// Very low output sample rate (5 Hz). On a fast trajectory this can produce only a
/// handful of output samples — borderline for the resampler's "n_samples ≥ 1" guard.
#[test]
fn very_low_output_sample_rate() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.1, -0.95, 1.15, 0.05, 0.05, 0.05]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.sample_rate_hz = 5.0;
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("5 Hz output:\n{}", diag);
    let traj = result.expect("5 Hz retime failed");
    assert!(traj.len() >= 2, "need at least 2 output samples, got {}", traj.len());
}

/// Closed-loop path: the joint pose returns to its start. The chord-length total is
/// nonzero, but `start.sd == end.sd == 0` rest-to-rest with the path returning to the
/// same configuration means the densified path crosses itself in joint space.
#[test]
fn closed_loop_returns_to_start() {
    let fk = common::dh_6dof();
    let mut wps = Vec::new();
    for i in 0..12 {
        let t = i as f64 / 12.0;
        let theta = std::f64::consts::TAU * t;
        wps.push(SRobotQ::from_array([
            0.5 * theta.cos(),
            -1.0 + 0.3 * theta.sin(),
            1.2,
            0.2 * (2.0 * theta).sin(),
            0.0,
            0.0,
        ]));
    }
    // Close the loop by appending the first waypoint.
    let first = wps[0];
    wps.push(first);
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    // Closed loops with rest-to-rest BCs are a known failure mode: the retimer treats
    // the path as open and the loop seam (qp[0] vs qp[m-1] for the same physical pose)
    // creates incompatible derivative constraints. Periodic BCs would fix it but are
    // out of scope. Cap iter low so this test fails fast instead of grinding for a
    // minute every CI run.
    cfg.solver.max_iterations = 400;
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("closed loop:\n{}", diag);
    assert!(result.is_ok(), "closed loop retime failed: {}", diag);
}

/// A zero-length path (every waypoint identical). `SRobotPath::try_new` itself only
/// checks `len ≥ 2`; the duplicate-waypoints check happens later in
/// `PathDerivatives::new`, so we route through the retimer and expect a clean
/// `DuplicateWaypoints` error rather than a crash.
#[test]
fn zero_length_path_rejected_by_retimer() {
    let fk = common::dh_6dof();
    let q = SRobotQ::<6, f64>::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]);
    let path = SRobotPath::<6, f64>::try_new(vec![q, q]).unwrap();
    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    let mut validator = common::wide_validator::<6>();
    let (result, _diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    match result {
        Err(deke_types::DekeError::DuplicateWaypoints) => {}
        Err(deke_types::DekeError::PathTooShort(_)) => {}
        other => panic!("expected DuplicateWaypoints or PathTooShort; got {:?}", other),
    }
}

/// Locked prefix equals chain length. The path must be entirely constant; this test
/// exercises whether the retimer handles "nothing to do" gracefully (or errors cleanly)
/// instead of producing a bogus trajectory.
#[test]
fn locked_prefix_equals_chain_length() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.001, -1.0, 1.2, 0.0, 0.0, 0.0]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.0, 5.0, 200.0);
    cfg.locked_prefix = 6; // every joint locked

    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("locked_prefix=N:\n{}", diag);
    // First joint moves 0.001 but is locked; this should be a `LockedPrefixViolation`.
    match result {
        Err(deke_types::DekeError::LockedPrefixViolation { joint: 0, .. }) => {}
        other => panic!(
            "expected LockedPrefixViolation on joint 0 with locked_prefix=N; got {:?}",
            other
        ),
    }
}

/// Densification disabled (`max_segment_step = None`) on a long sparse path. The
/// retimer must use the user's waypoints as-is, which means the integrator-consistent
/// initial guess has to handle large `ds` per segment.
#[test]
fn densification_disabled_long_sparse_path() {
    let fk = common::dh_6dof();
    let mut wps = Vec::new();
    for i in 0..15 {
        let t = i as f64 / 14.0;
        wps.push(SRobotQ::from_array([
            0.5 * t,
            -1.0 + 0.3 * (std::f64::consts::TAU * t).sin(),
            1.2,
            0.0,
            0.0,
            0.0,
        ]));
    }
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.densification.max_segment_step = None;
    cfg.densification.min_samples = 0;
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("densification disabled:\n{}", diag);
    assert!(result.is_ok(), "densification-off retime failed: {}", diag);
}

/// Multi-frequency joint motion: each joint oscillates at a different frequency. Lots
/// of independent extrema across the path; PCHIP fires its "extremum, slope=0" rule
/// at many places per dimension, none of which align across joints. Smaller waypoint
/// count than ideal so the test fails fast (Sleipnir's timeout option doesn't
/// interrupt the restoration phase reliably, so a path that lands the IPM in
/// restoration burns minutes of wall-time before reporting failure).
#[test]
fn multi_frequency_joint_motion() {
    let fk = common::dh_6dof();
    let mut wps = Vec::new();
    let n = 12;
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let phi = std::f64::consts::TAU * t;
        wps.push(SRobotQ::from_array([
            0.4 * (1.0 * phi).sin(),
            -1.0 + 0.2 * (2.0 * phi).cos(),
            1.2 + 0.15 * (3.0 * phi).sin(),
            0.3 * (5.0 * phi).cos(),
            0.2 * (7.0 * phi).sin(),
            0.4 * t,
        ]));
    }
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.solver.max_iterations = 500; // fail fast
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("multi-freq joint motion:\n{}", diag);
    assert!(result.is_ok(), "multi-freq retime failed: {}", diag);
}

/// Cusp: path comes in along one direction, reverses through a single waypoint, then
/// leaves in a *different* direction. Different from a 180° reversal because the
/// outgoing direction isn't the negative incoming.
#[test]
fn cusp_with_direction_change() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.8, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.6, -0.6, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.6, -0.6, 1.2, 0.5, 0.0, 0.0]), // cusp: reverse joint 1+2 trend
        SRobotQ::from_array([0.3, -0.4, 1.0, 0.5, 0.0, 0.0]),
    ])
    .unwrap();
    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("cusp:\n{}", diag);
    assert!(result.is_ok(), "cusp retime failed: {}", diag);
}

/// Tight solver tolerance (1e-10) to see if the IPM can actually achieve it on a
/// non-trivial problem, or whether it loses convergence trying.
#[test]
fn tight_solver_tolerance() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.7, 0.9, 0.2, 0.1, 0.3]),
        SRobotQ::from_array([0.6, -0.4, 0.6, 0.4, 0.2, 0.6]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.solver.tolerance = 1e-10;
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("tight tolerance:\n{}", diag);
    assert!(result.is_ok(), "tight tolerance retime failed: {}", diag);
}

/// Boundary slack disabled (`boundary_slack = 0`). Falls back to hard equalities at
/// rest-to-rest, which is the cone-tip case the slack box was added to avoid. Tests
/// whether the original hard-equality path still works.
#[test]
fn boundary_slack_disabled() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.7, 0.9, 0.2, 0.1, 0.3]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.solver.boundary_slack = 0.0;
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("boundary slack=0:\n{}", diag);
    assert!(result.is_ok(), "hard-boundary retime failed: {}", diag);
}

/// Joint with `v_max = 0`: the joint must literally not move. Test whether the
/// constraint forces sd=0 universally (deadlocking the path) or handles it gracefully
/// when the joint *isn't* changing across the path.
#[test]
fn one_joint_v_max_zero_static_joint() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -1.0, 1.2, 0.2, 0.1, 0.3]), // joint 1 unchanged
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.joint.v_max.0[1] = 0.0; // joint 1 frozen
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("v_max=0 on static joint:\n{}", diag);
    assert!(result.is_ok(), "v_max=0 on static joint retime failed: {}", diag);
}

/// 7-DOF: rail moves in lockstep with joint 1 (perfectly correlated dimensions). The
/// chord-length parameterization sees a single direction in joint space; joints 2..7
/// have qp ≈ 0 throughout. Tests that the redundancy doesn't wedge the NLP.
#[test]
fn seven_dof_perfectly_correlated_dims() {
    let fk = common::dh_7dof_prismatic();
    let mut wps = Vec::new();
    for i in 0..6 {
        let t = i as f64 / 5.0;
        wps.push(SRobotQ::from_array([
            0.5 * t,
            -1.0 + 0.5 * t, // joint 1 moves with rail
            1.2,
            0.0,
            0.0,
            0.0,
            0.0,
        ]));
    }
    let path = SRobotPath::<7, f64>::try_new(wps).unwrap();
    let cfg = Topp3Tcp6Constraints::<7>::symmetric(1.5, 5.0, 200.0);
    let mut validator = common::wide_validator::<7>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("7-DOF correlated:\n{}", diag);
    assert!(result.is_ok(), "7-DOF correlated retime failed: {}", diag);
}

/// Boundary velocity exactly at `v_max`. The joint v constraint `qp·sd ≤ v_max`
/// should be active at sample 0 from the start; combined with hard `sd[0] = start.sd`
/// (or the soft slack), this might create a degenerate dual.
#[test]
fn boundary_velocity_at_v_max_saturation() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([0.5]),
        SRobotQ::from_array([1.0]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<1>::symmetric(1.0, 5.0, 200.0);
    cfg.boundary = BoundaryConditions {
        v_start: SRobotQ::from_array([1.0]), // exactly v_max
        a_start: SRobotQ::zeros(),
        v_end: SRobotQ::from_array([1.0]),
        a_end: SRobotQ::zeros(),
        projection_tolerance: 1e-3,
    };
    let mut validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("v_max saturation at boundary:\n{}", diag);
    assert!(result.is_ok(), "v_max-saturation retime failed: {}", diag);
}

/// A path with multiple "near-duplicate" gaps interleaved with normal segments. After
/// the merging pre-pass these should collapse, but if a single near-dup slips through
/// the integrator equality goes wonky. Tests merging robustness.
#[test]
fn many_near_duplicates_interleaved() {
    let fk = common::dh_6dof();
    let base = [
        [0.0, -1.0, 1.2, 0.0, 0.0, 0.0],
        [0.0, -1.0, 1.2, 0.0, 0.0, 0.0], // dup of 0
        [0.3, -0.7, 0.9, 0.0, 0.0, 0.0],
        [0.3, -0.7, 0.9, 0.0, 0.0, 0.0], // dup of 2
        [0.5, -0.5, 0.7, 0.0, 0.0, 0.0],
        [0.5, -0.5, 0.7, 1e-7, 0.0, 0.0], // tiny gap
        [0.7, -0.3, 0.5, 0.2, 0.1, 0.2],
    ];
    // Add tiny perturbations so the path constructor accepts adjacent pairs (it rejects
    // exact duplicates).
    let wps: Vec<_> = base
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let mut a = *q;
            a[5] += i as f64 * 1e-9;
            SRobotQ::from_array(a)
        })
        .collect();
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.solver.max_iterations = 500; // fail fast
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("interleaved near-dups:\n{}", diag);
    assert!(result.is_ok(), "interleaved-dup retime failed: {}", diag);
}

/// Constant joint pose — every waypoint is identical. `SRobotPath::try_new` accepts it,
/// but the retimer should reject in `PathDerivatives::new`.
#[test]
fn constant_joint_pose_rejected() {
    let fk = common::dh_6dof();
    let q = SRobotQ::<6, f64>::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]);
    let path = SRobotPath::<6, f64>::try_new(vec![q; 5]).unwrap();
    let cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    let mut validator = common::wide_validator::<6>();
    let (result, _diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    match result {
        Err(deke_types::DekeError::DuplicateWaypoints) => {}
        other => panic!("expected DuplicateWaypoints; got {:?}", other),
    }
}

/// Boundary residual borderline — projection produces a residual exactly at the user's
/// tolerance. Tests whether `>` vs `>=` in the threshold check matters.
#[test]
fn boundary_residual_at_tolerance() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([1.0]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<1>::symmetric(1.0, 5.0, 200.0);
    cfg.boundary = BoundaryConditions {
        v_start: SRobotQ::from_array([0.5]),
        a_start: SRobotQ::zeros(),
        v_end: SRobotQ::zeros(),
        a_end: SRobotQ::zeros(),
        projection_tolerance: 1e-12, // essentially zero — any FP noise breaks it
    };
    let mut validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("boundary tol borderline:\n{}", diag);
    // No assertion on success here — this test is exploratory. The point is to see
    // whether the projection check rejects FP noise that the user clearly did not
    // intend to forbid.
    let _ = result;
}

/// Path with a sample where qp is *not* fully zero but *is* tiny in every dimension —
/// the relative threshold should consistently classify it. Tests the boundary of the
/// degenerate-qp detector.
#[test]
fn slow_section_with_tiny_qp() {
    let fk = common::dh_6dof();
    // Build a path where samples 5..10 are all within 1e-5 of each other (after
    // densification), and the rest are normal.
    let mut wps = Vec::new();
    wps.push(SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]));
    wps.push(SRobotQ::from_array([0.3, -1.0, 1.2, 0.0, 0.0, 0.0]));
    // A quasi-stationary segment.
    for i in 0..3 {
        wps.push(SRobotQ::from_array([
            0.3 + (i as f64 + 1.0) * 1e-4,
            -1.0,
            1.2,
            0.0,
            0.0,
            0.0,
        ]));
    }
    wps.push(SRobotQ::from_array([0.6, -0.7, 0.9, 0.0, 0.0, 0.0]));
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.solver.max_iterations = 500; // fail fast
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("slow tiny-qp section:\n{}", diag);
    assert!(result.is_ok(), "slow-section retime failed: {}", diag);
}

// ----------------------------------------------------------------------------
// Round 3: targeted stress for known weak spots (2026-05-14)
//
// Failure modes that have been observed in production and reduced to fixtures here, plus
// a small fuzz harness that generates fixed-seed random paths and asserts they all
// retime cleanly. Each test exercises a specific mechanism rather than reusing a captured
// trajectory — fast to triage when one breaks.
// ----------------------------------------------------------------------------
//
// Background on the PCHIP spike: the densifier subdivides every input segment uniformly,
// so densified samples within an input segment are colinear and PCHIP secants are
// constant *within* a segment. At each input waypoint, the slope value transitions from
// the left-segment secant to the right-segment secant, and the spline's 2nd/3rd
// derivatives spike at the adjacent densified samples (∝ secant-difference / h²). When
// adjacent input segments have very different lengths *and* very different secant
// magnitudes, the weighted harmonic mean PCHIP uses at the knot biases hard toward the
// smaller-magnitude secant — making the qppp spike at the immediately-following sample
// the dominant entry of the constraint Hessian. The fix is `two_stage_solve`'s
// stage-2-failure fallback to single-stage (see retimer.rs).

/// 4-waypoint joint path with input-segment length ratio ~10:1 and dramatically
/// different secant directions across the kink. Exactly the failure pattern we fixed:
/// PCHIP yields qppp ≈ 300 at one of the densified samples adjacent to wp1. If the
/// two-stage fallback is intact, this converges; if it regresses, the IPM bails with
/// `LocallyInfeasible` at ~100 iter.
#[test]
fn pchip_spike_uneven_segments_4wp() {
    let fk = common::dh_6dof();
    // Segment 0 is 10× longer than segment 1; segment 2 matches segment 1.
    let wps = vec![
        SRobotQ::from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([1.0, 0.3, -0.5, -0.1, 1.0, 1.5]),
        SRobotQ::from_array([1.02, 0.28, -0.55, -0.2, 1.1, 1.7]),
        SRobotQ::from_array([1.04, 0.26, -0.6, -0.3, 1.2, 1.9]),
    ];
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(2.0, 8.0, 300.0);
    cfg.tcp = Some(TcpLimits { v_max: 1.5, a_max: 12.0, j_max: 150.0 });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("PCHIP spike uneven 4wp:\n{}", diag);
    assert!(result.is_ok(), "uneven-seg 4wp retime failed: {}", diag);
}

/// 6-waypoint path where every adjacent-segment ratio is extreme (each segment about
/// half the previous). Stacks multiple qppp spikes within one path so the IPM Hessian
/// has several near-singular rows simultaneously.
#[test]
fn pchip_multi_knot_geometric_segment_ratios() {
    let fk = common::dh_6dof();
    // Joint deltas chosen so each segment's joint-space length is ~half the previous.
    let wps = vec![
        SRobotQ::from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.8, 0.4, -0.4, -0.2, 0.5, 0.7]),
        SRobotQ::from_array([1.2, 0.6, -0.6, -0.3, 0.75, 1.05]),
        SRobotQ::from_array([1.4, 0.7, -0.7, -0.35, 0.875, 1.225]),
        SRobotQ::from_array([1.5, 0.75, -0.75, -0.375, 0.9375, 1.3125]),
        SRobotQ::from_array([1.55, 0.775, -0.775, -0.3875, 0.96875, 1.35625]),
    ];
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(2.0, 8.0, 300.0);
    cfg.tcp = Some(TcpLimits { v_max: 1.5, a_max: 15.0, j_max: 200.0 });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("multi-knot geometric ratios:\n{}", diag);
    assert!(result.is_ok(), "geometric-ratio retime failed: {}", diag);
}

/// One joint changes direction sharply at the second waypoint while another joint
/// continues monotonically. PCHIP's small-flip detector decides per-joint; the
/// joint that flips gets centered FD slope, the joint that doesn't gets the
/// harmonic mean. The two co-located transitions sometimes interfere.
#[test]
fn mixed_per_joint_flip_versus_monotone() {
    let fk = common::dh_6dof();
    let wps = vec![
        // j0 monotonically increases; j5 reverses sharply at wp1.
        SRobotQ::from_array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        SRobotQ::from_array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0]),
        SRobotQ::from_array([1.0, 0.0, 0.0, 0.0, 0.0, -0.5]),
        SRobotQ::from_array([1.5, 0.0, 0.0, 0.0, 0.0, 0.8]),
    ];
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 6.0, 250.0);
    cfg.tcp = Some(TcpLimits { v_max: 1.0, a_max: 8.0, j_max: 100.0 });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("mixed flip vs monotone:\n{}", diag);
    assert!(result.is_ok(), "mixed-flip retime failed: {}", diag);
}

/// One joint's secants flip sign with magnitudes that differ by 100×. Forces PCHIP's
/// harmonic-mean rule into the regime where one secant dominates the slope at the
/// knot, biasing the spline derivative heavily toward zero on one side.
#[test]
fn extreme_secant_magnitude_ratio_at_knot() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([0.01]), // ~tiny first segment
        SRobotQ::from_array([1.0]),  // long second segment
        SRobotQ::from_array([1.005]), // tiny third segment
    ])
    .unwrap();
    let cfg = Topp3Tcp6Constraints::<1>::symmetric(1.0, 5.0, 200.0);
    let mut validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("extreme secant ratio:\n{}", diag);
    assert!(result.is_ok(), "extreme-ratio retime failed: {}", diag);
}

/// Path crosses near a wrist-singular configuration where joints 4 and 5 align so the
/// last two joint axes are parallel. TCP velocity in the wrist-roll direction has a
/// nullspace there — PCHIP yields |pp| → 0 at the singular sample.
#[test]
fn wrist_alignment_singularity_traverse() {
    let fk = common::dh_6dof();
    let wps = vec![
        SRobotQ::from_array([0.0, -1.0, 1.2,  0.4,  0.4, 0.0]),
        SRobotQ::from_array([0.1, -1.0, 1.2,  0.2,  0.2, 0.0]),
        SRobotQ::from_array([0.2, -1.0, 1.2,  0.0,  0.0, 0.0]), // wrist aligned
        SRobotQ::from_array([0.3, -1.0, 1.2, -0.2, -0.2, 0.0]),
        SRobotQ::from_array([0.4, -1.0, 1.2, -0.4, -0.4, 0.0]),
    ];
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 6.0, 200.0);
    cfg.tcp = Some(TcpLimits { v_max: 0.3, a_max: 3.0, j_max: 60.0 });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("wrist alignment traverse:\n{}", diag);
    assert!(result.is_ok(), "wrist-alignment retime failed: {}", diag);
}

/// Long path (24 wp) made of three distinct curvature regimes glued together: smooth
/// arc, sharp zigzag, smooth arc again. Stresses two-stage warm-start because stage-1's
/// (sd, sdd, sddd) profile fits the smooth ends but is the wrong shape across the
/// zigzag middle — exactly the kind of warm-start that pre-fix tripped stage 2.
#[test]
fn long_path_mixed_smooth_zigzag_smooth() {
    let fk = common::dh_6dof();
    let mut wps = Vec::new();
    // First arc: 8 wp smooth.
    for i in 0..8 {
        let t = i as f64 / 7.0;
        let theta = std::f64::consts::FRAC_PI_2 * t;
        wps.push(SRobotQ::from_array([
            0.3 * theta.sin(),
            -1.0 + 0.2 * t,
            1.2,
            0.0,
            0.0,
            0.0,
        ]));
    }
    // Zigzag middle: 8 wp small amplitude.
    let mid = *wps.last().unwrap();
    for i in 1..=8 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let mut q = mid.0;
        q[0] += 0.05 * i as f64;
        q[3] += 0.15 * sign;
        wps.push(SRobotQ::from_array(q));
    }
    // Second arc: 8 wp smooth.
    let mid2 = *wps.last().unwrap();
    for i in 1..=8 {
        let t = i as f64 / 8.0;
        let theta = std::f64::consts::FRAC_PI_2 * t;
        let mut q = mid2.0;
        q[0] += 0.3 * theta.sin();
        q[4] += 0.2 * t;
        wps.push(SRobotQ::from_array(q));
    }
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 6.0, 250.0);
    cfg.tcp = Some(TcpLimits { v_max: 0.5, a_max: 4.0, j_max: 80.0 });
    cfg.solver.max_iterations = 1500;
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("smooth+zigzag+smooth:\n{}", diag);
    assert!(result.is_ok(), "mixed-regime retime failed: {}", diag);
}

/// Final segment is ~100× shorter than the rest of the path. The retimer's boundary
/// slack absorbs `sdd` mismatch; a microscopic final segment compresses the slack
/// budget into a single tiny segment, forcing `sddd` on that segment to balloon.
#[test]
fn microscopic_final_segment_endpoint_squeeze() {
    let fk = common::dh_6dof();
    let wps = vec![
        SRobotQ::from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.4, 0.3, -0.3, 0.2, 0.2, 0.5]),
        SRobotQ::from_array([0.8, 0.6, -0.6, 0.4, 0.4, 1.0]),
        // Final segment ~1e-3 in joint distance vs 1.0 chord above.
        SRobotQ::from_array([0.8005, 0.6005, -0.6005, 0.4005, 0.4005, 1.0005]),
    ];
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 6.0, 250.0);
    cfg.tcp = Some(TcpLimits { v_max: 1.0, a_max: 6.0, j_max: 100.0 });
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("microscopic final segment:\n{}", diag);
    assert!(result.is_ok(), "microscopic-final retime failed: {}", diag);
}

/// Locked first three joints, curved motion in last three. After densification the
/// path is curved only in wrist DOFs — joint v/a/j constraints only apply to wrist
/// joints, but the boundary projection and integrator are full-rank. Tests the
/// `locked_prefix` skip-loop interaction with the qp-degeneracy detector.
#[test]
fn locked_base_curved_wrist() {
    let fk = common::dh_6dof();
    let wps = vec![
        SRobotQ::from_array([0.5, -0.8, 1.0,  0.0,  0.0,  0.0]),
        SRobotQ::from_array([0.5, -0.8, 1.0,  0.3,  0.5,  0.5]),
        SRobotQ::from_array([0.5, -0.8, 1.0,  0.6,  0.8,  1.2]),
        SRobotQ::from_array([0.5, -0.8, 1.0,  0.5,  1.0,  1.8]),
        SRobotQ::from_array([0.5, -0.8, 1.0,  0.2,  1.1,  2.4]),
    ];
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(2.0, 10.0, 400.0);
    cfg.locked_prefix = 3;
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("locked base + curved wrist:\n{}", diag);
    assert!(result.is_ok(), "locked-base retime failed: {}", diag);
}

/// Boundary projection produces a residual just under the user's tolerance ceiling.
/// The user requests start velocity nearly aligned with the chord, with a small
/// perpendicular component the projection has to absorb. Combined with TCP rows this
/// is a stress on the slack interplay near the soft-equality cone tip.
#[test]
fn boundary_residual_near_tolerance_with_tcp() {
    let fk = common::dh_6dof();
    let path = SRobotPath::<6, f64>::try_new(vec![
        SRobotQ::from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.6, 0.3, -0.3, 0.2, 0.2, 0.5]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 6.0, 200.0);
    cfg.tcp = Some(TcpLimits { v_max: 1.0, a_max: 5.0, j_max: 80.0 });
    // Chord direction is dominated by joint 0; v_start chosen to project cleanly onto it
    // with only a small (~5e-3) perpendicular residual against a tolerance of 1e-2.
    cfg.boundary = BoundaryConditions {
        v_start: SRobotQ::from_array([0.2, 0.1, -0.1, 0.066, 0.066, 0.166]),
        a_start: SRobotQ::zeros(),
        v_end: SRobotQ::zeros(),
        a_end: SRobotQ::zeros(),
        projection_tolerance: 1e-2,
    };
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("boundary residual near tol with TCP:\n{}", diag);
    assert!(result.is_ok(), "boundary-near-tol retime failed: {}", diag);
}

/// TCP a_max is so tight (relative to the joint-space curvature) that the SOC rows are
/// near-active across the entire path. Forces the IPM to walk along the cone boundary
/// at every sample — a known IPM weak spot.
#[test]
fn nearly_active_tcp_a_everywhere() {
    let fk = common::dh_6dof();
    let mut wps = Vec::new();
    for i in 0..12 {
        let t = i as f64 / 11.0;
        let theta = std::f64::consts::TAU * 0.4 * t;
        wps.push(SRobotQ::from_array([
            0.3 * theta.sin(),
            -1.0 + 0.15 * theta.cos(),
            1.2,
            0.2 * (2.0 * theta).cos(),
            0.0,
            0.0,
        ]));
    }
    let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(2.0, 10.0, 500.0);
    // Curvature-times-cruise-squared at peak ≈ 0.5; pick a_max just above that so the
    // optimizer wants to ride against the cone almost everywhere.
    cfg.tcp = Some(TcpLimits { v_max: f64::INFINITY, a_max: 0.6, j_max: f64::INFINITY });
    cfg.solver.max_iterations = 1500;
    let mut validator = common::wide_validator::<6>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("nearly-active TCP a everywhere:\n{}", diag);
    assert!(result.is_ok(), "near-active-cone retime failed: {}", diag);
}

/// Helper for the fuzz tests: deterministic xorshift64* random walk from a start
/// pose. Produces `n` random waypoints with each joint stepping by up to
/// `±delta` per step.
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
        ((*s as f64) / (u64::MAX as f64)) * 2.0 - 1.0
    };
    let mut wps = vec![start];
    let mut cur = start.0;
    for _ in 1..n {
        for j in 0..N {
            cur[j] += delta * next(&mut s);
        }
        wps.push(SRobotQ::from_array(cur));
    }
    wps
}

/// Pseudo-random fuzz: 6 distinct seeded paths, each 6 waypoints, joint deltas up to
/// `±0.4` per step from a fixed start pose. Asserts every one retimes cleanly. Fixed
/// seeds ensure reproducibility; the conservative delta keeps every path
/// physically-reasonable so success is achievable across all seeds. The companion
/// `fuzz_seeded_6wp_paths_aggressive` runs with a larger delta where some failures
/// are expected.
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
    for (i, &seed) in seeds.iter().enumerate() {
        let wps = fuzz_random_walk::<6>(seed, start, 6, 0.4);
        let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
        let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 6.0, 250.0);
        cfg.tcp = Some(TcpLimits { v_max: 1.0, a_max: 8.0, j_max: 100.0 });
        cfg.solver.max_iterations = 1500;
        let mut validator = common::wide_validator::<6>();
        let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
        eprintln!("fuzz seed {:#x} (#{}): status={:?}", seed, i, diag.status);
        if let Err(e) = result {
            failures.push(format!("seed {:#x}: {}", seed, e));
        }
    }
    assert!(failures.is_empty(), "fuzz failures:\n  {}", failures.join("\n  "));
}

/// Aggressive fuzz: same 6 seeds, delta `±0.6` (close to joint v_max in a unit-time
/// step) which produces sharp direction changes and uneven secant magnitudes that the
/// PCHIP machinery struggles with. The tolerance-relaxation retry in `Topp3Tcp6::retime`
/// catches the cases where the IPM's KKT factorization can't squeeze the last digit at
/// the user's tight tolerance — all 6 now succeed. Asserts all-pass; a regression that
/// drops any one will fail CI immediately.
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
    let mut failures: Vec<String> = Vec::new();
    for &seed in &seeds {
        let wps = fuzz_random_walk::<6>(seed, start, 6, 0.6);
        let path = SRobotPath::<6, f64>::try_new(wps).unwrap();
        let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 6.0, 250.0);
        cfg.tcp = Some(TcpLimits { v_max: 1.0, a_max: 8.0, j_max: 100.0 });
        cfg.solver.max_iterations = 1500;
        let mut validator = common::wide_validator::<6>();
        let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
        eprintln!(
            "aggressive fuzz seed {:#x}: status={:?} tol_used={:.0e}",
            seed, diag.status, diag.solver_tolerance_used,
        );
        if let Err(e) = result {
            failures.push(format!("seed {:#x}: {}", seed, e));
        }
    }
    assert!(failures.is_empty(), "aggressive fuzz failures:\n  {}", failures.join("\n  "));
}


/// Output trajectory must reproduce the user-requested start velocity within slack.
/// Stronger than `aligned_non_zero_velocity_is_feasible` (which uses 0.15 tolerance);
/// here we assert the slack we promised in `SolverOptions::boundary_slack`.
#[test]
fn output_start_velocity_matches_requested_within_slack() {
    let fk = common::dh_1dof();
    let path = SRobotPath::<1, f64>::try_new(vec![
        SRobotQ::from_array([0.0]),
        SRobotQ::from_array([1.0]),
    ])
    .unwrap();
    let mut cfg = Topp3Tcp6Constraints::<1>::symmetric(1.0, 5.0, 200.0);
    cfg.boundary = BoundaryConditions {
        v_start: SRobotQ::from_array([0.4]),
        a_start: SRobotQ::zeros(),
        v_end: SRobotQ::from_array([0.4]),
        a_end: SRobotQ::zeros(),
        projection_tolerance: 1e-3,
    };
    cfg.sample_rate_hz = 1000.0;
    let mut validator = common::wide_validator::<1>();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    let traj = result.expect("retime failed");
    let v0 = traj.velocity_at(0).unwrap().0[0];
    eprintln!("output_start_velocity test:\n{}\n  v0 = {}", diag, v0);
    assert!(
        (v0 - 0.4).abs() < 0.05,
        "start velocity drifted to {}, expected ~0.4",
        v0
    );
}
