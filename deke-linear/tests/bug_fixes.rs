//! Regression tests for the audit-confirmed deke-linear bugs.

mod common;

use std::time::Duration;

use deke_linear::{
    ConstantSpeedRetimer, JointLimits, LinearConstraints, PathConditioning, TcpLimits, condition,
};
use deke_types::glam::{DAffine3, DMat3, DVec3};
use deke_types::{FKChain, Retimer, SRobotPath, SRobotQ};

fn cons() -> LinearConstraints<6> {
    LinearConstraints {
        joint: JointLimits::symmetric(2.0, 8.0, 80.0),
        tcp: TcpLimits::speed(0.1),
        output_dt: Duration::from_millis(8),
        forbid_interior_dips: false,
        corner_smoothing: Some(0.01),
    }
}

fn two(a: SRobotQ<6, f64>, b: SRobotQ<6, f64>) -> SRobotPath<6, f64> {
    SRobotPath::try_new(vec![a, b]).unwrap()
}

// BUG-01: a near-zero-tip-arc joint move that still carries joint motion must NOT
// be emitted untimed — it has to pass verify_fd or be rejected.
#[test]
fn degenerate_wrist_move_is_not_emitted_over_limit() {
    let robot = common::ur();
    let q0 = common::anchor();
    let mut a1 = q0.0;
    a1[5] += 0.5; // spin the last joint: flange origin (tip) does not translate
    let q1 = SRobotQ::from_array(a1);

    let tip0 = robot.fk_end(&q0).unwrap().translation;
    let tip1 = robot.fk_end(&q1).unwrap().translation;
    assert!(
        tip0.distance(tip1) < 1e-9,
        "test assumes a zero-tip-arc wrist roll, got {}",
        tip0.distance(tip1)
    );

    let (r, _) =
        ConstantSpeedRetimer::new(&robot).retime(&cons(), &two(q0, q1), &common::noop(), &());
    // 0.5 rad / 8ms = 62.5 rad/s >> v_max=2 → must be rejected, never Ok.
    assert!(
        r.is_err(),
        "degenerate wrist move at 62 rad/s must be rejected"
    );
}

// A truly stationary degenerate path is still fine (Ok).
#[test]
fn stationary_degenerate_path_is_ok() {
    let robot = common::ur();
    let q = common::anchor();
    let (r, _) =
        ConstantSpeedRetimer::new(&robot).retime(&cons(), &two(q, q), &common::noop(), &());
    assert!(r.is_ok(), "stationary path should retime cleanly");
}

// BUG-02: zero / sub-microsecond output_dt rejected.
#[test]
fn rejects_zero_output_dt() {
    let robot = common::ur();
    let mut c = cons();
    c.output_dt = Duration::ZERO;
    let q0 = common::anchor();
    let mut a1 = q0.0;
    a1[0] += 0.3;
    let (r, _) = ConstantSpeedRetimer::new(&robot).retime(
        &c,
        &two(q0, SRobotQ::from_array(a1)),
        &common::noop(),
        &(),
    );
    assert!(r.is_err(), "zero output_dt must be rejected");
}

// BUG-06: a zero joint accel/jerk limit is rejected, not an OOM blowup.
#[test]
fn rejects_zero_joint_limit() {
    let robot = common::ur();
    let mut c = cons();
    c.joint.a_max = SRobotQ::from_array([8.0, 8.0, 0.0, 8.0, 8.0, 8.0]); // joint 2 accel = 0
    let q0 = common::anchor();
    let mut a1 = q0.0;
    a1[0] += 0.3;
    let (r, _) = ConstantSpeedRetimer::new(&robot).retime(
        &c,
        &two(q0, SRobotQ::from_array(a1)),
        &common::noop(),
        &(),
    );
    assert!(r.is_err(), "zero accel limit must be rejected, not OOM");
}

// BUG-03: a NaN joint value in the input path is rejected, not shipped Ok.
#[test]
fn rejects_non_finite_joint() {
    let robot = common::ur();
    let q0 = common::anchor();
    let mut a1 = q0.0;
    a1[3] = f64::NAN;
    let (r, _) = ConstantSpeedRetimer::new(&robot).retime(
        &cons(),
        &two(q0, SRobotQ::from_array(a1)),
        &common::noop(),
        &(),
    );
    assert!(r.is_err(), "NaN joint must be rejected");
}

// BUG-14: a non-finite input pose to Stage A errors cleanly instead of panicking.
#[test]
fn condition_rejects_non_finite_pose() {
    let good = DAffine3::from_mat3_translation(DMat3::IDENTITY, DVec3::new(0.0, 0.0, 0.5));
    let bad = DAffine3::from_mat3_translation(DMat3::IDENTITY, DVec3::new(f64::INFINITY, 0.0, 0.5));
    let r = condition(&[good, bad], &PathConditioning::default());
    assert!(r.is_err(), "non-finite pose must error, not panic");
}

// BUG-31: retime_weave with a mismatched seam_progress length errors, not panics.
#[test]
fn weave_rejects_mismatched_seam_progress() {
    let robot = common::ur();
    let q0 = common::anchor();
    let mut a1 = q0.0;
    a1[0] += 0.3;
    let path = two(q0, SRobotQ::from_array(a1)); // length 2
    let seam = [0.0, 0.05, 0.1]; // length 3 — mismatch
    let (r, _) =
        ConstantSpeedRetimer::new(&robot).retime_weave(&cons(), &path, &seam, &common::noop(), &());
    assert!(
        r.is_err(),
        "mismatched seam_progress length must error, not panic"
    );
}
