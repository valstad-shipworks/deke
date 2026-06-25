//! Regression tests for the audit-confirmed bugs.

mod common;

use std::time::Duration;

use deke_topp3_lp::{JointLimits, TcpLimits, Topp3Lp, Topp3LpConstraints};
use deke_types::{Retimer, SRobotPath, SRobotQ};

fn one(a: f64, b: f64) -> SRobotPath<1, f64> {
    SRobotPath::try_new(vec![SRobotQ::from_array([a]), SRobotQ::from_array([b])]).unwrap()
}

fn dt8() -> Duration {
    Duration::from_millis(8)
}

// BUG-01: a sub-microsecond / zero output_dt must be rejected (else the verified dt
// and the stamped dt disagree and the trajectory is over-limit over its own dt).
#[test]
fn rejects_sub_microsecond_dt() {
    let c = Topp3LpConstraints::<1>::symmetric(1.0, 2.0, 200.0, Duration::from_nanos(500));
    let (r, _) =
        Topp3Lp::<1>::new().retime(&c, &one(0.0, 1.0), &common::wide_validator::<1>(), &());
    assert!(r.is_err(), "sub-us output_dt must be rejected");
}

#[test]
fn rejects_zero_dt() {
    let c = Topp3LpConstraints::<1>::symmetric(1.0, 2.0, 200.0, Duration::ZERO);
    let (r, _) =
        Topp3Lp::<1>::new().retime(&c, &one(0.0, 1.0), &common::wide_validator::<1>(), &());
    assert!(r.is_err(), "zero output_dt must be rejected");
}

// BUG-03: non-finite waypoints must not panic; they are rejected (here or earlier).
#[test]
fn non_finite_waypoint_does_not_panic() {
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let path =
            match SRobotPath::try_new(vec![SRobotQ::from_array([0.0]), SRobotQ::from_array([bad])])
            {
                Ok(p) => p,
                Err(_) => continue, // rejected even earlier — fine
            };
        let c = Topp3LpConstraints::<1>::symmetric(1.0, 2.0, 200.0, dt8());
        let (r, _) = Topp3Lp::<1>::new().retime(&c, &path, &common::wide_validator::<1>(), &());
        assert!(
            r.is_err(),
            "non-finite waypoint ({bad}) must be rejected, not Ok"
        );
    }
}

// BUG-11: non-positive / non-finite limits must be rejected, not silently mishandled.
#[test]
fn rejects_nonpositive_limits() {
    for (v, a, j) in [
        (0.0, 2.0, 200.0),
        (1.0, -1.0, 200.0),
        (1.0, 2.0, f64::INFINITY),
    ] {
        let c = Topp3LpConstraints::<1>::symmetric(v, a, j, dt8());
        let (r, _) =
            Topp3Lp::<1>::new().retime(&c, &one(0.0, 1.0), &common::wide_validator::<1>(), &());
        assert!(r.is_err(), "invalid limits ({v},{a},{j}) must be rejected");
    }
}

#[test]
fn joint_only_still_rejects_tcp_cap() {
    let mut c = Topp3LpConstraints::<1>::symmetric(1.0, 2.0, 200.0, dt8());
    c.tcp = TcpLimits::speed(0.1);
    let (r, _) =
        Topp3Lp::<1>::new().retime(&c, &one(0.0, 1.0), &common::wide_validator::<1>(), &());
    assert!(r.is_err(), "joint-only retimer must reject a TCP cap");
}

// BUG-07: the terminal endpoint must be a genuine rest (v=0, a=0), not v ~ a_max*dt.
#[test]
fn terminal_endpoint_is_at_rest() {
    let c = Topp3LpConstraints::<1>::symmetric(1.0, 2.0, 200.0, dt8());
    let (r, _) =
        Topp3Lp::<1>::new().retime(&c, &one(0.0, 1.0), &common::wide_validator::<1>(), &());
    let traj = r.expect("retime");
    let n = traj.len();
    let v_end = traj.velocity_at(n - 1).unwrap().0[0].abs();
    let a_end = traj.acceleration_at(n - 1).unwrap().0[0].abs();
    assert!(v_end < 1e-9, "terminal velocity not zero: {v_end}");
    assert!(a_end < 1e-9, "terminal acceleration not zero: {a_end}");
    // and the start is at rest too (was already correct)
    let v0 = traj.velocity_at(0).unwrap().0[0].abs();
    assert!(v0 < 1e-9, "start velocity not zero: {v0}");
}

// A normal retime still succeeds after the added validation.
#[test]
fn valid_inputs_still_succeed() {
    let c = Topp3LpConstraints::<1>::symmetric(1.0, 2.0, 200.0, dt8());
    let (r, _) =
        Topp3Lp::<1>::new().retime(&c, &one(0.0, 1.5), &common::wide_validator::<1>(), &());
    assert!(r.is_ok());
    // unused import guard
    let _ = JointLimits::<1>::symmetric(1.0, 2.0, 200.0);
}
