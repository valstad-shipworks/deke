//! BUG-12 regression: TCP-cap timing on a long 7-DOF (RTU) chord that slides the
//! rail and reconfigures the arm a lot in one straight move. The TCP sweeps
//! several metres, so the linear-speed cap is the binding constraint.
//!
//! Before the fix `kappa_per_segment` evaluated the Jacobian only at each
//! segment's START knot, so on a long chord the mid-chord TCP gain was
//! under-estimated; the realised TCP peak overshot the cap by a few percent and
//! the derate loop could not recover it → `TcpLimitExceeded`. The fix samples the
//! Jacobian at several points along each segment and takes the max (a conservative
//! per-segment cap). See `docs/FUTURE.md`.
//!
//! Two real `nanopanel-material` chords × two limit sets. PATH_A under the fast
//! UNIFORM limits is the case that failed pre-fix (overshoot 2.0984 > 2.0).

mod common;

use std::time::Duration;

use deke_topp3_lp::{JointLimits, Topp3LpConstraints, Topp3LpTcp};
use deke_types::{Retimer, SRobotPath, SRobotQ};

const TCP_CAP: f64 = 2.0;

fn material_limits() -> JointLimits<7> {
    JointLimits {
        v_max: SRobotQ::from_array([1.422, 1.099557, 0.942478, 0.890118, 1.256637, 1.256637, 2.094395]),
        a_max: SRobotQ::from_array([3.262729, 3.096281, 2.653955, 2.506513, 3.538607, 3.538607, 5.897679]),
        j_max: SRobotQ::from_array([5.996099, 13.966876, 11.971608, 11.306519, 15.962144, 15.962144, 26.603575]),
    }
}

fn uniform_limits() -> JointLimits<7> {
    JointLimits {
        v_max: SRobotQ::from_array([2.0; 7]),
        a_max: SRobotQ::from_array([20.0; 7]),
        j_max: SRobotQ::from_array([200.0; 7]),
    }
}

fn constraints(lim: JointLimits<7>) -> Topp3LpConstraints<7> {
    // Raw (default) conditioning — what the production consumer uses; Collinear is
    // intentionally avoided (see docs/FUTURE.md: it is path-exact but slow).
    Topp3LpConstraints::<7> {
        joint: lim,
        tcp: deke_topp3_lp::TcpLimits::default(),
        output_dt: Duration::from_secs_f64(0.008),
        conditioning: deke_topp3_lp::Conditioning::Raw,
        sharp_corner_angle: Some(30.0_f64.to_radians()),
    }
    .with_tcp_speed(TCP_CAP)
}

fn assert_under_cap(name: &str, wps: &[[f64; 7]; 2], lim: JointLimits<7>) {
    let chain = common::material_7dof();
    let path = SRobotPath::<7, f64>::try_new(wps.iter().map(|w| SRobotQ::from_array(*w)).collect()).unwrap();
    let (res, diag) = Topp3LpTcp::new(&chain).retime(&constraints(lim), &path, &common::wide_validator::<7>(), &());
    let traj = res.unwrap_or_else(|e| panic!("{name}: retime failed under TCP cap: {e}"));
    assert!(traj.len() >= 2, "{name}: degenerate trajectory");
    assert!(
        diag.peak_tcp_speed <= TCP_CAP * (1.0 + 1e-3),
        "{name}: realised TCP peak {} exceeds cap {TCP_CAP}",
        diag.peak_tcp_speed
    );
}

#[test]
fn path_a_material_under_cap() {
    assert_under_cap("PATH_A material", &common::MATERIAL_PATH_A, material_limits());
}

#[test]
fn path_b_material_under_cap() {
    assert_under_cap("PATH_B material", &common::MATERIAL_PATH_B, material_limits());
}

// The pre-fix failure: fast uniform limits let the arm move quickly, so the TCP
// curves hard and the start-knot kappa under-estimate bites.
#[test]
fn path_a_uniform_under_cap() {
    assert_under_cap("PATH_A uniform", &common::MATERIAL_PATH_A, uniform_limits());
}

#[test]
fn path_b_uniform_under_cap() {
    assert_under_cap("PATH_B uniform", &common::MATERIAL_PATH_B, uniform_limits());
}
