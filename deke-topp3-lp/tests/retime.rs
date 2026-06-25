use std::f64::consts::PI;
use std::time::Duration;

use deke_kin::{DHJoint, JointLimits as KinJointLimits, Kinematics};
use deke_topp3_lp::{Topp3Lp, Topp3LpConstraints, Topp3LpTcp};
use deke_types::{DekeError, DekeResult, Retimer, SRobotPath, SRobotQ, SRobotQLike, Validator};

/// Accept-everything validator (the retimer's own backstops enforce the limits).
#[derive(Default, Clone, Debug)]
struct Noop;
impl Validator<6, (), f64> for Noop {
    type Context<'c> = ();
    fn validate<'c, E: Into<DekeError>, A: SRobotQLike<6, E, f64>>(
        &self,
        _q: A,
        _ctx: &(),
    ) -> DekeResult<()> {
        Ok(())
    }
    fn validate_motion<'c>(&self, _qs: &[SRobotQ<6, f64>], _ctx: &()) -> DekeResult<()> {
        Ok(())
    }
}

fn ur() -> Kinematics<6, f64> {
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

fn path(wps: &[[f64; 6]]) -> SRobotPath<6, f64> {
    SRobotPath::try_new(wps.iter().map(|w| SRobotQ::from_array(*w)).collect()).unwrap()
}

fn seg_dist(p: &[f64; 6], a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let ab: [f64; 6] = std::array::from_fn(|i| b[i] - a[i]);
    let ab2: f64 = ab.iter().map(|x| x * x).sum();
    let t = if ab2 > 1e-18 {
        ((0..6).map(|i| (p[i] - a[i]) * ab[i]).sum::<f64>() / ab2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0..6)
        .map(|i| {
            let d = p[i] - (a[i] + t * ab[i]);
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Worst joint-space distance from any output sample to the input polyline.
fn polyline_dev(traj_path: &SRobotPath<6, f64>, wps: &[[f64; 6]]) -> f64 {
    (0..traj_path.len())
        .map(|i| {
            let p = traj_path[i].0;
            (0..wps.len() - 1)
                .map(|s| seg_dist(&p, &wps[s], &wps[s + 1]))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

fn jt_constraints() -> Topp3LpConstraints<6> {
    Topp3LpConstraints::symmetric(2.0, 8.0, 80.0, Duration::from_millis(8))
}

#[test]
fn straight_joint_move_is_exact_and_within_limits() {
    let wps = [[0.0; 6], [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let (traj, diag) = Topp3Lp::<6>::new().retime(&jt_constraints(), &path(&wps), &Noop, &());
    let traj = traj.expect("retime");

    assert!(
        polyline_dev(traj.path(), &wps) < 1e-9,
        "zero chord deviation"
    );
    assert_eq!(traj.path()[0].0, wps[0], "starts at first waypoint");
    assert_eq!(
        traj.path()[traj.len() - 1].0,
        wps[1],
        "ends at last waypoint"
    );

    // Hard finite-difference bounds hold.
    assert!(diag.peak_joint_vel <= 2.0 * (1.0 + 1e-6));
    assert!(diag.peak_joint_accel <= 8.0 * (1.0 + 1e-6));
    assert!(diag.peak_joint_jerk <= 80.0 * (1.0 + 1e-6));
    // High utilization: the move is long enough to reach the velocity ceiling.
    assert!(
        diag.peak_joint_vel > 1.8,
        "velocity should saturate near 2.0, got {}",
        diag.peak_joint_vel
    );
}

#[test]
fn joint_corner_stays_on_both_legs() {
    let wps = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.8, 0.0, 0.0, 0.0, 0.0],
    ];
    let (traj, diag) = Topp3Lp::<6>::new().retime(&jt_constraints(), &path(&wps), &Noop, &());
    let traj = traj.expect("retime cornered path");

    assert!(
        polyline_dev(traj.path(), &wps) < 1e-9,
        "every sample on a leg (zero deviation)"
    );
    assert!(diag.peak_joint_vel <= 2.0 * (1.0 + 1e-6));
    assert!(diag.peak_joint_accel <= 8.0 * (1.0 + 1e-6));
    assert!(diag.peak_joint_jerk <= 80.0 * (1.0 + 1e-6));
}

#[test]
fn joint_only_rejects_tcp_cap() {
    let c = jt_constraints().with_tcp_speed(0.1);
    let wps = [[0.0; 6], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let (res, _) = Topp3Lp::<6>::new().retime(&c, &path(&wps), &Noop, &());
    assert!(res.is_err(), "joint-only retimer must reject a TCP cap");
}

#[test]
fn tcp_speed_cap_is_respected_and_binds() {
    let robot = ur();
    let q0 = [0.2, -1.0, 1.2, -1.3, -PI / 2.0, 0.3];
    let q1 = [0.6, -0.7, 1.0, -1.2, -1.3, 0.2];
    let wps = [q0, q1];

    let cap = 0.05;
    // Generous joint limits so the TCP cap is the binding constraint.
    let mut c = Topp3LpConstraints::symmetric(5.0, 40.0, 400.0, Duration::from_millis(8));
    c.tcp = deke_topp3_lp::TcpLimits::speed(cap);

    let retimer = Topp3LpTcp::new(&robot);
    let (traj, diag) = retimer.retime(&c, &path(&wps), &Noop, &());
    let traj = traj.expect("tcp retime");

    assert!(
        polyline_dev(traj.path(), &wps) < 1e-9,
        "zero chord deviation"
    );
    assert!(
        diag.peak_tcp_speed <= cap * (1.0 + 1e-6),
        "TCP speed {} over cap {}",
        diag.peak_tcp_speed,
        cap
    );
    assert!(
        diag.peak_tcp_speed > 0.7 * cap,
        "TCP cap should bind (peak {} vs cap {})",
        diag.peak_tcp_speed,
        cap
    );
}
