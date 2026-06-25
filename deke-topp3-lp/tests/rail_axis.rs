//! Validate the retimer on a 7-DOF rail-mounted chain: a prismatic rail axis
//! (q[0], metres) carrying a 6-DOF arm (q[1..7], radians), with realistic
//! ASYMMETRIC per-axis limits (the heavy rail is slower than the arm). Confirms
//! every axis — rail included — stays under its own v/a/j limit, the path is
//! exact, the rail's tight limit correctly throttles the coordinated motion, and
//! the TCP velocity cap accounts for the prismatic Jacobian column.

mod common;

use std::time::Duration;

use deke_topp3_lp::{JointLimits, Topp3Lp, Topp3LpConstraints, Topp3LpTcp};
use deke_types::{FKChain, Retimer, SRobotPath, SRobotQ, SRobotTraj};

fn dt8() -> Duration {
    Duration::from_millis(8)
}

/// Rail (axis 0): 0.5 m/s, 2 m/s², 20 m/s³.  Arm (axes 1..6): 2 rad/s, 10, 200.
fn rail_limits() -> JointLimits<7> {
    JointLimits {
        v_max: SRobotQ::from_array([0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        a_max: SRobotQ::from_array([2.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        j_max: SRobotQ::from_array([20.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]),
    }
}

fn path(wps: &[[f64; 7]]) -> SRobotPath<7, f64> {
    SRobotPath::try_new(wps.iter().map(|w| SRobotQ::from_array(*w)).collect()).unwrap()
}

/// Per-axis worst (value/limit) over v, a, j. Returns the per-axis peak utilization.
fn per_axis_peaks(traj: &SRobotTraj<7, f64>, lim: &JointLimits<7>) -> ([f64; 7], &'static str) {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let n = p.len();
    let mut peak = [0.0f64; 7];
    let mut worst_kind = "none";
    let mut worst = 0.0f64;
    let mut hit = |ax: usize, val: f64, limit: f64, kind: &'static str, peak: &mut [f64; 7]| {
        let u = val / limit;
        peak[ax] = peak[ax].max(u);
        if u > worst {
            worst = u;
            worst_kind = kind;
        }
    };
    for i in 1..n {
        for ax in 0..7 {
            hit(
                ax,
                (p[i].0[ax] - p[i - 1].0[ax]).abs() / dt,
                lim.v_max.0[ax],
                "v",
                &mut peak,
            );
        }
    }
    for i in 2..n {
        for ax in 0..7 {
            let a = (p[i].0[ax] - 2.0 * p[i - 1].0[ax] + p[i - 2].0[ax]).abs() / (dt * dt);
            hit(ax, a, lim.a_max.0[ax], "a", &mut peak);
        }
    }
    for i in 3..n {
        for ax in 0..7 {
            let j = (p[i].0[ax] - 3.0 * p[i - 1].0[ax] + 3.0 * p[i - 2].0[ax] - p[i - 3].0[ax])
                .abs()
                / (dt * dt * dt);
            hit(ax, j, lim.j_max.0[ax], "j", &mut peak);
        }
    }
    (peak, worst_kind)
}

fn seg_dist(p: &[f64; 7], a: &[f64; 7], b: &[f64; 7]) -> f64 {
    let ab: [f64; 7] = std::array::from_fn(|i| b[i] - a[i]);
    let ab2: f64 = ab.iter().map(|x| x * x).sum();
    let t = if ab2 > 1e-18 {
        ((0..7).map(|i| (p[i] - a[i]) * ab[i]).sum::<f64>() / ab2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0..7)
        .map(|i| (p[i] - (a[i] + t * ab[i])).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn polyline_dev(traj: &SRobotTraj<7, f64>, wps: &[[f64; 7]]) -> f64 {
    let p = traj.path();
    (0..p.len())
        .map(|i| {
            (0..wps.len() - 1)
                .map(|s| seg_dist(&p[i].0, &wps[s], &wps[s + 1]))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

fn assert_within(peaks: &[f64; 7]) {
    for (ax, u) in peaks.iter().enumerate() {
        assert!(*u <= 1.0 + 1e-3, "axis {ax} over its limit: {:.4}x", u);
    }
}

// Coordinated rail + arm move; every axis must stay under its own limit, path exact.
#[test]
fn rail_and_arm_each_under_own_limit() {
    let wps = [
        [0.0, 0.0, -1.3, 1.5, 0.0, 0.0, 0.0],
        [0.2, 0.2, -1.1, 1.3, -0.1, 0.1, 0.1],
        [0.4, 0.4, -0.9, 1.1, -0.2, 0.2, 0.2],
        [0.6, 0.6, -0.7, 0.9, -0.3, 0.1, 0.3],
        [0.8, 0.8, -0.5, 0.7, -0.4, 0.0, 0.4],
    ];
    let mut c = Topp3LpConstraints::<7>::symmetric(2.0, 10.0, 200.0, dt8());
    c.joint = rail_limits();
    let (r, diag) =
        Topp3Lp::<7>::new().retime(&c, &path(&wps), &common::wide_validator::<7>(), &());
    eprintln!("{diag}");
    let traj = r.expect("rail retime");
    let (peaks, _) = per_axis_peaks(&traj, &c.joint);
    assert_within(&peaks);
    assert!(polyline_dev(&traj, &wps) < 1e-9, "off-chord deviation");
    assert_eq!(traj.path()[0].0, wps[0]);
    assert_eq!(traj.path()[traj.len() - 1].0, wps[4]);
    // The rail (slow heavy axis) is the binding one here.
    assert!(
        peaks[0] > 0.98,
        "rail axis should bind, peak util {:.3}",
        peaks[0]
    );
}

// Rail-dominant move with the rail throttled: the tight rail limit must slow the
// WHOLE coordinated motion, so the (looser) arm axes run BELOW their limits.
#[test]
fn tight_rail_throttles_the_arm() {
    let wps = [
        [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        [0.5, 0.05, -1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.1, -1.0, 1.0, 0.0, 0.0, 0.0],
    ];
    let mut c = Topp3LpConstraints::<7>::symmetric(2.0, 10.0, 200.0, dt8());
    c.joint = rail_limits();
    let (r, _) = Topp3Lp::<7>::new().retime(&c, &path(&wps), &common::wide_validator::<7>(), &());
    let traj = r.expect("rail-dominant retime");
    let (peaks, _) = per_axis_peaks(&traj, &c.joint);
    assert_within(&peaks);
    assert!(peaks[0] > 0.98, "rail should bind, got {:.3}", peaks[0]);
    // Arm axis 1 moves 1/10th the rail distance under a 4x looser velocity limit —
    // it must stay well under its own limit because the rail throttles the timing.
    assert!(
        peaks[1] < 0.5,
        "arm axis 1 should be throttled by the rail, got {:.3}",
        peaks[1]
    );
}

// TCP velocity cap on a rail-dominant move: the prismatic rail moves the tip 1:1
// (axis along world +X), so the cap must throttle via the rail's Jacobian column.
#[test]
fn tcp_cap_accounts_for_prismatic_rail() {
    let fk = common::dh_7dof_prismatic();
    let wps = [
        [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        [0.4, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
    ];
    let cap = 0.3;
    // Loose joint limits so the TCP cap is the binding constraint.
    let mut c = Topp3LpConstraints::<7>::symmetric(5.0, 40.0, 400.0, dt8()).with_tcp_speed(cap);
    c.joint.v_max = SRobotQ::from_array([5.0; 7]);
    let (r, diag) =
        Topp3LpTcp::new(&fk).retime(&c, &path(&wps), &common::wide_validator::<7>(), &());
    eprintln!("{diag}");
    let traj = r.expect("rail tcp retime");

    // Independent tip-speed check via FK.
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let mut peak_tip = 0.0f64;
    let mut moved = 0.0f64;
    for i in 1..p.len() {
        let a = fk.fk_end(&p[i - 1]).unwrap().translation;
        let b = fk.fk_end(&p[i]).unwrap().translation;
        peak_tip = peak_tip.max(a.distance(b) / dt);
    }
    let p0 = fk.fk_end(&p[0]).unwrap().translation;
    let pn = fk.fk_end(&p[p.len() - 1]).unwrap().translation;
    moved += p0.distance(pn);

    assert!(
        peak_tip <= cap * (1.0 + 1e-3),
        "tip speed {peak_tip:.4} over cap {cap}"
    );
    assert!(
        peak_tip > 0.7 * cap,
        "TCP cap should bind (tip {peak_tip:.4} vs cap {cap})"
    );
    // The rail swept 0.8 m along +X — the tip must have moved ~that far, proving the
    // prismatic column drives the tip (arm is static).
    assert!(
        moved > 0.7,
        "tip should track the rail sweep (~0.8 m), moved {moved:.3}"
    );
    assert!(polyline_dev(&traj, &wps) < 1e-9, "off-chord deviation");
}
