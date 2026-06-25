mod common;

use std::time::Duration;

use deke_linear::{
    CartesianLinearPlanner, ConstantSpeedRetimer, JointLimits, LinearConstraints, NoopValidator,
    PathConditioning, PlannerOptions, RailAxis, RailConfig, RailLinearPlanner, RailMountedChain,
    RailOptions, RailRefine, TcpLimits, condition,
};
use deke_types::glam::DVec3;
use deke_types::{
    ContinuousFKChain, DekeError, FKChain, Planner, Retimer, SRobotPath, SRobotQ, SRobotTraj,
};

fn limits7() -> JointLimits<7> {
    JointLimits {
        v_max: SRobotQ::from_array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        a_max: SRobotQ::from_array([20.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
        j_max: SRobotQ::from_array([2000.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]),
    }
}

fn point_seg(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-18)).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// A straight seam longer than the arm's fixed-base reach, parallel to the rail.
/// The fixed-base 6-DOF planner cannot reach the far end; with the rail travelling
/// along the seam, the whole pass is planned and timed within all joint limits at
/// the commanded `speed`.
fn over_reach(speed: f64, output_dt: Duration) {
    let arm = common::ur();
    let reach = arm.max_reach().unwrap();
    // Run the seam *outward* along the rail (away from the base) so its far end
    // sits 0.6 m beyond the reach sphere (`max_reach` is an upper bound, so this is
    // firmly unreachable). Going radially out keeps the arm in one well-conditioned
    // configuration the whole traverse.
    let base_tcp = arm.fk_end(&common::anchor()).unwrap().translation;
    let sgn = if base_tcp.x >= 0.0 { 1.0 } else { -1.0 };
    let dir = DVec3::X * sgn;
    let len = (reach + 0.6 - base_tcp.x.abs()).max(1.2);
    let poses = common::straight(&arm, dir, len, ((len / 0.025).ceil() as usize).max(2));
    let runs = condition(&poses, &PathConditioning::default()).unwrap();
    assert_eq!(runs.len(), 1, "a straight seam conditions to one run");

    // Fixed base (no rail): the seam runs past the reachable workspace, so the
    // ordinary 6-DOF branch planner fails (the far end has no IK solution).
    let base = CartesianLinearPlanner::new(&arm);
    let bopts = PlannerOptions::<6>::default();
    let base_ok = runs.iter().all(|run| {
        base.plan::<DekeError, _>(&bopts, run, &NoopValidator::<6>, &())
            .0
            .is_ok()
    });
    assert!(
        !base_ok,
        "expected the {len:.2} m seam to exceed the fixed-base reach (~{reach:.2} m upper bound)"
    );

    let lim = limits7();
    let cfg = RailConfig::<6, 7> {
        planner: PlannerOptions {
            sample_ds: 5e-4,
            manip_weight: 1.0,
            max_branch_jump: 0.6,
            max_velocity: speed,
            joint_v_max: lim.v_max,
            reconfig_vel_fraction: 0.9,
        },
        rail: RailOptions {
            // `samples` is just a floor — the planner raises it to whatever the scan
            // speed needs for the reconfiguration test, so the caller doesn't size
            // the grid to the speed.
            axis: RailAxis::PosX,
            window: if sgn > 0.0 {
                (-0.2, len + 0.2)
            } else {
                (-(len + 0.2), 0.2)
            },
            samples: 21,
            dp_ds: 5e-3,
            rate_weight: 0.0, // the rail must follow the TCP; its velocity is still
            // capped by is_reconfiguration. A nonzero rate penalty would stall the
            // traverse and force the arm to over-reach.
            max_step: 0.05,
            centering_weight: 0.0, // a long traverse must not be pulled back to centre
            refine: RailRefine::Pchip,
        },
    };
    let chain = RailMountedChain::<6, 7, _>::new(&arm, RailAxis::PosX);
    let planner = RailLinearPlanner::<6, 7, _>::new(&arm);
    let retimer = ConstantSpeedRetimer::new(&chain);
    let cons = LinearConstraints {
        joint: lim.clone(),
        tcp: TcpLimits::speed(speed),
        output_dt,
        forbid_interior_dips: false,
        corner_smoothing: Some(0.01),
    };

    let mut all: Vec<SRobotQ<7, f64>> = Vec::new();
    for run in &runs {
        let (path, _) = planner.plan::<DekeError, _>(&cfg, run, &NoopValidator::<7>, &());
        let path = path.expect("rail plan should reach the whole over-reach seam");
        let (traj, _) = retimer.retime(&cons, &path, &NoopValidator::<7>, &());
        let traj = traj.expect("rail retime should succeed");
        let it = traj.path().iter().copied();
        if all.is_empty() {
            all.extend(it);
        } else {
            all.extend(it.skip(1));
        }
    }
    let traj = SRobotTraj::new(output_dt, SRobotPath::try_new(all).unwrap());
    let p = traj.path();
    let dt = traj.dt().as_secs_f64();
    let eps = 1.0 + 1e-9;

    for i in 1..p.len() {
        for j in 0..7 {
            assert!(
                (p[i].0[j] - p[i - 1].0[j]).abs() / dt <= lim.v_max.0[j] * eps,
                "velocity over limit on joint {j} at {speed} m/s"
            );
        }
    }
    for i in 2..p.len() {
        for j in 0..7 {
            let a = (p[i].0[j] - 2.0 * p[i - 1].0[j] + p[i - 2].0[j]).abs() / (dt * dt);
            assert!(
                a <= lim.a_max.0[j] * eps,
                "accel over limit on joint {j} at {speed} m/s"
            );
        }
    }
    for i in 3..p.len() {
        for j in 0..7 {
            let jk = (p[i].0[j] - 3.0 * p[i - 1].0[j] + 3.0 * p[i - 2].0[j] - p[i - 3].0[j]).abs()
                / (dt * dt * dt);
            assert!(
                jk <= lim.j_max.0[j] * eps,
                "jerk over limit on joint {j} at {speed} m/s"
            );
        }
    }

    let rail_lo = p.iter().map(|q| q.0[0]).fold(f64::INFINITY, f64::min);
    let rail_hi = p.iter().map(|q| q.0[0]).fold(f64::NEG_INFINITY, f64::max);
    let travel = rail_hi - rail_lo;
    assert!(
        travel > 0.7 * len,
        "rail travelled {travel:.3} m — too little to carry the arm along a {len:.2} m seam"
    );

    let p0 = chain.fk_end(&p[0]).unwrap().translation;
    let pn = chain.fk_end(&p[p.len() - 1]).unwrap().translation;
    let span = (pn - p0).length();
    assert!(
        span > len - 0.02,
        "TCP spanned {span:.3} m of the {len:.2} m seam"
    );

    let line: Vec<DVec3> = poses.iter().map(|q| q.translation).collect();
    let max_dev = (0..p.len())
        .map(|i| {
            let pt = chain.fk_end(&p[i]).unwrap().translation;
            (0..line.len() - 1)
                .map(|s| point_seg(pt, line[s], line[s + 1]))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max);
    assert!(
        max_dev < 2e-3,
        "max deviation {:.4} mm at {speed} m/s",
        max_dev * 1e3
    );

    println!(
        "{speed:.4} m/s: seam {len:.2} m > reach ~{reach:.2} m, rail travelled {travel:.3} m, TCP span {span:.3} m, dev {:.4} mm",
        max_dev * 1e3
    );
}

/// Scan feed.
#[test]
fn over_reach_scan_0p25_mps() {
    over_reach(0.25, Duration::from_millis(8));
}

/// Weld feed (30 inches per minute). The long seam needs many output ticks at this
/// slow feed, so a coarser tick keeps the timing LP tractable.
#[test]
fn over_reach_weld_30_ipm() {
    over_reach(30.0 * 0.0254 / 60.0, Duration::from_millis(20));
}
