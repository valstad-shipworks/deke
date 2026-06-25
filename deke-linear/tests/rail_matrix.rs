mod common;

use std::time::Duration;

use deke_kin::Kinematics;
use deke_linear::{
    ConstantSpeedRetimer, JointLimits, LinearConstraints, NoopValidator, PathConditioning,
    PlannerOptions, RailAxis, RailConfig, RailLinearPlanner, RailMountedChain, RailOptions,
    RailRefine, TcpLimits, condition,
};
use deke_types::glam::DVec3;
use deke_types::{DekeError, FKChain, Planner, Retimer, SRobotPath, SRobotQ, SRobotTraj};

const RAIL_V: f64 = 1.0;
const RAIL_A: f64 = 20.0;
const RAIL_J: f64 = 2000.0;
const ARM_V: f64 = 2.0;
const ARM_A: f64 = 8.0;
const ARM_J: f64 = 80.0;
const EPS: f64 = 1.0 + 1e-9;

fn tcp_speed() -> f64 {
    30.0 * 0.0254 / 60.0
}

fn vmax7() -> SRobotQ<7, f64> {
    SRobotQ::from_array([RAIL_V, ARM_V, ARM_V, ARM_V, ARM_V, ARM_V, ARM_V])
}

fn joint_limits7() -> JointLimits<7> {
    JointLimits {
        v_max: vmax7(),
        a_max: SRobotQ::from_array([RAIL_A, ARM_A, ARM_A, ARM_A, ARM_A, ARM_A, ARM_A]),
        j_max: SRobotQ::from_array([RAIL_J, ARM_J, ARM_J, ARM_J, ARM_J, ARM_J, ARM_J]),
    }
}

fn planner7() -> PlannerOptions<7> {
    PlannerOptions {
        sample_ds: 5e-4,
        manip_weight: 1.0,
        max_branch_jump: 0.6,
        max_velocity: tcp_speed(),
        joint_v_max: vmax7(),
        reconfig_vel_fraction: 0.9,
    }
}

fn constraints7() -> LinearConstraints<7> {
    LinearConstraints {
        joint: joint_limits7(),
        tcp: TcpLimits::speed(tcp_speed()),
        output_dt: Duration::from_millis(8),
        forbid_interior_dips: false,
        corner_smoothing: Some(0.01),
    }
}

struct FdPeaks {
    v: [f64; 7],
    a: [f64; 7],
    j: [f64; 7],
}

fn fd_peaks(traj: &SRobotTraj<7, f64>) -> FdPeaks {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let mut v = [0.0f64; 7];
    let mut a = [0.0f64; 7];
    let mut j = [0.0f64; 7];
    for i in 0..p.len().saturating_sub(1) {
        for (k, vk) in v.iter_mut().enumerate() {
            *vk = vk.max(((p[i + 1].0[k] - p[i].0[k]) / dt).abs());
        }
    }
    for i in 1..p.len().saturating_sub(1) {
        for (k, ak) in a.iter_mut().enumerate() {
            let acc = (p[i + 1].0[k] - 2.0 * p[i].0[k] + p[i - 1].0[k]) / (dt * dt);
            *ak = ak.max(acc.abs());
        }
    }
    for i in 3..p.len() {
        for (k, jk_acc) in j.iter_mut().enumerate() {
            let jk = (p[i].0[k] - 3.0 * p[i - 1].0[k] + 3.0 * p[i - 2].0[k] - p[i - 3].0[k])
                / (dt * dt * dt);
            *jk_acc = jk_acc.max(jk.abs());
        }
    }
    FdPeaks { v, a, j }
}

fn tcp_speeds(
    chain: &RailMountedChain<6, 7, Kinematics<6, f64>>,
    traj: &SRobotTraj<7, f64>,
) -> Vec<f64> {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    (0..p.len().saturating_sub(1))
        .map(|i| {
            let a = chain.fk_end(&p[i]).unwrap().translation;
            let b = chain.fk_end(&p[i + 1]).unwrap().translation;
            a.distance(b) / dt
        })
        .collect()
}

fn tcp_accel_jerk_peak(
    chain: &RailMountedChain<6, 7, Kinematics<6, f64>>,
    traj: &SRobotTraj<7, f64>,
) -> (f64, f64) {
    let dt = traj.dt().as_secs_f64();
    let sp = tcp_speeds(chain, traj);
    let mut a = 0.0f64;
    let mut j = 0.0f64;
    for i in 1..sp.len() {
        a = a.max(((sp[i] - sp[i - 1]) / dt).abs());
    }
    for i in 2..sp.len() {
        j = j.max(((sp[i] - 2.0 * sp[i - 1] + sp[i - 2]) / (dt * dt)).abs());
    }
    (a, j)
}

fn point_segment_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-18)).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

fn polyline_deviation(
    chain: &RailMountedChain<6, 7, Kinematics<6, f64>>,
    traj: &SRobotTraj<7, f64>,
    line: &[DVec3],
) -> f64 {
    let path = traj.path();
    (0..path.len())
        .map(|i| {
            let pt = chain.fk_end(&path[i]).unwrap().translation;
            (0..line.len() - 1)
                .map(|s| point_segment_distance(pt, line[s], line[s + 1]))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

struct CellResult {
    weld: &'static str,
    rail: &'static str,
    refine: RailRefine,
    pass: bool,
    v_ratio: f64,
    a_ratio: f64,
    j_ratio: f64,
    tcp_a_ratio: f64,
    tcp_j_ratio: f64,
    cruise_err_pct: f64,
    dev_mm: f64,
    note: String,
}

fn run_cell(
    arm: &Kinematics<6, f64>,
    weld: (&'static str, DVec3),
    rail: (&'static str, DVec3),
    refine: RailRefine,
) -> CellResult {
    let lim = joint_limits7();
    let tcp = tcp_speed();
    let poses = common::straight(arm, weld.1, 0.30, 24);
    let runs = match condition(&poses, &PathConditioning::default()) {
        Ok(r) => r,
        Err(e) => {
            return fail_cell(weld.0, rail.0, refine, format!("condition: {e}"));
        }
    };

    let rail_opts = RailOptions {
        axis: RailAxis::Custom(rail.1),
        window: (-0.3, 0.3),
        samples: 21,
        dp_ds: 5e-3,
        rate_weight: 0.5,
        max_step: 0.05,
        centering_weight: 0.05,
        refine,
    };
    let chain = RailMountedChain::<6, 7, _>::new(arm, rail_opts.axis);
    let planner = RailLinearPlanner::<6, 7, _>::new(arm);
    let retimer = ConstantSpeedRetimer::new(&chain);
    let validator = NoopValidator::<7>;

    let cfg = RailConfig::<6, 7> {
        planner: planner7(),
        rail: rail_opts.clone(),
    };
    let mut all: Vec<SRobotQ<7, f64>> = Vec::new();
    for (idx, run) in runs.iter().enumerate() {
        let (path, _diag) = planner.plan::<DekeError, _>(&cfg, run, &validator, &());
        let path = match path {
            Ok(p) => p,
            Err(e) => return fail_cell(weld.0, rail.0, refine, format!("plan run {idx}: {e}")),
        };
        let (traj, _rd) = retimer.retime(&constraints7(), &path, &validator, &());
        let traj = match traj {
            Ok(t) => t,
            Err(e) => return fail_cell(weld.0, rail.0, refine, format!("retime run {idx}: {e}")),
        };
        let samples = traj.path().iter().copied();
        if all.is_empty() {
            all.extend(samples);
        } else {
            all.extend(samples.skip(1));
        }
    }

    let path = SRobotPath::try_new(all).unwrap();
    let traj = SRobotTraj::new(Duration::from_millis(8), path);

    let pk = fd_peaks(&traj);
    let v_ratio = (0..7).map(|k| pk.v[k] / lim.v_max.0[k]).fold(0.0, f64::max);
    let a_ratio = (0..7).map(|k| pk.a[k] / lim.a_max.0[k]).fold(0.0, f64::max);
    let j_ratio = (0..7).map(|k| pk.j[k] / lim.j_max.0[k]).fold(0.0, f64::max);

    let (tcp_a, tcp_j) = tcp_accel_jerk_peak(&chain, &traj);

    let line: Vec<DVec3> = poses.iter().map(|p| p.translation).collect();
    let dev_mm = polyline_deviation(&chain, &traj, &line) * 1000.0;

    let speeds = tcp_speeds(&chain, &traj);
    let lo = speeds.len() / 5;
    let hi = speeds.len().saturating_sub(speeds.len() / 5);
    let cruise: Vec<f64> = speeds.get(lo..hi).unwrap_or(&[]).to_vec();
    let cruise_err_pct = cruise
        .iter()
        .map(|s| ((s - tcp) / tcp * 100.0).abs())
        .fold(0.0, f64::max);

    let fd_ok = v_ratio <= EPS && a_ratio <= EPS && j_ratio <= EPS;
    let cruise_ok = cruise_err_pct <= 2.0;
    let dev_ok = dev_mm < 2.0;
    let window_ok = {
        let mut ok = true;
        for q in traj.path().iter() {
            if q.0[0] < -0.3 - 1e-9 || q.0[0] > 0.3 + 1e-9 {
                ok = false;
            }
        }
        ok
    };
    let pass = fd_ok && cruise_ok && dev_ok && window_ok;

    let mut note = String::new();
    if !fd_ok {
        note.push_str("FD>limit ");
    }
    if !cruise_ok {
        note.push_str("cruise ");
    }
    if !dev_ok {
        note.push_str("dev ");
    }
    if !window_ok {
        note.push_str("window ");
    }

    // TCP accel/jerk have no explicit TcpLimits cap here (speed-only), so report
    // the realized magnitudes as ratios against the projected joint envelope is
    // not meaningful; report raw against a nominal so the table is populated.
    let tcp_a_ratio = tcp_a;
    let tcp_j_ratio = tcp_j;

    CellResult {
        weld: weld.0,
        rail: rail.0,
        refine,
        pass,
        v_ratio,
        a_ratio,
        j_ratio,
        tcp_a_ratio,
        tcp_j_ratio,
        cruise_err_pct,
        dev_mm,
        note,
    }
}

fn fail_cell(
    weld: &'static str,
    rail: &'static str,
    refine: RailRefine,
    note: String,
) -> CellResult {
    CellResult {
        weld,
        rail,
        refine,
        pass: false,
        v_ratio: f64::NAN,
        a_ratio: f64::NAN,
        j_ratio: f64::NAN,
        tcp_a_ratio: f64::NAN,
        tcp_j_ratio: f64::NAN,
        cruise_err_pct: f64::NAN,
        dev_mm: f64::NAN,
        note,
    }
}

#[test]
fn rail_matrix_weld_dir_x_rail_dir() {
    let arm = common::ur();
    let d45 = (DVec3::X + DVec3::Y).normalize();
    let dirs: [(&str, DVec3); 3] = [("X", DVec3::X), ("Y", DVec3::Y), ("D45", d45)];

    let mut results: Vec<CellResult> = Vec::new();
    for refine in [RailRefine::Linear, RailRefine::Pchip] {
        for weld in dirs.iter().copied() {
            for rail in dirs.iter().copied() {
                results.push(run_cell(&arm, weld, rail, refine));
            }
        }
    }

    println!(
        "\n{:<7} {:<4} {:<4} {:<6} | {:>7} {:>7} {:>7} | {:>8} {:>8} | {:>9} {:>7} | note",
        "refine",
        "weld",
        "rail",
        "pass",
        "v/lim",
        "a/lim",
        "j/lim",
        "tcp_a",
        "tcp_j",
        "cruise%",
        "dev_mm"
    );
    for r in &results {
        let refine = match r.refine {
            RailRefine::Linear => "Linear",
            RailRefine::Pchip => "Pchip",
        };
        println!(
            "{:<7} {:<4} {:<4} {:<6} | {:>7.4} {:>7.4} {:>7.4} | {:>8.3} {:>8.2} | {:>9.4} {:>7.4} | {}",
            refine,
            r.weld,
            r.rail,
            if r.pass { "PASS" } else { "FAIL" },
            r.v_ratio,
            r.a_ratio,
            r.j_ratio,
            r.tcp_a_ratio,
            r.tcp_j_ratio,
            r.cruise_err_pct,
            r.dev_mm,
            r.note,
        );
    }

    let failed: Vec<&CellResult> = results.iter().filter(|r| !r.pass).collect();
    assert!(
        failed.is_empty(),
        "{} cell(s) failed: {:?}",
        failed.len(),
        failed
            .iter()
            .map(|r| format!("{}/{}/{:?}:{}", r.weld, r.rail, r.refine, r.note))
            .collect::<Vec<_>>()
    );
}
