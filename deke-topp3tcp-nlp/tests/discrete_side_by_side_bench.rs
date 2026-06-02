//! Side-by-side comparison of the discrete retimer against the continuous-time
//! retimer on a fixture fleet.
//!
//! Run with:
//!   cargo test -p deke-topp3tcp-nlp --release --test discrete_side_by_side_bench \
//!     -- --ignored --nocapture
//!
//! Columns:
//!   T(s)    — output trajectory duration
//!   wall    — IPM wall time
//!   peak jv — peak |joint velocity| in output FD (joint-limit ratio)
//!   peak tv — peak |TCP velocity| in output FD (TCP-limit ratio if TCP active)
//!   fd_jv   — strict FD-V joint residual (observed/limit − 1, clipped ≥ 0)
//!   fd_ja   — strict FD-A joint residual
//!   fd_jj   — strict FD-J joint residual
//!   fd_tv   — strict FD-V TCP   residual
//!
//! The continuous crate's `peak_*` is the *analytical* peak (what the NLP
//! constrained); the discrete crate's `peak_*` is the *FD* peak (what the
//! consumer measures). They are not perfectly comparable but useful as
//! magnitudes.

#![allow(clippy::too_many_arguments)]

mod common;

use std::time::{Duration, Instant};

use deke_topp3tcp_nlp::continuous::{
    SolveStatus as ContSolveStatus, TcpLimits as ContTcpLimits, Topp3Tcp6,
    Topp3Tcp6Constraints,
};
use deke_topp3tcp_nlp::discrete::{
    SolveStatus as DiscSolveStatus, TcpLimits as DiscTcpLimits, Topp3Tcp6Discrete,
    Topp3Tcp6DiscreteConstraints,
};
use deke_topp_speed::{MotionSpec, StepStatus, ToppSolver};
use deke_types::{Retimer, SRobotPath, SRobotQ};

#[derive(Clone, Copy)]
struct LimitsSpec {
    v: f64,
    a: f64,
    j: f64,
    tcp: Option<(f64, f64, f64)>,
}

fn cont_cfg<const N: usize>(spec: LimitsSpec) -> Topp3Tcp6Constraints<N> {
    let mut c = Topp3Tcp6Constraints::<N>::symmetric(spec.v, spec.a, spec.j);
    if let Some((tv, ta, tj)) = spec.tcp {
        c.tcp = Some(ContTcpLimits {
            v_max: tv,
            a_max: ta,
            j_max: tj,
        });
    }
    c
}

fn disc_cfg<const N: usize>(spec: LimitsSpec) -> Topp3Tcp6DiscreteConstraints<N> {
    let mut c = Topp3Tcp6DiscreteConstraints::<N>::symmetric(spec.v, spec.a, spec.j);
    if let Some((tv, ta, tj)) = spec.tcp {
        c.tcp = Some(DiscTcpLimits {
            v_max: tv,
            a_max: ta,
            j_max: tj,
        });
    }
    c
}

#[allow(dead_code)]
struct Row {
    name: String,
    cont_t: f64,
    cont_wall: Duration,
    cont_status: ContSolveStatus,
    cont_peak_jv: f64,
    cont_peak_ja: f64,
    cont_peak_jj: f64,
    cont_peak_tv: f64,
    disc_t: f64,
    disc_wall: Duration,
    disc_status: DiscSolveStatus,
    disc_peak_jv: f64,
    disc_peak_ja: f64,
    disc_peak_jj: f64,
    disc_peak_tv: f64,
    disc_fd_jv: f64,
    disc_fd_ja: f64,
    disc_fd_jj: f64,
    disc_fd_tv: f64,
    disc_fd_ta: f64,
    disc_fd_tj: f64,
    disc_ok: bool,
    speed_t: f64,
    speed_wall: Duration,
    speed_status: StepStatus,
    speed_ok: bool,
    spec: LimitsSpec,
}

fn run_one<const N: usize, FK: deke_types::ContinuousFKChain<N, f64>>(
    name: &str,
    path: SRobotPath<N, f64>,
    fk: &FK,
    spec: LimitsSpec,
) -> Row {
    let validator = common::wide_validator::<N>();

    let cc = cont_cfg::<N>(spec);
    let t = Instant::now();
    let (rc, dc) = Topp3Tcp6::new(fk).retime(&cc, &path, &validator, &());
    let cont_wall = t.elapsed();
    let cont_t = rc
        .as_ref()
        .map(|t| t.duration().as_secs_f64())
        .unwrap_or(f64::NAN);

    let dc_cfg = disc_cfg::<N>(spec);
    let t = Instant::now();
    let (rd, dd) = Topp3Tcp6Discrete::new(fk).retime(&dc_cfg, &path, &validator, &());
    let disc_wall = t.elapsed();
    let disc_t = rd
        .as_ref()
        .map(|t| t.duration().as_secs_f64())
        .unwrap_or(f64::NAN);

    // Bare topp-speed (jerk-limited shaper, no IPM). Matches discrete's
    // sample rate (125 Hz default → dt = 8 ms) so the K count is directly
    // comparable across the three implementations. Topp-speed handles
    // joint V/A/J and TCP V; it does NOT constrain TCP A/J, so its peak
    // TCP a/j may overshoot the limits given to the IPM crates.
    let h = if dc_cfg.sample_rate_hz.is_finite() && dc_cfg.sample_rate_hz > 0.0 {
        1.0 / dc_cfg.sample_rate_hz
    } else {
        0.008
    };
    let mut mspec = MotionSpec::<N, f64>::new();
    mspec.current_pose = *path.first();
    mspec.goal_pose = *path.last();
    mspec.waypoint_poses.clear();
    for i in 1..path.len() - 1 {
        if let Some(p) = path.get(i) {
            mspec.waypoint_poses.push(*p);
        }
    }
    mspec.max_vel = SRobotQ::from_array([spec.v; N]);
    mspec.max_accel = SRobotQ::from_array([spec.a; N]);
    mspec.max_jerk = SRobotQ::from_array([spec.j; N]);
    if let Some((tv, _, _)) = spec.tcp {
        mspec.max_tcp_speed = Some(tv);
    }
    let speed_solver = ToppSolver::<N, f64, _>::new(Duration::from_secs_f64(h), fk);
    let t = Instant::now();
    let (rs, ds) = speed_solver.retime(&mspec, &path, &validator, &());
    let speed_wall = t.elapsed();
    let speed_t = rs
        .as_ref()
        .map(|t| t.duration().as_secs_f64())
        .unwrap_or(f64::NAN);

    Row {
        name: name.to_string(),
        cont_t,
        cont_wall,
        cont_status: dc.status,
        cont_peak_jv: dc.peak_joint_velocity,
        cont_peak_ja: dc.peak_joint_acceleration,
        cont_peak_jj: dc.peak_joint_jerk,
        cont_peak_tv: dc.peak_tcp_velocity,
        disc_t,
        disc_wall,
        disc_status: dd.status,
        disc_peak_jv: dd.peak_joint_velocity,
        disc_peak_ja: dd.peak_joint_acceleration,
        disc_peak_jj: dd.peak_joint_jerk,
        disc_peak_tv: dd.peak_tcp_velocity,
        disc_fd_jv: dd.output_fd_residual.joint_v,
        disc_fd_ja: dd.output_fd_residual.joint_a,
        disc_fd_jj: dd.output_fd_residual.joint_j,
        disc_fd_tv: dd.output_fd_residual.tcp_v,
        disc_fd_ta: dd.output_fd_residual.tcp_a,
        disc_fd_tj: dd.output_fd_residual.tcp_j,
        disc_ok: rd.is_ok(),
        speed_t,
        speed_wall,
        speed_status: ds.status,
        speed_ok: rs.is_ok(),
        spec,
    }
}

fn print_rows(rows: &[Row]) {
    println!();
    println!(
        "{:<22} | {:>8} {:>8} {:>8} | {:>9} {:>9} {:>9} | {:>4} {:>4} {:>4}",
        "fixture",
        "speed T",
        "cont T",
        "disc T",
        "speed wall",
        "cont wall",
        "disc wall",
        "spd",
        "cnt",
        "dsc",
    );
    println!("{}", "-".repeat(115));
    for r in rows {
        println!(
            "{:<22} | {:>8.3} {:>8.3} {:>8.3} | {:>8.3}s {:>8.3}s {:>8.3}s | {:>4} {:>4} {:>4}",
            r.name,
            r.speed_t,
            r.cont_t,
            r.disc_t,
            r.speed_wall.as_secs_f64(),
            r.cont_wall.as_secs_f64(),
            r.disc_wall.as_secs_f64(),
            if r.speed_ok { "ok" } else { "FAIL" },
            // continuous crate "ok" inferred from cont_t being finite
            if r.cont_t.is_finite() { "ok" } else { "FAIL" },
            if r.disc_ok { "ok" } else { "FAIL" },
        );
    }
    println!();
    println!("legend:");
    println!("  speed/cont/disc T — output trajectory duration (s)");
    println!("  *_wall            — solve wall time (release)");
    println!("  spd/cnt/dsc       — ok? for topp-speed / continuous topp3tcp6 / discrete topp3tcp6");
    println!();

    // Trajectory-duration ratios.
    println!("trajectory-time ratios (smaller = faster trajectory):");
    println!(
        "{:<22} | {:>10} {:>10} {:>10} | {:>10}",
        "fixture", "speed/cont", "disc/cont", "speed/disc", "best",
    );
    println!("{}", "-".repeat(75));
    for r in rows {
        let sc = if r.cont_t.is_finite() && r.speed_t.is_finite() {
            r.speed_t / r.cont_t
        } else {
            f64::NAN
        };
        let dc = if r.cont_t.is_finite() && r.disc_t.is_finite() {
            r.disc_t / r.cont_t
        } else {
            f64::NAN
        };
        let sd = if r.speed_t.is_finite() && r.disc_t.is_finite() {
            r.speed_t / r.disc_t
        } else {
            f64::NAN
        };
        let mut best = ("none", f64::INFINITY);
        if r.speed_t.is_finite() && r.speed_t < best.1 {
            best = ("speed", r.speed_t);
        }
        if r.cont_t.is_finite() && r.cont_t < best.1 {
            best = ("cont", r.cont_t);
        }
        if r.disc_t.is_finite() && r.disc_t < best.1 {
            best = ("disc", r.disc_t);
        }
        println!(
            "{:<22} | {:>10.3} {:>10.3} {:>10.3} | {:>10}",
            r.name, sc, dc, sd, best.0,
        );
    }
    println!();
    println!("wall-time ratios (smaller = faster solve):");
    println!(
        "{:<22} | {:>10} {:>10} {:>10}",
        "fixture", "cont/speed", "disc/speed", "disc/cont",
    );
    println!("{}", "-".repeat(60));
    for r in rows {
        let sw = r.speed_wall.as_secs_f64();
        let cw = r.cont_wall.as_secs_f64();
        let dw = r.disc_wall.as_secs_f64();
        println!(
            "{:<22} | {:>10.1} {:>10.1} {:>10.1}",
            r.name,
            if sw > 0.0 { cw / sw } else { f64::NAN },
            if sw > 0.0 { dw / sw } else { f64::NAN },
            if cw > 0.0 { dw / cw } else { f64::NAN },
        );
    }
    println!();
    println!("discrete crate's strict FD residuals (0 = perfect compliance):");
    println!(
        "{:<22} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "fixture", "fd jv", "fd ja", "fd jj", "fd tv", "fd ta", "fd tj",
    );
    println!("{}", "-".repeat(85));
    for r in rows {
        println!(
            "{:<22} | {:>+8.1e} {:>+8.1e} {:>+8.1e} | {:>+8.1e} {:>+8.1e} {:>+8.1e}",
            r.name,
            r.disc_fd_jv,
            r.disc_fd_ja,
            r.disc_fd_jj,
            r.disc_fd_tv,
            r.disc_fd_ta,
            r.disc_fd_tj,
        );
    }
    println!();
}

fn print_peaks(rows: &[Row]) {
    println!();
    println!(
        "{:<22} | {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} | status cont / disc",
        "fixture",
        "c pk_jv",
        "c pk_ja",
        "c pk_jj",
        "c pk_tv",
        "d pk_jv",
        "d pk_ja",
        "d pk_jj",
        "d pk_tv",
    );
    println!("{}", "-".repeat(140));
    for r in rows {
        println!(
            "{:<22} | {:>8.3} {:>8.3} {:>8.3} {:>8.3} | {:>8.3} {:>8.3} {:>8.3} {:>8.3} | {:?} / {:?}",
            r.name,
            r.cont_peak_jv,
            r.cont_peak_ja,
            r.cont_peak_jj,
            r.cont_peak_tv,
            r.disc_peak_jv,
            r.disc_peak_ja,
            r.disc_peak_jj,
            r.disc_peak_tv,
            r.cont_status,
            r.disc_status,
        );
    }
    println!();
    println!("legend:");
    println!("  c pk_*  — continuous crate analytical NLP peak (what the IPM bounded)");
    println!("  d pk_*  — discrete crate output-FD peak (what backward FD measures)");
    println!();
}

#[test]
#[ignore]
fn benchmark_side_by_side() {
    let fk6 = common::dh_6dof();

    let mut rows: Vec<Row> = Vec::new();

    // ── single joint ───────────────────────────────────────────────────
    {
        let fk = common::dh_1dof();
        let path = SRobotPath::<1, f64>::try_new(vec![
            SRobotQ::from_array([0.0]),
            SRobotQ::from_array([1.0]),
        ])
        .unwrap();
        rows.push(run_one(
            "1dof_rest_to_rest",
            path,
            &fk,
            LimitsSpec {
                v: 1.0,
                a: 2.0,
                j: 200.0,
                tcp: None,
            },
        ));
    }

    // ── 6-DOF straight (joint-limited) ─────────────────────────────────
    {
        let path = SRobotPath::<6, f64>::try_new(vec![
            SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]),
            SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]),
        ])
        .unwrap();
        rows.push(run_one(
            "6dof_2wp_straight_joint",
            path,
            &fk6,
            LimitsSpec {
                v: 1.0,
                a: 3.0,
                j: 300.0,
                tcp: None,
            },
        ));
    }

    // ── 6-DOF curved 5wp (joint-limited) ───────────────────────────────
    {
        let path = SRobotPath::<6, f64>::try_new(vec![
            SRobotQ::from_array([0.0, -1.3, 1.5, 0.0, 0.0, 0.0]),
            SRobotQ::from_array([0.2, -1.1, 1.3, -0.1, 0.1, 0.1]),
            SRobotQ::from_array([0.4, -0.9, 1.1, -0.2, 0.2, 0.2]),
            SRobotQ::from_array([0.6, -0.7, 0.9, -0.3, 0.1, 0.3]),
            SRobotQ::from_array([0.8, -0.5, 0.7, -0.4, 0.0, 0.4]),
        ])
        .unwrap();
        rows.push(run_one(
            "6dof_5wp_curved",
            path,
            &fk6,
            LimitsSpec {
                v: 1.5,
                a: 8.0,
                j: 400.0,
                tcp: None,
            },
        ));
    }

    // ── 6-DOF TCP-dominated ────────────────────────────────────────────
    {
        let path = SRobotPath::<6, f64>::try_new(vec![
            SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]),
            SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]),
        ])
        .unwrap();
        rows.push(run_one(
            "6dof_tcp_v_dominant",
            path,
            &fk6,
            LimitsSpec {
                v: 5.0,
                a: 30.0,
                j: 3000.0,
                tcp: Some((0.25, f64::INFINITY, f64::INFINITY)),
            },
        ));
    }

    // ── 6-DOF TCP-tight (V/A/J all bound) ──────────────────────────────
    {
        let path = SRobotPath::<6, f64>::try_new(vec![
            SRobotQ::from_array([0.0, -1.2, 1.5, -0.3, 0.5, 0.0]),
            SRobotQ::from_array([0.6, -0.6, 0.9, 0.3, -0.2, 0.8]),
        ])
        .unwrap();
        rows.push(run_one(
            "6dof_tcp_full",
            path,
            &fk6,
            LimitsSpec {
                v: 5.0,
                a: 30.0,
                j: 3000.0,
                tcp: Some((1.0, 5.0, 50.0)),
            },
        ));
    }

    // ── tight jerk (forces jerk to bind) ───────────────────────────────
    {
        let fk = common::dh_1dof();
        let path = SRobotPath::<1, f64>::try_new(vec![
            SRobotQ::from_array([0.0]),
            SRobotQ::from_array([1.0]),
        ])
        .unwrap();
        rows.push(run_one(
            "1dof_tight_jerk",
            path,
            &fk,
            LimitsSpec {
                v: 1.0,
                a: 2.0,
                j: 4.0,
                tcp: None,
            },
        ));
    }

    // ── 6-DOF 10wp captured trajectory ─────────────────────────────────
    {
        let path = SRobotPath::<6, f64>::try_new(vec![
            SRobotQ::from_array([-1.1967357, 0.6513940, 0.0649984, -0.7458407, -1.0254644, 1.9914096]),
            SRobotQ::from_array([-1.4218939, 0.7337620, 0.3250841, -0.5453823, -0.9866293, 1.6930232]),
            SRobotQ::from_array([-1.5209670, 0.7668492, 0.5091862, -0.3223294, -0.8368485, 1.4805338]),
            SRobotQ::from_array([-1.3353859, 0.6944350, 0.5843410, -0.0834244, -0.9241249, 1.1985058]),
            SRobotQ::from_array([-1.5004166, 0.7278752, 0.6759865, -0.2666964, -0.7699144, 1.4176370]),
            SRobotQ::from_array([-1.3280353, 0.5349262, 0.4342674, -0.4279996, -0.9543195, 1.6290450]),
            SRobotQ::from_array([-1.2149905, 0.3504188, 0.7067139, -0.3298242, -0.9069862, 1.6441734]),
            SRobotQ::from_array([-1.2035334, 0.3318515, 1.0019210, -0.2021268, -0.6941859, 1.5207703]),
            SRobotQ::from_array([-1.3864710, 0.3202283, 1.1828311, -0.2501720, -0.8450851, 1.3589037]),
            SRobotQ::from_array([-1.5619611, 0.5108429, 1.3277226, -0.0363543, -0.6535910, 1.3245343]),
        ])
        .unwrap();
        rows.push(run_one(
            "6dof_10wp_captured",
            path,
            &fk6,
            LimitsSpec {
                v: 1.5,
                a: 6.0,
                j: 25.0,
                tcp: Some((2.0, 20.0, 200.0)),
            },
        ));
    }

    print_rows(&rows);
    print_peaks(&rows);
}
