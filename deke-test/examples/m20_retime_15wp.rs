//! Retimes a hand-authored 15-waypoint joint-space path for the Fanuc M-20iD/12L and dumps
//! the full diagnostic report, per-waypoint timing, and per-joint/TCP analytical peaks.
//!
//! Run with `cargo run -p deke-test --release --example m20_retime_15wp`.

use std::error::Error;
use std::time::Duration;

use deke_test::m20id12l::*;
use deke_topp3tcp6::{BoundaryConditions, JointLimits, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{FKChain, JointValidator, Retimer, SRobotPath, SRobotQ, URDFChain};

fn main() -> Result<(), Box<dyn Error>> {
    let waypoints: Vec<SRobotQ<6>> = (0..15)
        .map(|i| {
            let t = i as f32 / 14.0;
            let pi = std::f32::consts::PI;
            let bump = 0.2 * (pi * t).sin();
            SRobotQ::from_array([
                -1.0 + 2.0 * t,
                -0.2 + 0.5 * bump,
                0.2 + 0.4 * bump,
                -0.3 + 0.6 * t,
                0.1 * (pi * t * 2.0).sin(),
                -0.5 + 1.0 * t,
            ])
        })
        .collect();
    let path = SRobotPath::<6>::try_new(waypoints)?;
    println!(
        "input path : {} waypoints, arc_length = {:.4} rad",
        path.len(),
        path.arc_length()
    );
    let seg_lens = path.segment_lengths();
    println!(
        "segment lens (rad): min {:.4}  max {:.4}  mean {:.4}",
        seg_lens.iter().cloned().fold(f32::INFINITY, f32::min),
        seg_lens.iter().cloned().fold(0.0_f32, f32::max),
        seg_lens.iter().sum::<f32>() / seg_lens.len() as f32,
    );
    println!();

    let fk = URDFChain::<6>::new(URDF_JOINTS)?;

    let joint_v_max: [f32; 6] = [
        210.0_f32.to_radians(),
        210.0_f32.to_radians(),
        265.0_f32.to_radians(),
        420.0_f32.to_radians(),
        450.0_f32.to_radians(),
        720.0_f32.to_radians(),
    ];
    let joint_a_max: [f32; 6] = [
        605.77_f32.to_radians(),
        605.77_f32.to_radians(),
        764.42_f32.to_radians(),
        1211.54_f32.to_radians(),
        1298.08_f32.to_radians(),
        2076.92_f32.to_radians(),
    ];
    let joint_j_max: [f32; 6] = [
        3494.82_f32.to_radians(),
        3494.82_f32.to_radians(),
        4410.13_f32.to_radians(),
        6989.65_f32.to_radians(),
        7488.91_f32.to_radians(),
        11982.25_f32.to_radians(),
    ];

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.0, 1.0, 1.0);
    cfg.joint = JointLimits {
        q_min: SRobotQ::from_array(JOINT_LOWER),
        q_max: SRobotQ::from_array(JOINT_UPPER),
        v_max: SRobotQ::from_array(joint_v_max),
        a_max: SRobotQ::from_array(joint_a_max),
        j_max: SRobotQ::from_array(joint_j_max),
    };
    cfg.tcp.v_max = 3.0;
    cfg.tcp.a_max = f32::INFINITY;
    cfg.tcp.j_max = f32::INFINITY;
    cfg.boundary = BoundaryConditions::rest_to_rest();
    cfg.sample_rate_hz = 250.0;
    cfg.solver.max_iterations = 4_000;
    cfg.densification.max_segment_step = Some(0.15);
    cfg.densification.max_samples = 80;
    cfg.solver.diagnostics = false;

    println!("constraints:");
    println!("  joint v_max : {}", fmt_deg_per_s(&joint_v_max));
    println!("  joint a_max : {}", fmt_deg_per_s(&joint_a_max));
    println!("  joint j_max : {}", fmt_deg_per_s(&joint_j_max));
    println!(
        "  tcp   v/a/j : {:.2} m/s / {:.1} m/s² / {:.0} m/s³",
        cfg.tcp.v_max, cfg.tcp.a_max, cfg.tcp.j_max
    );
    println!("  output rate : {:.0} Hz", cfg.sample_rate_hz);
    println!();

    let mut validator = JointValidator::<6>::new(
        SRobotQ::from_array(JOINT_LOWER),
        SRobotQ::from_array(JOINT_UPPER),
    );
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    print!("{}", diag);
    println!();

    let traj = result?;
    println!(
        "trajectory: {} samples at dt = {:.2} ms (total {:.4}s)",
        traj.len(),
        traj.dt().as_secs_f32() * 1000.0,
        traj.duration().as_secs_f32()
    );

    let dt_s = traj.dt().as_secs_f32();
    let mut peak_v_per_joint = [0.0_f32; 6];
    let mut peak_a_per_joint = [0.0_f32; 6];
    let mut peak_j_per_joint = [0.0_f32; 6];
    for i in 0..traj.len() {
        if let Some(v) = traj.velocity_at(i) {
            for j in 0..6 {
                peak_v_per_joint[j] = peak_v_per_joint[j].max(v.0[j].abs());
            }
        }
        if let Some(a) = traj.acceleration_at(i) {
            for j in 0..6 {
                peak_a_per_joint[j] = peak_a_per_joint[j].max(a.0[j].abs());
            }
        }
    }
    for i in 0..traj.len().saturating_sub(3) {
        let q0 = traj[i];
        let q1 = traj[i + 1];
        let q2 = traj[i + 2];
        let q3 = traj[i + 3];
        let inv = 1.0_f32 / (dt_s * dt_s * dt_s);
        for j in 0..6 {
            let jk = (q3.0[j] - 3.0 * q2.0[j] + 3.0 * q1.0[j] - q0.0[j]) * inv;
            peak_j_per_joint[j] = peak_j_per_joint[j].max(jk.abs());
        }
    }

    println!();
    println!(
        "per-joint peaks (output finite-difference, {} | {} | {}):",
        "velocity [deg/s]", "accel [deg/s²]", "jerk [deg/s³]"
    );
    for j in 0..6 {
        println!(
            "  J{}: v {:>7.1} / {:>7.1}  a {:>8.1} / {:>8.1}  j {:>9.1} / {:>9.1}",
            j + 1,
            peak_v_per_joint[j].to_degrees(),
            joint_v_max[j].to_degrees(),
            peak_a_per_joint[j].to_degrees(),
            joint_a_max[j].to_degrees(),
            peak_j_per_joint[j].to_degrees(),
            joint_j_max[j].to_degrees(),
        );
    }

    let mut tcp_speed_samples = Vec::with_capacity(traj.len());
    let mut prev: Option<[f32; 3]> = None;
    for q in traj.iter() {
        let p = fk.fk_end(q)?.translation;
        let cur = [p.x, p.y, p.z];
        if let Some(pr) = prev {
            let dx = cur[0] - pr[0];
            let dy = cur[1] - pr[1];
            let dz = cur[2] - pr[2];
            tcp_speed_samples.push((dx * dx + dy * dy + dz * dz).sqrt() / dt_s);
        }
        prev = Some(cur);
    }
    let tcp_peak = tcp_speed_samples.iter().copied().fold(0.0_f32, f32::max);
    let tcp_mean = tcp_speed_samples.iter().sum::<f32>() / tcp_speed_samples.len() as f32;
    println!();
    println!(
        "TCP speed : peak {:.3} m/s (limit {:.2}), mean {:.3} m/s",
        tcp_peak, cfg.tcp.v_max, tcp_mean
    );

    let cumulative = cumulative_waypoint_times(&traj, &path);
    println!();
    println!("per-waypoint timing (arrival time of each input waypoint):");
    println!("  {:>3} {:>10} {:>10}", "idx", "t [s]", "Δt [s]");
    let mut prev_t = Duration::ZERO;
    for (i, t) in cumulative.iter().enumerate() {
        println!(
            "  {:>3} {:>10.4} {:>10.4}",
            i,
            t.as_secs_f64(),
            (*t - prev_t).as_secs_f64()
        );
        prev_t = *t;
    }

    Ok(())
}

fn cumulative_waypoint_times(
    traj: &deke_types::SRobotTraj<6>,
    path: &SRobotPath<6>,
) -> Vec<Duration> {
    let dt = traj.dt();
    let mut out = Vec::with_capacity(path.len());
    let mut sample_cursor = 0_usize;
    for (wi, wp) in path.iter().enumerate() {
        let mut best = (f32::INFINITY, sample_cursor);
        for i in sample_cursor..traj.len() {
            let q = traj[i];
            let mut d = 0.0_f32;
            for j in 0..6 {
                d += (q.0[j] - wp.0[j]).powi(2);
            }
            if d < best.0 {
                best = (d, i);
            }
            if wi + 1 == path.len() && i + 1 == traj.len() {
                break;
            }
        }
        sample_cursor = best.1;
        out.push(dt.saturating_mul(best.1 as u32));
    }
    out
}

fn fmt_deg_per_s(v: &[f32; 6]) -> String {
    format!(
        "[{:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1}] deg/s",
        v[0].to_degrees(),
        v[1].to_degrees(),
        v[2].to_degrees(),
        v[3].to_degrees(),
        v[4].to_degrees(),
        v[5].to_degrees(),
    )
}
