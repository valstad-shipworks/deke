//! Plan + retime a path for the Fanuc M-20iD/12L, then render joint and TCP trajectories
//! as a PNG.
//!
//! Run with `cargo run -p deke-test --release --example m20_retime_plot` — the output goes
//! to `target/m20_retime.png`.

use std::error::Error;
use std::time::Duration;

use deke_test::m20id12l::*;
use deke_topp3tcp6::{BoundaryConditions, JointLimits, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{FKChain, JointValidator, Planner, Retimer, SRobotQ, URDFChain};
use plotters::prelude::*;
use plotters::style::RGBColor;

const START: [f32; 6] = [-1.0, 0.2, -0.3, 0.5, 0.1, -0.5];
const GOAL: [f32; 6] = [1.5, -0.8, 0.4, -1.0, 0.8, 2.0];

const JOINT_COLORS: [RGBColor; 6] = [
    RGBColor(220, 38, 38),
    RGBColor(234, 88, 12),
    RGBColor(202, 138, 4),
    RGBColor(22, 163, 74),
    RGBColor(37, 99, 235),
    RGBColor(139, 92, 246),
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut env = wreck::Collider::default();
    for &(x, y, z, r) in &[
        (0.5_f32, 0.0, 0.5, 0.15),
        (-0.3, 0.4, 0.3, 0.12),
        (0.0, -0.5, 0.6, 0.10),
    ] {
        env.add(wreck::Sphere::new(glam::Vec3::new(x, y, z), r));
    }

    let mut validator = validator();
    let ctx = ((), deke_wreck::WreckValidatorContext::new(&env));

    let rrtc_settings = deke_rrt::RrtcSettings::new(
        SRobotQ(JOINT_LOWER),
        SRobotQ(JOINT_UPPER),
    );
    let planner = rrtc();
    let (plan_result, plan_diag) = planner.plan(
        &rrtc_settings,
        SRobotQ(START),
        SRobotQ(GOAL),
        &mut validator,
        &ctx,
    );
    let path = plan_result.expect("rrtc failed");
    println!(
        "planned: {} waypoints, arc_length={:.3}, {}",
        path.len(),
        path.arc_length(),
        plan_diag
    );

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
    cfg.tcp.v_max = 2.0;
    cfg.tcp.a_max = 20.0;
    cfg.tcp.j_max = 500.0;
    cfg.boundary = BoundaryConditions::rest_to_rest();
    cfg.sample_rate_hz = 250.0;
    cfg.solver.max_iterations = 2_000;

    let mut joint_validator = JointValidator::<6>::new(
        SRobotQ::from_array(JOINT_LOWER),
        SRobotQ::from_array(JOINT_UPPER),
    );
    let (retime_result, retime_diag) =
        Topp3Tcp6.retime(&cfg, &path, &fk, &mut joint_validator, &());
    println!("{}", retime_diag);
    let traj = retime_result.expect("retime failed");

    let dt_s = traj.dt().as_secs_f32();
    let n = traj.len();
    let times: Vec<f32> = (0..n).map(|i| i as f32 * dt_s).collect();

    let mut positions = vec![vec![0.0_f32; n]; 6];
    for (i, q) in traj.iter().enumerate() {
        for j in 0..6 {
            positions[j][i] = q.0[j];
        }
    }

    let velocities = derivative(&positions, dt_s);
    let accelerations = derivative(&velocities, dt_s);
    let jerks = derivative(&accelerations, dt_s);

    let mut tcp_pos = vec![[0.0_f32; 3]; n];
    for (i, q) in traj.iter().enumerate() {
        let p = fk.fk_end(q)?.translation;
        tcp_pos[i] = [p.x, p.y, p.z];
    }
    let mut tcp_speed = vec![0.0_f32; n];
    for i in 0..n {
        let (prev, next, span) = if i == 0 {
            (tcp_pos[0], tcp_pos[1.min(n - 1)], 1.0)
        } else if i == n - 1 {
            (tcp_pos[n - 2], tcp_pos[n - 1], 1.0)
        } else {
            (tcp_pos[i - 1], tcp_pos[i + 1], 2.0)
        };
        let dx = (next[0] - prev[0]) / (span * dt_s);
        let dy = (next[1] - prev[1]) / (span * dt_s);
        let dz = (next[2] - prev[2]) / (span * dt_s);
        tcp_speed[i] = (dx * dx + dy * dy + dz * dz).sqrt();
    }

    let out_path = "target/m20_retime.png";
    let root = BitMapBackend::new(out_path, (1400, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let header = format!(
        "M-20iD/12L retime | {:.3}s | {} samples @ {:.0}Hz | {:.1}% avg utilization",
        retime_diag.total_time.as_secs_f32(),
        retime_diag.output_samples,
        1.0 / dt_s,
        retime_diag.average_utilization * 100.0,
    );
    let (title_area, plot_area) = root.split_vertically(40);
    title_area.titled(&header, ("sans-serif", 24).into_font())?;
    let panels = plot_area.split_evenly((2, 2));

    plot_joint_series(&panels[0], "Joint position [rad]", &times, &positions, None)?;
    plot_joint_series(
        &panels[1],
        "Joint velocity [rad/s]",
        &times,
        &velocities,
        Some(&joint_v_max),
    )?;
    plot_joint_series(
        &panels[2],
        "Joint acceleration [rad/s²]",
        &times,
        &accelerations,
        Some(&joint_a_max),
    )?;
    plot_tcp_panel(&panels[3], &times, &tcp_speed, cfg.tcp.v_max)?;

    root.present()?;
    println!("wrote {}", out_path);

    println!(
        "peak jerk from output finite-diff: {:.1} rad/s³ (analytical peak: {:.1})",
        jerks
            .iter()
            .flat_map(|r| r.iter().map(|x| x.abs()))
            .fold(0.0_f32, f32::max),
        retime_diag.peak_joint_jerk
    );

    Ok(())
}

fn derivative(series: &[Vec<f32>], dt: f32) -> Vec<Vec<f32>> {
    let n = series[0].len();
    let mut out = vec![vec![0.0; n]; series.len()];
    for j in 0..series.len() {
        for i in 0..n {
            let (a, b, span) = if i == 0 {
                (series[j][0], series[j][1.min(n - 1)], 1.0)
            } else if i == n - 1 {
                (series[j][n - 2], series[j][n - 1], 1.0)
            } else {
                (series[j][i - 1], series[j][i + 1], 2.0)
            };
            out[j][i] = (b - a) / (span * dt);
        }
    }
    out
}

fn plot_joint_series<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    title: &str,
    times: &[f32],
    joints: &[Vec<f32>],
    limits: Option<&[f32; 6]>,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    let t_min = 0.0_f32;
    let t_max = *times.last().unwrap_or(&1.0);
    let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for row in joints {
        for &v in row {
            y_min = y_min.min(v);
            y_max = y_max.max(v);
        }
    }
    if let Some(lims) = limits {
        for l in lims {
            y_min = y_min.min(-l);
            y_max = y_max.max(*l);
        }
    }
    let pad = (y_max - y_min).abs().max(1e-3) * 0.05;
    let (y_min, y_max) = (y_min - pad, y_max + pad);

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 18).into_font())
        .margin(10)
        .x_label_area_size(32)
        .y_label_area_size(48)
        .build_cartesian_2d(t_min..t_max, y_min..y_max)?;
    chart
        .configure_mesh()
        .x_desc("time [s]")
        .axis_desc_style(("sans-serif", 14))
        .draw()?;

    for (j, row) in joints.iter().enumerate() {
        let color = JOINT_COLORS[j];
        chart
            .draw_series(LineSeries::new(
                times.iter().copied().zip(row.iter().copied()),
                color.stroke_width(2),
            ))?
            .label(format!("J{}", j + 1))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color.stroke_width(2)));

        if let Some(lims) = limits {
            let l = lims[j];
            for sign in [1.0_f32, -1.0] {
                chart.draw_series(LineSeries::new(
                    [(t_min, sign * l), (t_max, sign * l)],
                    color.mix(0.25).stroke_width(1),
                ))?;
            }
        }
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    Ok(())
}

fn plot_tcp_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    times: &[f32],
    speed: &[f32],
    v_max: f32,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    let t_min = 0.0_f32;
    let t_max = *times.last().unwrap_or(&1.0);
    let y_peak = speed.iter().copied().fold(0.0_f32, f32::max).max(v_max);
    let y_max = y_peak * 1.1;

    let mut chart = ChartBuilder::on(area)
        .caption("TCP speed [m/s]", ("sans-serif", 18).into_font())
        .margin(10)
        .x_label_area_size(32)
        .y_label_area_size(48)
        .build_cartesian_2d(t_min..t_max, 0.0..y_max)?;
    chart
        .configure_mesh()
        .x_desc("time [s]")
        .axis_desc_style(("sans-serif", 14))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            times.iter().copied().zip(speed.iter().copied()),
            RGBColor(15, 23, 42).stroke_width(2),
        ))?
        .label("|ṗ|")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 18, y)], RGBColor(15, 23, 42).stroke_width(2))
        });

    if v_max.is_finite() {
        chart
            .draw_series(LineSeries::new(
                [(t_min, v_max), (t_max, v_max)],
                RGBColor(220, 38, 38).mix(0.6).stroke_width(1),
            ))?
            .label("v_max")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 18, y)],
                    RGBColor(220, 38, 38).mix(0.6).stroke_width(1),
                )
            });
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    Ok(())
}

#[allow(dead_code)]
fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}
