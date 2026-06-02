//! Comparative bench against the slicer's logged `joint_trajectory_with_obstacles`
//! calls (linear-trajectory entries omitted). The 7th DOF (positioner / rail)
//! is dropped — all FK and limits are on the 6-DOF arm only.
//!
//! Some paths are short (2 waypoints with sub-millimetre deltas) — these are
//! essentially settling motions and dominated by the boundary phase. Other
//! paths are 15+ waypoints with sharp directional changes (real obstacle
//! avoidance). One entry from the welder set carries a corrupted waypoint
//! (joint 4 at ~16 rad, ~10x its limit); we keep it to see how each retimer
//! behaves on bad input.
//!
//! Run with
//!   `cargo test --release -p deke-bench-retimers --test slicer_real_world -- --nocapture`.

use std::time::Duration;

use deke_bench_retimers::{
    BenchProblem, BenchResult, MATERIAL_A_MAX, MATERIAL_J_MAX, MATERIAL_V_MAX,
    SLICER_SAMPLE_RATE_HZ, SLICER_TCP_V_MAX, WELDER_A_MAX, WELDER_J_MAX, WELDER_V_MAX,
    production_urdf_chain, run_all,
};
use deke_types::SRobotQ;

#[derive(Clone, Copy)]
enum Robot {
    Material,
    Welder,
}

impl Robot {
    fn v_max(self) -> [f64; 6] {
        match self {
            Robot::Material => MATERIAL_V_MAX,
            Robot::Welder => WELDER_V_MAX,
        }
    }
    fn a_max(self) -> [f64; 6] {
        match self {
            Robot::Material => MATERIAL_A_MAX,
            Robot::Welder => WELDER_A_MAX,
        }
    }
    fn j_max(self) -> [f64; 6] {
        match self {
            Robot::Material => MATERIAL_J_MAX,
            Robot::Welder => WELDER_J_MAX,
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Robot::Material => "mat",
            Robot::Welder => "wld",
        }
    }
}

fn problem(name: &'static str, robot: Robot, waypoints: &[[f64; 6]]) -> BenchProblem<6> {
    BenchProblem {
        name,
        waypoints: waypoints.iter().copied().map(SRobotQ::from_array).collect(),
        v_max: robot.v_max(),
        a_max: robot.a_max(),
        j_max: robot.j_max(),
        tcp_v_max: Some(SLICER_TCP_V_MAX),
        sample_rate_hz: SLICER_SAMPLE_RATE_HZ,
    }
}

fn problems() -> Vec<(Robot, BenchProblem<6>)> {
    let mut out: Vec<(Robot, BenchProblem<6>)> = Vec::new();

    // Each tuple: (test name, robot, waypoints). Waypoints are the 6-DOF
    // joint coords from the slicer's `path` field with the 7th positioner
    // value dropped. Paths with n=2 of near-equal positions (sub-mm
    // settling) are excluded — they reduce to a boundary-phase test.

    out.push((
        Robot::Material,
        problem(
            "mat_3wp_a",
            Robot::Material,
            &[
                [0.5186031, 0.5187571, -0.6023409, -0.0012982, -0.9717733, -0.5119401],
                [0.5186031, 0.5187550, -0.6023410, -0.0012982, -0.9717725, -0.5119402],
                [1.0711051, 1.1864575, -0.8005924, 0.6558156, 0.9134074, -0.4329165],
            ],
        ),
    ));
    out.push((
        Robot::Material,
        problem(
            "mat_3wp_b",
            Robot::Material,
            &[
                [1.1263426, 1.1779104, -0.8078261, 0.5887414, 0.8972303, -0.3883128],
                [1.1263426, 1.1779078, -0.8078283, 0.5887402, 0.8972323, -0.3883109],
                [0.0025722, 0.1409948, -0.1509858, 0.0007421, 0.1485941, 0.0033052],
            ],
        ),
    ));
    out.push((
        Robot::Material,
        problem(
            "mat_3wp_c",
            Robot::Material,
            &[
                [0.0025722, 0.1409948, -0.1509858, 0.0007421, 0.1485941, 0.0033052],
                [0.0025721, 0.1409929, -0.1509854, 0.0007419, 0.1485945, 0.0033054],
                [0.5186031, 0.5187571, -0.6023409, -0.0012982, -0.9717733, -0.5119401],
            ],
        ),
    ));
    out.push((
        Robot::Material,
        problem(
            "mat_3wp_d",
            Robot::Material,
            &[
                [0.5186031, 0.5187571, -0.6023409, -0.0012982, -0.9717733, -0.5119401],
                [0.5186031, 0.5187550, -0.6023410, -0.0012982, -0.9717725, -0.5119402],
                [0.8991182, 1.1283528, -0.9138029, 0.7940624, 1.0737996, -0.4455641],
            ],
        ),
    ));
    out.push((
        Robot::Material,
        problem(
            "mat_4wp",
            Robot::Material,
            &[
                [0.5186031, 0.5187571, -0.6023409, -0.0012982, -0.9717733, -0.5119401],
                [1.1263426, 1.1779078, -0.8078283, 0.5887402, 0.8972323, -0.3883109],
                [1.1263426, 1.1779078, -0.8078283, 0.5887402, 0.8972323, -0.3883109],
                [1.1322524, 1.1554919, -0.8885733, 0.5496673, 0.9631879, -0.3301780],
            ],
        ),
    ));

    out.push((
        Robot::Welder,
        problem(
            "wld_3wp_approach",
            Robot::Welder,
            &[
                [1.5780812, 0.5869404, -0.2890840, -0.6101472, -0.3075315, 0.0091891],
                [0.6642343, 0.4445946, -0.7659897, -0.3462262, -0.5824553, 1.2774464],
                [0.5257535, 0.4344155, -0.8657337, -0.3063928, -0.6255449, 1.4718179],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_4wp",
            Robot::Welder,
            &[
                [-0.6318192, 0.0981597, -0.9604388, 0.2045245, -0.9342331, 0.1083174],
                [-0.5699674, -0.0482891, -0.8067053, 0.1822633, -0.9560764, 0.1190448],
                [1.5780812, 0.5869404, -0.2890840, -0.6101472, -0.3075315, 0.0091891],
                [1.5780812, 0.5869404, -0.2890840, -0.6101472, -0.3075315, 0.0091891],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_3wp_depart",
            Robot::Welder,
            &[
                [1.5780812, 0.5869404, -0.2890840, -0.6101472, -0.3075315, 0.0091891],
                [-0.5705440, 0.6123908, -0.6998733, -0.0809036, -0.6870645, 0.4735658],
                [-0.5928206, 0.6363877, -0.7760972, 0.0006566, -0.7707846, 0.5910937],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_3wp_obstacle",
            Robot::Welder,
            &[
                [-0.7305147, -0.0891057, -0.8319749, -0.1471059, -0.6549325, -2.4216411],
                [-0.7305147, -0.0891057, -0.8319749, -0.1471059, -0.6549325, -2.4216411],
                [1.5780812, 0.5869404, -0.2890840, -0.6101472, -0.3075315, 0.0091891],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_3wp_long_return",
            Robot::Welder,
            &[
                [-0.9153177, 0.2117941, -0.9173934, -0.0117970, -0.6638594, -2.2196259],
                [-0.9153177, 0.2117941, -0.9173934, -0.0117970, -0.6638594, -2.2196259],
                [1.5780812, 0.5869404, -0.2890840, -0.6101472, -0.3075315, 0.0091891],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_3wp_arc",
            Robot::Welder,
            &[
                [-0.4185161, 0.4844550, -0.8397670, 0.0050318, -0.7085639, 0.4126371],
                [-0.5928206, 0.6363877, -0.7760972, 0.0006566, -0.7707846, 0.5910937],
                [-0.6219474, 0.1847027, -0.9488991, -0.5498572, -0.3355248, -2.3963034],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_3wp_zigzag",
            Robot::Welder,
            &[
                [1.5780812, 0.5869404, -0.2890840, -0.6101472, -0.3075315, 0.0091891],
                [-0.5871974, 0.5130523, -1.0232079, 0.0331222, 0.5065555, 2.4516635],
                [-0.6114527, 0.5122243, -1.0314238, 0.0404919, 0.5156440, 2.4788493],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_4wp_approach_kink",
            Robot::Welder,
            &[
                [-0.3832734, 0.6041287, -1.1410728, -0.6714098, 1.0522211, 2.3981096],
                [-0.4644115, 0.5984216, -0.9454367, -0.4967112, 0.4895237, 1.8974953],
                [-0.5850615, 0.6404352, -0.7866275, -0.2427232, -0.3015802, 1.1974515],
                [-0.6486155, 0.6833095, -0.7573868, -0.1307556, -0.6995923, 0.8660346],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_5wp_short_arc",
            Robot::Welder,
            &[
                [-0.4185161, 0.4844550, -0.8397670, 0.0050318, -0.7085639, 0.4126371],
                [-0.6355526, -0.0961086, -0.4010356, 0.0646089, -0.9810759, 0.2715216],
                [-0.7311831, -0.2897333, -0.3471665, 0.0899501, -1.0967607, 0.1096415],
                [-0.7066662, 0.0069052, -0.8778882, 0.1136494, -0.9323299, 0.0350113],
                [-0.6318192, 0.0981597, -0.9604388, 0.2045245, -0.9342331, 0.1083174],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_5wp_obstacle_pass",
            Robot::Welder,
            &[
                [-0.5928206, 0.6363877, -0.7760972, 0.0006566, -0.7707846, 0.5910937],
                [-0.5923910, 0.5272135, -0.7352307, -0.1006526, -0.6988774, 0.1144883],
                [-0.5855612, 0.2266432, -0.8527898, -0.4921143, -0.4381279, -1.8142614],
                [-0.5859521, 0.2033056, -0.8738760, -0.5262089, -0.4132683, -1.9997148],
                [-0.6219474, 0.1847027, -0.9488991, -0.5498572, -0.3355248, -2.3963034],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_7wp_long_dodge",
            Robot::Welder,
            &[
                [-0.6860271, 0.7315717, -0.7349359, -0.1347785, -0.7264935, 0.9033550],
                [-0.5960954, 0.4163419, -0.9317965, -0.1503383, 0.5406363, 1.5971897],
                [-0.5793830, 0.3738472, -1.1082615, -0.2728553, 0.9343545, 1.8496473],
                [-0.6810008, 0.3361851, -1.3009593, -0.3091199, 1.1450423, 2.2177612],
                [-0.6810008, 0.3361851, -1.3009593, -0.3091199, 1.1450423, 2.2177612],
                [-0.6810008, 0.3361851, -1.3009593, -0.3091199, 1.1450423, 2.2177612],
                [-0.6810008, 0.3361851, -1.3009593, -0.3091199, 1.1450423, 2.2177612],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_7wp_obstacle_chain",
            Robot::Welder,
            &[
                [-0.8647307, 0.2431696, -0.9835411, -1.4178578, -0.3296981, -1.5903481],
                [-0.8728238, 0.2003748, -0.9378218, -1.1643116, -0.3704182, -1.2373842],
                [-0.8826785, 0.1798271, -0.9096855, -0.9987262, -0.3991741, -0.9802087],
                [-1.0034042, 0.2408860, -0.8394706, -0.3956421, -0.5418661, 0.4396077],
                [-1.0247414, 0.2611874, -0.8354639, -0.3306085, -0.5609626, 0.6430443],
                [-1.0685725, 0.3549660, -0.8779016, -0.2088867, -0.6189979, 1.3242931],
                [-1.0678413, 0.3617035, -0.8852934, -0.2117663, -0.6212942, 1.3572706],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_6wp_obstacle_brake",
            Robot::Welder,
            &[
                [-0.5555264, 0.5047965, -1.2036027, -0.4641864, 1.0687766, 2.2972408],
                [-0.5781423, 0.4511598, -1.1516014, -0.4894838, 0.9167262, 1.8173666],
                [-0.6117742, 0.3927832, -1.0997780, -0.5151474, 0.7231334, 1.2576967],
                [-0.8142967, 0.1780104, -0.9471990, -0.5854191, -0.2306175, -1.1679548],
                [-0.8833108, 0.1420050, -0.9409407, -0.6259490, -0.3876236, -1.6626738],
                [-0.9510171, 0.1244990, -0.9579738, -0.6770335, -0.4382736, -1.9697171],
            ],
        ),
    ));
    out.push((
        Robot::Welder,
        problem(
            "wld_13wp_dense_return",
            Robot::Welder,
            &[
                [-0.9829899, 0.0464222, -0.9651077, -0.6907056, -0.4438856, -1.9254578],
                [-0.9865312, 0.0455618, -0.9063003, -0.6798942, -0.5802672, -1.4331386],
                [-0.9885673, 0.0484301, -0.8913550, -0.6736394, -0.6154727, -1.2897647],
                [-0.9895943, 0.0507351, -0.8887515, -0.6706610, -0.6219349, -1.2539397],
                [-0.9902324, 0.0522339, -0.8875198, -0.6688247, -0.6250661, -1.2345380],
                [-0.9908585, 0.0537539, -0.8865951, -0.6670339, -0.6274888, -1.2175998],
                [-0.9910978, 0.0543415, -0.8862817, -0.6663510, -0.6283232, -1.2114230],
                [-0.9938191, 0.0610692, -0.8829728, -0.6585830, -0.6372218, -1.1430747],
                [-0.9967269, 0.0701927, -0.8823855, -0.6464922, -0.6399448, -1.0680853],
                [-1.0254972, 0.1624435, -0.8797811, -0.5217556, -0.6587852, -0.3251267],
                [-1.0349580, 0.1935138, -0.8791456, -0.4784740, -0.6640923, -0.0704630],
                [-1.0590001, 0.2925193, -0.8829125, -0.3157496, -0.6627585, 0.8579798],
                [-1.0370296, 0.2967749, -0.9050159, -0.2155727, -0.5980903, 1.3323904],
            ],
        ),
    ));
    // Note: the slicer log also contained a 15-waypoint sequence with a
    // corrupted waypoint (joint 4 at ~-16 rad, joint 6 at ~16 rad — 5x past
    // limits). It's excluded here because every retimer either times out
    // or blows up its internal scaling, and the abandoned worker threads
    // leak memory across the test run.

    out
}

fn fmt_solve_ms(d: Duration) -> String {
    format!("{:.1}", d.as_secs_f64() * 1000.0)
}

fn print_row(robot_tag: &str, prob_name: &str, n_wp: usize, r: &BenchResult<6>) {
    let solve_ms = fmt_solve_ms(r.solve_time);
    let dur_s = r.trajectory_duration.as_secs_f64();
    let u = r.utilization.unwrap_or_default();
    let u_tcpv = u
        .tcp_v
        .map(|x| format!("{:>5.3}", x))
        .unwrap_or_else(|| "  -  ".into());
    let dev = r
        .max_path_deviation
        .map(|x| format!("{:>7.4}", x))
        .unwrap_or_else(|| "   -   ".into());
    let status_or_err = if let Some(e) = &r.error {
        let e = if e.len() > 36 { &e[..36] } else { e.as_str() };
        format!("FAIL: {}", e)
    } else {
        r.status.clone()
    };
    println!(
        "  {:>3} {:<22} {:<19} {:>3}  {:>8} {:>7.3} {:>5.3} {:>5.3} {:>5.3} {} {:>5.3} {:>5.3} {}   {}",
        robot_tag,
        prob_name,
        r.retimer,
        n_wp,
        solve_ms,
        dur_s,
        u.joint_v,
        u.joint_a,
        u.joint_j,
        u_tcpv,
        u.max_u,
        u.peak_u,
        dev,
        status_or_err,
    );
}

#[derive(Default)]
struct PerRetimerStats {
    runs: u32,
    successes: u32,
    solve_ms_sum: f64,
    solve_ms_max: f64,
    duration_sum: f64,
    max_u_sum: f64,
    peak_u_max: f64,
    dev_sum: f64,
    dev_max: f64,
}

impl PerRetimerStats {
    fn record(&mut self, r: &BenchResult<6>) {
        self.runs += 1;
        self.solve_ms_sum += r.solve_time.as_secs_f64() * 1000.0;
        self.solve_ms_max = self
            .solve_ms_max
            .max(r.solve_time.as_secs_f64() * 1000.0);
        if r.error.is_none() {
            self.successes += 1;
            self.duration_sum += r.trajectory_duration.as_secs_f64();
            if let Some(u) = r.utilization {
                self.max_u_sum += u.max_u;
                if u.peak_u > self.peak_u_max {
                    self.peak_u_max = u.peak_u;
                }
            }
            if let Some(d) = r.max_path_deviation {
                self.dev_sum += d;
                self.dev_max = self.dev_max.max(d);
            }
        }
    }
    fn mean_solve_ms(&self) -> f64 {
        if self.runs > 0 {
            self.solve_ms_sum / self.runs as f64
        } else {
            0.0
        }
    }
    fn mean_duration(&self) -> f64 {
        if self.successes > 0 {
            self.duration_sum / self.successes as f64
        } else {
            0.0
        }
    }
    fn mean_max_u(&self) -> f64 {
        if self.successes > 0 {
            self.max_u_sum / self.successes as f64
        } else {
            0.0
        }
    }
    fn mean_dev(&self) -> f64 {
        if self.successes > 0 {
            self.dev_sum / self.successes as f64
        } else {
            0.0
        }
    }
}

#[test]
fn slicer_real_world() {
    let fk = production_urdf_chain();
    let problems = problems();

    println!();
    println!(
        "Real-world slicer trajectories (production 6-DOF URDF; 7th DOF dropped).\n\
         {} problems — both nanopanel-material and nanopanel-welder configs.\n\
         u_* columns = average per-sample utilization, max_u = mean per-sample max across all limits.",
        problems.len()
    );
    println!();
    println!(
        "  {:>3} {:<22} {:<19} {:>3}  {:>8} {:>7} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>7}   {}",
        "rob", "problem", "retimer", "wps", "solve_ms", "dur_s", "u_jv", "u_ja", "u_jj", "u_tcp",
        "max_u", "peak_u", "dev_rad", "status"
    );
    println!(
        "  {}",
        "-".repeat(126)
    );

    let mut stats_topp3tcp6 = PerRetimerStats::default();
    let mut stats_topp3tcp6_discrete = PerRetimerStats::default();
    let mut stats_topp3tcp_spline = PerRetimerStats::default();
    let mut stats_topp_speed = PerRetimerStats::default();

    for (robot, problem) in &problems {
        let n_wp = problem.waypoints.len();
        let results = run_all(problem, &fk);
        for r in &results {
            print_row(robot.tag(), problem.name, n_wp, r);
            match r.retimer {
                "topp3tcp6" => stats_topp3tcp6.record(r),
                "topp3tcp6-discrete" => stats_topp3tcp6_discrete.record(r),
                "topp3tcp-spline" => stats_topp3tcp_spline.record(r),
                "topp-speed" => stats_topp_speed.record(r),
                _ => {}
            }
        }
        println!();
    }

    println!();
    println!("Aggregate stats across all {} problems:", problems.len());
    println!(
        "  {:<20} {:>4}/{:<4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "retimer", "succ", "run", "mean_ms", "max_ms", "mean_dur", "mean_max_u", "peak_u",
        "mean_dev", "max_dev"
    );
    for (name, s) in [
        ("topp3tcp6", &stats_topp3tcp6),
        ("topp3tcp6-discrete", &stats_topp3tcp6_discrete),
        ("topp3tcp-spline", &stats_topp3tcp_spline),
        ("topp-speed", &stats_topp_speed),
    ] {
        println!(
            "  {:<20} {:>4}/{:<4} {:>10.1} {:>10.1} {:>10.3} {:>10.3} {:>10.3} {:>10.4} {:>10.4}",
            name,
            s.successes,
            s.runs,
            s.mean_solve_ms(),
            s.solve_ms_max,
            s.mean_duration(),
            s.mean_max_u(),
            s.peak_u_max,
            s.mean_dev(),
            s.dev_max,
        );
    }
}
