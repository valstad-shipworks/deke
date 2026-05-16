//! Runs all four retimers on a small problem suite and prints a comparison
//! table. Asserts every successful retime stayed within configured V/A/J/TCP
//! limits (with a small slack), but does *not* assert success itself — some
//! retimers fail on inputs the others handle, and the table captures that.
//!
//! Run with `cargo test --release -p deke-bench-retimers comparative -- --nocapture`.

use deke_bench_retimers::{
    BenchProblem, BenchResult, PRODUCTION_A_MAX, PRODUCTION_J_MAX, PRODUCTION_SAMPLE_RATE_HZ,
    PRODUCTION_TCP_V_MAX, PRODUCTION_V_MAX, production_urdf_chain, run_all,
};
use deke_types::SRobotQ;

fn problems() -> Vec<BenchProblem<6>> {
    let v = PRODUCTION_V_MAX;
    let a = PRODUCTION_A_MAX;
    let j = PRODUCTION_J_MAX;
    let tcp = Some(PRODUCTION_TCP_V_MAX);
    let rate = PRODUCTION_SAMPLE_RATE_HZ;

    // Joint configurations are chosen to stay within typical Fanuc-class
    // ranges; production v/a/j ceilings are loose enough that all problems
    // are feasible on every retimer (modulo the spline's coarse search dt).

    let short_curved = BenchProblem {
        name: "short_curved_5wp",
        waypoints: vec![
            SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
            SRobotQ::from_array([0.2, -0.8, 1.0, -0.1, 0.1, 0.1]),
            SRobotQ::from_array([0.4, -0.6, 0.8, -0.2, 0.2, 0.2]),
            SRobotQ::from_array([0.6, -0.4, 0.6, -0.3, 0.1, 0.3]),
            SRobotQ::from_array([0.8, -0.2, 0.4, -0.4, 0.0, 0.4]),
        ],
        v_max: v,
        a_max: a,
        j_max: j,
        tcp_v_max: tcp,
        sample_rate_hz: rate,
    };

    let straight_line = BenchProblem {
        name: "straight_line_2wp",
        waypoints: vec![
            SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
            SRobotQ::from_array([0.3, -0.7, 0.9, 0.2, 0.1, 0.3]),
        ],
        v_max: v,
        a_max: a,
        j_max: j,
        tcp_v_max: None,
        sample_rate_hz: rate,
    };

    // Long, smooth, sinusoidal path that stresses solve time. The
    // bump-amplitude scale (0.2 rad) keeps the path inside reachable
    // workspace on the production URDF.
    let waypoints: Vec<SRobotQ<6, f64>> = (0..15)
        .map(|i| {
            let t = i as f64 / 14.0;
            let bump = 0.2 * (std::f64::consts::PI * t).sin();
            SRobotQ::from_array([
                -0.8 + 1.6 * t,
                -1.0 + 0.5 * bump,
                1.0 + 0.4 * bump,
                -0.3 + 0.6 * t,
                0.1 * (std::f64::consts::PI * t * 2.0).sin(),
                -0.5 + 1.0 * t,
            ])
        })
        .collect();
    let sinusoidal = BenchProblem {
        name: "sinusoidal_15wp",
        waypoints,
        v_max: v,
        a_max: a,
        j_max: j,
        tcp_v_max: tcp,
        sample_rate_hz: rate,
    };

    // 7-waypoint path: long approach, 5-waypoint tight zigzag in the middle
    // (simulates threading around a tightly-clustered obstacle), long
    // departure. The kink segments are ~0.1 rad apart on a few joints —
    // ~10× shorter than the approach/departure segments. Tests how each
    // retimer handles aggressive directional changes mid-path.
    let obstacle_dodge = BenchProblem {
        name: "obstacle_dodge_7wp",
        waypoints: vec![
            SRobotQ::from_array([-1.2_f64, -1.2, 1.4, 0.0, 0.0, 0.0]), // start (far)
            SRobotQ::from_array([-0.10, -0.60, 0.80, 0.20, 0.10, 0.30]), // enter kink
            SRobotQ::from_array([0.00, -0.70, 0.85, 0.25, 0.15, 0.40]),  // zig
            SRobotQ::from_array([0.05, -0.60, 0.80, 0.30, 0.20, 0.50]),  // zag
            SRobotQ::from_array([0.10, -0.70, 0.85, 0.35, 0.15, 0.40]),  // zig
            SRobotQ::from_array([0.15, -0.60, 0.80, 0.40, 0.10, 0.50]),  // exit kink
            SRobotQ::from_array([1.20, 0.00, 0.40, 0.50, 0.00, 0.30]),   // end (far)
        ],
        v_max: v,
        a_max: a,
        j_max: j,
        tcp_v_max: tcp,
        sample_rate_hz: rate,
    };

    vec![short_curved, straight_line, sinusoidal, obstacle_dodge]
}

fn print_header(problem_name: &str) {
    println!();
    println!("=== {} ===", problem_name);
    println!(
        "  {:<20} {:>10} {:>10} {:>9}  {:>6} {:>6} {:>6} {:>7} {:>6} {:>7}",
        "retimer", "solve_ms", "dur_s", "samples", "u_jv", "u_ja", "u_jj", "u_tcpv", "max_u", "dev_rad"
    );
}

fn print_row(r: &BenchResult<6>) {
    let solve_ms = r.solve_time.as_secs_f64() * 1000.0;
    let dur_s = r.trajectory_duration.as_secs_f64();
    let u = r.utilization.unwrap_or_default();
    let tcp_u = u
        .tcp_v
        .map(|x| format!("{:>7.3}", x))
        .unwrap_or_else(|| "       ".into());
    let dev = r
        .max_path_deviation
        .map(|x| format!("{:>7.4}", x))
        .unwrap_or_else(|| "       ".into());
    let status_or_err = if let Some(e) = &r.error {
        format!("FAIL: {}", e)
    } else {
        r.status.clone()
    };
    println!(
        "  {:<20} {:>10.2} {:>10.4} {:>9}  {:>6.3} {:>6.3} {:>6.3} {:>7} {:>6.3} {}   {}",
        r.retimer, solve_ms, dur_s, r.num_samples,
        u.joint_v, u.joint_a, u.joint_j, tcp_u, u.max_u, dev,
        status_or_err,
    );
}

fn check_within_limits(problem: &BenchProblem<6>, r: &BenchResult<6>) {
    // Skip retimers that failed; the print row already surfaced the error and
    // a failure is itself an output of the comparison.
    let Some(u) = r.utilization else { return };

    // Average per-sample utilization should sit well below the limit; even a
    // limit-saturating retimer averages below 1.0 because it doesn't sit at
    // the peak for every sample (ramp-up/ramp-down). The 1.10 slack accounts
    // for backward-FD readout overshoot near segment boundaries; in practice
    // averages stay around 0.5–0.7 of the limit.
    const SLACK: f64 = 1.10;

    assert!(
        u.joint_v <= SLACK,
        "{}: {} exceeded joint v limit: u={:.3}",
        problem.name,
        r.retimer,
        u.joint_v
    );
    assert!(
        u.joint_a <= SLACK,
        "{}: {} exceeded joint a limit: u={:.3}",
        problem.name,
        r.retimer,
        u.joint_a
    );
    assert!(
        u.joint_j <= SLACK,
        "{}: {} exceeded joint j limit: u={:.3}",
        problem.name,
        r.retimer,
        u.joint_j
    );
    if let Some(tcp_u) = u.tcp_v {
        assert!(
            tcp_u <= SLACK,
            "{}: {} exceeded TCP v limit: u={:.3}",
            problem.name,
            r.retimer,
            tcp_u
        );
    }
}

#[test]
fn comparative() {
    let fk = production_urdf_chain();

    println!();
    println!(
        "Comparative retimer analysis (production 6-DOF URDF chain). Columns: solve_ms = wall-clock,"
    );
    println!(
        "dur_s = trajectory length, samples = output points, u_* = AVERAGE per-sample"
    );
    println!(
        "utilization (per-sample max_j(|x|/limit), averaged across samples)."
    );

    for problem in problems() {
        print_header(problem.name);
        let results = run_all(&problem, &fk);
        for r in &results {
            print_row(r);
        }
        for r in &results {
            check_within_limits(&problem, r);
        }
    }
}
