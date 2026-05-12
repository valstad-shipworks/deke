//! Benchmark suite — many synthetic trajectories against the external 6-DOF URDF chain
//! and constraints. Run with:
//!
//! ```text
//! cargo test --test external_bench -- --ignored --nocapture
//! ```
//!
//! It is `#[ignore]`d by default because a full sweep takes ~30–60 s and is not the
//! kind of thing you want running on every `cargo test` invocation. The output is a
//! markdown-formatted table summarising success rate, iteration count, and wall time
//! per category — paste the table into a notebook or PR description.
//!
//! All trajectories are generated with a deterministic LCG so successive runs are
//! comparable across versions. Bumping the seed (`SEED` const at the top) gives a
//! different draw of trajectories from the same shape distribution.

use std::time::{Duration, Instant};

use deke_topp3tcp6::{SolveStatus, TcpLimits, Topp3Tcp6, Topp3Tcp6Constraints};
use deke_types::{JointValidator, Retimer, SRobotPath, SRobotQ, URDFChain, URDFJoint};

// ----------------------------------------------------------------------------
// External FK chain (verbatim copy from the user's project, same as
// `external_failures.rs` — kept duplicated here so each test file is
// self-contained).
// ----------------------------------------------------------------------------

const URDF_JOINTS: [URDFJoint; 6] = [
    URDFJoint::revolute((0.0, 0.0, 0.152), (0.0, -0.0, 0.0), (0.0, 0.0, 1.0)),
    URDFJoint::revolute(
        (0.075, -0.105, 0.273),
        (-1.5708, -0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    URDFJoint::revolute(
        (-0.00000000000000625888, -0.84, 0.04028),
        (-3.14159, -0.0000000000000000252315, -0.00000000000000423966),
        (0.0, 0.0, 1.0),
    ),
    URDFJoint::revolute(
        (0.295618, 0.215, -0.0642),
        (-3.14159, 1.5708, 0.0),
        (0.0, 0.0, 1.0),
    ),
    URDFJoint::revolute(
        (-0.0501976, 0.000491285, -1.0445),
        (-3.13181, 1.5708, 0.0),
        (0.0, 0.0, 1.0),
    ),
    URDFJoint::revolute(
        (0.075182, -0.00000000205208, -0.0507),
        (3.14159, 1.5708, 0.0),
        (0.0, 0.0, 1.0),
    ),
];

const URDF_FIXED_SUFFIX: [URDFJoint; 1] = [URDFJoint::fixed(
    (
        -0.00000000000000056205,
        -0.000000000000000888178,
        -0.00000000000000310862,
    ),
    (-1.5708, 1.5708, 3.14159),
)];

fn external_chain() -> URDFChain<6, f64> {
    URDFChain::<6, f64>::new_f64(URDF_JOINTS)
        .expect("URDFChain::new_f64 fixture is broken")
        .with_fixed_suffix_f64(&URDF_FIXED_SUFFIX)
        .expect("URDF fixed-suffix fixture is broken")
}

fn external_cfg() -> Topp3Tcp6Constraints<6> {
    use deke_topp3tcp6::JointLimits;
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.0, 1.0, 1.0);
    cfg.joint = JointLimits {
        q_min: SRobotQ::from_array([f64::NEG_INFINITY; 6]),
        q_max: SRobotQ::from_array([f64::INFINITY; 6]),
        v_max: SRobotQ::from_array([
            2.748894, 2.748894, 3.468842, 5.497787, 5.890486, 9.424778,
        ]),
        a_max: SRobotQ::from_array([
            6.170564, 6.170564, 7.786665, 12.341129, 13.222637, 21.156220,
        ]),
        j_max: SRobotQ::from_array([
            22.897033, 22.897033, 28.893875, 45.794066, 49.065074, 78.504119,
        ]),
    };
    cfg.tcp = TcpLimits {
        v_max: 2.0,
        a_max: 20.0,
        j_max: 200.0,
    };
    cfg.sample_rate_hz = 125.0;
    cfg.post_validation = false;
    cfg
}

fn validator() -> JointValidator<6, f64> {
    JointValidator::<6, f64>::new(
        SRobotQ::from_array([-10.0; 6]),
        SRobotQ::from_array([10.0; 6]),
    )
}

// ----------------------------------------------------------------------------
// Deterministic trajectory generator (small LCG; we don't pull in the rand crate
// just for repeatable benchmark inputs).
// ----------------------------------------------------------------------------

const SEED: u64 = 0xC0FFEE_DECAF_F00D;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / (1u64 << 32) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
    fn joint_pose(&mut self, scale: f64) -> [f64; 6] {
        // Scale is in radians — peak per-joint amplitude. We constrain joint 1 (the
        // shoulder) to a narrower band because the URDF chain pulls its workspace down
        // to weird singular places at extreme shoulder angles.
        let mut q = [0.0_f64; 6];
        q[0] = self.range(-1.5, 1.5) * scale;
        q[1] = self.range(-0.5, 1.0) * scale;
        q[2] = self.range(-1.5, 0.5) * scale;
        q[3] = self.range(-1.5, 1.5) * scale;
        q[4] = self.range(-1.5, 1.5) * scale;
        q[5] = self.range(-3.0, 3.0) * scale;
        q
    }
    fn perturb(&mut self, base: [f64; 6], delta: f64) -> [f64; 6] {
        let mut out = base;
        for v in out.iter_mut() {
            *v += self.range(-delta, delta);
        }
        out
    }
}

// ----------------------------------------------------------------------------
// Per-category trajectory generators
// ----------------------------------------------------------------------------

fn gen_two_wp(rng: &mut Lcg, joint_delta: f64) -> Vec<SRobotQ<6, f64>> {
    let a = rng.joint_pose(1.0);
    let b = rng.perturb(a, joint_delta);
    vec![SRobotQ::from_array(a), SRobotQ::from_array(b)]
}

fn gen_wrist_only(rng: &mut Lcg, joint_delta: f64) -> Vec<SRobotQ<6, f64>> {
    let a = rng.joint_pose(1.0);
    let mut b = a;
    b[3] += rng.range(-joint_delta, joint_delta);
    b[4] += rng.range(-joint_delta, joint_delta);
    b[5] += rng.range(-joint_delta, joint_delta);
    vec![SRobotQ::from_array(a), SRobotQ::from_array(b)]
}

fn gen_base_only(rng: &mut Lcg, joint_delta: f64) -> Vec<SRobotQ<6, f64>> {
    let a = rng.joint_pose(1.0);
    let mut b = a;
    b[0] += rng.range(-joint_delta, joint_delta);
    b[1] += rng.range(-joint_delta, joint_delta);
    b[2] += rng.range(-joint_delta, joint_delta);
    vec![SRobotQ::from_array(a), SRobotQ::from_array(b)]
}

fn gen_multi_segment(rng: &mut Lcg, n_wp: usize, joint_delta: f64) -> Vec<SRobotQ<6, f64>> {
    let mut out = Vec::with_capacity(n_wp);
    let mut cur = rng.joint_pose(1.0);
    out.push(SRobotQ::from_array(cur));
    for _ in 1..n_wp {
        cur = rng.perturb(cur, joint_delta);
        out.push(SRobotQ::from_array(cur));
    }
    out
}

/// Smooth sinusoidal sweep across all 6 joints, each at a different frequency in
/// `[0.5, 2.0]` cycles per trajectory — a good stand-in for a "scan" or "follow a
/// curve" workload.
fn gen_sinusoidal(rng: &mut Lcg, n_wp: usize, amplitude: f64) -> Vec<SRobotQ<6, f64>> {
    let mut out = Vec::with_capacity(n_wp);
    let centre = rng.joint_pose(0.5);
    let amps: [f64; 6] = std::array::from_fn(|_| rng.range(0.3, 1.0) * amplitude);
    let freqs: [f64; 6] = std::array::from_fn(|_| rng.range(0.5, 2.0));
    let phases: [f64; 6] = std::array::from_fn(|_| rng.range(0.0, std::f64::consts::TAU));
    for i in 0..n_wp {
        let t = i as f64 / (n_wp - 1) as f64;
        let mut q = [0.0_f64; 6];
        for j in 0..6 {
            q[j] = centre[j] + amps[j] * (freqs[j] * std::f64::consts::TAU * t + phases[j]).sin();
        }
        out.push(SRobotQ::from_array(q));
    }
    out
}

// ----------------------------------------------------------------------------
// Benchmark machinery
// ----------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct PhaseSecs {
    densify: f64,
    derivatives: f64,
    nlp_build: f64,
    nlp_solve: f64,
    resample: f64,
}

impl PhaseSecs {
    fn total(&self) -> f64 {
        self.densify + self.derivatives + self.nlp_build + self.nlp_solve + self.resample
    }
}

#[derive(Clone, Copy)]
struct RunResult {
    iterations: i32,
    solve_time: Duration,
    wall_time: Duration,
    output_samples: usize,
    densified_samples: usize,
    status: SolveStatus,
    phases: PhaseSecs,
}

fn run_one(
    fk: &URDFChain<6, f64>,
    cfg: &Topp3Tcp6Constraints<6>,
    waypoints: Vec<SRobotQ<6, f64>>,
) -> RunResult {
    let path = match SRobotPath::<6, f64>::try_new(waypoints) {
        Ok(p) => p,
        Err(_) => {
            return RunResult {
                iterations: 0,
                solve_time: Duration::ZERO,
                wall_time: Duration::ZERO,
                output_samples: 0,
                densified_samples: 0,
                status: SolveStatus::NotAttempted,
                phases: PhaseSecs::default(),
            };
        }
    };
    let mut v = validator();
    let t0 = Instant::now();
    let (_result, diag) = Topp3Tcp6.retime(cfg, &path, fk, &mut v, &());
    let wall = t0.elapsed();
    RunResult {
        iterations: diag.iterations,
        solve_time: diag.solve_time,
        wall_time: wall,
        output_samples: diag.output_samples,
        densified_samples: diag.densified_samples,
        status: diag.status,
        phases: PhaseSecs {
            densify: diag.phase_timing.densify.as_secs_f64(),
            derivatives: diag.phase_timing.derivatives.as_secs_f64(),
            nlp_build: diag.phase_timing.nlp_build.as_secs_f64(),
            nlp_solve: diag.phase_timing.nlp_solve.as_secs_f64(),
            resample: diag.phase_timing.resample.as_secs_f64(),
        },
    }
}

#[derive(Default, Clone)]
struct CategorySummary {
    name: &'static str,
    n_runs: usize,
    successes: usize,
    iterations: Vec<i32>,
    solve_secs: Vec<f64>,
    wall_secs: Vec<f64>,
    output_samples: Vec<usize>,
    densified_samples: Vec<usize>,
    phase_totals: PhaseSecs,
}

impl CategorySummary {
    fn record(&mut self, r: RunResult) {
        self.n_runs += 1;
        if matches!(r.status, SolveStatus::Success) {
            self.successes += 1;
        }
        self.iterations.push(r.iterations);
        self.solve_secs.push(r.solve_time.as_secs_f64());
        self.wall_secs.push(r.wall_time.as_secs_f64());
        self.output_samples.push(r.output_samples);
        self.densified_samples.push(r.densified_samples);
        self.phase_totals.densify += r.phases.densify;
        self.phase_totals.derivatives += r.phases.derivatives;
        self.phase_totals.nlp_build += r.phases.nlp_build;
        self.phase_totals.nlp_solve += r.phases.nlp_solve;
        self.phase_totals.resample += r.phases.resample;
    }
}

fn mean_f64(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn median_f64(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f64> = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn max_f64(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(0.0_f64, f64::max)
}

fn mean_i32(xs: &[i32]) -> f64 {
    mean_f64(&xs.iter().map(|&x| x as f64).collect::<Vec<_>>())
}

fn median_i32(xs: &[i32]) -> f64 {
    median_f64(&xs.iter().map(|&x| x as f64).collect::<Vec<_>>())
}

fn max_i32(xs: &[i32]) -> i32 {
    xs.iter().copied().max().unwrap_or(0)
}

fn print_table(summaries: &[CategorySummary]) {
    eprintln!();
    eprintln!("┌──────────────────────────────────┬──────┬─────────┬───────────┬───────────┬───────────┬───────────┬──────────┐");
    eprintln!(
        "│ Category                         │ Runs │ Success │ Iter mean │ Iter med  │ Iter max  │ Solve mean│ Solve max│"
    );
    eprintln!("├──────────────────────────────────┼──────┼─────────┼───────────┼───────────┼───────────┼───────────┼──────────┤");
    for s in summaries {
        let pct = 100.0 * s.successes as f64 / s.n_runs.max(1) as f64;
        eprintln!(
            "│ {:32} │ {:>4} │ {:>5.1}%  │ {:>9.0} │ {:>9.0} │ {:>9} │ {:>8.3}s │ {:>7.3}s │",
            s.name,
            s.n_runs,
            pct,
            mean_i32(&s.iterations),
            median_i32(&s.iterations),
            max_i32(&s.iterations),
            mean_f64(&s.solve_secs),
            max_f64(&s.solve_secs),
        );
    }
    eprintln!("└──────────────────────────────────┴──────┴─────────┴───────────┴───────────┴───────────┴───────────┴──────────┘");
    eprintln!();

    let total_runs: usize = summaries.iter().map(|s| s.n_runs).sum();
    let total_success: usize = summaries.iter().map(|s| s.successes).sum();
    let total_solve: f64 = summaries.iter().flat_map(|s| s.solve_secs.iter()).sum();
    let total_wall: f64 = summaries.iter().flat_map(|s| s.wall_secs.iter()).sum();
    eprintln!(
        "Aggregate: {}/{} succeeded ({:.1}%), {:.2}s solve / {:.2}s wall (overhead {:.0}%)",
        total_success,
        total_runs,
        100.0 * total_success as f64 / total_runs.max(1) as f64,
        total_solve,
        total_wall,
        100.0 * (total_wall - total_solve) / total_wall.max(1e-9),
    );

    // Per-phase breakdown — total time across the entire sweep, with the share each
    // phase took. NLP solve almost always dominates; the other phases tell you whether
    // any of them grew unexpectedly (e.g. derivatives ballooning under a future
    // higher-order spline).
    let mut total_phases = PhaseSecs::default();
    for s in summaries {
        total_phases.densify += s.phase_totals.densify;
        total_phases.derivatives += s.phase_totals.derivatives;
        total_phases.nlp_build += s.phase_totals.nlp_build;
        total_phases.nlp_solve += s.phase_totals.nlp_solve;
        total_phases.resample += s.phase_totals.resample;
    }
    let phase_total = total_phases.total().max(1e-9);
    eprintln!();
    eprintln!("Phase breakdown across all {} runs:", total_runs);
    eprintln!(
        "  densify     : {:>7.3}s  ({:>4.1}%)",
        total_phases.densify,
        100.0 * total_phases.densify / phase_total
    );
    eprintln!(
        "  derivatives : {:>7.3}s  ({:>4.1}%)",
        total_phases.derivatives,
        100.0 * total_phases.derivatives / phase_total
    );
    eprintln!(
        "  NLP build   : {:>7.3}s  ({:>4.1}%)",
        total_phases.nlp_build,
        100.0 * total_phases.nlp_build / phase_total
    );
    eprintln!(
        "  NLP solve   : {:>7.3}s  ({:>4.1}%)",
        total_phases.nlp_solve,
        100.0 * total_phases.nlp_solve / phase_total
    );
    eprintln!(
        "  resample    : {:>7.3}s  ({:>4.1}%)",
        total_phases.resample,
        100.0 * total_phases.resample / phase_total
    );
    eprintln!(
        "  total       : {:>7.3}s   (vs aggregate solve_time = {:.3}s; gap is timer noise + per-phase fences)",
        phase_total, total_solve,
    );

    // Per-category mean phase time table — useful for spotting categories that load any
    // particular phase disproportionately.
    eprintln!();
    eprintln!("Per-category mean phase time (ms):");
    eprintln!("┌──────────────────────────────────┬────────┬────────┬────────┬────────┬────────┐");
    eprintln!("│ Category                         │  dens  │ deriv  │ build  │ solve  │ resamp │");
    eprintln!("├──────────────────────────────────┼────────┼────────┼────────┼────────┼────────┤");
    for s in summaries {
        let n = s.n_runs.max(1) as f64;
        eprintln!(
            "│ {:32} │ {:>5.1}  │ {:>5.1}  │ {:>5.1}  │ {:>5.1}  │ {:>5.1}  │",
            s.name,
            1000.0 * s.phase_totals.densify / n,
            1000.0 * s.phase_totals.derivatives / n,
            1000.0 * s.phase_totals.nlp_build / n,
            1000.0 * s.phase_totals.nlp_solve / n,
            1000.0 * s.phase_totals.resample / n,
        );
    }
    eprintln!("└──────────────────────────────────┴────────┴────────┴────────┴────────┴────────┘");
}

// ----------------------------------------------------------------------------
// The benchmark itself
// ----------------------------------------------------------------------------

#[test]
#[ignore = "long-running benchmark; run with: cargo test --test external_bench -- --ignored --nocapture"]
fn benchmark_typical_trajectories() {
    let fk = external_chain();
    let cfg = external_cfg();

    // (name, runs_per_category, generator)
    let categories: Vec<(&'static str, usize, Box<dyn Fn(&mut Lcg) -> Vec<SRobotQ<6, f64>>>)> = vec![
        ("p2p tiny (Δ≈0.05 rad)", 8, Box::new(|r| gen_two_wp(r, 0.05))),
        ("p2p small (Δ≈0.2 rad)", 8, Box::new(|r| gen_two_wp(r, 0.2))),
        ("p2p medium (Δ≈0.6 rad)", 8, Box::new(|r| gen_two_wp(r, 0.6))),
        ("p2p large (Δ≈1.5 rad)", 8, Box::new(|r| gen_two_wp(r, 1.5))),
        ("wrist-only (Δ≈0.5 rad)", 6, Box::new(|r| gen_wrist_only(r, 0.5))),
        ("base-only (Δ≈0.5 rad)", 6, Box::new(|r| gen_base_only(r, 0.5))),
        ("multi-seg 5wp (Δ≈0.3)", 6, Box::new(|r| gen_multi_segment(r, 5, 0.3))),
        ("multi-seg 10wp (Δ≈0.3)", 6, Box::new(|r| gen_multi_segment(r, 10, 0.3))),
        ("multi-seg 25wp (Δ≈0.15)", 4, Box::new(|r| gen_multi_segment(r, 25, 0.15))),
        ("multi-seg 50wp (Δ≈0.10)", 4, Box::new(|r| gen_multi_segment(r, 50, 0.10))),
        ("sinusoidal 10wp (A≈0.5)", 4, Box::new(|r| gen_sinusoidal(r, 10, 0.5))),
        ("sinusoidal 20wp (A≈0.5)", 4, Box::new(|r| gen_sinusoidal(r, 20, 0.5))),
    ];

    let mut summaries: Vec<CategorySummary> = Vec::with_capacity(categories.len());
    let mut rng = Lcg::new(SEED);

    for (name, n_runs, factory) in &categories {
        let mut summary = CategorySummary {
            name,
            ..Default::default()
        };
        for run_idx in 0..*n_runs {
            let waypoints = factory(&mut rng);
            let r = run_one(&fk, &cfg, waypoints.clone());
            if !matches!(r.status, SolveStatus::Success) {
                eprintln!(
                    "  [{}] run {}/{}: {:?} ({} iter, {:.3}s)",
                    name,
                    run_idx + 1,
                    n_runs,
                    r.status,
                    r.iterations,
                    r.solve_time.as_secs_f64()
                );
                eprintln!("    waypoints (n={}):", waypoints.len());
                for (i, wp) in waypoints.iter().enumerate() {
                    eprintln!(
                        "      {:>2}: [{:>11.7}, {:>11.7}, {:>11.7}, {:>11.7}, {:>11.7}, {:>11.7}]",
                        i, wp.0[0], wp.0[1], wp.0[2], wp.0[3], wp.0[4], wp.0[5],
                    );
                }
            }
            summary.record(r);
        }
        eprintln!(
            "{:32} done — {}/{} succeeded",
            name, summary.successes, summary.n_runs
        );
        summaries.push(summary);
    }

    print_table(&summaries);

    // Don't fail the benchmark on individual retime failures — the table is the result.
    // But do fail if ZERO retimes succeeded across the whole sweep, since that means
    // the fixture or solver is broken.
    let total_success: usize = summaries.iter().map(|s| s.successes).sum();
    assert!(
        total_success > 0,
        "no retimes succeeded across the entire benchmark — fixture or solver is broken"
    );
}
