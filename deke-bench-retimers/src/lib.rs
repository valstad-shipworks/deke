//! Comparative analysis across the `deke_types::Retimer` implementations:
//!
//! - [`deke_topp3_lp::Topp3LpTcp`] (joint-space convex-LP, exact path)
//! - [`deke_topp3tcp_spline::Topp3TcpSpline`] (B-spline + DFS over jerk)
//! - [`deke_topp_speed::ToppSolver`] (real-time jerk-limited shaper)
//!
//! A [`BenchProblem`] bundles a joint-space waypoint path with per-joint V/A/J
//! limits and a TCP velocity limit. [`run_all`] executes each retimer against
//! a problem and returns a [`BenchResult`] per retimer; metrics include solve
//! wall time, trajectory duration, joint V/A/J peaks (computed via the same
//! backward-difference stencils the consumer would use on the output samples),
//! TCP linear velocity peak (forward-difference on FK-evaluated positions),
//! and the utilization of each limit (peak ÷ limit).
//!
//! The crate is intentionally analysis-only: no published API. It compiles a
//! single binary test ([`tests::comparative`]) that runs the full sweep and
//! prints a comparison table.

#![allow(clippy::approx_constant)] // URDF fixture RPY data, not the math constant

use std::sync::mpsc;
use std::time::{Duration, Instant};

use deke_kin::{JointLimits as KinLimits, Kinematics, URDFJoint};
use deke_types::ContinuousFKChain;
use deke_types::{DekeResult, JointValidator, Retimer, SRobotPath, SRobotQ, SRobotTraj};

// Production 6-DOF URDF chain + canonical limits
//
// Lifted verbatim from `deke-topp3tcp-nlp/tests/continuous_external_failures.rs`. All bench
// problems target this chain so the comparison runs against the same FK and
// kinematic ceilings the captured-failure tests exercise. If the chain ever
// changes in the producing project, update both arrays here AND in the
// `external_failures.rs` fixture.

const URDF_JOINTS: [URDFJoint; 6] = [
    URDFJoint::revolute(
        (0f64, 0f64, 0.152f64),
        (0f64, -0f64, 0f64),
        (0f64, 0f64, 1f64),
    ),
    URDFJoint::revolute(
        (0.075f64, -0.105f64, 0.273f64),
        (-1.5708f64, -0f64, 0f64),
        (0f64, 0f64, 1f64),
    ),
    URDFJoint::revolute(
        (-0.00000000000000625888f64, -0.84f64, 0.04028f64),
        (
            -3.14159f64,
            -0.0000000000000000252315f64,
            -0.00000000000000423966f64,
        ),
        (0f64, 0f64, 1f64),
    ),
    URDFJoint::revolute(
        (0.295618f64, 0.215f64, -0.0642f64),
        (-3.14159f64, 1.5708f64, 0f64),
        (0f64, 0f64, 1f64),
    ),
    URDFJoint::revolute(
        (-0.0501976f64, 0.000491285f64, -1.0445f64),
        (-3.13181f64, 1.5708f64, 0f64),
        (0f64, 0f64, 1f64),
    ),
    URDFJoint::revolute(
        (0.075182f64, -0.00000000205208f64, -0.0507f64),
        (3.14159f64, 1.5708f64, 0f64),
        (0f64, 0f64, 1f64),
    ),
];

const URDF_FIXED_SUFFIX: [URDFJoint; 1] = [URDFJoint::fixed(
    (
        -0.00000000000000056205f64,
        -0.000000000000000888178f64,
        -0.00000000000000310862f64,
    ),
    (-1.5708f64, 1.5708f64, 3.14159f64),
)];

/// Builds the production 6-DOF URDF chain (`f64`). Panics on construction
/// failure — that would be a fixture bug, not a retimer bug.
pub fn production_urdf_chain() -> Kinematics<6, f64> {
    let mut joints: Vec<URDFJoint> = URDF_JOINTS.to_vec();
    joints.extend_from_slice(&URDF_FIXED_SUFFIX);
    Kinematics::<6, f64>::from_urdf(&joints, KinLimits::symmetric(f64::INFINITY), &[])
        .expect("URDF fixture build failed on the production joints")
}

/// Production per-joint velocity ceilings (rad/s).
pub const PRODUCTION_V_MAX: [f64; 6] = [
    2.748893571891069,
    2.748893571891069,
    3.46884188833873,
    5.497787143782138,
    5.8904862254808625,
    9.42477796076938,
];
/// Production per-joint acceleration ceilings (rad/s²).
pub const PRODUCTION_A_MAX: [f64; 6] = [
    6.17056448573788,
    6.17056448573788,
    7.786664511626387,
    12.34112897147576,
    13.222637422820858,
    21.156219876513376,
];
/// Production per-joint jerk ceilings (rad/s³).
pub const PRODUCTION_J_MAX: [f64; 6] = [
    22.897032833404705,
    22.897032833404705,
    28.893874634073196,
    45.79406566680941,
    49.065074313992284,
    78.50411890238763,
];
/// Production TCP linear-velocity ceiling (m/s).
pub const PRODUCTION_TCP_V_MAX: f64 = 2.0;
/// Production output sample rate (Hz).
pub const PRODUCTION_SAMPLE_RATE_HZ: f64 = 125.0;

// Slicer-logged kinematic configs (production data, 6-DOF subset)
//
// Two robot configurations appear in the slicer's `traj_gen` call logs:
//
//  - **nanopanel-material** — payload-derated material-handling cell.
//    Looser velocity/jerk ceilings used when the arm carries a workpiece.
//  - **nanopanel-welder** — welding cell with stiffer dynamics and tighter
//    travel.
//
// Both run at the slicer's `interval_s = 0.008` (125 Hz). TCP cap is
// `2.0 m/s / 20.0 m/s² / 200.0 m/s³`. The 7th DOF in the original calls is
// always the external rail / positioner — ignored here.

pub const MATERIAL_V_MAX: [f64; 6] = [
    0.8246680715673207,
    0.7068583470577035,
    0.667588438887831,
    0.9424777960769379,
    0.9424777960769379,
    1.5707963267948966,
];
pub const MATERIAL_A_MAX: [f64; 6] = [
    2.2436289071502484,
    1.9231107010914414,
    1.8162712561653547,
    2.564147610332459,
    2.564147610332459,
    4.2735795991158,
];
pub const MATERIAL_J_MAX: [f64; 6] = [
    9.874010285203688,
    8.463437427898949,
    7.993246679047906,
    11.284583341357788,
    11.284583341357788,
    18.807639470404013,
];

pub const WELDER_V_MAX: [f64; 6] = [
    1.6493361431346414,
    1.6493361431346414,
    2.081305133003238,
    3.2986722862692828,
    3.5342917352885173,
    5.654866776461628,
];
pub const WELDER_A_MAX: [f64; 6] = [
    3.702338691442728,
    3.702338691442728,
    4.671998706975832,
    7.404677382885456,
    7.933582453692515,
    12.693731925908025,
];
pub const WELDER_J_MAX: [f64; 6] = [
    13.738219700042823,
    13.738219700042823,
    17.33632478044392,
    27.476439400085646,
    29.439044588395372,
    47.10247134143259,
];

/// Slicer's TCP cap on every `traj_gen` call.
pub const SLICER_TCP_V_MAX: f64 = 2.0;
/// Slicer's output rate (`1.0 / interval_s = 1.0 / 0.008`).
pub const SLICER_SAMPLE_RATE_HZ: f64 = 125.0;

/// Wall-clock cap applied to each retimer call. The spline retimer's DFS has
/// no built-in time limit and can stall indefinitely on paths whose spline
/// approximation can't get under the deviation tube. We abandon any retimer
/// that exceeds this — the worker thread is detached (it keeps running in
/// the background and eats memory until the test process exits), so longer
/// caps + many timeouts will OOM. 10 s is enough to let the slowest healthy
/// retime finish while bounding the leak.
const PER_RETIMER_TIMEOUT: Duration = Duration::from_secs(10);

/// One comparative problem: an input path plus the kinematic limits all four
/// retimers are asked to respect.
#[derive(Debug, Clone)]
pub struct BenchProblem<const N: usize> {
    pub name: &'static str,
    pub waypoints: Vec<SRobotQ<N, f64>>,
    /// Per-joint velocity ceiling.
    pub v_max: [f64; N],
    /// Per-joint acceleration ceiling.
    pub a_max: [f64; N],
    /// Per-joint jerk ceiling.
    pub j_max: [f64; N],
    /// TCP linear-velocity ceiling (m/s). `None` skips TCP constraints.
    pub tcp_v_max: Option<f64>,
    /// Output sample rate, in Hz.
    pub sample_rate_hz: f64,
}

impl<const N: usize> BenchProblem<N> {
    pub fn path(&self) -> DekeResult<SRobotPath<N, f64>> {
        SRobotPath::try_new(self.waypoints.clone())
    }
}

/// Backward-difference V/A/J peaks per joint, evaluated on the trajectory's
/// output samples (`dt = traj.dt()`).
#[derive(Debug, Clone)]
pub struct JointFdMetrics<const N: usize> {
    pub peak_v: [f64; N],
    pub peak_a: [f64; N],
    pub peak_j: [f64; N],
}

/// Forward-difference TCP linear-velocity peak in m/s.
#[derive(Debug, Clone, Copy)]
pub struct TcpFdMetrics {
    pub peak_v: f64,
}

/// Per-limit utilization, **averaged across the trajectory**. For each output
/// sample we compute the per-limit ratio `max_j(|x_j| / limit_j)`, then mean
/// across samples. `tcp_v = None` means no TCP limit was set on the problem.
///
/// Note: this is an *average* — values >1 don't necessarily indicate a
/// per-sample violation; the trajectory might have a brief spike. For
/// pointwise violation checks, look at the peak metrics
/// (`JointFdMetrics::peak_*`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Utilization {
    pub joint_v: f64,
    pub joint_a: f64,
    pub joint_j: f64,
    pub tcp_v: Option<f64>,
    /// Average across samples of the per-sample tightest-limit reading:
    /// `mean_k(max(u_jv_k, u_ja_k, u_jj_k, u_tcpv_k))`. Captures how close the
    /// trajectory sits to *its* binding limit on average — a single number
    /// that doesn't get smeared by limits the retimer wasn't trying to honor.
    /// Only averaged over samples where every component is defined
    /// (`k in 3..n`, plus TCP availability if a TCP cap is set).
    pub max_u: f64,
    /// Peak (max over samples) of the per-sample tightest-limit reading. Same
    /// expression as [`max_u`](Self::max_u), but taken as the maximum across
    /// samples instead of the mean. The relevant figure for "did the
    /// backward-FD readout ever cross 1.05 × limit" checks.
    pub peak_u: f64,
}

/// Single retimer's run against one problem.
#[derive(Debug)]
pub struct BenchResult<const N: usize> {
    pub retimer: &'static str,
    pub status: String,
    pub solve_time: Duration,
    pub trajectory_duration: Duration,
    pub num_samples: usize,
    pub joint_fd: Option<JointFdMetrics<N>>,
    pub tcp_fd: Option<TcpFdMetrics>,
    pub utilization: Option<Utilization>,
    /// Maximum joint-space distance from any output sample to the closest
    /// point on the input polyline (radians, by Euclidean norm in joint
    /// space). Path-parameterized retimers track the polyline within
    /// densification interpolation noise (~1e-3 or less); retimers that
    /// treat waypoints as point constraints (e.g. `topp-speed`) can
    /// deviate substantially when joint-space chord directions change at
    /// interior waypoints.
    pub max_path_deviation: Option<f64>,
    pub error: Option<String>,
}

impl<const N: usize> BenchResult<N> {
    pub fn ok(&self) -> bool {
        self.error.is_none()
    }
}

/// Backward-difference V (and 3-point A, 4-point J) peaks per joint on a
/// trajectory.
pub fn joint_fd_metrics<const N: usize>(traj: &SRobotTraj<N, f64>) -> JointFdMetrics<N> {
    let dt = traj.dt().as_secs_f64();
    let n = traj.len();
    let mut peak_v = [0.0_f64; N];
    let mut peak_a = [0.0_f64; N];
    let mut peak_j = [0.0_f64; N];
    if dt <= 0.0 || n < 2 {
        return JointFdMetrics {
            peak_v,
            peak_a,
            peak_j,
        };
    }
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;

    for k in 1..n {
        let q0 = traj[k].0;
        let q1 = traj[k - 1].0;
        for j in 0..N {
            let v = (q0[j] - q1[j]) / dt;
            let a = v.abs();
            if a > peak_v[j] {
                peak_v[j] = a;
            }
        }
    }
    for k in 2..n {
        let q0 = traj[k].0;
        let q1 = traj[k - 1].0;
        let q2 = traj[k - 2].0;
        for j in 0..N {
            let a = (q0[j] - 2.0 * q1[j] + q2[j]) / dt2;
            let av = a.abs();
            if av > peak_a[j] {
                peak_a[j] = av;
            }
        }
    }
    for k in 3..n {
        let q0 = traj[k].0;
        let q1 = traj[k - 1].0;
        let q2 = traj[k - 2].0;
        let q3 = traj[k - 3].0;
        for j in 0..N {
            let jk = (q0[j] - 3.0 * q1[j] + 3.0 * q2[j] - q3[j]) / dt3;
            let av = jk.abs();
            if av > peak_j[j] {
                peak_j[j] = av;
            }
        }
    }

    JointFdMetrics {
        peak_v,
        peak_a,
        peak_j,
    }
}

/// Peak TCP linear-velocity magnitude `‖Δp/dt‖` across the output trajectory.
/// `p` is the end-effector translation evaluated via `fk_end` at each sample.
pub fn tcp_fd_metrics<const N: usize, FK: ContinuousFKChain<N, f64>>(
    traj: &SRobotTraj<N, f64>,
    fk: &FK,
) -> DekeResult<TcpFdMetrics> {
    let dt = traj.dt().as_secs_f64();
    let n = traj.len();
    if dt <= 0.0 || n < 2 {
        return Ok(TcpFdMetrics { peak_v: 0.0 });
    }
    let mut prev: Option<[f64; 3]> = None;
    let mut peak = 0.0_f64;
    for q in traj.iter() {
        let pose = fk.fk_end(q)?;
        let t = pose.translation;
        let cur = [t.x, t.y, t.z];
        if let Some(p) = prev {
            let dx = cur[0] - p[0];
            let dy = cur[1] - p[1];
            let dz = cur[2] - p[2];
            let speed = (dx * dx + dy * dy + dz * dz).sqrt() / dt;
            if speed > peak {
                peak = speed;
            }
        }
        prev = Some(cur);
    }
    Ok(TcpFdMetrics { peak_v: peak })
}

/// Maximum joint-space distance from any output trajectory sample to the
/// closest point on the input polyline. Path-parameterized retimers densify
/// the polyline and produce trajectories that lie on the densified polyline
/// (deviation ~0 modulo chord-linear interpolation); retimers that treat the
/// input as point waypoints connected by independent per-joint shapes can
/// bow off the chord between waypoints whenever the per-waypoint boundary
/// velocity is non-parallel to the next chord.
///
/// Distances are computed against the original input polyline, not any
/// densified / spline-interpolated version. The metric uses straight-line
/// point-to-segment distance in joint space (no orientation-aware norm).
pub fn max_path_deviation<const N: usize>(
    traj: &SRobotTraj<N, f64>,
    waypoints: &[SRobotQ<N, f64>],
) -> f64 {
    if waypoints.len() < 2 || traj.is_empty() {
        return 0.0;
    }
    let mut max_d = 0.0_f64;
    for sample in traj.iter() {
        let q = sample.0;
        let mut best_sq = f64::INFINITY;
        for k in 0..(waypoints.len() - 1) {
            let a = waypoints[k].0;
            let b = waypoints[k + 1].0;
            let mut ab_sq = 0.0_f64;
            let mut aq_dot_ab = 0.0_f64;
            for j in 0..N {
                let ab_j = b[j] - a[j];
                ab_sq += ab_j * ab_j;
                aq_dot_ab += (q[j] - a[j]) * ab_j;
            }
            let t = if ab_sq > 1e-18 {
                (aq_dot_ab / ab_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut d_sq = 0.0_f64;
            for j in 0..N {
                let ab_j = b[j] - a[j];
                let proj = a[j] + t * ab_j;
                let dj = q[j] - proj;
                d_sq += dj * dj;
            }
            if d_sq < best_sq {
                best_sq = d_sq;
            }
        }
        let d = best_sq.sqrt();
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

/// Average per-limit utilization across the trajectory. Each sample contributes
/// its `max_j(|x_j| / limit_j)` reading; we return the arithmetic mean across
/// samples. Velocity samples come from a 2-point backward FD (`k in 1..n`),
/// acceleration from 3-point (`k in 2..n`), jerk from 4-point (`k in 3..n`),
/// and TCP velocity from a forward FD on `fk_end`-evaluated positions.
pub fn average_utilization<const N: usize, FK: ContinuousFKChain<N, f64>>(
    traj: &SRobotTraj<N, f64>,
    fk: &FK,
    problem: &BenchProblem<N>,
) -> DekeResult<Utilization> {
    let dt = traj.dt().as_secs_f64();
    let n = traj.len();
    if dt <= 0.0 || n < 2 {
        return Ok(Utilization::default());
    }
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;

    let mut sum_jv = 0.0_f64;
    let mut cnt_jv = 0usize;
    for k in 1..n {
        let q0 = traj[k].0;
        let q1 = traj[k - 1].0;
        let mut sample_u = 0.0_f64;
        for j in 0..N {
            let lim = problem.v_max[j];
            if lim.is_finite() && lim > 0.0 {
                let v = (q0[j] - q1[j]) / dt;
                let u = v.abs() / lim;
                if u > sample_u {
                    sample_u = u;
                }
            }
        }
        sum_jv += sample_u;
        cnt_jv += 1;
    }

    let mut sum_ja = 0.0_f64;
    let mut cnt_ja = 0usize;
    for k in 2..n {
        let q0 = traj[k].0;
        let q1 = traj[k - 1].0;
        let q2 = traj[k - 2].0;
        let mut sample_u = 0.0_f64;
        for j in 0..N {
            let lim = problem.a_max[j];
            if lim.is_finite() && lim > 0.0 {
                let a = (q0[j] - 2.0 * q1[j] + q2[j]) / dt2;
                let u = a.abs() / lim;
                if u > sample_u {
                    sample_u = u;
                }
            }
        }
        sum_ja += sample_u;
        cnt_ja += 1;
    }

    let mut sum_jj = 0.0_f64;
    let mut cnt_jj = 0usize;
    for k in 3..n {
        let q0 = traj[k].0;
        let q1 = traj[k - 1].0;
        let q2 = traj[k - 2].0;
        let q3 = traj[k - 3].0;
        let mut sample_u = 0.0_f64;
        for j in 0..N {
            let lim = problem.j_max[j];
            if lim.is_finite() && lim > 0.0 {
                let jk = (q0[j] - 3.0 * q1[j] + 3.0 * q2[j] - q3[j]) / dt3;
                let u = jk.abs() / lim;
                if u > sample_u {
                    sample_u = u;
                }
            }
        }
        sum_jj += sample_u;
        cnt_jj += 1;
    }

    // Optional TCP velocity utilization per sample at index k (forward FD from
    // k-1 → k, defined for k in 1..n).
    let tcp_limit = problem.tcp_v_max.filter(|l| l.is_finite() && *l > 0.0);
    let tcp_u_per_sample: Option<Vec<f64>> = if let Some(limit) = tcp_limit {
        let mut positions = Vec::with_capacity(n);
        for q in traj.iter() {
            let pose = fk.fk_end(q)?;
            let t = pose.translation;
            positions.push([t.x, t.y, t.z]);
        }
        let mut out = vec![0.0_f64; n];
        for k in 1..n {
            let dx = positions[k][0] - positions[k - 1][0];
            let dy = positions[k][1] - positions[k - 1][1];
            let dz = positions[k][2] - positions[k - 1][2];
            out[k] = (dx * dx + dy * dy + dz * dz).sqrt() / dt / limit;
        }
        Some(out)
    } else {
        None
    };

    let tcp_v_avg = tcp_u_per_sample.as_ref().map(|v| {
        let s: f64 = v[1..].iter().sum();
        let c = (n - 1) as f64;
        if c > 0.0 { s / c } else { 0.0 }
    });

    // max_u: per-sample max across all four utilizations, averaged across
    // samples where every component is defined (k in 3..n — jerk needs four
    // points). TCP enters the max only when a TCP cap was set.
    let mut sum_max_u = 0.0_f64;
    let mut peak_max_u = 0.0_f64;
    let mut cnt_max_u = 0usize;
    for k in 3..n {
        let q0 = traj[k].0;
        let q1 = traj[k - 1].0;
        let q2 = traj[k - 2].0;
        let q3 = traj[k - 3].0;

        let mut u_v = 0.0_f64;
        let mut u_a = 0.0_f64;
        let mut u_j = 0.0_f64;
        for j in 0..N {
            let v_lim = problem.v_max[j];
            if v_lim.is_finite() && v_lim > 0.0 {
                let v = (q0[j] - q1[j]) / dt;
                let u = v.abs() / v_lim;
                if u > u_v {
                    u_v = u;
                }
            }
            let a_lim = problem.a_max[j];
            if a_lim.is_finite() && a_lim > 0.0 {
                let a = (q0[j] - 2.0 * q1[j] + q2[j]) / dt2;
                let u = a.abs() / a_lim;
                if u > u_a {
                    u_a = u;
                }
            }
            let j_lim = problem.j_max[j];
            if j_lim.is_finite() && j_lim > 0.0 {
                let jk = (q0[j] - 3.0 * q1[j] + 3.0 * q2[j] - q3[j]) / dt3;
                let u = jk.abs() / j_lim;
                if u > u_j {
                    u_j = u;
                }
            }
        }
        let mut sample_max = u_v.max(u_a).max(u_j);
        if let Some(ref tcps) = tcp_u_per_sample {
            sample_max = sample_max.max(tcps[k]);
        }
        sum_max_u += sample_max;
        if sample_max > peak_max_u {
            peak_max_u = sample_max;
        }
        cnt_max_u += 1;
    }

    Ok(Utilization {
        joint_v: if cnt_jv > 0 {
            sum_jv / cnt_jv as f64
        } else {
            0.0
        },
        joint_a: if cnt_ja > 0 {
            sum_ja / cnt_ja as f64
        } else {
            0.0
        },
        joint_j: if cnt_jj > 0 {
            sum_jj / cnt_jj as f64
        } else {
            0.0
        },
        tcp_v: tcp_v_avg,
        max_u: if cnt_max_u > 0 {
            sum_max_u / cnt_max_u as f64
        } else {
            0.0
        },
        peak_u: peak_max_u,
    })
}

/// Wide-open joint position validator (positions are unbounded for retiming
/// purposes; the bench is about kinematic-limit compliance, not joint stops).
pub fn wide_validator<const N: usize>() -> JointValidator<N, f64> {
    JointValidator::<N, f64>::new(
        SRobotQ::from_array([-1.0e9; N]),
        SRobotQ::from_array([1.0e9; N]),
    )
}

/// Build a per-trajectory [`BenchResult`] from a [`Retimer`] output.
fn finalize_result<const N: usize, FK: ContinuousFKChain<N, f64>>(
    name: &'static str,
    status: String,
    solve_time: Duration,
    result: DekeResult<SRobotTraj<N, f64>>,
    problem: &BenchProblem<N>,
    fk: &FK,
) -> BenchResult<N> {
    match result {
        Ok(traj) => {
            let joint = joint_fd_metrics(&traj);
            let tcp = tcp_fd_metrics(&traj, fk).ok();
            let util = average_utilization(&traj, fk, problem).unwrap_or_default();
            let dev = max_path_deviation(&traj, &problem.waypoints);
            BenchResult {
                retimer: name,
                status,
                solve_time,
                trajectory_duration: traj.duration(),
                num_samples: traj.len(),
                joint_fd: Some(joint),
                tcp_fd: tcp,
                utilization: Some(util),
                max_path_deviation: Some(dev),
                error: None,
            }
        }
        Err(e) => BenchResult {
            retimer: name,
            status,
            solve_time,
            trajectory_duration: Duration::ZERO,
            num_samples: 0,
            joint_fd: None,
            tcp_fd: None,
            utilization: None,
            max_path_deviation: None,
            error: Some(format!("{}", e)),
        },
    }
}

/// Runs `f` on a worker thread; returns `Some(result)` if it completes within
/// [`PER_RETIMER_TIMEOUT`], `None` otherwise. The worker is detached on timeout.
fn with_timeout<T, F>(f: F) -> Option<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = mpsc::sync_channel::<T>(1);
    let _detached = std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(PER_RETIMER_TIMEOUT).ok()
}

fn timeout_result<const N: usize>(name: &'static str) -> BenchResult<N> {
    BenchResult {
        retimer: name,
        status: "timeout".into(),
        solve_time: PER_RETIMER_TIMEOUT,
        trajectory_duration: Duration::ZERO,
        num_samples: 0,
        joint_fd: None,
        tcp_fd: None,
        utilization: None,
        max_path_deviation: None,
        error: Some(format!(
            "exceeded {:?} wall-clock budget",
            PER_RETIMER_TIMEOUT
        )),
    }
}

pub fn run_topp3_lp<const N: usize, FK: ContinuousFKChain<N, f64> + 'static>(
    problem: &BenchProblem<N>,
    fk: &FK,
) -> BenchResult<N> {
    let problem = problem.clone();
    let fk = fk.clone();
    with_timeout(move || run_topp3_lp_inner::<N, FK>(&problem, &fk))
        .unwrap_or_else(|| timeout_result::<N>("topp3-lp"))
}

fn run_topp3_lp_inner<const N: usize, FK: ContinuousFKChain<N, f64>>(
    problem: &BenchProblem<N>,
    fk: &FK,
) -> BenchResult<N> {
    use deke_topp3_lp::{JointLimits, TcpLimits, Topp3LpConstraints, Topp3LpTcp};

    let path = match problem.path() {
        Ok(p) => p,
        Err(e) => {
            return BenchResult {
                retimer: "topp3-lp",
                status: "path-construction".into(),
                solve_time: Duration::ZERO,
                trajectory_duration: Duration::ZERO,
                num_samples: 0,
                joint_fd: None,
                tcp_fd: None,
                utilization: None,
                max_path_deviation: None,
                error: Some(format!("{}", e)),
            };
        }
    };

    let dt = Duration::from_secs_f64(1.0 / problem.sample_rate_hz);
    let mut cfg = Topp3LpConstraints::<N>::symmetric(1.0, 1.0, 1.0, dt);
    cfg.joint = JointLimits {
        v_max: SRobotQ::from_array(problem.v_max),
        a_max: SRobotQ::from_array(problem.a_max),
        j_max: SRobotQ::from_array(problem.j_max),
    };
    cfg.tcp = TcpLimits {
        v_max: problem.tcp_v_max,
    };

    let validator = wide_validator::<N>();
    let t0 = Instant::now();
    // Topp3LpTcp covers both joint-only (tcp v_max None) and TCP-capped cases.
    let (result, _diag) = Topp3LpTcp::new(fk).retime(&cfg, &path, &validator, &());
    let solve_time = t0.elapsed();
    let status = if result.is_ok() {
        "ok".into()
    } else {
        "failed".into()
    };
    finalize_result("topp3-lp", status, solve_time, result, problem, fk)
}

pub fn run_topp3tcp_spline<const N: usize, FK: ContinuousFKChain<N, f64> + 'static>(
    problem: &BenchProblem<N>,
    fk: &FK,
) -> BenchResult<N> {
    let problem = problem.clone();
    let fk = fk.clone();
    with_timeout(move || run_topp3tcp_spline_inner::<N, FK>(&problem, &fk))
        .unwrap_or_else(|| timeout_result::<N>("topp3tcp-spline"))
}

fn run_topp3tcp_spline_inner<const N: usize, FK: ContinuousFKChain<N, f64>>(
    problem: &BenchProblem<N>,
    fk: &FK,
) -> BenchResult<N> {
    use deke_topp3tcp_spline::{
        JointLimits, SearchOptions, SplinePathOptions, TcpLimits, Topp3TcpSpline,
        Topp3TcpSplineConstraints,
    };

    let path = match problem.path() {
        Ok(p) => p,
        Err(e) => {
            return BenchResult {
                retimer: "topp3tcp-spline",
                status: "path-construction".into(),
                solve_time: Duration::ZERO,
                trajectory_duration: Duration::ZERO,
                num_samples: 0,
                joint_fd: None,
                tcp_fd: None,
                utilization: None,
                max_path_deviation: None,
                error: Some(format!("{}", e)),
            };
        }
    };

    let cfg = Topp3TcpSplineConstraints::<N> {
        joint: JointLimits {
            v_max: SRobotQ::from_array(problem.v_max),
            a_max: SRobotQ::from_array(problem.a_max),
            j_max: SRobotQ::from_array(problem.j_max),
        },
        // Spline retimer requires a TCP limit; default to a large cap when the
        // problem opted out of TCP so this column stays comparable.
        tcp: TcpLimits::new(
            problem.tcp_v_max.unwrap_or(1.0e6),
            f64::INFINITY,
            f64::INFINITY,
        ),
        path: SplinePathOptions {
            // Looser than the spline crate's default (1e-3) — at the default
            // tube width, even modestly-curved joint-space paths fail
            // refinement and the DFS then explores a non-feasible search
            // space until the per-retimer timeout fires.
            max_deviation: 1e-2,
            max_refine_iters: 8,
            start_direction: None,
            end_direction: None,
        },
        // The spline retimer's DFS scales as `branch_factor^(time / search.dt)`
        // — branch factor 12, so 1-second trajectories at the bench's 4 ms
        // output dt would need depth ~250 (intractable). The algorithm is
        // designed to run at a much coarser search dt (the reference impl
        // used `dt ≈ 0.120` s). The output trajectory then has fewer samples
        // than the other retimers; column comparisons reflect that.
        search: SearchOptions {
            dt: 0.12,
            // Tighter `verify_dt`: check fused-utilization at the output
            // sample step inside each DFS segment, not just at the
            // segment endpoint. Catches intra-segment analytical
            // violations that the endpoint check alone would miss.
            // Cost: ~30× DFS-work per node at this dt ratio, but DFS
            // typically has ≪ depth=20 so total impact is bounded.
            verify_dt: 1.0 / problem.sample_rate_hz,
            // Emit at the problem's configured sample rate so this
            // retimer's output cadence lines up with the others; the
            // coarse DFS dt above is just the internal search step.
            output_dt: Some(1.0 / problem.sample_rate_hz),
            jerk_smoothing_passes: 0,
            fd_safety_slack: 0.05,
            // Cap on `|sdddot[k+1] − sdddot[k]|` between DFS segments —
            // bounds the FD-jerk spike directly at source. Set to the
            // largest per-axis joint jerk limit; empirically this is a
            // good starting point (spike contribution scales as
            // `|qp| × Δsdddot`, and `|qp|` is bounded by 1 for the
            // unit-arclength spline path).
            max_jerk_jump: Some(problem.j_max.iter().copied().fold(0.0_f64, f64::max)),
            start_sdot: 0.0,
            end_sdot: 0.0,
            max_sdot: 10.0,
        },
    };

    let validator = wide_validator::<N>();
    let t0 = Instant::now();
    let (result, diag) = Topp3TcpSpline::new(fk).retime(&cfg, &path, &validator, &());
    let solve_time = t0.elapsed();
    let status = format!("{:?}", diag.status);
    finalize_result("topp3tcp-spline", status, solve_time, result, problem, fk)
}

pub fn run_topp_speed<const N: usize, FK: ContinuousFKChain<N, f64> + 'static>(
    problem: &BenchProblem<N>,
    fk: &FK,
) -> BenchResult<N> {
    let problem = problem.clone();
    let fk = fk.clone();
    with_timeout(move || run_topp_speed_inner::<N, FK>(&problem, &fk))
        .unwrap_or_else(|| timeout_result::<N>("topp-speed"))
}

fn run_topp_speed_inner<const N: usize, FK: ContinuousFKChain<N, f64>>(
    problem: &BenchProblem<N>,
    fk: &FK,
) -> BenchResult<N> {
    use deke_topp_speed::{Coordination, MotionSpec, ToppSolver};

    let path = match problem.path() {
        Ok(p) => p,
        Err(e) => {
            return BenchResult {
                retimer: "topp-speed",
                status: "path-construction".into(),
                solve_time: Duration::ZERO,
                trajectory_duration: Duration::ZERO,
                num_samples: 0,
                joint_fd: None,
                tcp_fd: None,
                utilization: None,
                max_path_deviation: None,
                error: Some(format!("{}", e)),
            };
        }
    };

    // topp-speed's `MotionSpec`: the offline `Retimer` impl pulls current/goal
    // poses + intermediate waypoints from the input path, so we only have to
    // populate limits + TCP.
    let mut spec = MotionSpec::<N, f64>::new();
    spec.max_vel = SRobotQ::from_array(problem.v_max);
    spec.max_accel = SRobotQ::from_array(problem.a_max);
    spec.max_jerk = SRobotQ::from_array(problem.j_max);
    spec.max_tcp_speed = problem.tcp_v_max;
    spec.current_pose = *path.first();
    spec.goal_pose = *path.last();
    // PhaseLocked: all joints share one normalized ramp shape so the
    // joint-space trajectory follows the chord between waypoints. The
    // default `TimeLocked` only enforces equal end times — joints then
    // bow off the chord because each runs its own jerk-limited shape.
    spec.coordination = Coordination::PhaseLocked;

    let dt = Duration::from_secs_f64(1.0 / problem.sample_rate_hz);
    let solver = ToppSolver::<N, f64, _>::new(dt, fk);

    let validator = wide_validator::<N>();
    let t0 = Instant::now();
    let (result, diag) = solver.retime(&spec, &path, &validator, &());
    let solve_time = t0.elapsed();
    let status = format!("{:?}", diag.status);
    finalize_result("topp-speed", status, solve_time, result, problem, fk)
}

/// Runs every retimer on `problem` and returns one [`BenchResult`] per retimer
/// in fixed order.
pub fn run_all<const N: usize, FK: ContinuousFKChain<N, f64> + 'static>(
    problem: &BenchProblem<N>,
    fk: &FK,
) -> Vec<BenchResult<N>> {
    vec![
        run_topp3_lp(problem, fk),
        run_topp3tcp_spline(problem, fk),
        run_topp_speed(problem, fk),
    ]
}
