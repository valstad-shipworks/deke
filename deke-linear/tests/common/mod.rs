// Shared across several test binaries; each uses a subset.
#![allow(dead_code)]

use std::time::Duration;

use deke_kin::{DHJoint, JointLimits as KinJointLimits, Kinematics};
use deke_linear::{
    CartesianLinearPlanner, ConstantSpeedRetimer, JointLimits, LinearConstraints,
    LinearPlannerDiagnostic, LinearRetimerDiagnostic, PathConditioning, PlannerOptions,
    RedundantConfig, RedundantDiagnostic, RedundantLinearPlanner, RedundantOptions, TcpLimits,
    condition,
};
use deke_types::glam::{DAffine3, DMat3, DVec3};
use deke_types::{
    DekeError, FKChain, Planner, Retimer, SRobotPath, SRobotQ, SRobotTraj, Validator,
};

/// Caller-owned bundle of the three stages' knobs. The library no longer ships
/// an end-to-end config; orchestration (and the config it needs) lives here.
#[derive(Clone)]
pub struct Cfg {
    pub conditioning: PathConditioning,
    pub planner: PlannerOptions<6>,
    pub redundant: Option<RedundantOptions>,
    pub constraints: LinearConstraints<6>,
}

impl Cfg {
    /// Arc-welding preset, quoted in inches per minute (typically 20–50 IPM):
    /// fine geometric sampling with the velocity-based reconfiguration test on.
    pub fn weld(ipm: f64, joint: JointLimits<6>, output_dt: Duration) -> Self {
        let tcp_speed = ipm * 0.0254 / 60.0;
        Self {
            conditioning: PathConditioning {
                sharp_corner_angle: 30.0_f64.to_radians(),
            },
            planner: PlannerOptions {
                sample_ds: 5e-4,
                manip_weight: 1.0,
                max_branch_jump: 0.6,
                max_velocity: tcp_speed,
                joint_v_max: joint.v_max,
                reconfig_vel_fraction: 0.9,
            },
            redundant: None,
            constraints: LinearConstraints {
                joint,
                tcp: TcpLimits::speed(tcp_speed),
                output_dt,
                forbid_interior_dips: false,
                corner_smoothing: Some(0.01),
            },
        }
    }

    pub fn with_redundancy(mut self, options: RedundantOptions) -> Self {
        self.redundant = Some(options);
        self
    }
}

/// Per-run diagnostics from [`follow`].
#[derive(Default, Debug)]
pub struct FollowOut {
    pub runs: usize,
    pub planner: Vec<LinearPlannerDiagnostic>,
    pub redundant: Vec<RedundantDiagnostic>,
    pub retimer: Vec<LinearRetimerDiagnostic>,
}

/// Caller-side orchestration of the three stages purely through the public
/// trait surface: condition the polyline into runs, then plan ([`Planner`]) and
/// retime ([`Retimer`]) each run and concatenate the trajectories. Runs are
/// planned independently — there is no cross-run seed stitching.
pub fn follow<V: Validator<6, (), f64>>(
    robot: &Kinematics<6, f64>,
    poses: &[DAffine3],
    cfg: &Cfg,
    validator: &V,
    ctx: &V::Context<'_>,
) -> Result<(SRobotTraj<6, f64>, FollowOut), DekeError> {
    let runs = condition(poses, &cfg.conditioning)?;
    let planner = CartesianLinearPlanner::new(robot);
    let retimer = ConstantSpeedRetimer::new(robot);
    let redundant = cfg
        .redundant
        .as_ref()
        .map(|_| RedundantLinearPlanner::new(robot));

    let mut all: Vec<SRobotQ<6, f64>> = Vec::new();
    let mut out = FollowOut {
        runs: runs.len(),
        ..Default::default()
    };
    for run in runs.iter() {
        let jpath = match (&redundant, &cfg.redundant) {
            (Some(rp), Some(ropts)) => {
                let rcfg = RedundantConfig {
                    planner: cfg.planner.clone(),
                    redundant: ropts.clone(),
                };
                let (path, diag) = rp.plan::<DekeError, _>(&rcfg, run, validator, ctx);
                out.redundant.push(diag);
                path?
            }
            _ => {
                let (path, diag) = planner.plan::<DekeError, _>(&cfg.planner, run, validator, ctx);
                out.planner.push(diag);
                path?
            }
        };
        let (traj, diag) = retimer.retime(&cfg.constraints, &jpath, validator, ctx);
        out.retimer.push(diag);
        let traj = traj?;
        let samples = traj.path().iter().copied();
        if all.is_empty() {
            all.extend(samples);
        } else {
            all.extend(samples.skip(1));
        }
    }

    let dt = cfg.constraints.output_dt;
    let path = SRobotPath::try_new(all)?;
    Ok((SRobotTraj::new(dt, path), out))
}

/// UR10-ish 6R chain (spherical wrist → analytic IK), generous joint limits.
pub fn ur() -> Kinematics<6, f64> {
    use std::f64::consts::PI;
    let alpha = [PI / 2.0, 0.0, 0.0, PI / 2.0, -PI / 2.0, 0.0];
    let a = [0.0, -0.612, -0.573, 0.0, 0.0, 0.0];
    let d = [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922];
    Kinematics::from_dh(
        std::array::from_fn(|i| DHJoint {
            a: a[i],
            alpha: alpha[i],
            d: d[i],
            theta_offset: 0.0,
        }),
        KinJointLimits::symmetric(2.0 * PI),
        &[],
    )
}

/// A well-conditioned mid-workspace configuration to anchor test paths.
pub fn anchor() -> SRobotQ<6, f64> {
    SRobotQ::from_array([0.2, -1.0, 1.2, -1.3, -std::f64::consts::FRAC_PI_2, 0.3])
}

/// A validator that accepts everything, for tests that don't exercise obstacles.
pub fn noop() -> deke_linear::NoopValidator<6> {
    deke_linear::NoopValidator
}

pub fn config(tcp_speed: f64) -> Cfg {
    config_flag(tcp_speed, false)
}

pub fn config_flag(tcp_speed: f64, forbid_interior_dips: bool) -> Cfg {
    Cfg {
        conditioning: PathConditioning::default(),
        planner: PlannerOptions::default(),
        redundant: None,
        constraints: LinearConstraints {
            joint: JointLimits::symmetric(2.0, 8.0, 80.0),
            tcp: TcpLimits::speed(tcp_speed),
            output_dt: Duration::from_millis(8),
            forbid_interior_dips,
            corner_smoothing: Some(0.01),
        },
    }
}

/// A straight Cartesian pose line of `n` vertices through the anchor's pose,
/// translating along `dir` for `len` metres with fixed orientation.
pub fn straight(robot: &Kinematics<6, f64>, dir: DVec3, len: f64, n: usize) -> Vec<DAffine3> {
    let base = robot.fk_end(&anchor()).unwrap();
    let rot = base.matrix3;
    let dir = dir.normalize();
    (0..n)
        .map(|i| {
            let f = i as f64 / (n - 1) as f64;
            DAffine3::from_mat3_translation(rot, base.translation + dir * (f * len))
        })
        .collect()
}

/// Two straight legs meeting at a corner, fixed orientation. `turn` is applied
/// in the XY plane of the world between the legs.
pub fn corner(
    robot: &Kinematics<6, f64>,
    leg: f64,
    turn_rad: f64,
    per_leg: usize,
) -> Vec<DAffine3> {
    let base = robot.fk_end(&anchor()).unwrap();
    let rot = base.matrix3;
    let d0 = DVec3::X;
    let d1 = DVec3::new(turn_rad.cos(), turn_rad.sin(), 0.0);
    let mut out = Vec::new();
    let p0 = base.translation;
    for i in 0..per_leg {
        let f = i as f64 / (per_leg - 1) as f64;
        out.push(DAffine3::from_mat3_translation(rot, p0 + d0 * (f * leg)));
    }
    let corner = p0 + d0 * leg;
    for i in 1..per_leg {
        let f = i as f64 / (per_leg - 1) as f64;
        out.push(DAffine3::from_mat3_translation(
            rot,
            corner + d1 * (f * leg),
        ));
    }
    out
}

/// TCP linear speed (m/s) between consecutive output samples via FK.
pub fn tcp_speeds(robot: &Kinematics<6, f64>, traj: &SRobotTraj<6, f64>) -> Vec<f64> {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    (0..p.len().saturating_sub(1))
        .map(|i| {
            let a = robot.fk_end(&p[i]).unwrap().translation;
            let b = robot.fk_end(&p[i + 1]).unwrap().translation;
            a.distance(b) / dt
        })
        .collect()
}

/// Peak per-axis joint velocity (rad/s) from the output trajectory.
pub fn joint_vel_peak(traj: &SRobotTraj<6, f64>) -> f64 {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let mut peak = 0.0f64;
    for i in 0..p.len().saturating_sub(1) {
        for j in 0..6 {
            peak = peak.max(((p[i + 1].0[j] - p[i].0[j]) / dt).abs());
        }
    }
    peak
}

/// Peak per-axis joint acceleration (rad/s²) via second difference.
pub fn joint_acc_peak(traj: &SRobotTraj<6, f64>) -> f64 {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let mut peak = 0.0f64;
    for i in 1..p.len().saturating_sub(1) {
        for j in 0..6 {
            let acc = (p[i + 1].0[j] - 2.0 * p[i].0[j] + p[i - 1].0[j]) / (dt * dt);
            peak = peak.max(acc.abs());
        }
    }
    peak
}

/// Peak per-axis joint jerk (rad/s³) via third difference — the quantity a
/// controller reconstructs from the position stream.
pub fn joint_jerk_peak(traj: &SRobotTraj<6, f64>) -> f64 {
    let dt = traj.dt().as_secs_f64();
    let p = traj.path();
    let mut peak = 0.0f64;
    for i in 3..p.len() {
        for j in 0..6 {
            let jk = (p[i].0[j] - 3.0 * p[i - 1].0[j] + 3.0 * p[i - 2].0[j] - p[i - 3].0[j])
                / (dt * dt * dt);
            peak = peak.max(jk.abs());
        }
    }
    peak
}

/// Peak TCP tangential acceleration and jerk (m/s², m/s³) from the FK speed
/// stream — the second/third difference of `tcp_speeds`.
pub fn tcp_accel_jerk_peak(robot: &Kinematics<6, f64>, traj: &SRobotTraj<6, f64>) -> (f64, f64) {
    let dt = traj.dt().as_secs_f64();
    let sp = tcp_speeds(robot, traj);
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

#[allow(dead_code)]
pub fn identity_rot() -> DMat3 {
    DMat3::IDENTITY
}

/// Deterministic SplitMix64 PRNG so fuzz cases are reproducible from a seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    /// Uniform integer in `[lo, hi]`.
    pub fn int(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }
    /// A random unit direction, rejection-sampled away from the origin.
    pub fn unit_dir(&mut self) -> DVec3 {
        loop {
            let v = DVec3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            if v.length() > 0.2 {
                return v.normalize();
            }
        }
    }
}

fn point_segment_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-18)).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// Worst-case Cartesian distance (metres) from any output TCP sample to the
/// commanded pose polyline — the path-fidelity error of the executed weld.
pub fn tcp_polyline_deviation(
    robot: &Kinematics<6, f64>,
    traj: &SRobotTraj<6, f64>,
    poses: &[DAffine3],
) -> f64 {
    let line: Vec<DVec3> = poses.iter().map(|p| p.translation).collect();
    let path = traj.path();
    (0..path.len())
        .map(|i| {
            let pt = robot.fk_end(&path[i]).unwrap().translation;
            (0..line.len() - 1)
                .map(|s| point_segment_distance(pt, line[s], line[s + 1]))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}
