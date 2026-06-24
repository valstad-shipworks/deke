//! Redundancy-resolving planner for an arm mounted on a prismatic first axis (a
//! linear rail / 7th external axis).
//!
//! The rail position `x` along a fixed world axis `â` is a continuous redundant
//! scalar DOF, resolved the same way the free tool yaw is in [`crate::redundant`]:
//! a coarse rail grid is searched by a single global DP over
//! `(station) × (rail × branch)` — exact, so it finds the globally optimal rail
//! track in one pass. A manipulability node cost steers off singularities, a
//! centering term keeps the carriage off its stops, a rail-rate edge penalty
//! keeps the slow axis from darting, and the velocity reconfiguration test (over
//! all seven joints, the rail included) rejects discontinuous edges. The coarse
//! `x(s)` schedule is then refined to fine arc-length spacing — either linearly
//! or with a monotone PCHIP cubic — and the inner arm IK is solved against the
//! rail-shifted target, picking the branch nearest the previous step.
//!
//! The output joint vector is rail-first: `q = [x_rail, q1..q6]`. The
//! [`RailMountedChain`] is a [`ContinuousFKChain`] over the full seven DOF, so a
//! seven-wide path flows through the existing [`crate::ConstantSpeedRetimer`]
//! untouched.

use deke_types::glam::{DAffine3, DQuat, DVec3};
use deke_types::{
    ContinuousFKChain, DekeError, DekeResult, FKChain, IkOutcome, IkSolver, JointSpec, KinSpec,
    Planner, SRobotPath, SRobotQ, Validator,
};

use crate::constraints::PlannerOptions;
use crate::diagnostic::RailDiagnostic;
use crate::error::LinearError;
use crate::path::CartesianRun;
use crate::planner::is_reconfiguration;
use crate::redundant::RedundantOptions;
use crate::util::{interp, ladder_dp, pchip};

/// The fixed world-frame axis the rail travels along. (World frame, unlike
/// [`crate::RedundantAxis`], which is in the tool frame.)
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum RailAxis {
    #[default]
    PosX,
    PosY,
    PosZ,
    /// An arbitrary unit axis in the world frame.
    Custom(DVec3),
}

impl RailAxis {
    /// Unit axis vector in the world frame.
    pub fn vector(&self) -> DVec3 {
        match self {
            RailAxis::PosX => DVec3::X,
            RailAxis::PosY => DVec3::Y,
            RailAxis::PosZ => DVec3::Z,
            RailAxis::Custom(v) => v.normalize_or_zero(),
        }
    }
}

/// How the resolved coarse rail schedule `x(s)` is smoothed to fine spacing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RailRefine {
    /// Piecewise-linear interpolation of the DP knots.
    Linear,
    /// Fritsch–Carlson monotone cubic — smoother rail velocity, no overshoot of
    /// the DP envelope.
    Pchip,
}

/// Knobs for the redundancy-resolving rail search.
#[derive(Clone, Debug)]
pub struct RailOptions {
    /// Fixed world rail axis.
    pub axis: RailAxis,
    /// Rail travel window `(min, max)` in metres — the machine's hard stops.
    pub window: (f64, f64),
    /// Rail samples across the whole window — a resolution *floor*. The planner
    /// raises it as needed so the grid is fine enough for the rail to keep up with
    /// the commanded speed (spacing ≤ `reconfig_vel_fraction·v_rail·dp_ds/speed`),
    /// so the caller never has to size the grid to the scan speed.
    pub samples: usize,
    /// Coarse DP station spacing (metres).
    pub dp_ds: f64,
    /// Edge penalty weight on rail rate `|Δx|/Δs` (firm — the rail is slow).
    pub rate_weight: f64,
    /// Maximum rail change between DP stations (metres).
    pub max_step: f64,
    /// Node cost pulling `x` toward the window centre (keeps the carriage off its
    /// stops where the arm is equally happy either way).
    pub centering_weight: f64,
    /// Smoothing of `x(s)` at fine spacing.
    pub refine: RailRefine,
}

impl Default for RailOptions {
    fn default() -> Self {
        Self {
            axis: RailAxis::PosX,
            window: (-0.5, 0.5),
            samples: 21,
            dp_ds: 5e-3,
            rate_weight: 0.5,
            max_step: 0.05,
            centering_weight: 0.05,
            refine: RailRefine::Pchip,
        }
    }
}

/// Bundles the branch-tracking knobs and the rail-search knobs so the rail
/// planner can satisfy the single-config [`Planner`] trait.
#[derive(Clone, Debug)]
pub struct RailConfig<const A: usize, const N: usize> {
    pub planner: PlannerOptions<N>,
    pub rail: RailOptions,
}

/// Bundles the rail and yaw knobs for the hierarchical rail+yaw planner.
#[derive(Clone, Debug)]
pub struct RailYawConfig<const A: usize, const N: usize> {
    pub planner: PlannerOptions<N>,
    pub rail: RailOptions,
    pub yaw: RedundantOptions,
}

/// A serial arm rigidly mounted on a prismatic first axis along a fixed world
/// direction `â`. The full chain is `q = [x_rail, q_arm…]` (rail first).
#[derive(Clone, Debug)]
pub struct RailMountedChain<'a, const A: usize, const N: usize, ARM> {
    arm: &'a ARM,
    axis: DVec3,
}

impl<'a, const A: usize, const N: usize, ARM> RailMountedChain<'a, A, N, ARM> {
    pub fn new(arm: &'a ARM, axis: RailAxis) -> Self {
        debug_assert!(N == A + 1, "rail-mounted chain DOF must be arm DOF + 1");
        Self {
            arm,
            axis: axis.vector(),
        }
    }

    fn split(q: &SRobotQ<N, f64>) -> (f64, SRobotQ<A, f64>) {
        (q.0[0], SRobotQ::from_fn(|i| q.0[i + 1]))
    }

    fn rail_tf(&self, x: f64) -> DAffine3 {
        DAffine3::from_translation(self.axis * x)
    }
}

impl<'a, const A: usize, const N: usize, ARM> FKChain<N, f64> for RailMountedChain<'a, A, N, ARM>
where
    ARM: ContinuousFKChain<A, f64>,
{
    type Error = DekeError;

    fn fk(&self, q: &SRobotQ<N, f64>) -> Result<[DAffine3; N], DekeError> {
        let (x, arm_q) = Self::split(q);
        let t = self.rail_tf(x);
        let af = self.arm.fk(&arm_q)?;
        Ok(std::array::from_fn(
            |i| if i == 0 { t } else { t * af[i - 1] },
        ))
    }

    fn ee_tf(&self) -> DAffine3 {
        self.arm.ee_tf()
    }

    fn fk_end(&self, q: &SRobotQ<N, f64>) -> Result<DAffine3, DekeError> {
        let (x, arm_q) = Self::split(q);
        Ok(self.rail_tf(x) * self.arm.fk_end(&arm_q)?)
    }
}

impl<'a, const A: usize, const N: usize, ARM> ContinuousFKChain<N, f64>
    for RailMountedChain<'a, A, N, ARM>
where
    ARM: ContinuousFKChain<A, f64>,
{
    fn structure(&self) -> KinSpec<f64, N> {
        let a = self.arm.structure();
        let joints: [(DAffine3, JointSpec<f64>); N] = std::array::from_fn(|i| match i {
            0 => (
                DAffine3::IDENTITY,
                JointSpec::Prismatic {
                    axis_local: self.axis,
                },
            ),
            1 => (a.base_to_first * a.joints[0].0, a.joints[0].1),
            k => a.joints[k - 1],
        });
        KinSpec {
            base_to_first: DAffine3::IDENTITY,
            joints,
            end_to_ee: a.end_to_ee,
        }
    }
}

struct RailNode<const N: usize> {
    x: f64,
    q: SRobotQ<N, f64>,
    cost: f64,
}

/// Multi-station rail-redundancy planner over a single conditioned run. `ARM` is
/// the inner serial arm (analytic IK over its `A` DOF); the planner emits the
/// `N = A + 1` rail-first joint path.
#[derive(Clone, Debug)]
pub struct RailLinearPlanner<'a, const A: usize, const N: usize, ARM> {
    arm: &'a ARM,
}

impl<'a, const A: usize, const N: usize, ARM> RailLinearPlanner<'a, A, N, ARM>
where
    ARM: ContinuousFKChain<A, f64> + IkSolver<A, f64>,
{
    pub fn new(arm: &'a ARM) -> Self {
        Self { arm }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_run<V: Validator<N, (), f64>>(
        &self,
        run: &CartesianRun,
        planner: &PlannerOptions<N>,
        rail: &RailOptions,
        validator: &V,
        ctx: &V::Context<'_>,
        run_idx: usize,
    ) -> Result<(SRobotPath<N, f64>, RailDiagnostic), LinearError> {
        let axis = rail.axis.vector();
        let length = run.length();
        let n_dp = ((length / rail.dp_ds).ceil() as usize).max(1) + 1;

        let stations = self.build_stations(
            run, rail, planner, axis, length, n_dp, validator, ctx, run_idx,
        )?;

        let (coarse_s, coarse_x) = solve_global(&stations, rail, planner, length, n_dp)
            .ok_or(LinearError::NoContinuousTrack { run: run_idx })?;
        let coarse_x = polish_rail(self.arm, run, rail, axis, &coarse_s, &coarse_x);
        let (coarse_s, coarse_x) = pad_linear(&coarse_s, &coarse_x);

        self.refine(
            run, planner, rail, axis, length, &coarse_s, &coarse_x, validator, ctx, run_idx,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_stations<V: Validator<N, (), f64>>(
        &self,
        run: &CartesianRun,
        rail: &RailOptions,
        planner: &PlannerOptions<N>,
        axis: DVec3,
        length: f64,
        n_dp: usize,
        validator: &V,
        ctx: &V::Context<'_>,
        run_idx: usize,
    ) -> Result<Vec<Vec<RailNode<N>>>, LinearError> {
        let (w0, w1) = rail.window;
        // `samples` is a resolution floor; raise it so the grid spacing satisfies
        // both requirements, neither of which the caller should have to size by hand:
        //   - reconfiguration: spacing ≤ frac·v_rail·dp_ds/speed, so the rail can
        //     advance a sample per DP station without tripping the velocity test;
        //   - resolution: spacing ≲ a few·dp_ds, so the coarse DP can actually
        //     resolve a long traverse (too coarse and it routes a degenerate track
        //     that leaves the arm over-reaching).
        let v_rail = planner.joint_v_max.0[0];
        let g_reconfig = if planner.max_velocity > 0.0 && v_rail.is_finite() && v_rail > 0.0 {
            planner.reconfig_vel_fraction * v_rail * rail.dp_ds / planner.max_velocity
        } else {
            f64::INFINITY
        };
        let g_max = g_reconfig.min(3.0 * rail.dp_ds).max(1e-9);
        let needed = (((w1 - w0).abs() / g_max).ceil() as usize + 1).min(4096);
        let samples = rail.samples.max(needed).max(1);
        let mid = 0.5 * (w0 + w1);
        let half = (0.5 * (w1 - w0)).max(1e-9);
        let x_at = |m: usize| -> f64 {
            if samples <= 1 {
                mid
            } else {
                w0 + (w1 - w0) * m as f64 / (samples - 1) as f64
            }
        };

        let mut stations = Vec::with_capacity(n_dp);
        for k in 0..n_dp {
            let s = (k as f64 * rail.dp_ds).min(length);
            let ref_pose = run.eval(s);
            let ref_rot = DQuat::from_mat3(&ref_pose.matrix3);
            let mut nodes = Vec::with_capacity(samples * 4);
            let mut had_ik = false;
            for m in 0..samples {
                let x = x_at(m);
                let target =
                    DAffine3::from_rotation_translation(ref_rot, ref_pose.translation - axis * x);
                if let IkOutcome::Solved(sols) = self.arm.ik(target)? {
                    had_ik |= !sols.is_empty();
                    for arm_q in sols {
                        let q7 = lift(x, &arm_q);
                        if validator.validate(q7, ctx).is_err() {
                            continue;
                        }
                        // Score the ARM's conditioning, not the 7-DOF chain's: the
                        // rail's prismatic Jacobian column keeps `det(J Jᵀ)` high
                        // even when the arm itself is singular, so a 7-DOF measure
                        // masks an arm singularity and the DP would not recruit the
                        // rail to escape it. The rail is the free DOF spent to keep
                        // the arm well away from its own singular sets.
                        let w = self.arm.manipulability(&arm_q).map_err(LinearError::Deke)?;
                        let center = (x - mid) / half;
                        nodes.push(RailNode {
                            x,
                            q: q7,
                            cost: planner.manip_weight / (w + 1e-9)
                                + rail.centering_weight * center * center,
                        });
                    }
                }
            }
            if nodes.is_empty() {
                return Err(if had_ik {
                    LinearError::Obstructed { run: run_idx, s }
                } else {
                    LinearError::Unreachable { run: run_idx, s }
                });
            }
            stations.push(nodes);
        }
        Ok(stations)
    }

    #[allow(clippy::too_many_arguments)]
    fn refine<V: Validator<N, (), f64>>(
        &self,
        run: &CartesianRun,
        planner: &PlannerOptions<N>,
        rail: &RailOptions,
        axis: DVec3,
        length: f64,
        coarse_s: &[f64],
        coarse_x: &[f64],
        validator: &V,
        ctx: &V::Context<'_>,
        run_idx: usize,
    ) -> Result<(SRobotPath<N, f64>, RailDiagnostic), LinearError> {
        let n_fine = ((length / planner.sample_ds).ceil() as usize).max(1) + 1;
        // Space the fine stations evenly over `[0, length]`. Clamping `i·sample_ds`
        // to `length` instead leaves a short remainder as the final interval, whose
        // large secant the retimer reads as a sharp end segment — a boundary jerk
        // spike that surfaces on a fast scan.
        let step = length / (n_fine - 1).max(1) as f64;
        let schedule = |s: f64| -> f64 {
            match rail.refine {
                RailRefine::Linear => interp(coarse_s, coarse_x, s),
                RailRefine::Pchip => pchip(coarse_s, coarse_x, s),
            }
        };

        // With the rail schedule fixed, the arm is an ordinary branch-tracking
        // problem: solve IK at each fine station against the rail-shifted target and
        // route a globally continuous branch with the same ladder DP the base
        // planner uses. A greedy nearest-branch walk is myopic and can strand the
        // track on a hard diagonal-rail case where a continuous route exists.
        // The rail position only changes how the arm reaches the target — the inner
        // IK still hits the exact weld pose for any `x` — so `x(s)` can be smoothed
        // as hard as needed for free. A residual sub-mm wobble (DP grid + branch
        // crossings) is negligible in space but, cubed by a fast scan speed, spikes
        // the joint jerk; a slope-preserving box filter removes it while a straight
        // scan's linear ramp passes through untouched.
        let mut xs: Vec<f64> = (0..n_fine)
            .map(|i| schedule((i as f64 * step).min(length)))
            .collect();
        let win = (0.03 / step.max(1e-9)).round() as usize;
        smooth_schedule(&mut xs, win.max(1), 4);

        let mut x_lo = f64::INFINITY;
        let mut x_hi = f64::NEG_INFINITY;
        let mut layers: Vec<Vec<(SRobotQ<N, f64>, f64, f64)>> = Vec::with_capacity(n_fine);
        for (i, &x) in xs.iter().enumerate() {
            let s = (i as f64 * step).min(length);
            x_lo = x_lo.min(x);
            x_hi = x_hi.max(x);
            let ref_pose = run.eval(s);
            let rot = DQuat::from_mat3(&ref_pose.matrix3);
            let target = DAffine3::from_rotation_translation(rot, ref_pose.translation - axis * x);
            let raw = match self.arm.ik(target)? {
                IkOutcome::Solved(sols) if !sols.is_empty() => sols,
                _ => return Err(LinearError::Unreachable { run: run_idx, s }),
            };
            let mut nodes = Vec::with_capacity(raw.len());
            for arm_q in raw {
                let q = lift(x, &arm_q);
                if validator.validate(q, ctx).is_err() {
                    continue;
                }
                let w = self.arm.manipulability(&arm_q).map_err(LinearError::Deke)?;
                nodes.push((q, planner.manip_weight / (w + 1e-9), w));
            }
            if nodes.is_empty() {
                return Err(LinearError::Obstructed { run: run_idx, s });
            }
            layers.push(nodes);
        }

        let ds_at = |k: usize| {
            ((k as f64 * step).min(length) - ((k - 1) as f64 * step).min(length)).max(1e-12)
        };
        let layer_sizes: Vec<usize> = layers.iter().map(Vec::len).collect();
        let (chosen, _) = ladder_dp(
            &layer_sizes,
            |k, i| layers[k][i].1,
            |k, p, c| {
                let qp = layers[k - 1][p].0;
                let qc = layers[k][c].0;
                if is_reconfiguration(&qp, &qc, ds_at(k), planner) {
                    None
                } else {
                    Some(qp.distance(&qc))
                }
            },
        )
        .ok_or(LinearError::NoContinuousTrack { run: run_idx })?;

        let mut min_manip = f64::INFINITY;
        let mut fine: Vec<SRobotQ<N, f64>> = chosen
            .iter()
            .enumerate()
            .map(|(k, &i)| {
                min_manip = min_manip.min(layers[k][i].2);
                layers[k][i].0
            })
            .collect();

        // Remove the per-sample floating-point jitter of the independent analytic IK
        // solves (~1e-5 rad — sub-mm at the TCP, so well within path tolerance). A
        // fast scan cubes that jitter into a spurious jerk spike. The window is a
        // *few samples* (jitter is per-sample noise), not a fixed distance: a wider
        // window would also smooth genuine high-frequency joint motion such as a
        // weave overlay, which must be preserved.
        for j in 0..N {
            let mut col: Vec<f64> = fine.iter().map(|q| q.0[j]).collect();
            smooth_schedule(&mut col, 1, 2);
            for (q, &c) in fine.iter_mut().zip(col.iter()) {
                q.0[j] = c;
            }
        }

        let path = SRobotPath::try_new(fine).map_err(LinearError::from)?;
        Ok((
            path,
            RailDiagnostic {
                samples: n_fine,
                min_manipulability: min_manip,
                rail_range: (x_lo, x_hi),
            },
        ))
    }
}

impl<'a, const A: usize, const N: usize, ARM> Planner<N, f64> for RailLinearPlanner<'a, A, N, ARM>
where
    ARM: ContinuousFKChain<A, f64> + IkSolver<A, f64>,
{
    type Diagnostic = RailDiagnostic;
    type Config = RailConfig<A, N>;
    type Waypoints = CartesianRun;

    fn plan<E: Into<DekeError>, V: Validator<N, (), f64>>(
        &self,
        config: &Self::Config,
        waypoints: &Self::Waypoints,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotPath<N, f64>>, Self::Diagnostic) {
        match self.plan_run(waypoints, &config.planner, &config.rail, validator, ctx, 0) {
            Ok((path, diag)) => (Ok(path), diag),
            Err(e) => (
                Err(e.into()),
                RailDiagnostic {
                    samples: 0,
                    min_manipulability: 0.0,
                    rail_range: (0.0, 0.0),
                },
            ),
        }
    }
}

/// Hierarchical rail + tool-yaw planner. The rail is resolved first with the
/// orientation pinned to the reference; the yaw is then resolved per fine
/// station with the rail held fixed at the resolved `x(s)`. This keeps the
/// search one-dimensional at each stage (no joint 2-D DP) while still composing
/// both redundant DOFs.
#[derive(Clone, Debug)]
pub struct RailYawPlanner<'a, const A: usize, const N: usize, ARM> {
    arm: &'a ARM,
}

impl<'a, const A: usize, const N: usize, ARM> RailYawPlanner<'a, A, N, ARM>
where
    ARM: ContinuousFKChain<A, f64> + IkSolver<A, f64>,
{
    pub fn new(arm: &'a ARM) -> Self {
        Self { arm }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan_run<V: Validator<N, (), f64>>(
        &self,
        run: &CartesianRun,
        planner: &PlannerOptions<N>,
        rail: &RailOptions,
        yaw: &RedundantOptions,
        validator: &V,
        ctx: &V::Context<'_>,
        run_idx: usize,
    ) -> Result<(SRobotPath<N, f64>, RailDiagnostic), LinearError> {
        let chain = RailMountedChain::<A, N, ARM>::new(self.arm, rail.axis);
        let rail_axis = rail.axis.vector();
        let yaw_axis = yaw.axis.vector();
        let length = run.length();

        let (coarse_s, coarse_x) = {
            let n_dp = ((length / rail.dp_ds).ceil() as usize).max(1) + 1;
            let stations = self.build_rail_stations(
                run, rail, planner, rail_axis, length, n_dp, validator, ctx, run_idx,
            )?;
            solve_global(&stations, rail, planner, length, n_dp)
                .ok_or(LinearError::NoContinuousTrack { run: run_idx })?
        };

        let rail_at = |s: f64| -> f64 {
            match rail.refine {
                RailRefine::Linear => interp(&coarse_s, &coarse_x, s),
                RailRefine::Pchip => pchip(&coarse_s, &coarse_x, s),
            }
        };

        let n_fine = ((length / planner.sample_ds).ceil() as usize).max(1) + 1;
        let samples = yaw.yaw_samples.max(1);
        let yaw_at = |m: usize| -> f64 {
            if samples <= 1 {
                0.5 * (yaw.yaw_window.0 + yaw.yaw_window.1)
            } else {
                yaw.yaw_window.0
                    + (yaw.yaw_window.1 - yaw.yaw_window.0) * m as f64 / (samples - 1) as f64
            }
        };

        let mut fine: Vec<SRobotQ<N, f64>> = Vec::with_capacity(n_fine);
        let mut min_manip = f64::INFINITY;
        let mut x_lo = f64::INFINITY;
        let mut x_hi = f64::NEG_INFINITY;
        let mut prev: Option<(SRobotQ<N, f64>, f64)> = None;

        for i in 0..n_fine {
            let s = (i as f64 * planner.sample_ds).min(length);
            let x = rail_at(s);
            x_lo = x_lo.min(x);
            x_hi = x_hi.max(x);
            let ref_pose = run.eval(s);
            let ref_rot = DQuat::from_mat3(&ref_pose.matrix3);
            let trans = ref_pose.translation - rail_axis * x;

            let mut candidates: Vec<SRobotQ<N, f64>> = Vec::new();
            for m in 0..samples {
                let psi = yaw_at(m);
                let rot = ref_rot * DQuat::from_axis_angle(yaw_axis, psi);
                let target = DAffine3::from_rotation_translation(rot, trans);
                if let IkOutcome::Solved(sols) = self.arm.ik(target)? {
                    for arm_q in sols {
                        let q7 = lift(x, &arm_q);
                        if validator.validate(q7, ctx).is_ok() {
                            candidates.push(q7);
                        }
                    }
                }
            }
            if candidates.is_empty() {
                return Err(LinearError::Obstructed { run: run_idx, s });
            }

            let q = match prev {
                Some((pq, _)) => candidates
                    .into_iter()
                    .min_by(|a, b| pq.distance(a).total_cmp(&pq.distance(b)))
                    .unwrap(),
                None => {
                    let mut best = candidates[0];
                    let mut best_w = -1.0;
                    for q in candidates {
                        let w = chain.manipulability(&q).map_err(LinearError::Deke)?;
                        if w > best_w {
                            best_w = w;
                            best = q;
                        }
                    }
                    best
                }
            };

            if let Some((pq, ps)) = prev {
                let dsf = (s - ps).max(1e-12);
                if is_reconfiguration(&pq, &q, dsf, planner) {
                    return Err(LinearError::NoContinuousTrack { run: run_idx });
                }
            }

            min_manip = min_manip.min(chain.manipulability(&q).map_err(LinearError::Deke)?);
            fine.push(q);
            prev = Some((q, s));
        }

        let path = SRobotPath::try_new(fine).map_err(LinearError::from)?;
        Ok((
            path,
            RailDiagnostic {
                samples: n_fine,
                min_manipulability: min_manip,
                rail_range: (x_lo, x_hi),
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_rail_stations<V: Validator<N, (), f64>>(
        &self,
        run: &CartesianRun,
        rail: &RailOptions,
        planner: &PlannerOptions<N>,
        axis: DVec3,
        length: f64,
        n_dp: usize,
        validator: &V,
        ctx: &V::Context<'_>,
        run_idx: usize,
    ) -> Result<Vec<Vec<RailNode<N>>>, LinearError> {
        let inner = RailLinearPlanner::<A, N, ARM>::new(self.arm);
        inner.build_stations(
            run, rail, planner, axis, length, n_dp, validator, ctx, run_idx,
        )
    }
}

impl<'a, const A: usize, const N: usize, ARM> Planner<N, f64> for RailYawPlanner<'a, A, N, ARM>
where
    ARM: ContinuousFKChain<A, f64> + IkSolver<A, f64>,
{
    type Diagnostic = RailDiagnostic;
    type Config = RailYawConfig<A, N>;
    type Waypoints = CartesianRun;

    fn plan<E: Into<DekeError>, V: Validator<N, (), f64>>(
        &self,
        config: &Self::Config,
        waypoints: &Self::Waypoints,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotPath<N, f64>>, Self::Diagnostic) {
        match self.plan_run(
            waypoints,
            &config.planner,
            &config.rail,
            &config.yaw,
            validator,
            ctx,
            0,
        ) {
            Ok((path, diag)) => (Ok(path), diag),
            Err(e) => (
                Err(e.into()),
                RailDiagnostic {
                    samples: 0,
                    min_manipulability: 0.0,
                    rail_range: (0.0, 0.0),
                },
            ),
        }
    }
}

fn lift<const A: usize, const N: usize>(x: f64, arm_q: &SRobotQ<A, f64>) -> SRobotQ<N, f64> {
    SRobotQ::from_fn(|i| if i == 0 { x } else { arm_q.0[i - 1] })
}

/// De-quantize the coarse rail track. `solve_global` chooses rail positions on a
/// discrete grid, so the schedule is a staircase whose occasional one-sample step
/// the arm absorbs as a near-discontinuity — fine at weld feed but a hard jerk
/// spike on a fast scan. The grid only ever served to seed a global, branch-safe
/// search; here each knot is polished to the *continuous* arm-manipulability
/// optimum near its grid value, which on a straight scan is exactly `x = s − const`
/// (perfectly smooth, no step at any speed). A light Laplacian pass removes any
/// residual jitter where the manipulability landscape is flat.
fn polish_rail<const A: usize, ARM>(
    arm: &ARM,
    run: &CartesianRun,
    rail: &RailOptions,
    axis: DVec3,
    coarse_s: &[f64],
    coarse_x: &[f64],
) -> Vec<f64>
where
    ARM: ContinuousFKChain<A, f64> + IkSolver<A, f64>,
{
    let (w0, w1) = rail.window;
    let manip_at = |s: f64, x: f64| -> f64 {
        let rp = run.eval(s);
        let rot = DQuat::from_mat3(&rp.matrix3);
        let target = DAffine3::from_rotation_translation(rot, rp.translation - axis * x);
        match arm.ik(target) {
            Ok(IkOutcome::Solved(sols)) => sols
                .iter()
                .map(|q| arm.manipulability(q).unwrap_or(0.0))
                .fold(0.0f64, f64::max),
            _ => 0.0,
        }
    };
    let spacing = (w1 - w0) / (rail.samples.max(2) - 1) as f64;
    let r = 1.5 * spacing;
    coarse_x
        .iter()
        .zip(coarse_s)
        .map(|(&x, &s)| {
            let lo = (x - r).max(w0);
            let hi = (x + r).min(w1);
            if hi - lo < 1e-9 {
                x
            } else {
                golden_max(|t| manip_at(s, t), lo, hi, 24)
            }
        })
        .collect()
}

/// Pad a coarse track with one linearly-extrapolated knot beyond each end, so the
/// PCHIP/linear schedule uses smooth interior slopes at the real endpoints instead
/// of a one-sided end slope, whose boundary curvature the arm would absorb as a
/// jerk spike on a fast scan. The fine schedule only samples `[s[0], s[last]]`, so
/// the padded knots are never emitted.
fn pad_linear(s: &[f64], x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = s.len();
    if n < 2 {
        return (s.to_vec(), x.to_vec());
    }
    let mut ss = Vec::with_capacity(n + 2);
    let mut xx = Vec::with_capacity(n + 2);
    ss.push(2.0 * s[0] - s[1]);
    xx.push(2.0 * x[0] - x[1]);
    ss.extend_from_slice(s);
    xx.extend_from_slice(x);
    ss.push(2.0 * s[n - 1] - s[n - 2]);
    xx.push(2.0 * x[n - 1] - x[n - 2]);
    (ss, xx)
}

/// Slope-preserving box smoothing of a sampled schedule. The array is padded with
/// linearly-extrapolated samples (slope from a short end fit) before each box pass,
/// so a linear ramp — including at the endpoints — is preserved exactly while
/// higher-frequency wobble is removed. A naive shrinking-window box instead
/// back-averages the ends and flattens the ramp, which makes the arm reconfigure.
fn smooth_schedule(x: &mut [f64], win: usize, passes: usize) {
    let n = x.len();
    if n < 3 || win == 0 {
        return;
    }
    let k = win.min(n - 1).max(1);
    let sl0 = (x[k] - x[0]) / k as f64;
    let sl1 = (x[n - 1] - x[n - 1 - k]) / k as f64;
    let pad = win * passes;
    let mut buf = Vec::with_capacity(n + 2 * pad);
    for j in (1..=pad).rev() {
        buf.push(x[0] - sl0 * j as f64);
    }
    buf.extend_from_slice(x);
    for j in 1..=pad {
        buf.push(x[n - 1] + sl1 * j as f64);
    }
    for _ in 0..passes {
        let prev = buf.clone();
        for i in win..buf.len() - win {
            buf[i] = prev[i - win..=i + win].iter().sum::<f64>() / (2 * win + 1) as f64;
        }
    }
    x.copy_from_slice(&buf[pad..pad + n]);
}

/// Golden-section search for the maximiser of a unimodal `f` on `[a, b]`.
fn golden_max(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64, iters: usize) -> f64 {
    let gr = (5.0f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - gr * (b - a), a + gr * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..iters {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

/// The single global DP over the rail grid. Returns the coarse `(s[], x[])` of
/// the minimum-cost continuous rail+branch track, or `None` if none exists.
fn solve_global<const N: usize>(
    stations: &[Vec<RailNode<N>>],
    rail: &RailOptions,
    planner: &PlannerOptions<N>,
    length: f64,
    n_dp: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let ds_at = |k: usize| {
        ((k as f64 * rail.dp_ds).min(length) - ((k - 1) as f64 * rail.dp_ds).min(length)).max(1e-9)
    };
    let layer_sizes: Vec<usize> = stations.iter().map(Vec::len).collect();
    let (chosen, _) = ladder_dp(
        &layer_sizes,
        |k, i| stations[k][i].cost,
        |k, p, i| {
            let na = &stations[k - 1][p];
            let nb = &stations[k][i];
            let dx = (nb.x - na.x).abs();
            let ds = ds_at(k);
            if dx > rail.max_step || is_reconfiguration(&na.q, &nb.q, ds, planner) {
                None
            } else {
                // Smoothness term measures ARM joint motion only; the rail carries
                // the TCP, so its travel is governed by `rate_weight` (smoothness),
                // `is_reconfiguration` (velocity cap) and the PCHIP refine — not
                // penalised as joint motion, which would stall a long traverse and
                // make the arm over-reach instead of letting the rail follow.
                let arm_dist = (1..N)
                    .map(|j| (na.q.0[j] - nb.q.0[j]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                Some(arm_dist + rail.rate_weight * dx / ds)
            }
        },
    )?;

    let s = (0..n_dp)
        .map(|k| (k as f64 * rail.dp_ds).min(length))
        .collect();
    let x = chosen
        .iter()
        .enumerate()
        .map(|(k, &i)| stations[k][i].x)
        .collect();
    Some((s, x))
}
