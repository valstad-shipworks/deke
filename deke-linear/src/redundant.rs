//! Redundancy-resolving planner for a tool that is rotationally symmetric about
//! one of its axes (a welding torch, spray head, …).
//!
//! The free rotation about the tool's symmetry axis (`yaw` `ψ`) is a continuous
//! redundant DOF. It is a smooth scalar, so it is gridded coarsely and resolved by
//! a single global DP over `(station) × (yaw × branch)` — exact, so it finds the
//! globally optimal yaw track in one pass. A manipulability node cost steers off
//! singularities; a yaw-rate edge penalty keeps the spin smooth; the velocity
//! reconfiguration test rejects discontinuous edges. The coarse `ψ(s)` schedule is
//! then refined: at fine arc-length spacing the orientation is
//! `R_ref(s) · Rot(â, ψ(s))` and analytic IK places the arm exactly, picking the
//! branch nearest the previous step (predictor–corrector).

use deke_types::glam::{DAffine3, DQuat, DVec3};
use deke_types::{
    ContinuousFKChain, DekeError, DekeResult, IkOutcome, IkSolver, Planner, SRobotPath, SRobotQ,
    Validator,
};

use crate::constraints::PlannerOptions;
use crate::diagnostic::RedundantDiagnostic;
use crate::error::LinearError;
use crate::path::CartesianRun;
use crate::planner::is_reconfiguration;
use crate::util::{interp, ladder_dp};

/// The tool-frame axis the tool is symmetric about (its free rotation DOF). The
/// sign matters only for the orientation convention; the rotation axis itself is
/// sign-agnostic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RedundantAxis {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
    /// An arbitrary unit axis in the tool frame.
    Custom(DVec3),
}

impl RedundantAxis {
    /// Unit axis vector in the tool frame.
    pub fn vector(&self) -> DVec3 {
        match self {
            RedundantAxis::PosX => DVec3::X,
            RedundantAxis::NegX => DVec3::NEG_X,
            RedundantAxis::PosY => DVec3::Y,
            RedundantAxis::NegY => DVec3::NEG_Y,
            RedundantAxis::PosZ => DVec3::Z,
            RedundantAxis::NegZ => DVec3::NEG_Z,
            RedundantAxis::Custom(v) => v.normalize_or_zero(),
        }
    }
}

/// Knobs for the redundancy-resolving yaw search.
#[derive(Clone, Debug)]
pub struct RedundantOptions {
    /// Tool symmetry axis (free rotation DOF).
    pub axis: RedundantAxis,
    /// Overall allowed yaw range (radians) relative to the reference orientation.
    pub yaw_window: (f64, f64),
    /// Yaw samples across the whole window (the DP grid resolution).
    pub yaw_samples: usize,
    /// DP station spacing (metres).
    pub dp_ds: f64,
    /// Edge penalty weight on yaw rate `|Δψ|/Δs` (smoother spin).
    pub yaw_rate_weight: f64,
    /// Maximum yaw change between DP stations (radians).
    pub max_yaw_step: f64,
}

impl Default for RedundantOptions {
    fn default() -> Self {
        Self {
            axis: RedundantAxis::PosZ,
            yaw_window: (-std::f64::consts::PI, std::f64::consts::PI),
            yaw_samples: 24,
            dp_ds: 5e-3,
            yaw_rate_weight: 0.2,
            max_yaw_step: 0.6,
        }
    }
}

/// Bundles the branch-tracking knobs and the yaw-search knobs so the redundant
/// planner can satisfy the single-config [`Planner`] trait.
#[derive(Clone, Debug)]
pub struct RedundantConfig<const N: usize> {
    pub planner: PlannerOptions<N>,
    pub redundant: RedundantOptions,
}

struct YawNode<const N: usize> {
    yaw: f64,
    q: SRobotQ<N, f64>,
    cost: f64,
}

/// Multi-anchor yaw planner over a single conditioned run.
#[derive(Clone, Debug)]
pub struct RedundantLinearPlanner<'a, const N: usize, FK> {
    fk: &'a FK,
}

impl<'a, const N: usize, FK> RedundantLinearPlanner<'a, N, FK>
where
    FK: ContinuousFKChain<N, f64> + IkSolver<N, f64>,
{
    pub fn new(fk: &'a FK) -> Self {
        Self { fk }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_run<V: Validator<N, (), f64>>(
        &self,
        run: &CartesianRun,
        planner: &PlannerOptions<N>,
        red: &RedundantOptions,
        validator: &V,
        ctx: &V::Context<'_>,
        seed: Option<&SRobotQ<N, f64>>,
        run_idx: usize,
    ) -> Result<(SRobotPath<N, f64>, RedundantDiagnostic), LinearError> {
        let axis = red.axis.vector();
        let length = run.length();
        let n_dp = ((length / red.dp_ds).ceil() as usize).max(1) + 1;

        let stations = self.build_stations(
            run, red, planner, axis, length, n_dp, validator, ctx, run_idx,
        )?;

        let (coarse_s, coarse_psi) = solve_global(&stations, red, planner, length, n_dp)
            .ok_or(LinearError::NoContinuousTrack { run: run_idx })?;

        self.refine(
            run,
            planner,
            axis,
            length,
            &coarse_s,
            &coarse_psi,
            validator,
            ctx,
            seed,
            run_idx,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_stations<V: Validator<N, (), f64>>(
        &self,
        run: &CartesianRun,
        red: &RedundantOptions,
        planner: &PlannerOptions<N>,
        axis: DVec3,
        length: f64,
        n_dp: usize,
        validator: &V,
        ctx: &V::Context<'_>,
        run_idx: usize,
    ) -> Result<Vec<Vec<YawNode<N>>>, LinearError> {
        let samples = red.yaw_samples.max(1);
        let yaw_at = |m: usize| -> f64 {
            if samples <= 1 {
                0.5 * (red.yaw_window.0 + red.yaw_window.1)
            } else {
                red.yaw_window.0
                    + (red.yaw_window.1 - red.yaw_window.0) * m as f64 / (samples - 1) as f64
            }
        };

        let mut stations = Vec::with_capacity(n_dp);
        for k in 0..n_dp {
            let s = (k as f64 * red.dp_ds).min(length);
            let ref_pose = run.eval(s);
            let ref_rot = DQuat::from_mat3(&ref_pose.matrix3);
            let mut nodes = Vec::with_capacity(samples * 4);
            let mut had_ik = false;
            for m in 0..samples {
                let psi = yaw_at(m);
                let rot = ref_rot * DQuat::from_axis_angle(axis, psi);
                let target = DAffine3::from_rotation_translation(rot, ref_pose.translation);
                if let IkOutcome::Solved(sols) = self.fk.ik(target)? {
                    had_ik |= !sols.is_empty();
                    for q in sols {
                        if validator.validate(q, ctx).is_err() {
                            continue;
                        }
                        let w = self.fk.manipulability(&q).map_err(LinearError::Deke)?;
                        nodes.push(YawNode {
                            yaw: psi,
                            q,
                            cost: planner.manip_weight / (w + 1e-9),
                        });
                    }
                }
            }
            if nodes.is_empty() {
                // No yaw at this station is both reachable and collision-free.
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
        axis: DVec3,
        length: f64,
        coarse_s: &[f64],
        coarse_psi: &[f64],
        validator: &V,
        ctx: &V::Context<'_>,
        seed: Option<&SRobotQ<N, f64>>,
        run_idx: usize,
    ) -> Result<(SRobotPath<N, f64>, RedundantDiagnostic), LinearError> {
        let n_fine = ((length / planner.sample_ds).ceil() as usize).max(1) + 1;
        let mut fine: Vec<SRobotQ<N, f64>> = Vec::with_capacity(n_fine);
        let mut min_manip = f64::INFINITY;
        let mut yaw_lo = f64::INFINITY;
        let mut yaw_hi = f64::NEG_INFINITY;
        let mut prev: Option<(SRobotQ<N, f64>, f64)> = None;

        for i in 0..n_fine {
            let s = (i as f64 * planner.sample_ds).min(length);
            let psi = interp(coarse_s, coarse_psi, s);
            yaw_lo = yaw_lo.min(psi);
            yaw_hi = yaw_hi.max(psi);
            let ref_pose = run.eval(s);
            let rot = DQuat::from_mat3(&ref_pose.matrix3) * DQuat::from_axis_angle(axis, psi);
            let target = DAffine3::from_rotation_translation(rot, ref_pose.translation);
            let raw = match self.fk.ik(target)? {
                IkOutcome::Solved(sols) if !sols.is_empty() => sols,
                _ => return Err(LinearError::Unreachable { run: run_idx, s }),
            };
            let sols: Vec<SRobotQ<N, f64>> = raw
                .into_iter()
                .filter(|q| validator.validate(*q, ctx).is_ok())
                .collect();
            if sols.is_empty() {
                return Err(LinearError::Obstructed { run: run_idx, s });
            }

            let q = match prev {
                Some((pq, _)) => sols
                    .into_iter()
                    .min_by(|a, b| pq.distance(a).total_cmp(&pq.distance(b)))
                    .unwrap(),
                None => match seed {
                    Some(s) => sols
                        .into_iter()
                        .filter(|q| !is_reconfiguration(s, q, f64::INFINITY, planner))
                        .min_by(|a, b| s.distance(a).total_cmp(&s.distance(b)))
                        .ok_or(LinearError::NoContinuousTrack { run: run_idx })?,
                    None => {
                        let mut best = sols[0];
                        let mut best_w = -1.0;
                        for q in sols {
                            let w = self.fk.manipulability(&q).map_err(LinearError::Deke)?;
                            if w > best_w {
                                best_w = w;
                                best = q;
                            }
                        }
                        best
                    }
                },
            };

            if let Some((pq, ps)) = prev {
                let dsf = (s - ps).max(1e-12);
                if is_reconfiguration(&pq, &q, dsf, planner) {
                    return Err(LinearError::NoContinuousTrack { run: run_idx });
                }
            }

            min_manip = min_manip.min(self.fk.manipulability(&q).map_err(LinearError::Deke)?);
            fine.push(q);
            prev = Some((q, s));
        }

        let path = SRobotPath::try_new(fine).map_err(LinearError::from)?;
        Ok((
            path,
            RedundantDiagnostic {
                samples: n_fine,
                min_manipulability: min_manip,
                yaw_range: (yaw_lo, yaw_hi),
            },
        ))
    }
}

impl<'a, const N: usize, FK> Planner<N, f64> for RedundantLinearPlanner<'a, N, FK>
where
    FK: ContinuousFKChain<N, f64> + IkSolver<N, f64>,
{
    type Diagnostic = RedundantDiagnostic;
    type Config = RedundantConfig<N>;
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
            &config.redundant,
            validator,
            ctx,
            None,
            0,
        ) {
            Ok((path, diag)) => (Ok(path), diag),
            Err(e) => (
                Err(e.into()),
                RedundantDiagnostic {
                    samples: 0,
                    min_manipulability: 0.0,
                    yaw_range: (0.0, 0.0),
                },
            ),
        }
    }
}

/// The single global DP over the yaw grid. Returns the coarse `(s[], ψ[])` of the
/// minimum-cost continuous yaw+branch track, or `None` if none exists.
fn solve_global<const N: usize>(
    stations: &[Vec<YawNode<N>>],
    red: &RedundantOptions,
    planner: &PlannerOptions<N>,
    length: f64,
    n_dp: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let ds_at = |k: usize| {
        ((k as f64 * red.dp_ds).min(length) - ((k - 1) as f64 * red.dp_ds).min(length)).max(1e-9)
    };
    let layer_sizes: Vec<usize> = stations.iter().map(Vec::len).collect();
    let (chosen, _) = ladder_dp(
        &layer_sizes,
        |k, i| stations[k][i].cost,
        |k, p, i| {
            let na = &stations[k - 1][p];
            let nb = &stations[k][i];
            let dyaw = (nb.yaw - na.yaw).abs();
            let ds = ds_at(k);
            if dyaw > red.max_yaw_step || is_reconfiguration(&na.q, &nb.q, ds, planner) {
                None
            } else {
                Some(na.q.distance(&nb.q) + red.yaw_rate_weight * dyaw / ds)
            }
        },
    )?;

    let s = (0..n_dp)
        .map(|k| (k as f64 * red.dp_ds).min(length))
        .collect();
    let psi = chosen
        .iter()
        .enumerate()
        .map(|(k, &i)| stations[k][i].yaw)
        .collect();
    Some((s, psi))
}
