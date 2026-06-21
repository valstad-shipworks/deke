//! Stage A→B→C orchestration plus the `deke-types` `Planner`/`Retimer` trait
//! wiring.

use std::time::Duration;

use deke_types::glam::DAffine3;
use deke_types::{
    ContinuousFKChain, DekeError, DekeResult, IkSolver, Planner, Retimer, SRobotPath, SRobotQ,
    SRobotQLike, SRobotTraj, Validator,
};

use crate::constraints::{FollowConfig, LinearConstraints, PlannerOptions};
use crate::diagnostic::{LinearFollowDiagnostic, LinearPlannerDiagnostic, LinearRetimerDiagnostic};
use crate::error::LinearError;
use crate::path::{CartesianRun, condition};
use crate::planner::CartesianLinearPlanner;
use crate::retimer::ConstantSpeedRetimer;

/// End-to-end constant-TCP-speed Cartesian follower.
#[derive(Clone, Debug)]
pub struct LinearFollower<'a, const N: usize, FK> {
    fk: &'a FK,
}

impl<'a, const N: usize, FK> LinearFollower<'a, N, FK>
where
    FK: ContinuousFKChain<N, f64> + IkSolver<N, f64>,
{
    pub fn new(fk: &'a FK) -> Self {
        Self { fk }
    }

    /// Follow `poses` (a Cartesian polyline of full TCP poses) at a constant TCP
    /// speed, splitting at sharp corners and stopping at rest there.
    ///
    /// Every candidate configuration is checked against `validator` *inside* the
    /// planner DP, so the planner routes through collision-free configurations
    /// (and, for a redundant tool axis, rotates the yaw to keep the arm clear).
    /// Pass [`NoopValidator`] + `&()` to plan without obstacle checks. The
    /// stitched output trajectory is re-checked with `validate_motion` as a
    /// backstop.
    pub fn follow<V: Validator<N, (), f64>>(
        &self,
        poses: &[DAffine3],
        cfg: &FollowConfig<N>,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> Result<(SRobotTraj<N, f64>, LinearFollowDiagnostic), LinearError> {
        let runs = condition(poses, &cfg.conditioning)?;
        let planner = CartesianLinearPlanner::new(self.fk);
        let retimer = ConstantSpeedRetimer::new(self.fk);

        let mut all: Vec<SRobotQ<N, f64>> = Vec::new();
        let mut diag = LinearFollowDiagnostic {
            runs: runs.len(),
            ..Default::default()
        };

        let redundant = cfg
            .redundant
            .as_ref()
            .map(|_| crate::redundant::RedundantLinearPlanner::new(self.fk));

        let mut seed: Option<SRobotQ<N, f64>> = None;
        for (i, run) in runs.iter().enumerate() {
            let jpath = match (&redundant, &cfg.redundant) {
                (Some(rp), Some(ropts)) => {
                    let (path, rdiag) =
                        rp.plan_run(run, &cfg.planner, ropts, validator, ctx, seed.as_ref(), i)?;
                    diag.redundant.push(rdiag);
                    path
                }
                _ => {
                    let (path, pdiag) =
                        planner.plan_run(run, &cfg.planner, validator, ctx, seed.as_ref(), i)?;
                    diag.planner.push(pdiag);
                    path
                }
            };
            seed = Some(*jpath.last());
            let (traj, rdiag) = retimer.retime_path(&cfg.constraints, &jpath, i)?;
            let samples = traj.path().iter().copied();
            if all.is_empty() {
                all.extend(samples);
            } else {
                all.extend(samples.skip(1));
            }
            diag.retimer.push(rdiag);
        }

        // Backstop: the retimer interpolates between planned (validated) waypoints,
        // so re-check the stitched trajectory as continuous motion.
        validator
            .validate_motion(&all, ctx)
            .map_err(LinearError::from)?;

        let dt = cfg.constraints.output_dt;
        diag.total_samples = all.len();
        diag.total_duration =
            Duration::from_secs_f64(all.len().saturating_sub(1) as f64 * dt.as_secs_f64());
        let path = SRobotPath::try_new(all).map_err(LinearError::from)?;
        Ok((SRobotTraj::new(dt, path), diag))
    }
}

impl<'a, const N: usize, FK> Planner<N, f64> for CartesianLinearPlanner<'a, N, FK>
where
    FK: ContinuousFKChain<N, f64> + IkSolver<N, f64>,
{
    type Diagnostic = LinearPlannerDiagnostic;
    type Config = PlannerOptions<N>;
    type Waypoints = CartesianRun;

    fn plan<E: Into<DekeError>, V: Validator<N, (), f64>>(
        &self,
        config: &Self::Config,
        waypoints: &Self::Waypoints,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotPath<N, f64>>, Self::Diagnostic) {
        match self.plan_run(waypoints, config, validator, ctx, None, 0) {
            Ok((path, diag)) => (Ok(path), diag),
            Err(e) => (
                Err(e.into()),
                LinearPlannerDiagnostic {
                    samples: 0,
                    min_manipulability: 0.0,
                    total_cost: f64::INFINITY,
                },
            ),
        }
    }
}

impl<'a, const N: usize, FK> Retimer<N, f64> for ConstantSpeedRetimer<'a, N, FK>
where
    FK: ContinuousFKChain<N, f64>,
{
    type Diagnostic = LinearRetimerDiagnostic;
    type Constraints = LinearConstraints<N>;

    fn retime<V: Validator<N, (), f64>>(
        &self,
        constraints: &Self::Constraints,
        path: &SRobotPath<N, f64>,
        validator: &V,
        ctx: &V::Context<'_>,
    ) -> (DekeResult<SRobotTraj<N, f64>>, Self::Diagnostic) {
        match self.retime_path(constraints, path, 0) {
            Ok((traj, diag)) => {
                let samples: Vec<SRobotQ<N, f64>> = traj.path().iter().copied().collect();
                if let Err(e) = validator.validate_motion(&samples, ctx) {
                    return (Err(e), diag);
                }
                (Ok(traj), diag)
            }
            Err(e) => (
                Err(e.into()),
                LinearRetimerDiagnostic {
                    output_samples: 0,
                    duration: Duration::ZERO,
                    arc_length: 0.0,
                    commanded_speed: constraints.tcp_speed,
                    peak_speed: 0.0,
                },
            ),
        }
    }
}

/// A validator that accepts everything — for callers that handle collision
/// checking elsewhere (or not at all).
#[derive(Debug, Clone, Default)]
pub struct NoopValidator<const N: usize>;

impl<const N: usize> Validator<N, (), f64> for NoopValidator<N> {
    type Context<'ctx> = ();

    fn validate<'ctx, E: Into<DekeError>, A: SRobotQLike<N, E, f64>>(
        &self,
        _q: A,
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        Ok(())
    }

    fn validate_motion<'ctx>(
        &self,
        _qs: &[SRobotQ<N, f64>],
        _ctx: &Self::Context<'ctx>,
    ) -> DekeResult<()> {
        Ok(())
    }
}
