//! Stage B — resolve a conditioned [`CartesianRun`] into a continuous joint-space
//! path by analytic-IK branch tracking.
//!
//! Each densely-sampled pose is inverted with the chain's analytic IK, which
//! returns every isolated branch already filtered to joint limits — no Jacobian
//! inversion, so it cannot blow up near singularities. A dynamic program over the
//! branch layers chooses a globally continuous, well-conditioned track: a
//! manipulability term steers away from singular configurations and an edge term
//! penalises joint motion while rejecting discontinuous wrist flips.

use deke_types::{
    ContinuousFKChain, DekeError, DekeResult, IkOutcome, IkSolver, Planner, SRobotPath, SRobotQ,
    Validator,
};

use crate::constraints::PlannerOptions;
use crate::diagnostic::LinearPlannerDiagnostic;
use crate::error::LinearError;
use crate::path::CartesianRun;
use crate::util::ladder_dp;

/// Backstop on the per-run sample count so a misconfigured `sample_ds` cannot
/// drive an unbounded allocation (validation already rejects non-positive values).
const MAX_SAMPLES: usize = 2_000_000;

/// Analytic-IK branch-tracking planner over a single conditioned run.
#[derive(Clone, Debug)]
pub struct CartesianLinearPlanner<'a, const N: usize, FK> {
    fk: &'a FK,
}

impl<'a, const N: usize, FK> CartesianLinearPlanner<'a, N, FK>
where
    FK: ContinuousFKChain<N, f64> + IkSolver<N, f64>,
{
    pub fn new(fk: &'a FK) -> Self {
        Self { fk }
    }

    /// Plan a continuous joint path through `run`, sampled every
    /// `opts.sample_ds` of arc length. Every candidate configuration is checked
    /// against `validator`; rejected ones are dropped from the DP, so the planner
    /// routes through collision-free IK branches.
    ///
    /// `seed` anchors the first sample to a configuration the run must continue
    /// from (the previous run's final pose), so stitched runs stay continuous in
    /// joint space across a corner.
    pub(crate) fn plan_run<V: Validator<N, (), f64>>(
        &self,
        run: &CartesianRun,
        opts: &PlannerOptions<N>,
        validator: &V,
        ctx: &V::Context<'_>,
        seed: Option<&SRobotQ<N, f64>>,
        run_idx: usize,
    ) -> Result<(SRobotPath<N, f64>, LinearPlannerDiagnostic), LinearError> {
        let length = run.length();
        if !(opts.sample_ds.is_finite() && opts.sample_ds > 0.0) {
            return Err(LinearError::InvalidConfig { field: "sample_ds" });
        }
        let n = ((length / opts.sample_ds).ceil() as usize).clamp(1, MAX_SAMPLES) + 1;

        // Per sample: analytic IK + manipulability node cost for each branch that
        // passes the validator. `(q, node_cost, manipulability)`.
        let mut layers: Vec<Vec<(SRobotQ<N, f64>, f64, f64)>> = Vec::with_capacity(n);
        for i in 0..n {
            let s = (i as f64 * opts.sample_ds).min(length);
            let pose = run.eval(s);
            let sols = match self.fk.ik(pose)? {
                IkOutcome::Solved(sols) if !sols.is_empty() => sols,
                _ => return Err(LinearError::Unreachable { run: run_idx, s }),
            };
            let mut nodes = Vec::with_capacity(sols.len());
            for q in sols {
                if validator.validate(q, ctx).is_err() {
                    continue;
                }
                let w = self.fk.manipulability(&q).map_err(LinearError::Deke)?;
                nodes.push((q, opts.manip_weight / (w + 1e-9), w));
            }
            if nodes.is_empty() {
                return Err(LinearError::Obstructed { run: run_idx, s });
            }
            layers.push(nodes);
        }

        let min_manip = layers
            .iter()
            .flat_map(|l| l.iter().map(|&(_, _, w)| w))
            .fold(f64::INFINITY, f64::min);

        let ds_at =
            |k: usize| (k as f64 * opts.sample_ds).min(length) - ((k - 1) as f64 * opts.sample_ds);
        let layer_sizes: Vec<usize> = layers.iter().map(Vec::len).collect();
        let (chosen, total) = ladder_dp(
            &layer_sizes,
            |k, i| {
                let (q0, nc, _) = layers[k][i];
                match seed {
                    Some(s) if k == 0 => {
                        if is_reconfiguration(s, &q0, f64::INFINITY, opts) {
                            f64::INFINITY
                        } else {
                            nc + s.distance(&q0)
                        }
                    }
                    _ => nc,
                }
            },
            |k, pi, ci| {
                let qp = layers[k - 1][pi].0;
                let qc = layers[k][ci].0;
                if is_reconfiguration(&qp, &qc, ds_at(k), opts) {
                    None
                } else {
                    Some(qp.distance(&qc))
                }
            },
        )
        .ok_or(LinearError::NoContinuousTrack { run: run_idx })?;

        let track: Vec<SRobotQ<N, f64>> = chosen
            .iter()
            .enumerate()
            .map(|(k, &i)| layers[k][i].0)
            .collect();
        let path = SRobotPath::try_new(track).map_err(LinearError::from)?;
        Ok((
            path,
            LinearPlannerDiagnostic {
                samples: n,
                min_manipulability: min_manip,
                total_cost: total,
            },
        ))
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

/// Is the joint move `a → b` over Cartesian distance `ds` a reconfiguration?
///
/// Trips on the absolute continuity guard (`max_branch_jump`) or, when the
/// velocity test is enabled, when executing the move at `max_velocity` would
/// drive any joint past `reconfig_vel_fraction` of its velocity limit — at weld
/// speeds, the signature of a singularity or wrist flip.
pub(crate) fn is_reconfiguration<const N: usize>(
    a: &SRobotQ<N, f64>,
    b: &SRobotQ<N, f64>,
    ds: f64,
    opts: &PlannerOptions<N>,
) -> bool {
    if (*a - *b).linf_norm() > opts.max_branch_jump {
        return true;
    }
    if opts.max_velocity > 0.0 && ds > 1e-12 {
        for j in 0..N {
            let vmax = opts.joint_v_max.0[j];
            if vmax.is_finite() {
                let req = (b.0[j] - a.0[j]).abs() * opts.max_velocity / ds;
                if req > opts.reconfig_vel_fraction * vmax {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::is_reconfiguration;
    use crate::constraints::PlannerOptions;
    use deke_types::SRobotQ;

    fn opts(max_velocity: f64, v_max: f64) -> PlannerOptions<2> {
        PlannerOptions {
            sample_ds: 2e-3,
            manip_weight: 1.0,
            max_branch_jump: 100.0, // disable the absolute guard for these cases
            max_velocity,
            joint_v_max: SRobotQ::splat(v_max),
            reconfig_vel_fraction: 0.9,
        }
    }

    #[test]
    fn velocity_criterion_flags_fast_joint() {
        let a = SRobotQ::<2, f64>::from_array([0.0, 0.0]);
        // 0.5 rad over 2 mm at 0.01 m/s → 2.5 rad/s required on joint 0.
        let b = SRobotQ::<2, f64>::from_array([0.5, 0.0]);
        // 2.5 > 0.9 * 2.0 = 1.8 → reconfiguration.
        assert!(is_reconfiguration(&a, &b, 2e-3, &opts(0.01, 2.0)));
        // A gentle move: 0.01 rad over 2 mm → 0.05 rad/s ≪ 1.8 → fine.
        let c = SRobotQ::<2, f64>::from_array([0.01, 0.0]);
        assert!(!is_reconfiguration(&a, &c, 2e-3, &opts(0.01, 2.0)));
    }

    #[test]
    fn velocity_criterion_disabled_when_speed_zero() {
        let a = SRobotQ::<2, f64>::from_array([0.0, 0.0]);
        let b = SRobotQ::<2, f64>::from_array([0.5, 0.0]);
        assert!(!is_reconfiguration(&a, &b, 2e-3, &opts(0.0, 2.0)));
    }
}
